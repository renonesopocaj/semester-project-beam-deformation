"""
beam_fea_solver.py
Implementation of a 3D beam finite element solver using FEniCS.
Note the idea is that one dataset has forces only in the y direction, while the "rotated" in various directions.
"""

import argparse
import os

import dolfin as df
import matplotlib
import numpy as np
import torch
from fenics import project
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ufl import lhs, rhs
from src.fem_generation.fem2torch_helpers import (extract_edges, evaluate_u_at_vertices,
                                                  evaluate_velocity_at_vertices,
                                                  evaluate_acceleration_at_vertices,
                                                  evaluate_stress_at_vertices_fast,
                                                  dof_vector_to_vertex_array, extract_mesh_bc_nodemask)
from src.fem_generation.visualization_fea import plot_check, make_beam_gif

# Use a non-interactive backend so plots can be generated in headless environments.
matplotlib.use("agg")
CONSTANT_FOR_CUTOFF_TIME = 4.0

def extract_node_masses(V, rho, dx_lump):
    """
    Extract lumped node masses.  Because the mass matrix is assembled with
    vertex quadrature (dx_lump), it is already diagonal -- get_diagonal
    returns the true row-sum lumped masses.

    Args:
        V: dolfin.FunctionSpace
            Vector function space for displacement.
        rho: dolfin.Constant
            Material density.
        dx_lump: dolfin.Measure
            Vertex-quadrature measure that produces a diagonal mass matrix.
        return: np.ndarray
            Array of shape (num_vertices, 3) containing lumped mass values
            for x, y, z displacement components at each mesh vertex.
    """
    du = df.TrialFunction(V)
    u_ = df.TestFunction(V)
    M = df.assemble(rho * df.inner(du, u_) * dx_lump)
    diag = df.Vector()
    M.init_vector(diag, 0)
    M.get_diagonal(diag)
    return dof_vector_to_vertex_array(diag, V)

def save_graphs(lst_graph_tstep, save_path, verbose):
    """
    Save a list of graph snapshots as a PyTorch DataLoader object.

    Args:
        lst_graph_tstep: list
            List of torch_geometric.data.Data objects, one per time step.
        save_path: str
            Output path for the serialized .pt file.
        return: None
            Saves the graph data to disk.
    """
    # Wrap graphs in a DataLoader so downstream code can load them directly.
    gph_dataloader = DataLoader(lst_graph_tstep, batch_size=1, shuffle=False)
    torch.save(gph_dataloader, save_path)
    if verbose:
        print(f"saved graphs(dataloader) to {save_path}")


def sigma(r, mu, lmbda):
    """
    Compute linear elastic Cauchy stress from displacement field.

    Args:
        r: ufl expression
            Displacement-like field.
        mu: dolfin.Constant
            First Lamé parameter in shear form.
        lmbda: dolfin.Constant
            Second Lamé parameter.
        return: ufl expression
            Stress tensor expression.
    """
    return 2.0 * mu * df.sym(df.grad(r)) + lmbda * df.tr(df.sym(df.grad(r))) * df.Identity(len(r))


def m(uA, uB, rho, dx_lump):
    """
    Mass bilinear form using lumped (vertex-quadrature) integration.

    Args:
        uA: ufl expression
            First vector field.
        uB: ufl expression
            Second vector field.
        rho: dolfin.Constant
            Material density.
        dx_lump: dolfin.Measure
            Vertex-quadrature measure that makes the mass matrix diagonal.
        return: ufl form
            Lumped mass form rho * <uA, uB>.
    """
    return rho * df.inner(uA, uB) * dx_lump


def k(uA, uB, mu, lmbda):
    """
    Stiffness bilinear form.

    Args:
        uA: ufl expression
            First vector field.
        uB: ufl expression
            Second vector field.
        mu: dolfin.Constant
            First Lamé parameter in shear form.
        lmbda: dolfin.Constant
            Second Lamé parameter.
        return: ufl form
            Elastic stiffness form.
    """
    return df.inner(sigma(uA, mu, lmbda), df.sym(df.grad(uB))) * df.dx


def c(uA, uB, eta_m, eta_k, rho, mu, lmbda, dx_lump):
    """
    Rayleigh damping bilinear form.  The mass-proportional part uses
    the lumped mass measure so that damping is consistent with the
    diagonal mass matrix.

    Args:
        uA: ufl expression
            First vector field.
        uB: ufl expression
            Second vector field.
        eta_m: dolfin.Constant
            Mass-proportional damping coefficient.
        eta_k: dolfin.Constant
            Stiffness-proportional damping coefficient.
        rho: dolfin.Constant
            Material density.
        mu: dolfin.Constant
            First Lamé parameter in shear form.
        lmbda: dolfin.Constant
            Second Lamé parameter.
        dx_lump: dolfin.Measure
            Vertex-quadrature measure for the lumped mass part.
        return: ufl form
            Damping form eta_m * M_lump + eta_k * K.
    """
    return eta_m * m(uA, uB, rho, dx_lump) + eta_k * k(uA, uB, mu, lmbda)


def Wext(uA, p, dss):
    """
    External virtual work of the applied traction.

    Args:
        uA: ufl expression
            Test function.
        p: dolfin.Expression
            Applied traction.
        dss: dolfin.Measure
            Boundary measure with marked right boundary.
        return: ufl form
            Surface traction work on the marked right boundary.
    """
    return df.dot(uA, p) * dss(3)


def update_a(uNEW, uOLD, vOLD, aOLD, dt, beta, use_ufl):
    """
    Compute the new acceleration from displacement update.

    Args:
        uNEW: object
            New displacement-like quantity, either UFL object or vector.
        uOLD: object
            Previous displacement-like quantity.
        vOLD: object
            Previous velocity-like quantity.
        aOLD: object
            Previous acceleration-like quantity.
        dt: dolfin.Constant or float
            Time step size.
        beta: dolfin.Constant or float
            Generalized-alpha / Newmark beta parameter.
        use_ufl: bool
            Whether to use symbolic FEniCS constants or plain Python floats.
        return: object
            Updated acceleration expression or array.
    """
    if use_ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)

    return (uNEW - uOLD - dt_ * vOLD) / (beta_ * dt_**2) - (1 - 2 * beta_) / (2 * beta_) * aOLD


def update_v(aNEW, uOLD, vOLD, aOLD, dt, gamma, use_ufl):
    """
    Compute the new velocity from acceleration update.

    Args:
        aNEW: object
            New acceleration-like quantity.
        uOLD: object
            Previous displacement-like quantity.
        vOLD: object
            Previous velocity-like quantity.
        aOLD: object
            Previous acceleration-like quantity.
        dt: dolfin.Constant or float
            Time step size.
        gamma: dolfin.Constant or float
            Generalized-alpha / Newmark gamma parameter.
        use_ufl: bool
            Whether to use symbolic FEniCS constants or plain Python floats.
        return: object
            Updated velocity expression or array.
    """
    if use_ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)

    return vOLD + dt_ * ((1 - gamma_) * aOLD + gamma_ * aNEW)


def update_fields(uNEW, uOLD, vOLD, aOLD, dt, beta, gamma):
    """
    Update the state vectors in place after solving for the new displacement.

    Args:
        uNEW: dolfin.Function
            Newly solved displacement field.
        uOLD: dolfin.Function
            Previous displacement field to overwrite.
        vOLD: dolfin.Function
            Previous velocity field to overwrite.
        aOLD: dolfin.Function
            Previous acceleration field to overwrite.
        dt: dolfin.Constant
            Time step size.
        beta: dolfin.Constant
            Generalized-alpha / Newmark beta parameter.
        gamma: dolfin.Constant
            Generalized-alpha / Newmark gamma parameter.
        return: None
            Updates the old state variables in place.
    """
    # Extract underlying vectors.
    u_vec, u0_vec = uNEW.vector(), uOLD.vector()
    v0_vec, a0_vec = vOLD.vector(), aOLD.vector()

    # Update acceleration and velocity in algebraic form.
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, dt, beta, use_ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, dt, gamma, use_ufl=False)

    # Copy updated values into the FEniCS functions.
    vOLD.vector()[:] = v_vec
    aOLD.vector()[:] = a_vec
    uOLD.vector()[:] = uNEW.vector()


def avg(xOLD, xNEW, alpha):
    """
    Weighted average used in generalized-alpha time integration.

    Args:
        xOLD: object
            Old value.
        xNEW: object
            New value.
        alpha: object
            Weight parameter.
        return: object
            Weighted average alpha * old + (1 - alpha) * new.
    """
    return alpha * xOLD + (1 - alpha) * xNEW


def local_project(v, V_, uOut):
    """
    Perform a local projection onto a target function space.

    Args:
        v: ufl expression
            Quantity to project.
        V_: dolfin.FunctionSpace
            Target function space.
        uOut: dolfin.Function or None
            Optional output function to reuse.
        return: dolfin.Function or None
            Returns a new projected function if uOut is None, otherwise
            writes into uOut and returns None.
    """
    dv_ = df.TrialFunction(V_)
    vT_ = df.TestFunction(V_)

    # Local projection system.
    a_proj = df.inner(dv_, vT_) * df.dx
    b_proj = df.inner(v, vT_) * df.dx

    loc_solver = df.LocalSolver(a_proj, b_proj)
    loc_solver.factorize()

    if uOut is None:
        uOut = df.Function(V_)
        loc_solver.solve_local_rhs(uOut)
        return

    loc_solver.solve_local_rhs(uOut)
    return


def fea_simulation(L, W, D, NL, NW, ND, elastic_params, density, damping_params, newmark_params, initial_force_x,
                   initial_force_y, initial_force_z, cutoff_time_factor, total_time, num_steps, mode, output_root,
                   verbose, first_trajectory_debug, linear_solver):
    """
    Run a transient 3D beam finite element simulation and convert each time step
    into a graph object suitable for graph learning workflows.
    =0.0
    =1.0
    =0.0
    =1 / 5
    =4.0
    =50,


    Args:
        L: float
            Beam length along x.
        W: float
            Beam width along y.
        D: float
            Beam depth along z.
        NL: int
            Number of mesh elements along the beam length.
        NW: int
            Number of mesh elements along the beam width.
        ND: int
            Number of mesh elements along the beam depth.
        elastic_params: tuple
            Tuple (E, nu) with Young's modulus and Poisson's ratio.
        density: float
            Material density.
        damping_params: tuple
            Tuple (eta_m, eta_k) for Rayleigh damping coefficients.
        newmark_params: tuple
            Tuple (alpha_m, alpha_f) for generalized-alpha time integration.
        initial_force_x: float
            Peak x-component of the ramped external traction.
        initial_force_y: float
            Peak y-component of the ramped external traction.
        initial_force_z: float
            Peak z-component of the ramped external traction.
        cutoff_time_factor: float
            Fraction of the total time during which the force ramps up.
        total_time: float
            Final simulation time.
        num_steps: int
            Number of time steps.
        mode: str
            Subfolder name used to store outputs.
        return: (list, str, str)
            A tuple containing:
            - list of graph snapshots for all time steps
            - directory where graph data is stored
            - simulation name string used in output filenames
        verbose: bool
            enable verbose printing
        output_root: str
            output root
        first_trajectory_debug: bool
            when True, print average force / acc / vel at t=0, t=1, t=2
        linear_solver: str
            "mumps" for direct LU (MUMPS), "krylov" for CG + AMG.
    """
    # Enable FEniCS form compiler optimizations.
    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.parameters["form_compiler"]["optimize"] = True

    # Unpack elastic parameters and compute Lamé constants.
    E, nu = elastic_params
    mu = df.Constant(E / (2.0 * (1.0 + nu)))
    lmbda = df.Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    # Material density
    rho = df.Constant(density)
    # Rayleigh damping coefficients.
    eta_m_val, eta_k_val = damping_params
    eta_m = df.Constant(eta_m_val)
    eta_k = df.Constant(eta_k_val)

    # Generalized-alpha / Newmark-like integration parameters.
    alpha_m_val, alpha_f_val = newmark_params
    alpha_m = df.Constant(alpha_m_val)
    alpha_f = df.Constant(alpha_f_val)
    gamma = df.Constant(0.5 + alpha_f - alpha_m)
    beta = df.Constant((gamma + 0.5) ** 2 / 4.0)

    # Build a structured 3D box mesh for the beam geometry.
    mesh = df.BoxMesh(df.Point(0.0, 0.0, 0.0), df.Point(L, W, D), NL, NW, ND)

    # Define clamped boundary at x = 0.
    def left(x, on_boundary):
        return df.near(x[0], 0.0) and on_boundary

    # Define loaded boundary at x = L.
    def right(x, on_boundary):
        return df.near(x[0], L) and on_boundary

    # Displacement space and stress space.
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    Vsig = df.TensorFunctionSpace(mesh, "DG", 0)

    # Lumped-mass quadrature rule (vertex scheme for CG-1).
    # This makes the mass matrix diagonal so that the per-node equation
    #   M_i * a_i  =  f_ext_i  -  f_int_i  -  f_damp_i
    # holds exactly for every node i.
    dx_lump = df.dx(scheme="vertex", metadata={"degree": 1, "representation": "quadrature"})

    # Convert mesh topology into graph connectivity.
    bidir_edge_index, mesh_connectivity = extract_edges(mesh)

    # Trial and test functions.
    du = df.TrialFunction(V)
    u_ = df.TestFunction(V)

    # Unknown displacement and previous time-step state variables.
    u = df.Function(V, name="Displacement")
    u_old = df.Function(V)
    v_old = df.Function(V)
    a_old = df.Function(V)

    # Mark boundaries to identify the loaded surface.
    boundary_subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    force_boundary = df.AutoSubDomain(right)
    force_boundary.mark(boundary_subdomains, 3)
    dss = df.ds(subdomain_data=boundary_subdomains)

    # Apply clamped boundary condition on the left side.
    zero = df.Constant((0.0, 0.0, 0.0))
    bc = df.DirichletBC(V, zero, left)

    # Build a vertex-wise binary mask indicating constrained nodes.
    node_scalar_bc_mask = extract_mesh_bc_nodemask(bc, V)

    # Time discretization.
    T = total_time
    Nsteps = num_steps
    dt = df.Constant(T / Nsteps)

    # External traction parameters.
    p0_x, p0_y, p0_z = initial_force_x, initial_force_y, initial_force_z
    cutoff_Tc = CONSTANT_FOR_CUTOFF_TIME * cutoff_time_factor

    # Define a time-dependent traction that ramps linearly up to tc and then vanishes.
    p = df.Expression(("t <= tc ? p0x * t / tc : 0", "t <= tc ? p0y * t / tc : 0", "t <= tc ? p0z * t / tc : 0"),
                      t=0, tc=cutoff_Tc, p0x=p0_x, p0y=p0_y, p0z=p0_z, degree=0)

    # Build acceleration and velocity update expressions symbolically.
    a_new = update_a(du, u_old, v_old, a_old, dt, beta, use_ufl=True)
    v_new = update_v(a_new, u_old, v_old, a_old, dt, gamma, use_ufl=True)

    # Residual of the semi-discrete dynamic system (mass and damping use lumped dx_lump).
    res = (m(avg(a_old, a_new, alpha_m), u_, rho, dx_lump)
           + c(avg(v_old, v_new, alpha_f), u_, eta_m, eta_k, rho, mu, lmbda, dx_lump)
           + k(avg(u_old, du, alpha_f), u_, mu, lmbda)
           - Wext(u_, p, dss))

    # Extract left-hand and right-hand side for linear solve at each time step.
    a_form = lhs(res)
    L_form = rhs(res)

    # Assemble system matrix once.
    K, res_assembled = df.assemble_system(a_form, L_form, bc)

    if linear_solver == "mumps":
        solver = df.LUSolver(K, "mumps")
        solver.parameters["symmetric"] = True
    elif linear_solver == "krylov":
        solver = df.KrylovSolver("cg", "amg")
        solver.parameters["absolute_tolerance"] = 1e-7
        solver.parameters["relative_tolerance"] = 1e-5
        solver.parameters["maximum_iterations"] = 1000
        solver.set_operator(K)
    else:
        raise ValueError("linear_solver must be 'mumps' or 'krylov', got %r" % (linear_solver,))

    # Allocate time and diagnostic arrays.
    time = np.linspace(0, T, Nsteps + 1)
    u_tip = np.zeros((Nsteps + 1,))
    energies = np.zeros((Nsteps + 1, 4))
    E_damp = 0
    sig = df.Function(Vsig, name="sigma")

    # Compute lumped masses once because they do not change in time.
    lumped_masses_nodes = extract_node_masses(V, rho, dx_lump)

    # Corner points used for optional displacement tracking.
    corners = [(0.0, 0.0, 0.0), (0.0, W, 0.0), (0.0, 0.0, D), (0.0, W, D)]
    u_corners = np.zeros((Nsteps + 1, len(corners))) # only assigned (later), never used

    # Store one graph snapshot per time step.
    lst_graph_tstep = []

    # Node-wise stress projection space.
    V_stress_nodes = df.TensorFunctionSpace(mesh, "CG", 1)
    # Save undeformed coordinates once.
    initial_coordinates = mesh.coordinates()

    # Record t = 0 state (all zeros: beam is at rest).
    nverts = mesh.num_vertices()
    graph_t0 = Data(
        x=torch.tensor(initial_coordinates, dtype=torch.float),
        edge_index=bidir_edge_index,
        x_vel_t=torch.zeros((nverts, 3), dtype=torch.float),
        y_acc_t=torch.zeros((nverts, 3), dtype=torch.float),
        y_sig_t=torch.zeros((nverts, 9), dtype=torch.float),
        x_node_force=torch.zeros((nverts, 3), dtype=torch.float),
        x_bc=node_scalar_bc_mask,
        y_node_lumped_masses=torch.tensor(lumped_masses_nodes, dtype=torch.float),
        x_element_connectivity=mesh_connectivity,
        time=torch.tensor([0.0], dtype=torch.float),
    )
    lst_graph_tstep.append(graph_t0)

    if first_trajectory_debug:
        print("=== DEBUG t=0 ===")
        print(f"  avg force: {np.zeros(3)}")
        print(f"  avg acc:   {np.zeros(3)}")
        print(f"  avg vel:   {np.zeros(3)}")

    for i in range(Nsteps):
        dt_i = time[i + 1] - time[i]
        t_now = time[i + 1]
        if verbose:
            print("Time: ", t_now)

        # Evaluate the force at t_{n+1-alpha_f}.
        p.t = t_now - float(alpha_f) * dt_i

        # Assemble and solve the linear system for displacement.
        res_assembled = df.assemble(L_form)
        bc.apply(res_assembled)
        solver.solve(u.vector(), res_assembled)

        # Update state variables after the displacement solve.
        update_fields(u, u_old, v_old, a_old, dt, beta, gamma)

        # Project stress to a discontinuous space and then to node-compatible space.
        local_project(sigma(u, mu, lmbda), Vsig, sig)
        stress_nodewise = project(sig, V_stress_nodes,
                                  solver_type="cg", preconditioner_type="jacobi")

        # Correct nodal force extraction:
        # Set p.t to t_{n+1}, then assemble the external-work linear form to get the
        # true FE-consistent nodal force vector  f_ext = int N^T p dS.
        p.t = t_now
        f_ext_vec = df.assemble(Wext(u_, p, dss))
        f_ext_nodal = dof_vector_to_vertex_array(f_ext_vec, V)
        node_force_t = torch.tensor(f_ext_nodal, dtype=torch.float)

        # Evaluate displacement and build deformed coordinates.
        disp_evaluated = evaluate_u_at_vertices(u, mesh)
        deformed_coordinates = initial_coordinates + disp_evaluated
        pos_t = torch.tensor(deformed_coordinates, dtype=torch.float)
        # Evaluate velocity and acceleration at vertices.
        vel_values = evaluate_velocity_at_vertices(v_old, mesh)
        vel_t = torch.tensor(vel_values, dtype=torch.float)
        acc_values = evaluate_acceleration_at_vertices(a_old, mesh)
        acc_t = torch.tensor(acc_values, dtype=torch.float)

        # Evaluate stress tensor at vertices (fast vectorized path).
        stress_array = evaluate_stress_at_vertices_fast(stress_nodewise)
        sig_t = torch.tensor(stress_array, dtype=torch.float)

        # Build one PyG graph snapshot for this time step.
        graph = Data(
            x=pos_t,
            edge_index=bidir_edge_index,
            x_vel_t=vel_t,
            y_acc_t=acc_t,
            y_sig_t=sig_t,
            x_node_force=node_force_t,
            x_bc=node_scalar_bc_mask,
            y_node_lumped_masses=torch.tensor(lumped_masses_nodes, dtype=torch.float),
            x_element_connectivity=mesh_connectivity,
            time=torch.tensor([t_now], dtype=torch.float),
        )
        lst_graph_tstep.append(graph)

        if first_trajectory_debug and i < 2:
            print(f"=== DEBUG t={i + 1} (time={t_now:.4f}) ===")
            print(f"  avg force: {np.mean(f_ext_nodal, axis=0)}")
            print(f"  avg acc:   {np.mean(acc_values, axis=0)}")
            print(f"  avg vel:   {np.mean(vel_values, axis=0)}")

        # Track tip displacement only in serial (z-component at the tip).
        if df.MPI.comm_world.size == 1:
            u_tip[i + 1] = u(L, W, 0.0)[2]
            for j, cpt in enumerate(corners):
                u_corners[i + 1, j] = u(*cpt)[2]

        # Compute and accumulate energies.
        E_elas = df.assemble(0.5 * k(u_old, u_old, mu, lmbda))
        E_kin = df.assemble(0.5 * m(v_old, v_old, rho, dx_lump))
        E_damp += dt_i * df.assemble(c(v_old, v_old, eta_m, eta_k, rho, mu, lmbda, dx_lump))
        E_tot = E_elas + E_kin + E_damp
        energies[i + 1, :] = np.array([E_elas, E_kin, E_damp, E_tot])

    # Create a descriptive name encoding the main simulation parameters.
    simulation_name = (f"L{L}_W{W}_D{D}_NL{NL}_NW{NW}_ND{ND}_"
                       f"E{E}_nu{nu}_rho{density}_"
                       f"em{eta_m_val}_ek{eta_k_val}"
                       f"_Pix{initial_force_x}_Piy{initial_force_y}_Piz{initial_force_z}_"
                       f"T{total_time}_Tc{cutoff_time_factor * total_time}_Nsteps{num_steps}")

    # Define output folders.
    save_path = output_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path_graph = os.path.join(save_path, mode, "graphs")
    save_path_plot_check = os.path.join(save_path, mode, "plot_check")
    os.makedirs(save_path_graph, exist_ok=True)
    os.makedirs(save_path_plot_check, exist_ok=True)

    # Build output file paths.
    plot_file_path = os.path.join(save_path_plot_check, f"plot_{simulation_name}.png")
    graph_file_path = os.path.join(save_path_graph, f"graphs{simulation_name}.pt")

    # Save diagnostics and graph dataset.
    plot_check(time, u_tip, energies, plot_file_path, verbose=False)
    save_graphs(lst_graph_tstep, graph_file_path, verbose=False)

    return lst_graph_tstep, save_path_graph, simulation_name