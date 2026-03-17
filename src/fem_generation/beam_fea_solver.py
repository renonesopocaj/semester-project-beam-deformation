import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from dolfin import *
from fenics import project
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ufl import lhs, rhs
from src.fem_generation.fem2torch_helpers import (extract_edges, evaluate_u_at_vertices, evaluate_force_at_vertices,
                                                  evaluate_stress_at_vertices, evaluate_velocity_at_vertices,
                                                  evaluate_acceleration_at_vertices)
from src.fem_generation.visualization_fea import plot_check, make_beam_gif

# Use a non-interactive backend so plots can be generated in headless environments.
matplotlib.use("agg")


def extract_node_masses(u_, du, rho):
    """
    Compute node-wise lumped masses in 3D and reorder them so they match the
    vertex ordering given by mesh.coordinates().

    Args:
        u_: dolfin.TestFunction
            Test function defined on a vector-valued function space.
        du: dolfin.TrialFunction
            Trial function defined on the same vector-valued function space.
        rho: dolfin.Constant
            Material density used in the mass matrix.
        return: np.ndarray
            Array of shape (num_vertices, 3) containing lumped mass values
            for x, y, z displacement components at each mesh vertex.
    """

    # Recover the vector function space from the test function.
    V = u_.function_space()

    # Build and assemble the consistent mass matrix:
    # M = integral(rho * u . du dx)
    m_form = rho * inner(u_, du) * dx
    M = assemble(m_form)

    # Extract the diagonal entries of the mass matrix.
    # These diagonal terms are used as lumped masses per DOF.
    diag = Vector()
    M.init_vector(diag, 0)
    M.get_diagonal(diag)
    lumps_per_dof = diag.get_local()

    mesh = V.mesh()
    nverts = mesh.num_vertices()

    # Allocate output array storing one lumped mass per component per vertex.
    lumps_vertex = np.zeros((nverts, 3), dtype=float)

    # Reorder each displacement component independently.
    # In FEniCS, the global DOF ordering is not always the same as the
    # vertex ordering from mesh.coordinates().
    for comp_i in range(3):
        # Collapse the selected subspace into a scalar space.
        V_sub = V.sub(comp_i).collapse()

        # Map scalar-space DOFs to mesh vertex indices.
        d2v = dof_to_vertex_map(V_sub)

        sub_dim = V_sub.dim()
        lumps_sub = np.zeros(sub_dim, dtype=float)

        # Gather the diagonal entries corresponding to the current component.
        # Since the vector space has 3 components, global indexing is assumed
        # to be interleaved as [x0, y0, z0, x1, y1, z1, ...].
        for sub_dof in range(sub_dim):
            global_dof = sub_dof * 3 + comp_i
            lumps_sub[sub_dof] = lumps_per_dof[global_dof]

        # Reorder the component masses into vertex order.
        lumps_sub_vertex = lumps_sub[d2v]
        lumps_vertex[:, comp_i] = lumps_sub_vertex

    return lumps_vertex


def extract_mesh_bc_nodemask(bc, V):
    """
    Extract a binary node mask describing which mesh vertices are fixed by a Dirichlet boundary condition.

    A node is marked as fixed if the corresponding degree of freedom in the
    first component belongs to the boundary condition.

    Args:
        bc: dolfin.DirichletBC
            Boundary condition applied on the vector function space.
        V: dolfin.FunctionSpace
            Vector function space on which the boundary condition is defined.
        return: torch.Tensor
            Tensor of shape (num_vertices, 1) with float values 0.0 or 1.0, where 1.0 indicates a constrained vertex.
    """

    # Retrieve the constrained DOFs from the boundary condition.
    bc_dofs_dict = bc.get_boundary_values()
    bc_dof_indices = np.array(list(bc_dofs_dict.keys()), dtype=int)

    # Determine the number of vector components.
    # If num_sub_spaces() is zero for some reason, fall back to element value size.
    num_components = V.num_sub_spaces() or V.ufl_element().value_size()

    # Keep only component-0 DOFs so we can identify the corresponding vertices.
    # This assumes that if one component is fixed in a fully clamped BC, the
    # others are fixed too, which is true in this script.
    sub0_dofs = bc_dof_indices[bc_dof_indices % num_components == 0]
    sub0_dofs = sub0_dofs // num_components

    # Collapse subspace 0 to a scalar space and obtain DOF-to-vertex map.
    V_sub0 = V.sub(0).collapse()
    d2v = dof_to_vertex_map(V_sub0)

    # Convert the constrained scalar DOFs into vertex indices.
    pinned_vertices = d2v[sub0_dofs]
    pinned_vertices = np.unique(pinned_vertices)

    # Build a binary array marking fixed vertices.
    nverts = V.mesh().num_vertices()
    is_fixed = np.zeros(nverts, dtype=int)
    is_fixed[pinned_vertices] = 1

    # Return as a column tensor for graph node features.
    node_scalar = torch.tensor(is_fixed, dtype=torch.float).reshape(-1, 1)
    return node_scalar

def save_graphs(lst_graph_tstep, save_path):
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
    print(f"saved graphs(dataloader) to {save_path}")


def fea_simulation(L, W, D, NL, NW, ND, elastic_params, density, damping_params, newmark_params, initial_force_x=0.0,
                   initial_force_y=1.0, initial_force_z=0.0, cutoff_time_factor=1 / 5, total_time=4.0, num_steps=50,
                   mode="train"):
    """
    Run a transient 3D beam finite element simulation and convert each time step
    into a graph object suitable for graph learning workflows.

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
    """
    # Enable FEniCS form compiler optimizations.
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["optimize"] = True

    # Unpack elastic parameters and compute Lamé constants.
    E, nu = elastic_params
    mu = Constant(E / (2.0 * (1.0 + nu)))
    lmbda = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    # Material density
    rho = Constant(density)
    # Rayleigh damping coefficients.
    eta_m_val, eta_k_val = damping_params
    eta_m = Constant(eta_m_val)
    eta_k = Constant(eta_k_val)

    # Generalized-alpha / Newmark-like integration parameters.
    alpha_m_val, alpha_f_val = newmark_params
    alpha_m = Constant(alpha_m_val)
    alpha_f = Constant(alpha_f_val)
    gamma = Constant(0.5 + alpha_f - alpha_m)
    beta = Constant((gamma + 0.5) ** 2 / 4.0)

    # Build a structured 3D box mesh for the beam geometry.
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(L, W, D), NL, NW, ND)
    # Define clamped boundary at x = 0.
    def left(x, on_boundary):
        return near(x[0], 0.0) and on_boundary
    # Define loaded boundary at x = L.
    def right(x, on_boundary):
        return near(x[0], L) and on_boundary

    # Displacement space and stress space.
    V = VectorFunctionSpace(mesh, "CG", 1)
    Vsig = TensorFunctionSpace(mesh, "DG", 0)

    # Convert mesh topology into graph connectivity.
    bidir_edge_index, mesh_connectivity = extract_edges(mesh)

    # Trial and test functions.
    du = TrialFunction(V)
    u_ = TestFunction(V)

    # Unknown displacement and previous time-step state variables.
    u = Function(V, name="Displacement")
    u_old = Function(V)
    v_old = Function(V)
    a_old = Function(V)

    # Mark boundaries to identify the loaded surface.
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    force_boundary = AutoSubDomain(right)
    force_boundary.mark(boundary_subdomains, 3)
    dss = ds(subdomain_data=boundary_subdomains)

    # Apply clamped boundary condition on the left side.
    zero = Constant((0.0, 0.0, 0.0))
    bc = DirichletBC(V, zero, left)

    # Build a vertex-wise binary mask indicating constrained nodes.
    node_scalar_bc_mask = extract_mesh_bc_nodemask(bc, V)

    # Time discretization.
    T = total_time
    Nsteps = num_steps
    dt = Constant(T / Nsteps)

    # External traction parameters.
    p0_x, p0_y, p0_z = initial_force_x, initial_force_y, initial_force_z
    cutoff_Tc = T * cutoff_time_factor

    # Define a time-dependent traction that ramps linearly up to tc and then vanishes.
    p = Expression(("t <= tc ? p0x * t / tc : 0", "t <= tc ? p0y * t / tc : 0", "t <= tc ? p0z * t / tc : 0"),
                   t=0, tc=cutoff_Tc, p0x=p0_x, p0y=p0_y, p0z=p0_z, degree=0)

    def sigma(r):
        """
        Compute linear elastic Cauchy stress from displacement field.

        Args:
            r: ufl expression
                Displacement-like field.
            return: ufl expression
                Stress tensor expression.
        """
        return 2.0 * mu * sym(grad(r)) + lmbda * tr(sym(grad(r))) * Identity(len(r))

    def m(uA, uB):
        """
        Mass bilinear form.

        Args:
            uA: ufl expression
                First vector field.
            uB: ufl expression
                Second vector field.
            return: ufl form
                Mass form rho * <uA, uB>.
        """
        return rho * inner(uA, uB) * dx

    def k(uA, uB):
        """
        Stiffness bilinear form.

        Args:
            uA: ufl expression
                First vector field.
            uB: ufl expression
                Second vector field.
            return: ufl form
                Elastic stiffness form.
        """
        return inner(sigma(uA), sym(grad(uB))) * dx

    def c(uA, uB):
        """
        Rayleigh damping bilinear form.

        Args:
            uA: ufl expression
                First vector field.
            uB: ufl expression
                Second vector field.
            return: ufl form
                Damping form eta_m * M + eta_k * K.
        """
        return eta_m * m(uA, uB) + eta_k * k(uA, uB)

    def Wext(uA):
        """
        External virtual work of the applied traction.

        Args:
            uA: ufl expression
                Test function.
            return: ufl form
                Surface traction work on the marked right boundary.
        """
        return dot(uA, p) * dss(3)

    def update_a(uNEW, uOLD, vOLD, aOLD, ufl=True):
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
            ufl: bool
                Whether to use symbolic FEniCS constants or plain Python floats.
            return: object
                Updated acceleration expression or array.
        """
        if ufl:
            dt_ = dt
            beta_ = beta
        else:
            dt_ = float(dt)
            beta_ = float(beta)

        return (uNEW - uOLD - dt_ * vOLD) / (beta_ * dt_**2) - (1 - 2 * beta_) / (
            2 * beta_
        ) * aOLD

    def update_v(aNEW, uOLD, vOLD, aOLD, ufl=True):
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
            ufl: bool
                Whether to use symbolic FEniCS constants or plain Python floats.
            return: object
                Updated velocity expression or array.
        """
        if ufl:
            dt_ = dt
            gamma_ = gamma
        else:
            dt_ = float(dt)
            gamma_ = float(gamma)

        return vOLD + dt_ * ((1 - gamma_) * aOLD + gamma_ * aNEW)

    def update_fields(uNEW, uOLD, vOLD, aOLD):
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
            return: None
                Updates the old state variables in place.
        """
        # Extract underlying vectors.
        u_vec, u0_vec = uNEW.vector(), uOLD.vector()
        v0_vec, a0_vec = vOLD.vector(), aOLD.vector()

        # Update acceleration and velocity in algebraic form.
        a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

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

    # Build acceleration and velocity update expressions symbolically.
    a_new = update_a(du, u_old, v_old, a_old, ufl=True)
    v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

    # Residual of the semi-discrete dynamic system.
    res = (
        m(avg(a_old, a_new, alpha_m), u_)
        + c(avg(v_old, v_new, alpha_f), u_)
        + k(avg(u_old, du, alpha_f), u_)
        - Wext(u_)
    )

    # Extract left-hand and right-hand side for linear solve at each time step.
    a_form = lhs(res)
    L_form = rhs(res)

    # Assemble system matrix once.
    K, res_assembled = assemble_system(a_form, L_form, bc)

    # Use a direct LU solver with MUMPS backend.
    solver = LUSolver(K, "mumps")
    solver.parameters["symmetric"] = True

    # Allocate time and diagnostic arrays.
    time = np.linspace(0, T, Nsteps + 1)
    u_tip = np.zeros((Nsteps + 1,))
    energies = np.zeros((Nsteps + 1, 4))
    E_damp = 0
    sig = Function(Vsig, name="sigma")

    # Compute lumped masses once because they do not change in time.
    lumped_masses_nodes = extract_node_masses(u_, du, rho)

    def local_project(v, V_, uOut=None):
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
        dv_ = TrialFunction(V_)
        vT_ = TestFunction(V_)

        # Local projection system.
        a_proj = inner(dv_, vT_) * dx
        b_proj = inner(v, vT_) * dx

        loc_solver = LocalSolver(a_proj, b_proj)
        loc_solver.factorize()

        if uOut is None:
            uOut = Function(V_)
            loc_solver.solve_local_rhs(uOut)
            return uOut
        else:
            loc_solver.solve_local_rhs(uOut)
            return

    # Corner points used for optional displacement tracking.
    corners = [(0.0, 0.0, 0.0), (0.0, W, 0.0), (0.0, 0.0, D), (0.0, W, D)]
    u_corners = np.zeros((Nsteps + 1, len(corners)))

    # Store one graph snapshot per time step.
    lst_graph_tstep = []

    # Additional spaces for external force and node-wise stress projection.
    V_p = VectorFunctionSpace(mesh, "CG", 1)
    V_stress_nodes = TensorFunctionSpace(mesh, "CG", 1)
    p_function = Function(V_p)
    # Save undeformed coordinates once.
    initial_coordinates = mesh.coordinates()

    for i in range(Nsteps):
        dt_i = time[i + 1] - time[i]
        t_now = time[i + 1]
        print("Time: ", t_now)

        # Evaluate the force at t_{n+1-alpha_f}.
        p.t = t_now - float(alpha_f) * dt_i

        # Assemble and solve the linear system for displacement.
        res_assembled = assemble(L_form)
        bc.apply(res_assembled)
        solver.solve(K, u.vector(), res_assembled)

        # Update state variables after the displacement solve.
        update_fields(u, u_old, v_old, a_old)

        # Project stress to a discontinuous space and then to node-compatible space.
        local_project(sigma(u), Vsig, sig)
        stress_nodewise = project(sig, V_stress_nodes)

        # Build a boundary-restricted force field for logging / graph storage.
        bcs_force = DirichletBC(V_p, p, boundary_subdomains, 3)
        bcs_force.apply(p_function.vector())

        # Evaluate force at vertices in mesh vertex order.
        force_vertex_array = evaluate_force_at_vertices(p_function, mesh)
        node_force_t = torch.tensor(force_vertex_array, dtype=torch.float)

        # Evaluate displacement and build deformed coordinates.
        disp_evaluated = evaluate_u_at_vertices(u, mesh)
        deformed_coordinates = initial_coordinates + disp_evaluated
        pos_t = torch.tensor(deformed_coordinates, dtype=torch.float)

        # Evaluate velocity and acceleration at vertices.
        vel_values = evaluate_velocity_at_vertices(v_old, mesh)
        vel_t = torch.tensor(vel_values, dtype=torch.float)

        acc_values = evaluate_acceleration_at_vertices(a_old, mesh)
        acc_t = torch.tensor(acc_values, dtype=torch.float)

        # Evaluate stress tensor at vertices and flatten it.
        stress_array = evaluate_stress_at_vertices(stress_nodewise, mesh)
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

        # Reset to actual current time for any later use.
        p.t = t_now

        # Track tip displacement only in serial.
        if MPI.comm_world.size == 1:
            u_tip[i + 1] = u(L, W, 0.0)[1]
            for j, cpt in enumerate(corners):
                u_corners[i + 1, j] = u(*cpt)[1]

        # Compute and accumulate energies.
        E_elas = assemble(0.5 * k(u_old, u_old))
        E_kin = assemble(0.5 * m(v_old, v_old))
        E_damp += dt_i * assemble(c(v_old, v_old))
        E_tot = E_elas + E_kin + E_damp
        energies[i + 1, :] = np.array([E_elas, E_kin, E_damp, E_tot])

    # Create a descriptive name encoding the main simulation parameters.
    simulation_name = (
        f"L{L}_W{W}_D{D}_NL{NL}_NW{NW}_ND{ND}_"
        f"E{E}_nu{nu}_rho{density}_"
        f"em{eta_m_val}_ek{eta_k_val}"
        f"_Pix{initial_force_x}_Piy{initial_force_y}_Piz{initial_force_z}_"
        f"T{total_time}_Tc{cutoff_time_factor * total_time}_Nsteps{num_steps}"
    )

    # Define output folders.
    save_path = "./Results_large_10_pow_3"
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
    plot_check(time, u_tip, energies, plot_file_path)
    save_graphs(lst_graph_tstep, graph_file_path)

    return lst_graph_tstep, save_path_graph, simulation_name


if __name__ == "__main__":
    """
    Entry point for command-line execution.

    The script parses user-provided simulation parameters, runs the transient FEA simulation, saves graph snapshots 
    and diagnostics, and then creates a GIF visualization of the beam deformation and stress field.
    """
    # Create an argument parser describing the available command-line options.
    parser = argparse.ArgumentParser(description="FEA Simulation Parameters")
    # Geometrical and discretization parameters.
    parser.add_argument("--L", type=float, default=1.0, help="Length of the beam (default: 1.0)")
    parser.add_argument("--W", type=float, default=0.1, help="Width of the beam (default: 0.1)")
    parser.add_argument("--D", type=float, default=0.04, help="Depth of the beam (default: 0.04)")
    parser.add_argument("--NL", type=int, default=8, help="Number of elements along length (default: 8)")
    parser.add_argument("--NW", type=int, default=2, help="Number of elements along width (default: 2)")
    parser.add_argument("--ND", type=int, default=2, help="Number of elements along depth (default: 2)")
    # Material properties.
    parser.add_argument("--E", type=float, default=1000.0, help="Young's modulus (default: 1000.0)")
    parser.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio (default: 0.3)")
    parser.add_argument("--rho", type=float, default=1.0, help="Density (default: 1.0)")
    # Rayleigh damping parameters.
    parser.add_argument("--eta_m", type=float, default=0.01, help="Mass proportional damping (default: 0.01)")
    parser.add_argument("--eta_k", type=float, default=0.01, help="Stiffness proportional damping (default: 0.01)",)
    # Generalized-alpha / Newmark parameters.
    parser.add_argument("--alpha_m", type=float, default=0.0, help="Alpha mass (default: 0.0)")
    parser.add_argument("--alpha_f", type=float, default=0.0, help="Alpha force (default: 0.0)")
    # Simulation force and time parameters.
    parser.add_argument("--initial_force_y", type=float, default=1.0, help="Initial force (default: 1.0)",)
    parser.add_argument("--initial_force_x", type=float, default=0.0, help="Initial force in x direction (default: 0.0)",)
    parser.add_argument("--initial_force_z", type=float, default=0.0, help="Initial force in z direction (default: 0.0)")
    parser.add_argument("--cutoff_time_factor", type=float, default=1 / 5, help="Cutoff time factor (default: 0.2)")
    parser.add_argument("--total_time", type=float, default=4.0, help="Total simulation time (default: 4.0)")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of time steps (default: 50)")
    # Output mode used as subfolder name.
    parser.add_argument("--mode", type=str, default="train", help="subfolder name e.g. test,train")
    # Parse arguments from the command line.
    args = parser.parse_args()

    # Extract individual parameters from the parsed namespace.
    L = args.L
    W = args.W
    D = args.D
    NL = args.NL
    NW = args.NW
    ND = args.ND
    E = args.E
    nu = args.nu
    rho = args.rho
    eta_m = args.eta_m
    eta_k = args.eta_k
    alpha_m = args.alpha_m
    alpha_f = args.alpha_f
    initial_force_x = args.initial_force_x
    initial_force_y = args.initial_force_y
    initial_force_z = args.initial_force_z
    cutoff_time_factor = args.cutoff_time_factor
    total_time = args.total_time
    num_steps = args.num_steps
    mode = args.mode

    # Pack related parameters into tuples expected by fea_simulation().
    elastic_params = (E, nu)
    density = rho
    damping_params = (eta_m, eta_k)
    newmark_params = (alpha_m, alpha_f)

    # Run the transient simulation and collect graph snapshots.
    lst_graphs, graph_save_path, simulation_name = fea_simulation(L, W, D, NL, NW, ND, elastic_params, density,
                                                                  damping_params, newmark_params, initial_force_x,
                                                                  initial_force_y, initial_force_z, cutoff_time_factor,
                                                                  total_time, num_steps, mode)
    # Build GIF output path.
    out_gif = os.path.join(graph_save_path, "BEAM_" + simulation_name + ".gif")
    # Render the animated visualization.
    make_beam_gif(lst_graphs, L, W, D, out_gif, fps=4)