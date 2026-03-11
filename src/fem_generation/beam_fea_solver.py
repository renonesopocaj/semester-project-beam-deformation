import argparse
import os

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from dolfin import *
from fenics import project
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

matplotlib.use("agg")


def extract_edges(mesh):
    # Extract the element-to-node connectivity from the mesh
    mesh_connectivity = mesh.cells()  # Get element-to-node mapping
    edges = set()
    for element in mesh_connectivity:
        for i in range(len(element)):
            for j in range(i + 1, len(element)):
                edges.add(tuple(sorted((element[i], element[j]))))
    edge_index = list(edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    bidir_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return bidir_edge_index, mesh_connectivity


def extract_node_masses(u_, du, rho):
    """
    Compute node-wise lumped masses in 3D, re-ordered
    to match the same vertex indexing as mesh.coordinates().
    """
    import numpy as np
    from dolfin import dof_to_vertex_map

    # Get the VectorFunctionSpace from u_
    V = u_.function_space()

    # 1) Assemble the mass form
    m_form = rho * inner(u_, du) * dx
    M = assemble(m_form)

    # 2) Extract the diagonal lumps in global DOF order
    diag = Vector()
    M.init_vector(diag, 0)
    M.get_diagonal(diag)
    lumps_per_dof = diag.get_local()  # shape => 3 * num_vertices

    mesh = V.mesh()
    nverts = mesh.num_vertices()

    # We'll build lumps_vertex => (nverts,3)
    lumps_vertex = np.zeros((nverts, 3), dtype=float)

    # 3) For each subcomponent 0..2, reorder lumps for that subspace
    for comp_i in range(3):
        # Collapse subspace i => a scalar function space
        V_sub = V.sub(comp_i).collapse()
        d2v = dof_to_vertex_map(V_sub)  # reorders DOFs => vertex index

        sub_dim = V_sub.dim()
        lumps_sub = np.zeros(sub_dim, dtype=float)

        # Fill lumps_sub from lumps_per_dof
        for sub_dof in range(sub_dim):
            global_dof = sub_dof * 3 + comp_i
            lumps_sub[sub_dof] = lumps_per_dof[global_dof]

        # Reorder lumps_sub => vertex order
        lumps_sub_vertex = lumps_sub[d2v]
        lumps_vertex[:, comp_i] = lumps_sub_vertex

    return lumps_vertex


def extract_mesh_bc(bc, V):
    """
    Return a (num_vertices, 1) torch tensor of 0/1 indicating
    which vertices are pinned (fully constrained in all components).
    """
    import numpy as np
    from dolfin import dof_to_vertex_map

    # All dofs from the BC
    bc_dofs_dict = bc.get_boundary_values()
    bc_dof_indices = np.array(list(bc_dofs_dict.keys()), dtype=int)

    num_components = V.num_sub_spaces() or V.ufl_element().value_size()

    # Filter to subcomponent=0 DOFs to identify the vertex
    sub0_dofs = bc_dof_indices[bc_dof_indices % num_components == 0]
    sub0_dofs = sub0_dofs // num_components

    # Collapse to scalar subspace 0
    V_sub0 = V.sub(0).collapse()
    d2v = dof_to_vertex_map(V_sub0)

    pinned_vertices = d2v[sub0_dofs]
    pinned_vertices = np.unique(pinned_vertices)

    nverts = V.mesh().num_vertices()
    is_fixed = np.zeros(nverts, dtype=int)
    is_fixed[pinned_vertices] = 1

    node_scalar = torch.tensor(is_fixed, dtype=torch.float).reshape(-1, 1)
    return node_scalar


def plot_check(time, u_tip, energies, save_path):
    if MPI.comm_world.rank == 0:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # Tip displacement
        ax[0].plot(time, u_tip)
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Tip displacement")
        ax[0].set_title("Tip Displacement Evolution")

        # Energies
        ax[1].plot(time, energies)
        ax[1].legend(("Elastic", "Kinetic", "Damping", "Total"))
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Energies")
        ax[1].set_title("Energies Evolution")

        plt.tight_layout()
        plt.savefig(save_path)
        print("plot saved to :", save_path)
        # plt.show()


def save_graphs(lst_graph_tstep, save_path):
    gph_dataloader = DataLoader(lst_graph_tstep, batch_size=1, shuffle=False)
    torch.save(gph_dataloader, save_path)
    print(f"saved graphs(dataloader) to {save_path}")


def evaluate_u_at_vertices(u, mesh):
    """
    Evaluate the solution 'u' (VectorFunctionSpace) at each vertex
    in mesh.coordinates().

    Returns a NumPy array (num_vertices, 3) with the displacement
    for each vertex.  (In 3D.)
    """
    coords = mesh.coordinates()
    nverts = coords.shape[0]
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = u(*coords[i])  # Evaluate the solution at vertex i
    return out


def evaluate_velocity_at_vertices(v_fun, mesh):
    """
    Evaluate velocity 'v_fun' (VectorFunction) at each mesh vertex.
    Returns (num_vertices, 3).
    """
    coords = mesh.coordinates()
    nverts = coords.shape[0]
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = v_fun(*coords[i])  # Evaluate v at (x,y,z)
    return out


def evaluate_acceleration_at_vertices(a_fun, mesh):
    # same approach
    coords = mesh.coordinates()
    nverts = coords.shape[0]
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = a_fun(*coords[i])
    return out


# ****** Evaluate force at vertices *******
def evaluate_force_at_vertices(f, mesh):
    coords = mesh.coordinates()
    nverts = coords.shape[0]
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = f(*coords[i])
    return out


def evaluate_stress_at_vertices(stress_fun, mesh):
    """
    Evaluate a 3D TensorFunction 'stress_fun' at each mesh vertex,
    returning shape (num_vertices, 9).

    'stress_fun(*coords)' should return a 3x3 or (3,3) matrix
    in python. We'll flatten row-major for each node.
    """
    coords = mesh.coordinates()
    nverts = coords.shape[0]
    out = np.zeros((nverts, 9), dtype=float)

    for i in range(nverts):
        # Evaluate stress at vertex i => 3x3 matrix
        sigma_3x3 = stress_fun(*coords[i])  # shape (3,3)
        # Flatten in row-major order
        out[i] = np.array(sigma_3x3).flatten()

    return out


def fea_simulation(
    L,
    W,
    D,
    NL,
    NW,
    ND,
    elastic_params,
    density,
    damping_params,
    newmark_params,
    initial_force_x=0.0,
    initial_force_y=1.0,
    initial_force_z=0.0,
    cutoff_time_factor=1 / 5,
    total_time=4.0,
    num_steps=50,
    mode="train",
):
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["optimize"] = True

    # Elastic parameters
    E, nu = elastic_params
    mu = Constant(E / (2.0 * (1.0 + nu)))
    lmbda = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Mass density
    rho = Constant(density)

    # Damping
    eta_m_val, eta_k_val = damping_params
    eta_m = Constant(eta_m_val)
    eta_k = Constant(eta_k_val)

    # Generalized-alpha
    alpha_m_val, alpha_f_val = newmark_params
    alpha_m = Constant(alpha_m_val)
    alpha_f = Constant(alpha_f_val)
    gamma = Constant(0.5 + alpha_f - alpha_m)
    beta = Constant((gamma + 0.5) ** 2 / 4.0)

    # Mesh
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(L, W, D), NL, NW, ND)

    def left(x, on_boundary):
        return near(x[0], 0.0) and on_boundary

    def right(x, on_boundary):
        return near(x[0], L) and on_boundary

    V = VectorFunctionSpace(mesh, "CG", 1)
    Vsig = TensorFunctionSpace(mesh, "DG", 0)

    bidir_edge_index, mesh_connectivity = extract_edges(mesh)

    du = TrialFunction(V)
    u_ = TestFunction(V)

    # PDE unknown fields
    u = Function(V, name="Displacement")
    u_old = Function(V)
    v_old = Function(V)
    a_old = Function(V)

    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    force_boundary = AutoSubDomain(right)
    force_boundary.mark(boundary_subdomains, 3)
    dss = ds(subdomain_data=boundary_subdomains)

    # Clamped BC at x=0
    zero = Constant((0.0, 0.0, 0.0))
    bc = DirichletBC(V, zero, left)

    node_scalar = extract_mesh_bc(bc, V)

    T = total_time
    Nsteps = num_steps
    dt = Constant(T / Nsteps)

    # Force expression (ramp up in y-direction)
    p0_x = initial_force_x
    p0_y = initial_force_y
    p0_z = initial_force_z
    cutoff_Tc = T * cutoff_time_factor
    p = Expression(
        (
            "t <= tc ? p0x * t / tc : 0",
            "t <= tc ? p0y * t / tc : 0",
            "t <= tc ? p0z * t / tc : 0",
        ),
        t=0,
        tc=cutoff_Tc,
        p0x=p0_x,
        p0y=p0_y,
        p0z=p0_z,
        degree=0,
    )

    # Expression(("0", "0","t <= tc ? p0 * t / tc : 0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)
    # PDE forms
    def sigma(r):
        return 2.0 * mu * sym(grad(r)) + lmbda * tr(sym(grad(r))) * Identity(len(r))

    def m(uA, uB):
        return rho * inner(uA, uB) * dx

    def k(uA, uB):
        return inner(sigma(uA), sym(grad(uB))) * dx

    def c(uA, uB):
        return eta_m * m(uA, uB) + eta_k * k(uA, uB)

    def Wext(uA):
        return dot(uA, p) * dss(3)

    def update_a(uNEW, uOLD, vOLD, aOLD, ufl=True):
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
        if ufl:
            dt_ = dt
            gamma_ = gamma
        else:
            dt_ = float(dt)
            gamma_ = float(gamma)
        return vOLD + dt_ * ((1 - gamma_) * aOLD + gamma_ * aNEW)

    def update_fields(uNEW, uOLD, vOLD, aOLD):
        u_vec, u0_vec = uNEW.vector(), uOLD.vector()
        v0_vec, a0_vec = vOLD.vector(), aOLD.vector()
        a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

        vOLD.vector()[:] = v_vec
        aOLD.vector()[:] = a_vec
        uOLD.vector()[:] = uNEW.vector()

    def avg(xOLD, xNEW, alpha):
        return alpha * xOLD + (1 - alpha) * xNEW

    a_new = update_a(du, u_old, v_old, a_old, ufl=True)
    v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
    res = (
        m(avg(a_old, a_new, alpha_m), u_)
        + c(avg(v_old, v_new, alpha_f), u_)
        + k(avg(u_old, du, alpha_f), u_)
        - Wext(u_)
    )

    from ufl import lhs, rhs

    a_form = lhs(res)
    L_form = rhs(res)

    K, res_assembled = assemble_system(a_form, L_form, bc)
    solver = LUSolver(K, "mumps")
    solver.parameters["symmetric"] = True

    time = np.linspace(0, T, Nsteps + 1)
    u_tip = np.zeros((Nsteps + 1,))
    energies = np.zeros((Nsteps + 1, 4))
    E_damp = 0
    sig = Function(Vsig, name="sigma")

    # Lumped masses
    lumped_masses_nodes = extract_node_masses(u_, du, rho)

    # Local projection helper
    def local_project(v, V_, uOut=None):
        dv_ = TrialFunction(V_)
        vT_ = TestFunction(V_)
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

    # corners
    corners = [(0.0, 0.0, 0.0), (0.0, W, 0.0), (0.0, 0.0, D), (0.0, W, D)]
    u_corners = np.zeros((Nsteps + 1, len(corners)))

    lst_graph_tstep = []

    V_p = VectorFunctionSpace(mesh, "CG", 1)
    V_stress_nodes = TensorFunctionSpace(mesh, "CG", 1)

    p_function = Function(V_p)
    initial_coordinates = mesh.coordinates()

    for i in range(Nsteps):
        dt_i = time[i + 1] - time[i]
        t_now = time[i + 1]
        print("Time: ", t_now)

        # Evaluate force expression at t_{n+1 - alpha_f}
        p.t = t_now - float(alpha_f) * dt_i

        # PDE solve
        res_assembled = assemble(L_form)
        bc.apply(res_assembled)
        solver.solve(K, u.vector(), res_assembled)

        # Update old fields
        update_fields(u, u_old, v_old, a_old)

        # Stress
        local_project(sigma(u), Vsig, sig)
        stress_nodewise = project(sig, V_stress_nodes)

        # Build boundary function for logging
        bcs_force = DirichletBC(V_p, p, boundary_subdomains, 3)
        bcs_force.apply(p_function.vector())

        # *** Evaluate that force in vertex order ***
        force_vertex_array = evaluate_force_at_vertices(p_function, mesh)
        node_force_t = torch.tensor(force_vertex_array, dtype=torch.float)

        # Evaluate displacement at each vertex
        disp_evaluated = evaluate_u_at_vertices(u, mesh)
        deformed_coordinates = initial_coordinates + disp_evaluated
        pos_t = torch.tensor(deformed_coordinates, dtype=torch.float)

        vel_values = evaluate_velocity_at_vertices(v_old, mesh)
        vel_t = torch.tensor(vel_values, dtype=torch.float)

        acc_values = evaluate_acceleration_at_vertices(a_old, mesh)
        acc_t = torch.tensor(acc_values, dtype=torch.float)

        stress_array = evaluate_stress_at_vertices(stress_nodewise, mesh)
        sig_t = torch.tensor(stress_array, dtype=torch.float)

        graph = Data(
            x=pos_t,
            edge_index=bidir_edge_index,
            x_vel_t=vel_t,
            y_acc_t=acc_t,
            y_sig_t=sig_t,
            x_node_force=node_force_t,  # <-- Vertex-based force
            x_bc=node_scalar,
            y_node_lumped_masses=torch.tensor(lumped_masses_nodes, dtype=torch.float),
            x_element_connectivity=mesh_connectivity,
            time=torch.tensor([t_now], dtype=torch.float),
        )
        lst_graph_tstep.append(graph)

        p.t = t_now

        # Tip displacement
        if MPI.comm_world.size == 1:
            u_tip[i + 1] = u(L, W, 0.0)[1]
            for j, cpt in enumerate(corners):
                u_corners[i + 1, j] = u(*cpt)[1]

        # energies
        E_elas = assemble(0.5 * k(u_old, u_old))
        E_kin = assemble(0.5 * m(v_old, v_old))
        E_damp += dt_i * assemble(c(v_old, v_old))
        E_tot = E_elas + E_kin + E_damp
        energies[i + 1, :] = np.array([E_elas, E_kin, E_damp, E_tot])

    simulation_name = (
        f"L{L}_W{W}_D{D}_NL{NL}_NW{NW}_ND{ND}_"
        f"E{E}_nu{nu}_rho{density}_"
        f"em{eta_m_val}_ek{eta_k_val}"
        f"_Pix{initial_force_x}_Piy{initial_force_y}_Piz{initial_force_z}_"
        f"T{total_time}_Tc{cutoff_time_factor * total_time}_Nsteps{num_steps}"
    )

    save_path = "./Results_large_10_pow_3"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path_graph = os.path.join(save_path, mode, "graphs")
    save_path_plot_check = os.path.join(save_path, mode, "plot_check")
    os.makedirs(save_path_graph, exist_ok=True)
    os.makedirs(save_path_plot_check, exist_ok=True)

    # Define file paths for plot and graph data using the simulation_name
    plot_file_path = os.path.join(save_path_plot_check, f"plot_{simulation_name}.png")
    graph_file_path = os.path.join(
        save_path_graph, f"graphs{simulation_name}.pt"
    )  # Assuming graph data is saved in .pt format

    # Call the plot_check function and save the plot with the simulation name
    plot_check(time, u_tip, energies, plot_file_path)

    # Save the graphs using the simulation name
    save_graphs(lst_graph_tstep, graph_file_path)
    return lst_graph_tstep, save_path_graph, simulation_name


def compute_von_mises_3d(stress_9):
    sxx = stress_9[:, 0]
    sxy = stress_9[:, 1]
    sxz = stress_9[:, 2]
    syx = stress_9[:, 3]
    syy = stress_9[:, 4]
    syz = stress_9[:, 5]
    szx = stress_9[:, 6]
    szy = stress_9[:, 7]
    szz = stress_9[:, 8]
    vm = 0.5 * (
        (sxx - syy) ** 2
        + (syy - szz) ** 2
        + (szz - sxx) ** 2
        + 6.0 * (sxy**2 + syz**2 + sxz**2)
    )
    return np.sqrt(vm)


def find_boundary_triangles(tetra_connectivity):
    """
    tetra_connectivity: (num_tetra, 4) array of node indices (integers)

    Returns boundary_triangles: (num_faces, 3) array of node indices
    that appear exactly once among all tetra faces => "outside" faces.
    """
    from collections import defaultdict

    face_count = defaultdict(int)
    for tet in tetra_connectivity:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for f in faces:
            face_count[f] += 1
    boundary_faces = [list(face) for face, count in face_count.items() if count == 1]
    return np.array(boundary_faces, dtype=int)


def make_beam_gif(lst_graphs, L, W, D, out_gif, fps=4):
    """
    lst_graphs: list of PyG Data objects, each has:
        .x => (n,3) node positions (deformed)
        .y_sig_t => (n,9) stress
        .x_element_connectivity => (num_tetra,4) tetra connectivity
    out_gif: output filename for the GIF
    fps: frames per second
    """
    import shutil

    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    # 1) Get boundary triangles from first graph
    first_graph = lst_graphs[0]
    cells = first_graph.x_element_connectivity
    if hasattr(cells, "cpu"):
        cells = cells.cpu().numpy()
    boundary_triangles = find_boundary_triangles(cells)

    # 2) Collect bounding box across all frames
    all_coords = np.concatenate([g.x.cpu().numpy() for g in lst_graphs], axis=0)
    x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
    y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    z_min, z_max = np.min(all_coords[:, 2]), np.max(all_coords[:, 2])

    # Compute global von Mises stress range
    vm_min, vm_max = float("inf"), 0.0
    for g in lst_graphs:
        stress_9 = g.y_sig_t.cpu().numpy()
        vm = compute_von_mises_3d(stress_9)
        vm_min = min(vm_min, np.min(vm))
        vm_max = max(vm_max, np.max(vm))

    for i, graph in enumerate(lst_graphs):
        print(f"Rendering frame {i}/{len(lst_graphs) - 1}")

        coords = graph.x.cpu().numpy()  # shape (n,3)
        stress_9 = graph.y_sig_t.cpu().numpy()  # shape (n,9)
        vm = compute_von_mises_3d(stress_9)

        # Build polygon list
        polys = []
        face_values = []
        for tri in boundary_triangles:
            tri_coords = coords[tri]  # (3,3)
            polys.append(tri_coords)
            face_values.append(np.mean(vm[tri]))
        face_values = np.array(face_values)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Color map
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=vm_min, vmax=vm_max)
        face_colors = cmap(norm(face_values))

        polycoll = Poly3DCollection(polys)
        polycoll.set_facecolors(face_colors)
        polycoll.set_edgecolor("none")
        ax.add_collection3d(polycoll)

        # Set axis limits and aspect ratio
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Viewpoint
        ax.view_init(elev=30, azim=45)

        # Add title and colorbar
        ax.set_title(f"Time step {i}")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_values)
        cb = plt.colorbar(mappable, ax=ax, shrink=0.6)
        cb.set_label("von Mises Stress")

        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100)
        plt.close(fig)

        frames.append(imageio.imread(frame_path))

    # Combine frames into a GIF
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f"GIF saved => {out_gif}")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("Done.")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="FEA Simulation Parameters")

    # Geometrical and discretization parameters
    parser.add_argument(
        "--L", type=float, default=1.0, help="Length of the beam (default: 1.0)"
    )
    parser.add_argument(
        "--W", type=float, default=0.1, help="Width of the beam (default: 0.1)"
    )
    parser.add_argument(
        "--D", type=float, default=0.04, help="Depth of the beam (default: 0.04)"
    )
    parser.add_argument(
        "--NL", type=int, default=8, help="Number of elements along length (default: 8)"
    )
    parser.add_argument(
        "--NW", type=int, default=2, help="Number of elements along width (default: 2)"
    )
    parser.add_argument(
        "--ND", type=int, default=2, help="Number of elements along depth (default: 2)"
    )

    # Material properties
    parser.add_argument(
        "--E", type=float, default=1000.0, help="Young's modulus (default: 1000.0)"
    )
    parser.add_argument(
        "--nu", type=float, default=0.3, help="Poisson's ratio (default: 0.3)"
    )
    parser.add_argument("--rho", type=float, default=1.0, help="Density (default: 1.0)")

    # Damping parameters
    parser.add_argument(
        "--eta_m",
        type=float,
        default=0.01,
        help="Mass proportional damping (default: 0.01)",
    )
    parser.add_argument(
        "--eta_k",
        type=float,
        default=0.01,
        help="Stiffness proportional damping (default: 0.01)",
    )

    # Newmark method parameters
    parser.add_argument(
        "--alpha_m", type=float, default=0.0, help="Alpha mass (default: 0.0)"
    )
    parser.add_argument(
        "--alpha_f", type=float, default=0.0, help="Alpha force (default: 0.0)"
    )

    # Simulation parameters
    parser.add_argument(
        "--initial_force_y",
        type=float,
        default=1.0,
        help="Initial force (default: 1.0)",
    )
    parser.add_argument(
        "--initial_force_x",
        type=float,
        default=0.0,
        help="Initial force in x direction (default: 0.0)",
    )
    parser.add_argument(
        "--initial_force_z",
        type=float,
        default=0.0,
        help="Initial force in z direction (default: 0.0)",
    )
    parser.add_argument(
        "--cutoff_time_factor",
        type=float,
        default=1 / 5,
        help="Cutoff time factor (default: 0.2)",
    )
    parser.add_argument(
        "--total_time",
        type=float,
        default=4.0,
        help="Total simulation time (default: 4.0)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of time steps (default: 50)"
    )

    # flag
    parser.add_argument(
        "--mode", type=str, default="train", help="subfolder name e.g. test,train"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Extract parameters from args
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

    # Pack parameters into tuples
    elastic_params = (E, nu)
    density = rho
    damping_params = (eta_m, eta_k)
    newmark_params = (alpha_m, alpha_f)

    # Run simulation
    lst_graphs, graph_save_path, simulation_name = fea_simulation(
        L,
        W,
        D,
        NL,
        NW,
        ND,
        elastic_params,
        density,
        damping_params,
        newmark_params,
        initial_force_x,
        initial_force_y,
        initial_force_z,
        cutoff_time_factor,
        total_time,
        num_steps,
        mode,
    )

    out_gif = os.path.join(graph_save_path, "BEAM_" + simulation_name + ".gif")

    make_beam_gif(lst_graphs, L, W, D, out_gif, fps=4)
