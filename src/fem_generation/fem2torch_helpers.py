"""
fem2torch_helpers.py
This file contains helper functions for converting FEniCS meshes to PyTorch Geometric graphs.
"""

import numpy as np
import torch
import dolfin as df

def evaluate_u_at_vertices(u, mesh):
    """
    Evaluate the displacement field at each mesh vertex.

    Args:
        u: dolfin.Function
            Vector-valued displacement field.
        mesh: dolfin.Mesh
            Mesh whose vertices define the evaluation points.
        return: np.ndarray
            Array of shape (num_vertices, 3) containing the displacement
            evaluated at each vertex.
    """
    # obtain mesh coordinates and number of vertices
    coords = mesh.coordinates()
    nverts = coords.shape[0]

    # Evaluate the field at each vertex coordinate.
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = u(*coords[i])
    return out


def evaluate_velocity_at_vertices(v_fun, mesh):
    """
    Evaluate the velocity field at each mesh vertex.

    Args:
        v_fun: dolfin.Function
            Vector-valued velocity field.
        mesh: dolfin.Mesh
            Mesh whose vertices define the evaluation points.
        return: np.ndarray
            Array of shape (num_vertices, 3) containing vertex-wise velocity.
    """
    # obtain mesh coordinates and number of vertices
    coords = mesh.coordinates()
    nverts = coords.shape[0]

    # Evaluate velocity at every vertex position.
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = v_fun(*coords[i])

    return out


def evaluate_acceleration_at_vertices(a_fun, mesh):
    """
    Evaluate the acceleration field at each mesh vertex.

    Args:
        a_fun: dolfin.Function
            Vector-valued acceleration field.
        mesh: dolfin.Mesh
            Mesh whose vertices define the evaluation points.
        return: np.ndarray
            Array of shape (num_vertices, 3) containing vertex-wise acceleration.
    """
    # obtain mesh coordinates and number of vertices
    coords = mesh.coordinates()
    nverts = coords.shape[0]

    # Evaluate acceleration at every vertex position.
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = a_fun(*coords[i])
    return out


def evaluate_force_at_vertices(f, mesh):
    """
    Evaluate the given force field at each mesh vertex.

    Args:
        f: dolfin.Function
            Vector-valued force field.
        mesh: dolfin.Mesh
            Mesh whose vertices define the evaluation points.
        return: np.ndarray
            Array of shape (num_vertices, 3) containing vertex-wise force values.
    """
    # obtain mesh coordinates and number of vertices
    coords = mesh.coordinates()
    nverts = coords.shape[0]

    # Evaluate external force field at each vertex coordinate.
    out = np.zeros((nverts, 3), dtype=float)
    for i in range(nverts):
        out[i] = f(*coords[i])
    return out


def evaluate_stress_at_vertices(stress_fun, mesh):
    """
    Evaluate a tensor-valued stress field at each mesh vertex and flatten it.

    Args:
        stress_fun: dolfin.Function
            Tensor-valued stress field. Each evaluation should return a 3x3 tensor.
        mesh: dolfin.Mesh
            Mesh whose vertices define the evaluation points.
        return: np.ndarray
            Array of shape (num_vertices, 9) containing the flattened stress tensor
            at each vertex in row-major order.
    """
    # obtain mesh coordinates and number of vertices
    coords = mesh.coordinates()
    nverts = coords.shape[0]

    # Evaluate and flatten the 3x3 stress tensor at every vertex.
    out = np.zeros((nverts, 9), dtype=float)
    for i in range(nverts):
        sigma_3x3 = stress_fun(*coords[i])
        out[i] = np.array(sigma_3x3).flatten()
    return out

def extract_edges(mesh):
    """
    Extract the undirected mesh edges from element connectivity and return them
    in PyTorch Geometric format.

    Args:
        mesh: dolfin.Mesh
            FEniCS mesh containing tetrahedral element connectivity.
        return: (torch.Tensor, np.ndarray)
            A tuple containing:
            - bidirectional edge index with shape (2, num_edges * 2), dtype long
            - raw element-to-node connectivity as returned by mesh.cells()
    """
    # Get element-to-node connectivity.
    # For a tetrahedral mesh, each row usually contains 4 vertex indices.
    mesh_connectivity = mesh.cells()

    # Use a set to avoid duplicate undirected edges.
    edges = set()

    # For each element, connect every pair of nodes in that element.
    # In a tetrahedron, this generates all 6 local edges.
    for element in mesh_connectivity:
        for i in range(len(element)):
            for j in range(i + 1, len(element)):
                edges.add(tuple(sorted((element[i], element[j]))))

    # Convert the set of edge tuples to a tensor of shape (2, num_edges).
    edge_index = list(edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Duplicate edges in both directions because PyTorch Geometric message passing
    # typically expects directed edges.
    bidir_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return bidir_edge_index, mesh_connectivity

def extract_mesh_bc_nodemask(bc, V):
    """
    Extract a binary node mask describing which mesh vertices are fixed by a Dirichlet boundary condition.
    A node is marked as fixed if the corresponding degree of freedom in the first component belongs to the boundary
     condition. Used only to pass it to the PyG object.

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
    d2v = df.dof_to_vertex_map(V_sub0)

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
