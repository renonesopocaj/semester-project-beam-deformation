"""
clofnet.py

Implements the ClofNet architecture as a whole, using the ClofGCL Block as message passing block
- The basis vectors are "soft-normalized" with denominators like sqrt(radial) + 1 rather than exact unit normalization
    with a tiny epsilon. This is a stability choice present in the original code.
- The scalarization does more than the paper's minimal coefficient tuple: it also computes a pseudo-angle
    feature from the projected coefficients.
   This enriches the invariant edge descriptor passed to the message-passing block.
"""

import numpy as np
import torch
from torch import nn
from src.gnn.models.clofnet.gcl_clofnet import ClofMPBlock
from src.gnn.models.clofnet.clof_helpers import compute_angle_feat, forward_pass_generic, coord2localframe


class ClofNet(nn.Module):
    """
    Local-frame ClofNet: without explicit velocity scalarization, and with MLP edge encoding.

    High-level pipeline:
        1. Embed node attributes.
        2. Center coordinates per graph to remove global translation.
        3. Build an edge-local complete frame F_ij.
        4. Scalarize geometry into invariant edge features.
        5. Pass invariant features through stacked Clof_GCL blocks.
        6. Add the graph centroid back.

    Relative to the paper, this class implements the "Scalarize -> invariant message passing -> Vectorize" recipe
    from Eq. (9), but the actual vectorization happens inside Clof_GCL.coord_model rather than in this file.
    """

    def __init__(self, in_dim_node, in_dim_edge, hidden_dim,
                 act_fn,
                 n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, out_coord_mlp_dim, clamp_enable,
                 embedding_edge_dim):
        """
        Initializes the model
        Def: out_dim_edge_emb=8

        Args:
            in_dim_node: int
                Input dimensionality of node features h.
            in_dim_edge: int
                Input dimensionality of edge attributes e_ij.
            hidden_dim: int
                Hidden dimensionality used across embeddings and Clof_GCL.
            device: torch.device or str
                Device where the module is moved.
            act_fn: torch.nn.Module
                Activation used in the small edge/node MLPs.
            n_layers: int
                Number of stacked Clof_GCL blocks.
            coords_weight: float
                Multiplicative factor applied to coordinate updates inside
                Clof_GCL.
            recurrent: bool
                Whether the feature update uses residual accumulation.
            enable_norm_diff: bool
                Whether to normalize edge-local frame directions.
            tanh: bool
                Whether Clof_GCL should bound coordinate coefficients with tanh.
        """
        super().__init__()
        # message passing layers info
        self.graph_mp_block_list = []
        self.n_layers = n_layers
        # info about hidden dimension in general
        self.hidden_dim = hidden_dim
        self.embedding_edge_dim = embedding_edge_dim
        self.in_dim_edge_mlp = self.hidden_dim // 2
        self.hiddendim_fuse_mlp = self.hidden_dim // 2
        edge_embed_dim = in_dim_edge + 8 # 2 pseudo-angle scalars + 3 coeffs for x_i + 3 coeffs for x_j = 8. # Concatenated with the 2-dim edge_attr used in the original code path.
        self.fuse_in_dim = edge_embed_dim
        # Embedding layers mlp
        self.embedding_node_mlp = nn.Linear(in_dim_node, self.hidden_dim)
        # NOTE: this function is not used because edges are embedded directly together with scalarized coords via
        #   fuse_mlp
        # self.embedding_edge_mlp = nn.Sequential(nn.Linear(in_dim_edge, self.embedding_edge_dim),
        #                                         act_fn)
        # Additional MLP to embed coefficients of projected x_i, x_j on frame F_ij + pseudo angles AND e_ij edge attr
        self.fuse_mlp_embed = nn.Sequential(nn.Linear(self.fuse_in_dim, self.hiddendim_fuse_mlp),
                                            act_fn,
                                            nn.Linear(self.hiddendim_fuse_mlp, self.hiddendim_fuse_mlp),
                                            act_fn)
        self.enable_norm_diff = enable_norm_diff
        # Initialize the ClofGCL layers (message passing layers)
        for i in range(0, self.n_layers):
            self.graph_mp_block_list.append(
                ClofMPBlock(in_node_mlp_dim=self.hidden_dim, output_node_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim, in_mij_mlp_dim=self.in_dim_edge_mlp,
                            act_fn=act_fn,
                            recurrent=recurrent,
                            coords_weight=coords_weight, attention=attention_enable, norm_diff=enable_norm_diff,
                            tanh=tanh, vel_mlp_dim_out=out_coord_mlp_dim, out_basis_dim=3, clamp=clamp_enable,
                            enable_vel=False))
        # Print model params
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)

    def coord2localframe(self, edge_index, coord_xi):
        """
        Build the three edge-local basis directions used by ClofNet.

        Paper correspondence:
            This is the implementation analogue of Eq. (2):
                a_ij = normalized(x_i - x_j)
                b_ij = normalized(x_i x x_j)
                c_ij = a_ij x b_ij

        Practical variant:
            The original implementation divides by sqrt(radial) + 1 and
            ||cross|| + 1 rather than by a strict norm + epsilon. That makes
            the frame less sensitive to tiny distances and keeps training more
            stable, but it also means these are not exact unit vectors.

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col), with one entry per directed edge.
            coord_xi: torch.Tensor
                Coordinates of shape [N, 3].
        return: (coord_diff, coord_cross, coord_vertical)
            coord_diff: torch.Tensor
                First basis direction, shape [E, 1, 3].
            coord_cross: torch.Tensor
                Second basis direction, shape [E, 1, 3].
            coord_vertical: torch.Tensor
                Third basis direction, shape [E, 1, 3].
        """
        # These tensors ar [E, 3]
        radial, coord_diff, coord_cross, coord_vertical = coord2localframe(edge_index, coord_xi,
                                                                           enable_norm_diff=self.enable_norm_diff,
                                                                           crossdim1=True)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1) # (E, 1, 3)

    def scalarization(self, edges, x, vel):
        """
        Project endpoint coordinates onto the edge-local frame.
        What this implementation returns:
                        coff_feat = [pseudo_sin, pseudo_cos, coff_i, coff_j]
        where:
                        coord_frame_xi = [<a_ij, x_i>, <b_ij, x_i>, <c_ij, x_i>]
                        coord_frame_xj = [<a_ij, x_j>, <b_ij, x_j>, <c_ij, x_j>]
        The extra pseudo-angle feature is not explicitly written as one of the paper's numbered equations.

        Args:
            edges: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col).
            x: torch.Tensor
                Coordinates of shape [N, 3].
        return: torch.Tensor
            Invariant edge features of shape [E, 8].
        """
        EPS = 1e-5
        coord_diff_aij, coord_cross_bij, coord_vertical_cij = self.coord2localframe(edges, x) # [E, 1, 3]
        row, col = edges
        # Concatenate the basis vectors
        frame_basis_fij = torch.cat([coord_diff_aij, coord_cross_bij, coord_vertical_cij], dim=1) # [E, 3, 3]
        x_i, x_j = x[row], x[col] # [E, 3]
        # PROJECT x_i and x_j on the frame basis f_ij and obtain fram coefficient
        # note: dimension is E because the frame f_ij is for each (i,j) connected by an edge
        coord_frame_xi = torch.matmul(frame_basis_fij, x_i.unsqueeze(-1)).squeeze(-1) # [E, 3]
        coord_frame_xj = torch.matmul(frame_basis_fij, x_j.unsqueeze(-1)).squeeze(-1) # [E, 3]
        # Additional invariant angle-like descriptor built from the projected coefficient triplets.
        pseudo_angle = compute_angle_feat(coord_frame_xi, coord_frame_xj, EPS)  # [E, 2]
        # Concatenate everything
        coords_frame_xij = torch.cat([pseudo_angle, coord_frame_xi, coord_frame_xj], dim=-1)
        return coords_frame_xij # [E, 8]

    def forward(self, h, x, edges, vel, edge_attr, node_attr, n_nodes):
        """
        Run the ClofNet stack and return updated coordinates.
        node_attr=None, n_nodes=5

        Args:
            h: torch.Tensor
                Node features of shape [N, in_node_nf].
            x: torch.Tensor
                Coordinates of shape [N, 3].
            edges: tuple[torch.Tensor, torch.Tensor]
                Directed edge index tuple (row, col).
            vel: torch.Tensor
                Velocity tensor of shape [N, 3]. Included only for API compatibility with the other variants.
            edge_attr: torch.Tensor
                Edge attributes of shape [E, in_edge_nf].
            node_attr: torch.Tensor or None
                Optional node-side attributes concatenated inside Clof_GCL.
            n_nodes: int
                Number of particles per graph in the batch. Needed to recover graphwise centroids after flattening.
        return: torch.Tensor
            Updated coordinates of shape [N, 3].
        """
        x = forward_pass_generic(self, x, h, n_nodes, edges, edge_attr, vel, node_attr)
        return x # [N, 3]