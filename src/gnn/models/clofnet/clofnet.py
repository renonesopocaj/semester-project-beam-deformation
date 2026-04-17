"""
clofnet.py

Implements the ClofNet architecture as a whole, using the ClofGCL Block as message passing block
Important implementation note
-----------------------------
1. The basis vectors are "soft-normalized" with denominators like sqrt(radial) + 1 rather than exact unit normalization
    with a tiny epsilon. This is a stability choice present in the original code.
2. The scalarization does more than the paper's minimal coefficient tuple: it also computes a pseudo-angle
    feature from the projected coefficients.
   This enriches the invariant edge descriptor passed to the message-passing block.
"""

import logging
import numpy as np
import torch
from torch import nn

from src.gnn.models.clofnet.gcl_clofnet import ClofGCL


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
                 device,
                 act_fn,
                 n_layers,
                 coords_weight, recurrent, norm_diff, tanh,
                 out_dim_edge_emb=8):
        """
        Initializes the model

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
            norm_diff: bool
                Whether to normalize edge-local frame directions.
            tanh: bool
                Whether Clof_GCL should bound coordinate coefficients with tanh.
        """
        super(ClofNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = n_layers
        # Embedding layers
        self.embedding_node = nn.Linear(in_dim_node, self.hidden_dim)
        self.embedding_edge = nn.Sequential(nn.Linear(in_dim_edge, out_dim_edge_emb),
                                            act_fn)
        # 2 pseudo-angle scalars + 3 coeffs for x_i + 3 coeffs for x_j = 8.
        # Concatenated with the 2-dim edge_attr used in the original code path.
        edge_embed_dim = 10
        # FIXME: what is fuse_edge?
        self.fuse_edge = nn.Sequential(nn.Linear(edge_embed_dim, self.hidden_dim // 2),
                                       act_fn,
                                       nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
                                       act_fn)
        self.norm_diff = norm_diff
        # Initialize the ClofGCL layers (message passing layers)
        for i in range(0, self.n_layers):
            self.add_module("gcl_%d" % i,
                            ClofGCL(input_dim_node=self.hidden_dim, output_node_dim=self.hidden_dim,
                                    hidden_dim=self.hidden_dim, edges_in_d=self.hidden_dim // 2, act_fn=act_fn,
                                    recurrent=recurrent, coords_weight=coords_weight, norm_diff=norm_diff, tanh=tanh), )
        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        """
        return: str
            Trainable-parameter count as a string.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)
        logging.info("Network Size {}".format(params))
        return str(params)

    def coord2localframe(self, edge_index, coord):
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
            coord: torch.Tensor
                Coordinates of shape [N, 3].
        return: (coord_diff, coord_cross, coord_vertical)
            coord_diff: torch.Tensor
                First basis direction, shape [E, 1, 3].
            coord_cross: torch.Tensor
                Second basis direction, shape [E, 1, 3].
            coord_vertical: torch.Tensor
                Third basis direction, shape [E, 1, 3].
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        # Cross product of the endpoint position vectors.
        # This matches the spirit of b_ij in Eq. (2).
        coord_cross = torch.cross(coord[row], coord[col], dim=1)
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = torch.sqrt(torch.sum(coord_cross ** 2, 1).unsqueeze(1)) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross, dim=1)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def scalarization(self, edges, x):
        """
        Project endpoint coordinates onto the edge-local frame.

        Paper correspondence:
            - Eq. (3): project a vector onto the local frame with inner products.
            - End of Sec. 4: practical scalarization includes both x_i and x_j.

        What this implementation returns:
            coff_feat = [pseudo_sin, pseudo_cos, coff_i, coff_j]

        where:
            coff_i = [<a_ij, x_i>, <b_ij, x_i>, <c_ij, x_i>]
            coff_j = [<a_ij, x_j>, <b_ij, x_j>, <c_ij, x_j>]

        The extra pseudo-angle feature is not explicitly written as one of the
        paper's numbered equations. It is an implementation-side enrichment of
        the invariant descriptor derived from the scalarized coefficients.

        Args:
            edges: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col).
            x: torch.Tensor
                Coordinates of shape [N, 3]. In forward(), these are the
                centered coordinates.
        return: torch.Tensor
            Invariant edge features of shape [E, 8].
        """
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)
        r_i = x[row]
        r_j = x[col]
        # Frame coefficients from Eq. (3), applied to both endpoints.
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)
        # Additional invariant angle-like descriptor built from the projected
        # coefficient triplets. This is an implementation detail, not a paper
        # equation.
        coff_mul = coff_i * coff_j
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) + 1e-5
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) + 1e-5
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / coff_i_norm / coff_j_norm
        pseudo_sin = torch.sqrt(torch.clamp(1 - pseudo_cos ** 2, min=0.0))
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)
        coff_feat = torch.cat([pseudo_angle, coff_i, coff_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None, n_nodes=5):
        """
        Run the ClofNet stack and return updated coordinates.

        Paper correspondence:
            - Centralization corresponds to Fig. 2 / Sec. 5 translation handling.
            - scalarization corresponds to Eq. (3) and the practical
              Scalarize(x_i, x_j, chi_i, chi_j, F_ij).
            - The stacked Clof_GCL layers realize the invariant message passing
              plus vectorized coordinate update from Eqs. (16)-(19).

        Notes:
            vel is accepted to keep a uniform API with velocity-aware variants.
            In this class it is passed through to Clof_GCL but not included in
            the edge scalarization itself.

        Args:
            h: torch.Tensor
                Node features of shape [N, in_node_nf].
            x: torch.Tensor
                Coordinates of shape [N, 3].
            edges: tuple[torch.Tensor, torch.Tensor]
                Directed edge index tuple (row, col).
            vel: torch.Tensor
                Velocity tensor of shape [N, 3]. Included only for API
                compatibility with the other variants.
            edge_attr: torch.Tensor
                Edge attributes of shape [E, in_edge_nf].
            node_attr: torch.Tensor or None
                Optional node-side attributes concatenated inside Clof_GCL.
            n_nodes: int
                Number of particles per graph in the batch. Needed to recover
                graphwise centroids after flattening.
        return: torch.Tensor
            Updated coordinates of shape [N, 3].
        """
        # Embed node
        h = self.embedding_node(h)
        # Translation handling: move every graph to its centroid-centered frame before constructing
        # complete local frames.
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)
        # Invariant scalarization of centered geometry.
        coff_feat = self.scalarization(edges, x_center) # FIXME: what is this
        # In the original implementation the edge attributes are concatenated to the scalarized frame coefficients
        # and fused by a small MLP. FIXME: waht is fuse doing
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)
        # Pass through message passing layers
        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](h, edges, x_center, vel, edge_attr=edge_feat,
                                                         node_attr=node_attr)
        # Restore the global translation.
        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x