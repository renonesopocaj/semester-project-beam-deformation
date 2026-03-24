"""
clof.py
ClofNet variants used for Newtonian rollouts.

Paper mapping
-------------
This module is the closest implementation of the high-level ClofNet pipeline
described in Sec. 4 and Algorithm 1 of:

    Du et al., "SE(3) Equivariant Graph Neural Networks with Complete Local
    Frames", ICML 2022.

The relevant paper pieces are:

- Eq. (2): construction of the local complete frame
    a_ij = (x_i - x_j) / ||x_i - x_j||
    b_ij = (x_i x x_j) / ||x_i x x_j||
    c_ij = a_ij x b_ij

- Eqs. (3)-(7): scalarization by projecting vectors/tensors onto that frame.

- End of Sec. 4: practical scalarization
    Scalarize(x_i, x_j, chi_i, chi_j, F_ij)
  where chi_i denotes additional vector-valued features such as velocity.

- Eq. (8): vectorization back to an equivariant vector by pairing scalar
  coefficients with the frame basis.

- Fig. 2 / Algorithm 1 / Sec. 5: centralization of coordinates to preserve
  translation equivariance.

Important implementation note
-----------------------------
The code below is not a literal symbolic transcription of the paper. It keeps
the original runtime behavior and documents where the implementation is a
practical variant:

1. The basis vectors are "soft-normalized" with denominators like
   sqrt(radial) + 1 rather than exact unit normalization with a tiny epsilon.
   This is a stability choice present in the original code.

2. The scalarization does more than the paper's minimal coefficient tuple:
   it also computes a pseudo-angle feature from the projected coefficients.
   This enriches the invariant edge descriptor passed to the message-passing
   block.

3. The velocity-aware variants instantiate the paper's generic extra vector
   features chi_i with the specific choice chi_i = velocity_i.

4. The Gaussian-basis edge encoder is an implementation extension, not a
   formula explicitly named in the main paper.
"""

import logging
import numpy as np
import torch
from torch import nn

from src.gnn.models.clofnet.gcl import Clof_GCL
from src.gnn.models.clofnet.layers import GaussianLayer


class ClofNet(nn.Module):
    """
    Local-frame ClofNet without explicit velocity scalarization.

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

    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device, act_fn,
                 n_layers, coords_weight, recurrent, norm_diff, tanh):
        """
        Initializes the model

        Args:
            in_node_nf: int
                Input dimensionality of node features h.
            in_edge_nf: int
                Input dimensionality of edge attributes e_ij.
            hidden_nf: int
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
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_edge = nn.Sequential(nn.Linear(in_edge_nf, 8), act_fn)

        # 2 pseudo-angle scalars + 3 coeffs for x_i + 3 coeffs for x_j = 8.
        # Concatenated with the 2-dim edge_attr used in the original code path.
        edge_embed_dim = 10
        self.fuse_edge = nn.Sequential(nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
                                       nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        self.norm_diff = norm_diff

        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                Clof_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf // 2,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    coords_weight=coords_weight,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )

        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        """
        Args:
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
        h = self.embedding_node(h)

        # Translation handling from Fig. 2 / Sec. 5:
        # move every graph to its centroid-centered frame before constructing
        # complete local frames.
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        # Invariant scalarization of centered geometry.
        coff_feat = self.scalarization(edges, x_center)

        # In the original implementation the edge attributes are concatenated to
        # the scalarized frame coefficients and fused by a small MLP.
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr
            )

        # Restore the global translation.
        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class ClofNetVel(nn.Module):
    """
    ClofNet variant that explicitly scalarizes velocities in the local frame.

    Paper correspondence:
        This is the practical realization of the paper's notation
            Scalarize(x_i, x_j, chi_i, chi_j, F_ij)
        where chi_i is instantiated as the velocity vector v_i.

    Compared with ClofNet:
        The frame coefficients now include both projected coordinates and
        projected velocities for source and target nodes.
    """

    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device="cpu",
                 act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=True, norm_diff=True, tanh=False):
        """
        Args:
            in_node_nf: int
                Input node feature dimension.
            in_edge_nf: int
                Input edge feature dimension. Kept for API symmetry even though
                the fused edge dimensionality is fixed by the original code.
            hidden_nf: int
                Hidden dimensionality.
            device: str or torch.device
                Target device.
            act_fn: torch.nn.Module
                Activation used inside MLPs.
            n_layers: int
                Number of Clof_GCL blocks.
            coords_weight: float
                Coordinate update scaling inside Clof_GCL.
            recurrent: bool
                Whether feature residuals are enabled.
            norm_diff: bool
                Whether local-frame directions are normalized.
            tanh: bool
                Whether coordinate coefficients are bounded by tanh.
            return: None
                Initializes the module.
        """
        super(ClofNetVel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)

        # 2 pseudo-angle scalars + 3 coeffs for x_i + 3 coeffs for x_j
        # + 3 coeffs for v_i + 3 coeffs for v_j = 14.
        # With the original code path's 2-dim edge_attr, the fused input is 16.
        edge_embed_dim = 16
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn
        )

        # Preserves the original behavior: the constructor argument is accepted,
        # but the module stores True unconditionally.
        self.norm_diff = True

        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                Clof_GCL(
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=self.hidden_nf // 2,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    coords_weight=coords_weight,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )

        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        """
        Args:
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
        Same local complete frame construction as in ClofNet.

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col).
            coord: torch.Tensor
                Coordinates of shape [N, 3].
            return: (coord_diff, coord_cross, coord_vertical)
                Edge-local basis tensors, each with shape [E, 1, 3].
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col], dim=1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

            cross_norm = torch.sqrt(torch.sum(coord_cross ** 2, 1).unsqueeze(1)) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross, dim=1)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def scalarization(self, edges, x, vel):
        """
        Scalarize both positions and velocities in the same local frame.

        Paper correspondence:
            This matches the practical multi-vector scalarization at the end of
            Sec. 4, where the paper denotes generic extra vector features by
            chi_i in
                Scalarize(x_i, x_j, chi_i, chi_j, F_ij).

        Args:
            edges: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col).
            x: torch.Tensor
                Centered coordinates of shape [N, 3].
            vel: torch.Tensor
                Velocities of shape [N, 3].
            return: torch.Tensor
                Invariant edge descriptor of shape [E, 14].
        """
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)

        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)

        r_i = x[row]
        r_j = x[col]
        v_i = vel[row]
        v_j = vel[col]

        # Position coefficients.
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)

        # Velocity coefficients: this is the chi_i instantiation.
        vel_i = torch.matmul(edge_basis, v_i.unsqueeze(-1)).squeeze(-1)
        vel_j = torch.matmul(edge_basis, v_j.unsqueeze(-1)).squeeze(-1)

        coff_mul = coff_i * coff_j
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pseudo_sin = torch.sqrt(torch.clamp(1 - pseudo_cos ** 2, min=0.0))
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)

        coff_feat = torch.cat([pseudo_angle, coff_i, coff_j, vel_i, vel_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None, n_nodes=5):
        """
        Forward pass for the velocity-aware ClofNet.

        Args:
            h: torch.Tensor
                Node features of shape [N, in_node_nf].
            x: torch.Tensor
                Coordinates of shape [N, 3].
            edges: tuple[torch.Tensor, torch.Tensor]
                Directed edge index tuple.
            vel: torch.Tensor
                Velocities of shape [N, 3].
            edge_attr: torch.Tensor
                Edge attributes of shape [E, in_edge_nf].
            node_attr: torch.Tensor or None
                Optional node attributes.
            n_nodes: int
                Number of nodes per graph.
            return: torch.Tensor
                Updated coordinates of shape [N, 3].
        """
        h = self.embedding_node(h)

        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center, vel)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr
            )

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class ClofNet_vel_gbf(nn.Module):
    """
    Velocity-aware ClofNet with Gaussian basis edge encoding.

    Paper relation:
        The core frame/scalarization/vectorization logic is still ClofNet.
        The extra Gaussian basis edge encoder is an implementation extension.
        Appendix A.3.1 Eq. (15) discusses Fourier or MLP embeddings for
        scalars; this class instead uses a Gaussian radial basis expansion for
        the edge-type/distance pathway.
    """

    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device="cpu",
                 act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=True, norm_diff=True, tanh=False):
        """
        Args:
            in_node_nf: int
                Input node feature dimension.
            in_edge_nf: int
                Input edge feature dimension.
            hidden_nf: int
                Hidden dimensionality.
            device: str or torch.device
                Target device.
            act_fn: torch.nn.Module
                Activation function.
            n_layers: int
                Number of Clof_GCL blocks.
            coords_weight: float
                Coordinate update scaling.
            recurrent: bool
                Whether residual feature updates are enabled.
            norm_diff: bool
                Whether frame directions are normalized.
            tanh: bool
                Whether coordinate coefficients are tanh-bounded.
            return: None
                Initializes the module.
        """
        super(ClofNet_vel_gbf, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)

        self.gbf = GaussianLayer(K=self.hidden_nf // 2, edge_types=8)

        # coff_feat is 14-dim here.
        edge_embed_dim = 14
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn
        )

        # Preserves the original behavior.
        self.norm_diff = True

        for i in range(0, self.n_layers):
            clof_gcl = Clof_GCL(input_nf=self.hidden_nf,
                                output_nf=self.hidden_nf,
                                hidden_nf=self.hidden_nf,
                                edges_in_d=self.hidden_nf // 2,
                                                    act_fn=act_fn,
                                recurrent=recurrent,
                                                    coords_weight=coords_weight,
                                                    norm_diff=norm_diff,
                                                    tanh=tanh)
            self.add_module("gcl_%d" % i, clof_gcl,)

        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        """
        Args:
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
        Construct the local complete frame exactly as in the other variants.

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col).
            coord: torch.Tensor
                Coordinates of shape [N, 3].
            return: (coord_diff, coord_cross, coord_vertical)
                Edge-local basis tensors, each with shape [E, 1, 3].
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col], dim=1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

            cross_norm = torch.sqrt(torch.sum(coord_cross ** 2, 1).unsqueeze(1)) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross, dim=1)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def embed_edge(self, edge_types, dist):
        """
        Encode discrete edge types and distances with Gaussian bases.

        Args:
            edge_types: torch.Tensor
                Edge-type indicator per edge.
            dist: torch.Tensor
                Scalar distance or distance-like attribute per edge.
            return: torch.Tensor
                Edge embedding of shape [E, hidden_nf // 2].
        """
        edge_types = edge_types * 0.5 + 0.5
        return self.gbf(dist, edge_types.long())

    def scalarization(self, edges, x, vel):
        """
        Scalarize positions and velocities in the complete local frame.

        Args:
            edges: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col).
            x: torch.Tensor
                Centered coordinates of shape [N, 3].
            vel: torch.Tensor
                Velocities of shape [N, 3].
            return: torch.Tensor
                Invariant feature tensor of shape [E, 14].
        """
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)

        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)

        r_i = x[row]
        r_j = x[col]
        v_i = vel[row]
        v_j = vel[col]

        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)
        vel_i = torch.matmul(edge_basis, v_i.unsqueeze(-1)).squeeze(-1)
        vel_j = torch.matmul(edge_basis, v_j.unsqueeze(-1)).squeeze(-1)

        coff_mul = coff_i * coff_j
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pseudo_sin = torch.sqrt(torch.clamp(1 - pseudo_cos ** 2, min=0.0))
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)

        coff_feat = torch.cat([pseudo_angle, coff_i, coff_j, vel_i, vel_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr=None, n_nodes=5):
        """
        Forward pass for the Gaussian-basis velocity variant.

        Args:
            h: torch.Tensor
                Node features of shape [N, in_node_nf].
            x: torch.Tensor
                Coordinates of shape [N, 3].
            edges: tuple[torch.Tensor, torch.Tensor]
                Directed edge index tuple.
            vel: torch.Tensor
                Velocities of shape [N, 3].
            edge_attr: torch.Tensor
                Edge attributes. The original code assumes:
                    edge_attr[:, 0] -> edge type
                    edge_attr[:, 1] -> distance-like scalar
            node_attr: torch.Tensor or None
                Optional node attributes.
            n_nodes: int
                Number of nodes per graph.
            return: torch.Tensor
                Updated coordinates of shape [N, 3].
        """
        h = self.embedding_node(h)

        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center, vel)

        # The scalarized geometric features and the Gaussian basis edge encoder
        # are fused additively in the hidden edge space.
        edge_embed = self.embed_edge(edge_attr[:, 0], edge_attr[:, 1])
        edge_feat = self.fuse_edge(coff_feat)
        edge_feat = edge_feat + edge_embed

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr
            )

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x