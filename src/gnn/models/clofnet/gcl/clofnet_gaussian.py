import logging
import numpy as np
import torch
from torch import nn

from src.gnn.models.clofnet.gcl.clofnet_gcl import ClofGCL
from src.gnn.models.clofnet.gaussian import GaussianLayer


class ClofNetVelocityGaussian(nn.Module):
    """
    Velocity-aware ClofNet with Gaussian basis edge encoding instead of MLP or fourier.

    Paper relation:
        The core frame/scalarization/vectorization logic is still ClofNet.
        Appendix A.3.1 Eq. (15) discusses Fourier or MLP embeddings for scalars; this class instead uses a Gaussian
        radial basis expansion for the edge-type/distance pathway.
    """

    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device="cpu", act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
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
        super(ClofNetVelocityGaussian, self).__init__()
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
            clof_gcl = ClofGCL(input_nf=self.hidden_nf,
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
            h, x_center, _ = self._modules["gcl_%d" % i](h, edges, x_center, vel, edge_attr=edge_feat,
                                                         node_attr=node_attr)

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x