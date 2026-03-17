"""
clof.py
ClofNet variants used for Newtonian rollouts.

The module exposes three stacks sharing the same Clof_GCL backbone but
different edge featurizations:
- ClofNet: geometry-only local-frame scalarization.
- ClofNetVel: adds velocity projections into the same local frame.
- ClofNet_vel_gbf: velocity variant with Gaussian basis edge encoding.
"""

from typing import Optional, Tuple

import torch
from torch import nn
import numpy as np
import logging
from src.gnn.models.clofnet.gcl import Clof_GCL
from src.gnn.models.clofnet.layers import GaussianLayer

class ClofNet(nn.Module):
    """Local-frame ClofNet without velocity inputs.

    Builds an MLP embedding for nodes/edges, computes local-frame scalar
    features per edge, and stacks Clof_GCL layers to update node states and
    coordinates.
    """

    def __init__(self, in_node_nf: int, in_edge_nf: int, hidden_nf: int, device: torch.device, act_fn: nn.Module,
                 n_layers: int, coords_weight: float, recurrent: bool, norm_diff: bool, tanh: bool):
        """Initialize a velocity-free ClofNet stack.

        Args:
            in_node_nf: Input node feature dimension.
            in_edge_nf: Input edge feature dimension (distance, type, etc.).
            hidden_nf: Hidden size used for embeddings and Clof_GCL.
            device: Torch device string or torch.device.
            act_fn: Activation function for all MLPs.
            n_layers: Number of Clof_GCL blocks.
            coords_weight: Scaling for coordinate updates inside Clof_GCL.
            recurrent: Whether to use residual accumulation in Clof_GCL.
            norm_diff: Normalize edge direction vectors before use.
            tanh: Clamp coordinate outputs in Clof_GCL with tanh.
        """
        super(ClofNet, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_edge = nn.Sequential(nn.Linear(in_edge_nf, 8), act_fn)

        edge_embed_dim = 10
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
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

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Network Size', params)
        logging.info('Network Size {}'.format(params))
        return str(params)

    def coord2localframe(self, edge_index: Tuple[torch.Tensor, torch.Tensor], coord: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build an orthonormal-ish local frame per edge.

        Returns edge-aligned basis vectors (diff, cross, vertical) with optional
        normalization to avoid exploding updates.
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(
                torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def scalarization(self, edges: Tuple[torch.Tensor, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Project node coordinates into edge-local bases and angles."""
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)
        r_i = x[row]
        r_j = x[col]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) + 1e-5
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) + 1e-5
        pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / coff_i_norm / coff_j_norm
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j], dim=-1)
        return coff_feat

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges: Tuple[torch.Tensor, torch.Tensor], vel: torch.Tensor,
                edge_attr: torch.Tensor, node_attr: Optional[torch.Tensor] = None, n_nodes: int = 5) -> torch.Tensor:
        """Run message passing and predict updated coordinates.

        Args:
            h: Node features [N, in_node_nf].
            x: Flattened coordinates [N, 3] arranged as (B * n_nodes, 3).
            edges: Edge index tuple (row, col) following PyG convention.
            vel: Velocity tensor passed through unchanged (kept for API match).
            edge_attr: Edge attributes [E, in_edge_nf].
            node_attr: Optional auxiliary node features.
            n_nodes: Nodes per graph in the batch.
        Returns:
            Updated coordinates flattened back to [N, 3].
        """
        h = self.embedding_node(h)
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)  # Normalize positions around per-graph centroid.
        coff_feat = self.scalarization(edges, x_center)  # Local-frame geometric scalars per edge.
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class ClofNetVel(nn.Module):
    """ClofNet variant that conditions on velocities."""
    def __init__(self, in_node_nf: int, in_edge_nf: int, hidden_nf: int, device: str = 'cpu',
                 act_fn: nn.Module = nn.SiLU(), n_layers: int = 4, coords_weight: float = 1.0,
                 recurrent: bool = True, norm_diff: bool = True, tanh: bool = False):
        super(ClofNetVel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)

        edge_embed_dim = 16
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                Clof_GCL(input_nf=self.hidden_nf, output_nf=self.hidden_nf, hidden_nf=self.hidden_nf,
                         edges_in_d=self.hidden_nf // 2, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight,
                         norm_diff=norm_diff, tanh=tanh),
            )
        self.to(self.device)
        self.params = self.__str__()

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Network Size', params)
        logging.info('Network Size {}'.format(params))
        return str(params)

    def coord2localframe(self, edge_index: Tuple[torch.Tensor, torch.Tensor], coord: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(
                torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def scalarization(self, edges: Tuple[torch.Tensor, torch.Tensor], x: torch.Tensor, vel: torch.Tensor
                     ) -> torch.Tensor:
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        # Geometric Vectors Scalarization
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
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j, vel_i, vel_j],
                              dim=-1)  #[E, 14]
        return coff_feat

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges: Tuple[torch.Tensor, torch.Tensor], vel: torch.Tensor,
                edge_attr: torch.Tensor, node_attr: Optional[torch.Tensor] = None, n_nodes: int = 5) -> torch.Tensor:
        h = self.embedding_node(h)
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)  # Work in centered coordinates.

        coff_feat = self.scalarization(edges, x_center, vel)
        edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_feat = self.fuse_edge(edge_feat)

        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x


class ClofNet_vel_gbf(nn.Module):
    """Velocity-aware ClofNet with Gaussian basis edge encoding."""
    def __init__(self, in_node_nf: int, in_edge_nf: int, hidden_nf: int, device: str = 'cpu',
                 act_fn: nn.Module = nn.SiLU(), n_layers: int = 4, coords_weight: float = 1.0,
                 recurrent: bool = True, norm_diff: bool = True, tanh: bool = False):
        super(ClofNet_vel_gbf, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        self.gbf = GaussianLayer(K=self.hidden_nf // 2, edge_types=8)
        edge_embed_dim = 14
        self.fuse_edge = nn.Sequential(
            nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
            nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

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

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Network Size', params)
        logging.info('Network Size {}'.format(params))
        return str(params)

    def coord2localframe(self, edge_index: Tuple[torch.Tensor, torch.Tensor], coord: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (torch.sqrt(
                torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
        return coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)

    def embed_edge(self, edge_types: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        edge_types = edge_types * 0.5 + 0.5
        return self.gbf(dist, edge_types.long())

    def scalarization(self, edges: Tuple[torch.Tensor, torch.Tensor], x: torch.Tensor, vel: torch.Tensor
                     ) -> torch.Tensor:
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        # Geometric Vectors Scalarization
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)
        r_i = x[row]
        r_j = x[col]
        v_i = vel[row]
        v_j = vel[col]
        coff_i = torch.matmul(edge_basis,
                              r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis,
                              r_j.unsqueeze(-1)).squeeze(-1)
        vel_i = torch.matmul(edge_basis,
                             v_i.unsqueeze(-1)).squeeze(-1)
        vel_j = torch.matmul(edge_basis,
                             v_j.unsqueeze(-1)).squeeze(-1)
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(
            dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j, vel_i, vel_j],
                              dim=-1)  #[E, 14]
        return coff_feat

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges: Tuple[torch.Tensor, torch.Tensor], vel: torch.Tensor,
                edge_attr: torch.Tensor, node_attr: Optional[torch.Tensor] = None, n_nodes: int = 5) -> torch.Tensor:
        h = self.embedding_node(h)
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)

        coff_feat = self.scalarization(edges, x_center, vel)
        # edge_feat = torch.cat([edge_attr, coff_feat], dim=-1)
        edge_embed = self.embed_edge(edge_attr[:, 0], edge_attr[:, 1])
        edge_feat = self.fuse_edge(coff_feat)
        edge_feat = edge_feat + edge_embed
        for i in range(0, self.n_layers):
            h, x_center, _ = self._modules["gcl_%d" % i](
                h, edges, x_center, vel, edge_attr=edge_feat, node_attr=node_attr)

        x = x_center.reshape(-1, n_nodes, 3) + centroid
        x = x.reshape(-1, 3)
        return x

