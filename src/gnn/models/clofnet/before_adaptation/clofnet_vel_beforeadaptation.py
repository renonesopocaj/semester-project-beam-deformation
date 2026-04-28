"""
clofnet_vel_beforeadaptation.py

ClofNet variant that explicitly scalarizes velocities in the local frame.
Paper correspondence:
    This is the practical realization of the paper's notation
                         Scalarize(x_i, x_j, chi_i, chi_j, F_ij)
    where chi_i is instantiated as the velocity vector v_i.
Compared with ClofNet:
    The frame coefficients now include both projected coordinates and projected velocities for source and target nodes.
The extra term
                             coord += coord_mlp_vel(h) * vel
is not a paper equation. It is a repository-level dynamics addition.
"""

import torch
from torch import nn
from src.gnn.models.clofnet.gcl_clofnet import ClofMPBlock
import numpy as np
import logging
from src.gnn.models.clofnet.clof_helpers import coord2localframe, compute_angle_feat, forward_pass_generic


class ClofNetVel(nn.Module):

    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device, act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
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
        self.fuse_edge = nn.Sequential(nn.Linear(edge_embed_dim, self.hidden_nf // 2), act_fn,
                                       nn.Linear(self.hidden_nf // 2, self.hidden_nf // 2), act_fn)

        # Preserves the original behavior: the constructor argument is accepted,
        # but the module stores True unconditionally.
        self.norm_diff = True
        for i in range(0, self.n_layers):
            self.add_module("gcl_%d" % i, ClofMPBlock(input_dim_node=self.hidden_nf, output_node_dim=self.hidden_nf,
                                                      hidden_dim=self.hidden_nf, edges_in_d=self.hidden_nf // 2,
                                                      act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight,
                                                      norm_diff=norm_diff, tanh=tanh))

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

    def coord2localframe(self, edge_index, coord_xi):
        """
        Construct the complete local frame per edge.
        This is the code counterpart of paper Eq. (2):
            a_ij = (x_i - x_j) / ||x_i - x_j||
            b_ij = (x_i x x_j) / ||x_i x x_j||
            c_ij = a_ij x b_ij

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge list as (row, col).
            coord_xi: torch.Tensor
                Coordinates of shape [N, 3].
        return: (coord_diff, coord_cross, coord_vertical)
            Edge-local basis tensors, each with shape [E, 1, 3].
        """
        radial, coord_diff, coord_cross, coord_vertical = coord2localframe(edge_index, coord_xi, self.norm_diff,
                                                                           crossdim1=True)
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
        EPS = 1e-5
        coord_diff, coord_cross, coord_vertical = self.coord2localframe(edges, x)
        row, col = edges
        edge_basis = torch.cat([coord_diff, coord_cross, coord_vertical], dim=1)
        r_i, r_j = x[row], x[col]
        v_i, v_j = vel[row], vel[col]
        # Position coefficients
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)
        # Velocity coefficients, i.e. coordinates of the projection on the local frame: this is chi_i
        vel_i = torch.matmul(edge_basis, v_i.unsqueeze(-1)).squeeze(-1)
        vel_j = torch.matmul(edge_basis, v_j.unsqueeze(-1)).squeeze(-1)
        # Obtain angles
        pseudo_angle = compute_angle_feat(coff_i, coff_j, EPS)
        # Concatenate everything
        coff_feat = torch.cat([pseudo_angle, coff_i, coff_j, vel_i, vel_j], dim=-1)
        return coff_feat

    def forward(self, h, x, edges, vel, edge_attr, node_attr, n_nodes):
        """
        Forward pass for the velocity-aware ClofNet.
        Default n_nodes=5, node_attr=None

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
            node_attr: torch.Tensor
                Optional node attributes.
            n_nodes: int
                Number of nodes per graph.
        return: torch.Tensor
            Updated coordinates of shape [N, 3].
        """
        x = forward_pass_generic(self, x, h, n_nodes, edges, edge_attr, vel, node_attr)
        return x