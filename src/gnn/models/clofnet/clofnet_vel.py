"""
clofnet_vel.py

ClofNet variant that explicitly scalarizes velocities in the local frame.
Paper correspondence:
    This is the practical realization of the paper's notation
        Scalarize(x_i, x_j, chi_i, chi_j, F_ij)
    where chi_i is instantiated as the velocity vector v_i.
Compared with ClofNet:
    The frame coefficients now include both projected coordinates and projected velocities for source and target nodes.
"""

import torch
from torch import nn

class ClofNetVel(nn.Module):

    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device="cpu", act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
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
                ClofGCL(input_dim_node=self.hidden_nf, output_node_dim=self.hidden_nf, hidden_dim=self.hidden_nf,
                        edges_in_d=self.hidden_nf // 2, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight,
                        norm_diff=norm_diff, tanh=tanh),
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