"""Auxiliary layers used by ClofNet variants.

Exposes GaussianLayer for edge basis expansion and an attention-heavy E_GCL_vel
variant used in alternative experiments.
"""

from typing import Optional, Tuple

from torch import nn
import torch
import pickle
import numpy as np
import math
import torch.nn.functional as F
from src.gnn.models.clofnet.gcl import unsorted_segment_sum, unsorted_segment_mean


def gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian basis values with broadcasted mean/std."""
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Distance encoder using learnable Gaussian bases per edge type."""

    def __init__(self, K: int = 128, edge_types: int = 1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        # Scale and shift distances per edge type then expand into K Gaussian bins.
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class E_GCL_vel(nn.Module):
    """Attention-enhanced equivariant block (not the same as gcl.E_GCL_vel)."""

    def __init__(self, input_node_nf: int, input_edge_nf: int, output_nf: int, hidden_nf: int,
                 nodes_att_dim: int = 0, act_fn: nn.Module = nn.Softplus(), recurrent: bool = True,
                 coords_weight: float = 1.0, attention: bool = False, norm_diff: bool = False,
                 tanh: bool = False, nhead: int = 3, n_points: int = 5):
        super(E_GCL_vel, self).__init__()
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.nhead = nhead
        self.n_points = n_points
        assert (hidden_nf % nhead) == 0

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_node_nf * 2 + input_edge_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.edge_mlp_enhance = nn.Sequential(
            nn.Linear(hidden_nf * 2 + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(input_node_nf + hidden_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        self.coord_mlp_inner = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        self.fc_query = nn.Linear(hidden_nf, hidden_nf)
        self.fc_key = nn.Linear(2 * hidden_nf, hidden_nf)
        self.fc_value = nn.Linear(hidden_nf, hidden_nf)
        self.norm1 = nn.LayerNorm(hidden_nf)
        self.norm2 = nn.LayerNorm(hidden_nf)
        self.constant = 1

    def edge_model(self, source: torch.Tensor, target: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def edge_model_enhance(self, source: torch.Tensor, target: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp_enhance(out)
        return out

    def node_model(self, x: torch.Tensor, edge_index: Tuple[torch.Tensor, torch.Tensor], edge_attr: torch.Tensor,
                  node_attr: Optional[torch.Tensor]):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = self.norm1(agg)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
            out = self.norm2(out)

        return out, agg

    def coord2radial(self, edge_index: Tuple[torch.Tensor, torch.Tensor], coord: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + self.constant
            coord_diff = coord_diff / norm
            cross_norm = (
                torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1)) + self.constant
            )
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical


    def acc_model_inner(self, coord: torch.Tensor, edge_index: Tuple[torch.Tensor, torch.Tensor],
                        coord_diff: torch.Tensor, coord_cross: torch.Tensor, coord_vertical: torch.Tensor,
                        edge_feat: torch.Tensor) -> torch.Tensor:
        """Project edge features to basis coefficients and aggregate acceleration."""
        row, col = edge_index
        basis_coff = self.coord_mlp_inner(edge_feat)
        trans = (
            coord_diff * basis_coff[:, :1]
            + coord_cross * basis_coff[:, 1:2]
            + coord_vertical * basis_coff[:, 2:3]
        )
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        acc = agg * self.coords_weight

        return acc

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (B, nhead, N, C/nhead) for attention score computation."""
        new_shape = x.shape[:-1] + (self.nhead, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, h: torch.Tensor, edge_index: Tuple[torch.Tensor, torch.Tensor], coord: torch.Tensor,
                coord_pre: torch.Tensor, vel: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                node_attr: Optional[torch.Tensor] = None, short_cut: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Message enhancement
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], edge_attr)

        # Attention-based message passing, inspired by SE3-Fold
        B, N, C = int(h.shape[0]//self.n_points), self.n_points, edge_feat.shape[-1]
        query = h.reshape(B, N, C)  # [B, N, C]
        m = edge_feat.reshape(B, N, -1, C)  # [B, N, N-1, C]
        h_m = torch.cat([query.unsqueeze(2).repeat(1, 1, N - 1, 1), m], dim=-1).reshape(
            B, N * (N - 1), -1
        )  # [B, N*N-1, 2C]

        query = self.transpose_for_scores(
            self.fc_query(query)
        )  # [B, N, C] -> [B, A, N, C/A]
        key = self.transpose_for_scores(
            self.fc_key(h_m)
        )  # [B, N*(N-1), C] -> [B, A, N*(N-1), C/A]
        value = self.transpose_for_scores(self.fc_value(edge_feat.reshape(B, N * (N - 1), C)))  # [B, N*(N-1), C] -> [B, A, N*(N-1), C/A]

        key = key.reshape(B, self.nhead, N, N - 1, -1)  # [B, A, N, (N-1), C/A]
        attention_scores = torch.matmul(query.unsqueeze(-2), key.transpose(-1, -2)).squeeze(-2)  # [B, A, N, N-1]
        attention_scores = attention_scores / math.sqrt(C / self.nhead)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, A, N, N-1]
        m_update = (attention_weights.reshape(B, self.nhead, -1).unsqueeze(-1) * value)  # [B, A, N*(N-1), C/A]
        att_edge_feat = m_update.transpose(-3, -2)  # [B, A, N*(N-1), C/A]
        att_edge_feat = att_edge_feat.reshape(*att_edge_feat.shape[:-2], -1)  # [B, N*(N-1), C]
        att_edge_feat = att_edge_feat.reshape(-1, C)  # [B*N*(N-1), C]
        h, agg = self.node_model(h, edge_index, att_edge_feat, node_attr)

        # Equivariant message aggregation
        edge_feat = edge_feat + self.edge_model_enhance(h[row], h[col], edge_attr)
        coord_diff, coord_cross, coord_vertical = self.coord2radial(edge_index, coord_pre)
        acc1 = self.acc_model_inner(coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat)

        if short_cut:
            ACC = coord + acc1
        else:
            ACC = acc1

        return ACC, h, edge_feat

