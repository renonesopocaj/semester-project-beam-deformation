"""
layers.py

Auxiliary layers used by ClofNet variants.

This file contains two kinds of components:
1. Scalar/edge embedding helpers, especially `GaussianLayer`.
2. An attention-heavy equivariant block used in alternative experiments for
   molecular conformation generation.

Paper mapping used below:
- Eq. (13)-(15): feature embedding before message passing.
- Eq. (20)-(24): Graph Transformer Block (GTB) attention refinement.
- Eq. (2), Eq. (8), Eq. (19): complete-frame construction, vectorization and
  coordinate update.
- Appendix A.3.3: scalarization adds low extra cost compared with message
  passing, so feature expansion around those scalars is practical.

Some modules here are explicit implementation extensions rather than exact paper
objects. In particular, `GaussianLayer` is a code-level alternative to the
paper's "Fourier(s_ij) or MLP(s_ij)" embedding choice from Eq. (15), and the
multi-head attention implementation below is a richer realization of the GTB
idea from Eq. (20)-(24).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn.models.clofnet.gcl import unsorted_segment_mean, unsorted_segment_sum


def gaussian(x, mean, std):
    """
    Evaluate Gaussian basis functions with broadcasted parameters.

    The functional form is the standard normal-basis expansion:

        exp(-0.5 * ((x - mean) / std)^2) / (sqrt(2 pi) * std)

    Args:
        x: torch.Tensor
            Input values to expand.
        mean: torch.Tensor
            Gaussian means broadcastable to `x`.
        std: torch.Tensor
            Gaussian standard deviations broadcastable to `x`.
        return: torch.Tensor
            Gaussian basis values with the same broadcasted shape.
    """
    pi = 3.14159
    normalizer = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (normalizer * std)


class GaussianLayer(nn.Module):
    """
    Learnable Gaussian basis expansion for scalar distances.

    Paper relation:
    - The closest place in the paper is Eq. (15), where the scalarized features
      s_ij are embedded with either `Fourier(s_ij)` or `MLP(s_ij)` before graph
      message passing.
    - This class is not a literal equation from the paper. It is a practical
      code-level replacement for that embedding step when distances are treated
      as typed scalar inputs.

    Each edge type learns an affine transform

        x_typed = mul[type] * x + bias[type]

    before the scalar is expanded into K Gaussian channels.
    """

    def __init__(self, K=128, edge_types=1024):
        """
        Args:
            K: int
                Number of Gaussian basis channels.
            edge_types: int
                Number of discrete edge types that receive independent affine
                scale and bias parameters.
            return: None
                Initializes the learnable basis layer.
        """
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

    def forward(self, x, edge_types):
        """
        Args:
            x: torch.Tensor
                Scalar distances with shape [E] or [E, 1].
            edge_types: torch.Tensor
                Integer edge type ids with shape [E].
            return: torch.Tensor
                Gaussian-expanded distance features with shape [E, K].
        """
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, self.K)

        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class E_GCL_vel(nn.Module):
    """
    Attention-enhanced equivariant block used in alternative ClofNet variants.

    This block is most closely related to the Graph Transformer Block (GTB) in
    Appendix A.3.1:
    - `edge_model` corresponds to the first message computation before attention.
    - `fc_query`, `fc_key`, `fc_value`, the softmax weights and `node_model`
      implement the spirit of Eq. (20)-(24).
    - `coord2radial` plus `acc_model_inner` then apply ClofNet's complete-frame
      vectorization idea from Eq. (2), Eq. (8) and Eq. (19) to produce an
      equivariant coordinate or score update.

    Important implementation note:
    - The paper writes the GTB with single-head equations.
    - This module implements a multi-head variant and then feeds the refined
      node embeddings into an extra edge residual update before the coordinate
      aggregation. That is a faithful engineering extension, not a verbatim
      transcription of a printed equation.
    """

    def __init__(self, input_node_nf, input_edge_nf, output_nf, hidden_nf, nodes_att_dim=0,
                 act_fn=nn.Softplus(), recurrent=True, coords_weight=1.0, attention=False,
                 norm_diff=False, tanh=False, nhead=3, n_points=5):
        """
        Args:
            input_node_nf: int
                Input node feature dimension.
            input_edge_nf: int
                Input edge feature dimension.
            output_nf: int
                Output node feature dimension.
            hidden_nf: int
                Hidden width shared by the message, attention and node blocks.
            nodes_att_dim: int
                Optional extra node conditioning dimension.
            act_fn: torch.nn.Module
                Activation used throughout the block.
            recurrent: bool
                If True, applies residual node updates.
            coords_weight: float
                Global scaling on the aggregated coordinate update.
            attention: bool
                If True, applies an additional scalar edge gate before the GTB.
            norm_diff: bool
                If True, normalizes the complete-frame basis vectors.
            tanh: bool
                Legacy flag kept for API compatibility.
            nhead: int
                Number of attention heads for the multi-head GTB realization.
            n_points: int
                Number of nodes per dense graph instance. The forward pass uses
                this to reshape the flattened edge list into per-graph tensors.
            return: None
                Initializes the attention-enhanced equivariant block.
        """
        super().__init__()
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

    def edge_model(self, source, target, edge_attr):
        """
        Initial edge-message computation before attention refinement.

        This corresponds to the paper's first edgewise embedding stage before
        the GTB attention equations in Eq. (20)-(24).

        Args:
            source: torch.Tensor
                Source-node features with shape [E, input_node_nf].
            target: torch.Tensor
                Target-node features with shape [E, input_node_nf].
            edge_attr: torch.Tensor or None
                Edge features with shape [E, input_edge_nf].
            return: torch.Tensor
                Initial edge messages with shape [E, hidden_nf].
        """
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def edge_model_enhance(self, source, target, edge_attr):
        """
        Residual edge update after node refinement.

        Appendix A.3.1 states that after transformer-style node refinement, the
        block updates edgewise messages with a residue block. This method is
        that post-attention edge refinement.

        Args:
            source: torch.Tensor
                Source-side refined node features with shape [E, hidden_nf].
            target: torch.Tensor
                Target-side refined node features with shape [E, hidden_nf].
            edge_attr: torch.Tensor or None
                Previous edge messages with shape [E, hidden_nf].
            return: torch.Tensor
                Refined edge messages with shape [E, hidden_nf].
        """
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp_enhance(out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        Aggregate attention-refined messages and update node embeddings.

        This method corresponds most closely to Eq. (22)-(24):
        - sum_j alpha_ij v_ij
        - LayerNorm
        - phi_h(h_i, M_i)
        - residual + LayerNorm

        Args:
            x: torch.Tensor
                Node features with shape [N, input_node_nf] or [N, hidden_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge index in COO form as (row, col).
            edge_attr: torch.Tensor
                Attention-refined edge messages with shape [E, hidden_nf].
            node_attr: torch.Tensor or None
                Optional extra node conditioning.
            return: tuple[torch.Tensor, torch.Tensor]
                Updated node features and the concatenated node-MLP input.
        """
        row, col = edge_index
        del col
        agg = unsorted_segment_sum(edge_attr, row, x.size(0))
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

    def coord2radial(self, edge_index, coord):
        """
        Build the complete local frame used by the coordinate update.

        Although the method name is inherited from older radial-update code, the
        returned objects are the three ClofNet basis vectors:
            a_ij ~ x_i - x_j
            b_ij ~ x_i x x_j
            c_ij = a_ij x b_ij

        This is the same frame construction pattern as Eq. (2).

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge index in COO form as (row, col).
            coord: torch.Tensor
                Coordinates with shape [N, 3].
            return: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                a_ij, b_ij and c_ij, each with shape [E, 3].
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col], dim=1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + self.constant
            coord_diff = coord_diff / norm
            cross_norm = torch.sqrt(torch.sum(coord_cross ** 2, dim=1).unsqueeze(1)) + self.constant
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross, dim=1)
        return coord_diff, coord_cross, coord_vertical

    def acc_model_inner(self, coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat):
        """
        Vectorize edgewise scalar coefficients into a 3D acceleration/update.

        This is the same complete-frame vectorization idea as Eq. (8) and the
        coordinate update in Eq. (19), implemented after the GTB-style message
        refinement. The coordinate MLP predicts three scalar coefficients per
        edge and combines them with the three local basis vectors.

        Args:
            coord: torch.Tensor
                Coordinates with shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge index in COO form as (row, col).
            coord_diff: torch.Tensor
                First basis vector a_ij with shape [E, 3].
            coord_cross: torch.Tensor
                Second basis vector b_ij with shape [E, 3].
            coord_vertical: torch.Tensor
                Third basis vector c_ij with shape [E, 3].
            edge_feat: torch.Tensor
                Edge features with shape [E, hidden_nf].
            return: torch.Tensor
                Aggregated equivariant update with shape [N, 3].
        """
        row, col = edge_index
        del col

        basis_coff = self.coord_mlp_inner(edge_feat)
        trans = (
            coord_diff * basis_coff[:, :1]
            + coord_cross * basis_coff[:, 1:2]
            + coord_vertical * basis_coff[:, 2:3]
        )
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, coord.size(0))
        acc = agg * self.coords_weight
        return acc

    def transpose_for_scores(self, x):
        """
        Reshape `[B, N, C]` or `[B, M, C]` into multi-head format.

        Args:
            x: torch.Tensor
                Tensor whose last dimension is the hidden width C.
            return: torch.Tensor
                Tensor reshaped to expose the head dimension for attention.
        """
        new_shape = x.shape[:-1] + (self.nhead, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, h, edge_index, coord, coord_pre, vel, edge_attr=None, node_attr=None, short_cut=False):
        """
        Run the GTB-style equivariant block.

        High-level sequence:
        1. Build initial edge messages from node and edge features.
        2. Apply multi-head attention, following the spirit of Eq. (20)-(24):
           query from nodes, key from node+message, value from message,
           softmax weights, neighborhood aggregation, residual node update.
        3. Re-enhance edge messages with the refined node states.
        4. Rebuild a complete local frame on `coord_pre` and vectorize the edge
           coefficients into an equivariant coordinate update.

        Notes on arguments:
        - `coord_pre` is the coordinate state used to construct the frame. This
          is useful when the block predicts an update/acceleration from a prior
          coordinate reference rather than from the tensor being accumulated.
        - `vel` is accepted for interface compatibility with rollout code, but
          is not used directly in this block.
        - If `short_cut` is True, the method returns `coord + acc`; otherwise it
          returns only the predicted update `acc`.

        Args:
            h: torch.Tensor
                Node features with shape [B * n_points, hidden_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Edge index in COO form as (row, col).
            coord: torch.Tensor
                Current coordinate-like tensor being updated, shape [N, 3].
            coord_pre: torch.Tensor
                Coordinates used to construct the local frames, shape [N, 3].
            vel: torch.Tensor
                Velocity tensor kept for API compatibility.
            edge_attr: torch.Tensor or None
                Optional edge features with shape [E, input_edge_nf].
            node_attr: torch.Tensor or None
                Optional extra node conditioning.
            short_cut: bool
                If True, adds the predicted update back to `coord`.
            return: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                Updated coordinate-like tensor, updated node features and refined
                edge features.
        """
        del vel
        row, col = edge_index

        edge_feat = self.edge_model(h[row], h[col], edge_attr)

        B = int(h.shape[0] // self.n_points)
        N = self.n_points
        C = edge_feat.shape[-1]

        query = h.reshape(B, N, C)
        m = edge_feat.reshape(B, N, -1, C)

        h_m = torch.cat(
            [query.unsqueeze(2).repeat(1, 1, N - 1, 1), m],
            dim=-1,
        ).reshape(B, N * (N - 1), -1)

        query = self.transpose_for_scores(self.fc_query(query))
        key = self.transpose_for_scores(self.fc_key(h_m))
        value = self.transpose_for_scores(self.fc_value(edge_feat.reshape(B, N * (N - 1), C)))

        key = key.reshape(B, self.nhead, N, N - 1, -1)
        attention_scores = torch.matmul(query.unsqueeze(-2), key.transpose(-1, -2)).squeeze(-2)
        attention_scores = attention_scores / math.sqrt(C / self.nhead)
        attention_weights = F.softmax(attention_scores, dim=-1)

        m_update = attention_weights.reshape(B, self.nhead, -1).unsqueeze(-1) * value
        att_edge_feat = m_update.transpose(-3, -2)
        att_edge_feat = att_edge_feat.reshape(*att_edge_feat.shape[:-2], -1)
        att_edge_feat = att_edge_feat.reshape(-1, C)

        h, agg = self.node_model(h, edge_index, att_edge_feat, node_attr)
        del agg

        edge_feat = edge_feat + self.edge_model_enhance(h[row], h[col], edge_attr)
        coord_diff, coord_cross, coord_vertical = self.coord2radial(edge_index, coord_pre)
        acc1 = self.acc_model_inner(coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat)

        if short_cut:
            acc = coord + acc1
        else:
            acc = acc1

        return acc, h, edge_feat