"""
gcl.py

Message-passing blocks used by ClofNet and related equivariant GNN baselines.

Paper mapping for the parts that implement ClofNet most directly:
- EGNN baseline update: Eq. (1), Section 2.
- Complete local frame construction: Eq. (2), Section 4.
- Scalarization of vectors into frame coefficients: Eqs. (3)-(4), Section 4.
- Invertibility / vector reconstruction intuition: Eq. (7), Section 4.
- Vectorization back to an equivariant vector: Eq. (8), Section 4.
- Naive ClofNet message-passing block: Eqs. (16)-(19), Appendix A.3.1.
- GTB / attention refinement: Eqs. (20)-(24), Appendix A.3.1.

Important implementation notes:
- `Clof_GCL` is the closest code realization of the ClofNet update rule in this
  file. It learns scalar message features, predicts three scalar coefficients,
  and reconstructs an equivariant vector in the learned local frame.
- The additive velocity term used by `E_GCL_vel` and `Clof_GCL` is a repository
  design choice for dynamics modeling. The paper discusses vector features such
  as velocity in the scalarization inputs, but not this exact additive formula.
- Several classes in this file are baseline or utility layers and are only
  partially related to the ClofNet paper. Their docstrings say so explicitly.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class MLP(nn.Module):
    """
    Small fully-connected adapter network.

    This helper is not a named block in the paper. It is a generic MLP utility
    used to parameterize the learnable functions that the paper denotes with
    symbols such as phi_m, phi_h, phi_q, phi_k and phi_v in Appendix A.3.1.

    The architecture is:
        Linear -> LeakyReLU -> Linear -> LeakyReLU
        -> Linear -> LeakyReLU -> Linear

    Args:
        nin: int
            Input feature dimension.
        nout: int
            Output feature dimension.
        nh: int
            Hidden feature dimension used in all intermediate layers.
    """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        """
        Run the MLP.

        Args:
            x: torch.Tensor
                Input tensor of shape [..., nin].

            return: torch.Tensor
                Output tensor of shape [..., nout].
        """
        return self.net(x)


class GCL_basic(nn.Module, ABC):
    """
    Abstract base interface for edge-first graph convolution layers.

    This is a general graph message-passing skeleton:
        1. build edge messages from source/target node states
        2. aggregate edge messages back to nodes
        3. produce updated node states

    In ClofNet terminology, this mirrors the high-level decomposition of
    message computation and node aggregation used in Eqs. (16)-(17), but it is
    only a generic software abstraction, not a paper-defined module.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def edge_model(self, source, target, edge_attr):
        """
        Compute edgewise messages/features.

        Args:
            source: torch.Tensor
                Source-node features for each directed edge.
            target: torch.Tensor
                Target-node features for each directed edge.
            edge_attr: torch.Tensor or None
                Optional per-edge attributes.

            return: torch.Tensor
                Edgewise features/messages.
        """
        raise NotImplementedError

    @abstractmethod
    def node_model(self, h, edge_index, edge_attr):
        """
        Aggregate edge messages and update nodes.

        Args:
            h: torch.Tensor
                Node features.
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col). `row` is the destination index
                used for aggregation in this repository.
            edge_attr: torch.Tensor
                Per-edge messages/features produced by `edge_model`.

            return: torch.Tensor
                Updated node features.
        """
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        """
        Execute one message-passing step.

        Args:
            x: torch.Tensor
                Node features.
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor or None
                Optional per-edge features.

            return: (node_features, edge_features)
                node_features: torch.Tensor
                    Updated node features.
                edge_features: torch.Tensor
                    Edge messages/features computed by `edge_model`.
        """
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat


class GCL(GCL_basic):
    """
    Generic edge-MLP / node-MLP graph block.

    Relation to the paper:
    - This is not the ClofNet-specific layer.
    - Conceptually, it is closest to the message and node updates in
      Appendix A.3.1, Eqs. (16)-(17), but without the complete-frame
      vectorization step of Eqs. (18)-(19).
    - When `attention=True`, it adds a lightweight message gate, but this is
      not the GTB attention mechanism from Eqs. (20)-(24).

    Update rule implemented here:
        m_ij = edge_mlp([h_i, h_j, e_ij])
        agg_i = sum_j m_ij
        h_i' = node_mlp([h_i, agg_i])

    If `recurrent=True`, the final node update is residual:
        h_i <- h_i + h_i'

    Args:
        input_nf: int
            Input node feature dimension.
        output_nf: int
            Output node feature dimension.
        hidden_nf: int
            Hidden feature dimension.
        edges_in_nf: int
            Extra per-edge input feature dimension concatenated to node pairs.
        act_fn: nn.Module
            Activation used in the MLPs.
        bias: bool
            Whether linear layers use bias terms.
        attention: bool
            Whether to apply a scalar gate to each edge message.
        t_eq: bool
            Legacy flag kept for compatibility. It is unused in this code.
        recurrent: bool
            Whether to add a residual connection on node updates.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_nf=0,
        act_fn=nn.ReLU(),
        bias=True,
        attention=False,
        t_eq=False,
        recurrent=True,
    ):
        super().__init__()
        self.attention = attention
        self.t_eq = t_eq
        self.recurrent = recurrent

        input_edge_nf = input_nf * 2

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn,
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid(),
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias),
        )

    def edge_model(self, source, target, edge_attr):
        """
        Build edge messages from source, target and optional edge attributes.

        Args:
            source: torch.Tensor
                Source-node features for each edge, shape [E, input_nf].
            target: torch.Tensor
                Target-node features for each edge, shape [E, input_nf].
            edge_attr: torch.Tensor or None
                Optional edge attributes, shape [E, edges_in_nf].

            return: torch.Tensor
                Edge message tensor, shape [E, hidden_nf].
        """
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)

        out = self.edge_mlp(edge_in)

        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att

        return out

    def node_model(self, h, edge_index, edge_attr):
        """
        Aggregate edge messages and update node states.

        This is the software analogue of the aggregation in paper Eq. (17):
            h_i^{l+1} = phi_h(h_i^l, sum_j m_ij)

        Args:
            h: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col). Messages are aggregated into
                the indices in `row`.
            edge_attr: torch.Tensor
                Edge messages, shape [E, hidden_nf].

            return: torch.Tensor
                Updated node features, shape [N, output_nf].
        """
        row, col = edge_index
        del col

        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)

        if self.recurrent:
            out = out + h

        return out


class GCL_rf(GCL_basic):
    """
    Radial-force coordinate update block without learned node features.

    Relation to the paper:
    - This is a baseline-style block, not a ClofNet block.
    - It is closest to the radial-only coordinate update idea used by EGNN
      in Eq. (1), where motion is constrained to the relative direction
      x_i - x_j.
    - It does not implement the complete local frame of Eq. (2), so it cannot
      express the richer vectorization used by ClofNet Eq. (8).

    Implemented edge force:
        x_diff = x_i - x_j
        r_ij = ||x_i - x_j||
        e_out = phi([r_ij, e_ij])
        m_ij = x_diff * e_out

    Node update:
        x_i <- x_i + mean_j m_ij - reg * x_i

    Args:
        nf: int
            Hidden dimension inside the radial force MLP.
        edge_attr_nf: int
            Number of extra edge attributes appended to the radial distance.
        reg: float
            Damping coefficient applied directly to coordinates.
        act_fn: nn.Module
            Activation used in the radial force MLP.
        clamp: bool
            Whether to clamp force vectors elementwise to [-100, 100].
    """

    def __init__(
        self,
        nf=64,
        edge_attr_nf=0,
        reg=0,
        act_fn=nn.LeakyReLU(0.2),
        clamp=False,
    ):
        super().__init__()
        self.clamp = clamp
        self.reg = reg

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi = nn.Sequential(
            nn.Linear(edge_attr_nf + 1, nf),
            act_fn,
            layer,
        )

    def edge_model(self, source, target, edge_attr):
        """
        Predict radial force vectors per edge.

        Args:
            source: torch.Tensor
                Source coordinates, shape [E, 3].
            target: torch.Tensor
                Target coordinates, shape [E, 3].
            edge_attr: torch.Tensor
                Extra edge attributes, shape [E, edge_attr_nf].

            return: torch.Tensor
                Edge force vectors, shape [E, 3].
        """
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out

        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)

        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        """
        Aggregate radial force vectors and update coordinates.

        Args:
            x: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor
                Per-edge force vectors, shape [E, 3].

            return: torch.Tensor
                Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x * self.reg
        return x_out


class GCL_rf_vel(nn.Module):
    """
    Radial-force block with an additional learned velocity drift term.

    Relation to the paper:
    - This is a dynamics-oriented utility block in the repository.
    - The radial part is still closest to EGNN Eq. (1), not to ClofNet's
      complete-frame vectorization.
    - The additive velocity term is an implementation choice.

    Update rule:
        m_ij = (x_i - x_j) * phi([||x_i - x_j||, e_ij])
        x_i <- x_i + coords_weight * mean_j m_ij + vel_i * psi(||vel_i||)

    Args:
        nf: int
            Hidden dimension in the MLPs.
        edge_attr_nf: int
            Number of extra edge attributes.
        act_fn: nn.Module
            Activation used in the MLPs.
        coords_weight: float
            Scaling factor for the aggregated coordinate update.
    """

    def __init__(
        self,
        nf=64,
        edge_attr_nf=0,
        act_fn=nn.LeakyReLU(0.2),
        coords_weight=1.0,
    ):
        super().__init__()
        self.coords_weight = coords_weight

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1),
        )

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi = nn.Sequential(
            nn.Linear(1 + edge_attr_nf, nf),
            act_fn,
            layer,
            nn.Tanh(),
        )

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        """
        Update coordinates using radial messages plus a velocity drift.

        Args:
            x: torch.Tensor
                Coordinates, shape [N, 3].
            vel_norm: torch.Tensor
                Velocity norms, shape [N, 1].
            vel: torch.Tensor
                Velocities, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor or None
                Optional per-edge attributes.

            return: (coord, edge_attr)
                coord: torch.Tensor
                    Updated coordinates, shape [N, 3].
                edge_attr: torch.Tensor or None
                    Returned unchanged for interface compatibility.
        """
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x = x + vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        """
        Build radial coordinate messages.

        Args:
            source: torch.Tensor
                Source coordinates, shape [E, 3].
            target: torch.Tensor
                Target coordinates, shape [E, 3].
            edge_attr: torch.Tensor
                Extra per-edge attributes, shape [E, edge_attr_nf].

            return: torch.Tensor
                Per-edge coordinate messages, shape [E, 3].
        """
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        """
        Aggregate radial messages and update coordinates.

        Args:
            x: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_m: torch.Tensor
                Per-edge coordinate messages, shape [E, 3].

            return: torch.Tensor
                Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


class E_GCL(nn.Module):
    """
    Equivariant graph convolution layer with EGNN-style radial coordinate update.

    Relation to the paper:
    - This block is closest to the EGNN baseline summarized in Eq. (1), which
      the ClofNet paper explicitly references before introducing complete local
      frames.
    - It is not yet the ClofNet block because coordinate updates are restricted
      to the normalized relative direction `coord_diff` rather than the full
      local frame `(a, b, c)` of Eq. (2).

    Core equations represented here:
        m_ij = phi_m(h_i, h_j, r_ij, e_ij)
        x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        h_i' = phi_h(h_i, sum_j m_ij)

    Here:
        r_ij = ||x_i - x_j||^2

    Args:
        input_nf: int
            Input node feature dimension.
        output_nf: int
            Output node feature dimension.
        hidden_nf: int
            Hidden dimension.
        edges_in_d: int
            Extra edge feature dimension.
        nodes_att_dim: int
            Extra node attribute dimension concatenated before node update.
        act_fn: nn.Module
            Activation function used by the MLPs.
        recurrent: bool
            Whether node updates are residual.
        coords_weight: float
            Scaling factor applied to the aggregated coordinate update.
        attention: bool
            Whether to gate edge features with a learned scalar attention.
        clamp: bool
            Legacy flag kept for compatibility. Coordinate updates are always
            clamped in this implementation.
        norm_diff: bool
            Whether to normalize coordinate differences by sqrt(radial) + 1.
        tanh: bool
            Whether to squash the coordinate coefficient with tanh.
        out_basis_dim: int
            Output dimension of `coord_mlp`. For EGNN-style updates this is 1;
            for ClofNet frame-based vectorization this becomes 3.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        clamp=False,
        norm_diff=False,
        tanh=False,
        out_basis_dim=1,
    ):
        super().__init__()
        input_edge = input_nf * 2
        edge_coords_nf = 1

        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.clamp = clamp

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, out_basis_dim, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        ]

        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3

        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

    def edge_model(self, source, target, radial, edge_attr):
        """
        Compute edge features/messages from node states and radial geometry.

        Args:
            source: torch.Tensor
                Source-node features, shape [E, input_nf].
            target: torch.Tensor
                Target-node features, shape [E, input_nf].
            radial: torch.Tensor
                Squared pairwise distances, shape [E, 1].
            edge_attr: torch.Tensor or None
                Optional extra edge attributes, shape [E, edges_in_d].

            return: torch.Tensor
                Edge features/messages, shape [E, hidden_nf].
        """
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)

        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        Aggregate messages and update node embeddings.

        This is the code-level realization of the node update role of Eq. (17):
            h_i^{l+1} = phi_h(h_i^l, sum_j m_ij)

        Args:
            x: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor
                Edge messages/features, shape [E, hidden_nf].
            node_attr: torch.Tensor or None
                Optional extra node attributes concatenated before the node MLP.

            return: (out, agg_input)
                out: torch.Tensor
                    Updated node features, shape [N, output_nf].
                agg_input: torch.Tensor
                    The concatenated tensor fed to the node MLP.
        """
        row, col = edge_index
        del col

        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)

        out = self.node_mlp(agg)

        if self.recurrent:
            out = x + out

        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        Update coordinates along the relative direction only.

        This is the EGNN-style coordinate update corresponding to Eq. (1):
            x_i <- x_i + sum_j (x_i - x_j) * phi_x(m_ij)

        In this implementation:
            trans_ij = coord_diff_ij * coord_mlp(edge_feat_ij)

        and nodewise updates are the mean over incoming edge transforms.

        Args:
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord_diff: torch.Tensor
                Relative directions, shape [E, 3].
            edge_feat: torch.Tensor
                Edge messages/features, shape [E, hidden_nf].

            return: torch.Tensor
                Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)

        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        """
        Compute squared distances and relative directions for each edge.

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].

            return: (radial, coord_diff)
                radial: torch.Tensor
                    Squared distances, shape [E, 1].
                coord_diff: torch.Tensor
                    Relative directions x_i - x_j, shape [E, 3].
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        """
        Run one equivariant message-passing layer.

        Args:
            h: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            edge_attr: torch.Tensor or None
                Optional edge features, shape [E, edges_in_d].
            node_attr: torch.Tensor or None
                Optional node features, shape [N, nodes_att_dim].

            return: (h, coord, edge_attr)
                h: torch.Tensor
                    Updated node features.
                coord: torch.Tensor
                    Updated coordinates.
                edge_attr: torch.Tensor or None
                    Returned unchanged for interface compatibility.
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        del agg

        return h, coord, edge_attr


class E_GCL_vel(E_GCL):
    """
    EGNN-style equivariant block with an additional learned velocity term.

    Relation to the paper:
    - Closest to the EGNN baseline update and not the full ClofNet block.
    - The velocity addition is a repository-side dynamics adaptation.

    Coordinate update:
        x_i <- x_i
             + mean_j ((x_i - x_j) * phi_x(m_ij))
             + psi(h_i) * vel_i

    Args:
        input_nf: int
            Input node feature dimension.
        output_nf: int
            Output node feature dimension.
        hidden_nf: int
            Hidden feature dimension.
        edges_in_d: int
            Extra edge feature dimension.
        nodes_att_dim: int
            Extra node attribute dimension.
        act_fn: nn.Module
            Activation used by the MLPs.
        recurrent: bool
            Whether node updates are residual.
        coords_weight: float
            Scaling for the aggregated coordinate update.
        attention: bool
            Whether to gate edge features.
        norm_diff: bool
            Whether to normalize relative directions.
        tanh: bool
            Whether to apply tanh on the scalar coefficient used in the radial
            coordinate update.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        norm_diff=False,
        tanh=False,
    ):
        super().__init__(
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_att_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
            norm_diff=norm_diff,
            tanh=tanh,
        )
        self.norm_diff = norm_diff

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        """
        Run one layer with both radial equivariant updates and velocity drift.

        Args:
            h: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            vel: torch.Tensor
                Velocities, shape [N, 3].
            edge_attr: torch.Tensor or None
                Optional edge features.
            node_attr: torch.Tensor or None
                Optional node features.

            return: (h, coord, edge_attr)
                h: torch.Tensor
                    Updated node features.
                coord: torch.Tensor
                    Updated coordinates.
                edge_attr: torch.Tensor or None
                    Returned unchanged.
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        coord = coord + self.coord_mlp_vel(h) * vel

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        del agg

        return h, coord, edge_attr


class Clof_GCL(E_GCL):
    """
    ClofNet graph block with complete local frames.

    This is the key ClofNet layer in the file.

    Paper mapping:
    - Eq. (2): build a complete local frame
          a_ij = (x_i - x_j) / ||x_i - x_j||
          b_ij = (x_i x x_j) / ||x_i x x_j||
          c_ij = a_ij x b_ij
    - Eqs. (3)-(4): scalarization turns vectors into invariant coefficients
      relative to that frame.
    - Eq. (8): vectorization reconstructs an equivariant vector from three
      scalar coefficients and the basis vectors.
    - Eqs. (16)-(19): message passing learns invariant messages, then those
      messages are vectorized back into coordinate updates.

    How the code realizes Eq. (8):
    - `edge_model(...)` produces invariant edge embeddings `edge_feat`.
    - `coord_mlp(edge_feat)` predicts three scalar coefficients:
          coff = [alpha_ij, beta_ij, gamma_ij]
    - `coord_model(...)` reconstructs the vector update
          trans_ij = alpha_ij * a_ij + beta_ij * b_ij + gamma_ij * c_ij
      where this repository uses:
          a_ij = coord_diff
          b_ij = coord_cross
          c_ij = coord_vertical
    - Those vectorized edge updates are averaged per node and added to the
      coordinates.

    The extra term
        coord += coord_mlp_vel(h) * vel
    is not a paper equation. It is a repository-level dynamics addition.

    Args:
        input_nf: int
            Input node feature dimension.
        output_nf: int
            Output node feature dimension.
        hidden_nf: int
            Hidden feature dimension.
        edges_in_d: int
            Extra edge feature dimension.
        nodes_att_dim: int
            Extra node attribute dimension.
        act_fn: nn.Module
            Activation used by the MLPs.
        recurrent: bool
            Whether node updates are residual.
        coords_weight: float
            Scaling for vectorized coordinate updates.
        attention: bool
            Whether to gate edge features.
        norm_diff: bool
            Whether to normalize basis vectors using sqrt(radial) + 1 and the
            analogous norm for the cross-product direction.
        tanh: bool
            Whether to tanh-squash the three basis coefficients predicted by
            `coord_mlp`.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        norm_diff=False,
        tanh=False,
    ):
        super().__init__(
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_att_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
            norm_diff=norm_diff,
            tanh=tanh,
            out_basis_dim=3,
        )
        self.norm_diff = norm_diff

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.layer_norm = nn.LayerNorm(hidden_nf)

    def coord2localframe(self, edge_index, coord):
        """
        Construct the complete local frame per edge.

        This is the code counterpart of paper Eq. (2):
            a_ij = (x_i - x_j) / ||x_i - x_j||
            b_ij = (x_i x x_j) / ||x_i x x_j||
            c_ij = a_ij x b_ij

        Implementation details:
        - The repository uses `+ 1` instead of a tiny epsilon in the
          normalization denominator, mirroring the broader codebase style.
        - The frame is "complete" because it spans three directions instead of
          only the radial direction, which addresses the directional
          degeneracy problem discussed in Section 3.

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].

            return: (radial, coord_diff, coord_cross, coord_vertical)
                radial: torch.Tensor
                    Squared distances, shape [E, 1].
                coord_diff: torch.Tensor
                    First basis vector a_ij, shape [E, 3].
                coord_cross: torch.Tensor
                    Second basis vector b_ij, shape [E, 3].
                coord_vertical: torch.Tensor
                    Third basis vector c_ij, shape [E, 3].
        """
        row, col = edge_index

        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1).unsqueeze(1)

        coord_cross = torch.cross(coord[row], coord[col])

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

            cross_norm = torch.sqrt(torch.sum(coord_cross ** 2, dim=1).unsqueeze(1)) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return radial, coord_diff, coord_cross, coord_vertical

    def coord_model(self, coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat):
        """
        Vectorize learned scalar coefficients back into a 3D coordinate update.

        This is the main implementation of the paper's vectorization idea in
        Eq. (8). The learned edge embedding `edge_feat` is mapped to three
        scalar coefficients:
            coff_ij = [alpha_ij, beta_ij, gamma_ij]

        Then the update is reconstructed in the local frame:
            trans_ij =
                alpha_ij * a_ij
              + beta_ij  * b_ij
              + gamma_ij * c_ij

        Finally, edgewise transforms are averaged per destination node and
        added to the coordinates.

        Args:
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord_diff: torch.Tensor
                First basis vector a_ij, shape [E, 3].
            coord_cross: torch.Tensor
                Second basis vector b_ij, shape [E, 3].
            coord_vertical: torch.Tensor
                Third basis vector c_ij, shape [E, 3].
            edge_feat: torch.Tensor
                Invariant edge embeddings/messages, shape [E, hidden_nf].

            return: torch.Tensor
                Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        coff = self.coord_mlp(edge_feat)

        trans = (
            coord_diff * coff[:, :1]
            + coord_cross * coff[:, 1:2]
            + coord_vertical * coff[:, 2:3]
        )

        trans = torch.clamp(trans, min=-100, max=100)

        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg * self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        """
        Run one ClofNet block.

        Flow relative to the paper:
            1. rebuild the local frame from current coordinates
            2. compute invariant edge messages
            3. vectorize those messages into coordinate updates
            4. add an implementation-specific velocity drift
            5. aggregate messages into node embeddings

        Notes on residuals:
        - `node_model(...)` may already add a residual when `recurrent=True`.
        - This class then adds another residual:
              h = residue + h
          followed by `LayerNorm`.
        - This double-residual structure is preserved exactly from the original
          repository code.

        Args:
            h: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            vel: torch.Tensor
                Velocities, shape [N, 3].
            edge_attr: torch.Tensor or None
                Optional edge features, shape [E, edges_in_d].
            node_attr: torch.Tensor or None
                Optional node features, shape [N, nodes_att_dim].

            return: (h, coord, edge_attr)
                h: torch.Tensor
                    Updated node features.
                coord: torch.Tensor
                    Updated coordinates.
                edge_attr: torch.Tensor or None
                    Returned unchanged for interface compatibility.
        """
        row, col = edge_index
        residue = h

        radial, coord_diff, coord_cross, coord_vertical = self.coord2localframe(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat)

        coord = coord + self.coord_mlp_vel(h) * vel

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        del agg

        h = residue + h
        h = self.layer_norm(h)

        return h, coord, edge_attr


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    TensorFlow-style unsorted segment sum using `scatter_add_`.

    Relation to the paper:
    - This helper implements the neighborhood aggregation pattern appearing in
      the sum over neighbors in Eqs. (17), (19) and (22).
    - The input `segment_ids` indicates which node each edge contribution is
      aggregated into.

    Args:
        data: torch.Tensor
            Values to aggregate, shape [E, D].
        segment_ids: torch.Tensor
            Destination node indices, shape [E].
        num_segments: int
            Number of output segments / nodes.

        return: torch.Tensor
            Aggregated tensor, shape [num_segments, D].
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)

    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """
    TensorFlow-style unsorted segment mean using `scatter_add_`.

    Relation to the paper:
    - This helper implements normalized neighbor aggregation for coordinate
      updates, analogous to the averaged neighbor contribution used in code for
      Eqs. (19) and (22).

    Args:
        data: torch.Tensor
            Values to aggregate, shape [E, D].
        segment_ids: torch.Tensor
            Destination node indices, shape [E].
        num_segments: int
            Number of output segments / nodes.

        return: torch.Tensor
            Mean-aggregated tensor, shape [num_segments, D].
    """
    result_shape = (num_segments, data.size(1))

    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))

    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)

    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))

    return result / count.clamp(min=1)