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

- The additive velocity term used by `E_GCL_vel` and `Clof_GCL` is a repository design choice for dynamics modeling.
  The paper discusses vector features such as velocity in the scalarization inputs, but not this exact additive formula.

- Several classes in this file are baseline or utility layers and are only partially related to the ClofNet paper.
  Their docstrings say so explicitly.
"""

from abc import ABC, abstractmethod
from src.gnn.models.clofnet.clof_aggregators import unsorted_segment_sum
import torch
from torch import nn

class AbstractGCL(nn.Module, ABC):
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
    def edge_model_forward(self, source, target, edge_attr):
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
    def node_model_forward(self, h, edge_index, edge_attr):
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
        edge_feat = self.edge_model_forward(x[row], x[col], edge_attr)
        x = self.node_model_forward(x, edge_index, edge_feat)
        return x, edge_feat


class VanillaGCL(AbstractGCL):
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
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False,
                 t_eq=False, skip_connection=True):
        """
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
            skip_connection: bool
                Whether to add a residual connection on node updates.
        """
        super().__init__()
        self.attention = attention
        self.t_eq = t_eq
        self.skip_connection = skip_connection
        input_edge_nf = input_nf * 2
        # Edge MLP
        self.edge_mlp = nn.Sequential(nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
                                      act_fn,
                                      nn.Linear(hidden_nf, hidden_nf, bias=bias),
                                      act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(input_nf, hidden_nf, bias=bias),
                                         act_fn,
                                         nn.Linear(hidden_nf, 1, bias=bias),
                                         nn.Sigmoid())

        self.node_mlp = nn.Sequential(nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
                                      act_fn,
                                      nn.Linear(hidden_nf, output_nf, bias=bias))

    def edge_model_forward(self, source, target, edge_attr):
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
        out_pred = self.edge_mlp(edge_in)
        if self.attention:
            # I do a term by term mul: this is basically a "soft importance" weight on each edge message, for which given
            # an edge e_ij i have a weight and if the weight is near to 0 the edge is ignored. Since the gate is
            # a scalar computed from invariant edge features, it does not break the equivariance argument.
            att = self.att_mlp(torch.abs(source - target))
            out_pred = out_pred * att

        return out_pred

    def node_model_forward(self, h, edge_index, edge_attr):
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
        row = edge_index[0]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        out_pred = self.node_mlp(in_node)
        if self.skip_connection: # residual connection
            out_pred = out_pred + h
        return out_pred