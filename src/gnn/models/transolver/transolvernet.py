"""
transolvernet.py

PyG adapter around the irregular-mesh Transolver Model for one-step prediction of
normalized acceleration (3) and von Mises stress (1), matching the MeshGraphNet output layout.
"""

import torch
from torch import nn

from src.gnn.models.transolver.transolver_irregular_mesh import Model as TransolverIrregularModel


def _num_nodes_from_template(graph_template):
    """
    Infer the number of mesh nodes from one PyG Data or Batch (same convention as MAGNet).

    Args:
        graph_template: object
            PyG Data or Batch used only to read ptr or single-graph node count.

    return: int
        Node count N for one graph in the template.
    """
    if hasattr(graph_template, "ptr") and graph_template.ptr is not None:
        return int((graph_template.ptr[1] - graph_template.ptr[0]).item())
    return int(graph_template.x.shape[0])


class TransolverNet(nn.Module):
    """
    Wraps irregular-mesh Transolver: geometry g from x_pos_t, observed field u from x,
    returns (N_total, 4) for trainer and rollout.
    """

    num_nodes: int
    fun_dim: int
    out_dim: int
    time_input: bool
    model: TransolverIrregularModel

    def __init__(self, graph_template, node_channels, n_layers, n_hidden, dropout, n_head, time_input, act,
                 mlp_ratio, slice_num, ref, unified_pos):
        """
        Args:
            graph_template: object
                One collated batch or single graph; defines fixed N per graph.
            node_channels: int
                Width of batch.x (BC + valued node features).
            n_layers: int
                Number of Transolver blocks.
            n_hidden: int
                Hidden channel width C.
            dropout: float
                Dropout inside attention and MLP.
            n_head: int
                Attention head count.
            time_input: bool
                If True, pass per-graph scalar time into the backbone time embedding branch.
            act: str
                Activation name for MLP blocks.
            mlp_ratio: int
                FFN expansion ratio inside each block.
            slice_num: int
                Physics slice count M.
            ref: int
                Reference grid side for unified_pos distance features.
            unified_pos: bool
                If True, backbone replaces raw coords with ref-grid distance features (2D ref grid in backbone).
        """
        super().__init__()
        self.num_nodes = _num_nodes_from_template(graph_template)
        self.fun_dim = int(node_channels)
        self.out_dim = 4
        self.time_input = bool(time_input)
        self.model = TransolverIrregularModel(
            space_dim=3,
            n_layers=int(n_layers),
            n_hidden=int(n_hidden),
            dropout=float(dropout),
            n_head=int(n_head),
            Time_Input=self.time_input,
            act=str(act),
            mlp_ratio=int(mlp_ratio),
            fun_dim=self.fun_dim,
            out_dim=self.out_dim,
            slice_num=int(slice_num),
            ref=int(ref),
            unified_pos=bool(unified_pos),
        )

    def forward(self, ingraph):
        """
        Args:
            ingraph: object
                PyG Data or Batch with x_pos_t, x, num_graphs, and optionally ptr and t.

        return: torch.Tensor
            Predictions with shape (N_total, 4): acceleration channels then von Mises scalar.
        """
        if hasattr(ingraph, "ptr") and ingraph.ptr is not None:
            batch_size = int(ingraph.num_graphs)
            node_counts = ingraph.ptr[1:] - ingraph.ptr[:-1]
            if not torch.all(node_counts == self.num_nodes):
                raise ValueError("TransolverNet requires every batched graph to have the same node count as the template.")
        else:
            batch_size = 1
            if int(ingraph.x.shape[0]) != self.num_nodes:
                raise ValueError("TransolverNet single-graph input does not match the template node count.")

        g = ingraph.x_pos_t.reshape(batch_size, self.num_nodes, 3)
        u = ingraph.x.reshape(batch_size, self.num_nodes, self.fun_dim)
        if self.time_input:
            time_vec = ingraph.t.view(-1).float()
            pred = self.model(g, u, time_vec)
        else:
            pred = self.model(g, u, None)
        return pred.reshape(batch_size * self.num_nodes, self.out_dim)
