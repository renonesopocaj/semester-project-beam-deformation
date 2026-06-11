import torch
from torch_geometric.data import Data

from src.gnn.models.transolver.transolvernet import TransolverNet
from src.gnn.scripts.attention_transolver_visualization import _effective_node_attention_matrix


def _make_graph(num_nodes, node_channels):
    edge_src = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 3, 4, 0, 0, 1, 2], dtype=torch.long)
    return Data(
        x=torch.randn(num_nodes, node_channels),
        x_pos_t=torch.rand(num_nodes, 3),
        edge_index=torch.stack([edge_src, edge_dst], dim=0),
        t=torch.tensor([0.25], dtype=torch.float32),
    )


def test_transolver_attention_capture_exposes_effective_node_matrix():
    graph = _make_graph(num_nodes=5, node_channels=7)
    model = TransolverNet(
        graph_template=graph,
        node_channels=7,
        n_layers=2,
        n_hidden=16,
        dropout=0.0,
        n_head=4,
        time_input=True,
        act="gelu",
        mlp_ratio=1,
        slice_num=3,
        ref=2,
        unified_pos=False,
        stress_out_dim=6,
    )
    model.set_attention_capture(True)

    pred = model(graph)
    assert tuple(pred.shape) == (5, 9)

    attention_states = model.get_last_attention_states()
    assert len(attention_states) == 2
    for state in attention_states:
        assert tuple(state["slice_weights"].shape) == (1, 4, 5, 3)
        assert tuple(state["slice_norm"].shape) == (1, 4, 3)
        assert tuple(state["slice_attention"].shape) == (1, 4, 3, 3)

    matrix = _effective_node_attention_matrix(attention_states[-1], head_index=None)
    assert tuple(matrix.shape) == (5, 5)
    assert torch.allclose(matrix.sum(dim=1), torch.ones(5), atol=1e-5)
