"""
rollout.py
Takes care of doing rollout.
"""

import torch
from torch_geometric.data import Data
from src.gnn.preprocessing.build_features import build_edge_features, build_valued_node_features

def do_rollout(model, test_loader, node_stats, edge_stats, target_stats, dt, device, skip_first,
               rollout_steps, normalizer, rest_minus_nonrest_pos):
    """
    Perform autoregressive rollout using the model.
    1. Predict --> denormalize prediction --> extract von mises vs acceleration
    2. Semi-implicit Euler integration: obtain v_predict_tp1, x_predict_tp1 --> enforce bcs on fixed nodes
    3. Build the whole predicted graph, but enforce ground-truth node force (which acts as actuator)
    4. Add edges

    Args:
        model: torch.nn.Module
            Trained MeshGraphNet
        test_loader: DataLoader
            for test data
        node_stats: Stats
            Normalization statistics
        edge_stats: Stats
            Normalization statistics
        target_stats: Stats
            Normalization statistics
        dt: float
            Time step size
        device: torch.device
            Computation device
        skip_first: int
            Number of initial steps to skip
        rollout_steps: int
            Maximum number of steps to rollout
        rest_minus_nonrest_pos: bool
            if true adds xij-uij for both node and edge features

    Returns:
        true_rollout: list[Data]
            List of ground truth graphs
        pred_rollout: list[Data]
            List of predicted graphs
    """
    with torch.no_grad():
        true_rollout, pred_rollout = [], []
        rest_pos = None
        node_stats.to(device)
        edge_stats.to(device)
        target_stats.to(device)

        for t, graph_ground_truth in enumerate(test_loader):
            if rest_pos is None:
                rest_pos = graph_ground_truth.rest_pos
            if t < skip_first:
                continue
            if t > rollout_steps:
                break
            if t == skip_first:
                # First prediction t=0: the input graph_ground_truth is the same as the ground truth graph_ground_truth
                input_graph = graph_ground_truth
            # graph_ground_truth is ground truth, input_graph comes from autoregressive

            # Predict --> denormalize prediction --> extract von mises vs acceleration
            pred = model(input_graph.to(device))
            pred_denorm = normalizer.denormalize(pred, target_stats)
            a = pred_denorm[:, 0:3]
            von_mises = pred_denorm[:, 3]
            # Get ground truth (graph_ground_truth) targets
            von_mises_true = graph_ground_truth.y_von_mises
            x_true = graph_ground_truth.x_next
            v_true = graph_ground_truth.v_next
            # Input features (previous time step t) to obtain next step t+1
            x = input_graph.x_pos_t
            v = input_graph.x_vel_t
            # Semi-implicit Euler integration: obtain v_predict_tp1, x_predict_tp1 --> enforce bcs on fixed nodes
            boundary_condition = graph_ground_truth.x[:, 0]  # 1 = fixed, 0 = free
            v_predict_tp1 = v + a * dt
            x_predict_tp1 = x + v_predict_tp1 * dt
            x_predict_tp1[boundary_condition == 1] = x[boundary_condition == 1]
            v_predict_tp1[boundary_condition == 1] = 0.0
            # Prepare input for next timestep
            input_graph = _build_next_graph(ground_truth_graph=graph_ground_truth, x_predict_tp1=x_predict_tp1,
                                            v_predict_tp1=v_predict_tp1, rest_pos=rest_pos, node_stats=node_stats,
                                            edge_stats=edge_stats, normalizer=normalizer,
                                            rest_minus_nonrest_pos=rest_minus_nonrest_pos)
            # Store rollouts
            true_rollout.append(Data(x_pos_t=x_true,
                                     x_vel_t=v_true,
                                     acceleration=graph_ground_truth.y_acc_t,
                                     von_mises=von_mises_true,
                                     edge_index=graph_ground_truth.edge_index,
                                     x_element_connectivity=graph_ground_truth.x_element_connectivity,
                                     rest_pos=rest_pos))
            pred_rollout.append(Data(x_pos_t=x_predict_tp1,
                                     x_vel_t=v_predict_tp1,
                                     acceleration=a,
                                     von_mises=von_mises,
                                     edge_index=graph_ground_truth.edge_index,
                                     x_element_connectivity=graph_ground_truth.x_element_connectivity,
                                     rest_pos=rest_pos))
    return true_rollout, pred_rollout

def _build_next_graph(ground_truth_graph, x_predict_tp1, v_predict_tp1, rest_pos, node_stats, edge_stats, normalizer,
                     rest_minus_nonrest_pos):
    """
    Builds a graph for building the graph of next timestep from predicted state. Helper for do_rollout.

    Args:
        ground_truth_graph: Data
            current step graph
        x_predict_tp1: torch.Tensor
            position predicted for time step t+1 (via integration from acceleration prediction)
        v_predict_tp1: torch.Tensor
            velocity predicted for time step t+1 (via integration from acceleration prediction)
        rest_pos: torch.Tensor
            rest position (always the same)
        node_stats: Stats
        edge_stats: Stats
        normalizer: Normalizer

    :return input_graph
    """
    input_graph = ground_truth_graph
    # ---------------------- Nodes ----------------------
    # Obtain BCs (cannot change so always ground truth)
    # FIXME: hardcoded 0
    boundary_condition = ground_truth_graph.x[:, 0]
    categorical_node_features = boundary_condition.unsqueeze(1)
    # Obtain next-step node force (acts as an actuator, so always take from ground truth) and build next time step graph
    # (valued node features) with it, but the rest is predicted
    ground_truth_force_tp1 = ground_truth_graph.node_force_next
    valued_node_features = build_valued_node_features(x_predict_tp1, v_predict_tp1, rest_pos,
                                                      ground_truth_force_tp1,
                                                      rest_minus_nonrest_pos)
    _x = torch.concat([categorical_node_features, normalizer.normalize(valued_node_features, node_stats)],
                      dim=1)
    # Keep force node features unchanged
    # _x[-1] = ground_truth_graph.x[-1]
    # --------------- Edges ----------------------
    edge_features = build_edge_features(x_predict_tp1, rest_pos, ground_truth_graph.edge_index, rest_minus_nonrest_pos)
    _edge_attr_normalized = normalizer.normalize(edge_features, edge_stats)

    # -------------- Final: create next-time–step input graph -----------------
    input_graph.x = _x
    input_graph.edge_attr = _edge_attr_normalized
    input_graph.x_pos_t = x_predict_tp1
    input_graph.x_vel_t = v_predict_tp1
    # only thing from ground truth graph, i.e. is not predicted
    input_graph.node_force = ground_truth_force_tp1
    return input_graph

