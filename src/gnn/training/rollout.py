"""
rollout.py
Takes care of doing rollout.
Only thing from ground truth graph, i.e. is not predicted, is the node force, that acts as an actuator.
"""

import torch
from torch_geometric.data import Data
from src.gnn.preprocessing.build_features import build_edge_features, build_node_features

def do_rollout(model, test_loader, node_stats, edge_stats, target_stats, dt, device, skip_first_num,
               rollout_steps, normalizer, preprocess_cfg, time_discretizer):
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
        skip_first_num: int
            Number of initial steps to skip
        rollout_steps: int
            Maximum number of steps to rollout
        preprocess_cfg: PreprocessYaml
        time_discretizer: TimeDiscretizer
            time discretizer strategy to integrate
        normalizer: Normalizer
            Normalizer

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

        # Reset discretizer internal state (needed for Newmark which tracks previous acceleration)
        time_discretizer.reset()
        # Rollout loop
        for t, graph_ground_truth in enumerate(test_loader):
            graph_ground_truth = graph_ground_truth.to(device, non_blocking=True)
            if rest_pos is None:
                rest_pos = graph_ground_truth.rest_pos
            if t < skip_first_num:
                # skip all steps before this
                continue
            if t > rollout_steps:
                break
            if t == skip_first_num:
                # First prediction t=0: the input graph_ground_truth is the same as the ground truth graph_ground_truth
                input_graph = graph_ground_truth
            # graph_ground_truth is ground truth, input_graph comes from autoregressive
            # Predict --> denormalize prediction --> extract von mises vs acceleration
            pred = model(input_graph.to(device))
            pred_denorm = normalizer.denormalize(pred, target_stats)
            # FIXME: hardcoded prediction target
            a_t = pred_denorm[:, 0:3]
            von_mises_t = pred_denorm[:, 3]
            # Get ground truth (graph_ground_truth) targets
            von_mises_true = graph_ground_truth.y_von_mises
            x_true = graph_ground_truth.x_next
            v_true = graph_ground_truth.v_next
            # Input features (previous time step t) to obtain next step t+1
            x_t = input_graph.x_pos_t
            v_t = input_graph.x_vel_t
            # Time integration: obtain v_predict_tp1, x_predict_tp1 + enforce bcs on fixed nodes
            boundary_condition = graph_ground_truth.x[:, 0]  # 1 = fixed, 0 = free
            x_predict_tp1, v_predict_tp1 = time_discretizer.integrate_acceleration(a_t, v_t, dt, x_t)
            # enforce bcs
            x_predict_tp1[boundary_condition == 1] = x_t[boundary_condition == 1]
            v_predict_tp1[boundary_condition == 1] = 0.0
            # Prepare input for next timestep
            input_graph = _build_next_graph(ground_truth_graph=graph_ground_truth, x_predict_tp1=x_predict_tp1,
                                            v_predict_tp1=v_predict_tp1, x_t=x_t, rest_pos=rest_pos,
                                            node_stats=node_stats,
                                            edge_stats=edge_stats, normalizer=normalizer, preprocess_cfg=preprocess_cfg)
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
                                     acceleration=a_t,
                                     von_mises=von_mises_t,
                                     edge_index=graph_ground_truth.edge_index,
                                     x_element_connectivity=graph_ground_truth.x_element_connectivity,
                                     rest_pos=rest_pos))
    return true_rollout, pred_rollout

def _build_next_graph(ground_truth_graph, x_predict_tp1, v_predict_tp1, x_t, rest_pos, node_stats, edge_stats,
                      normalizer, preprocess_cfg):
    """
    Builds a graph for building the graph of next timestep t+1 from predicted state. Helper for do_rollout.

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
        preprocess_cfg: PreprocessYaml

    :return input_graph
    """
    input_graph = ground_truth_graph.clone()
    # ---------------------- Nodes ----------------------
    # Obtain BCs (cannot change so always ground truth)
    boundary_condition = ground_truth_graph.x[:, 0] # NOTE: hardcoded 0 for bc feature idx, but ok by convention
    categorical_node_features = boundary_condition.unsqueeze(1)
    # Obtain next-step node force (acts as an actuator, so always take from ground truth) and build next time step graph
    # (valued node features) with it, but the rest is predicted.
    ground_truth_force_tp1 = ground_truth_graph.node_force_next
    valued_node_features_tp1 = build_node_features(x_predict_tp1, v_predict_tp1, x_t, rest_pos,
                                                   ground_truth_force_tp1, preprocess_cfg)
    x_node_features = torch.concat(
        [categorical_node_features, normalizer.normalize(valued_node_features_tp1, node_stats)], dim=1)
    # Keep force node features unchanged: deleted
    # _x[-1] = ground_truth_graph.x[-1]

    # --------------- Edges ----------------------
    edge_features_tp1 = build_edge_features(x_predict_tp1, v_predict_tp1, x_t, rest_pos,
                                            ground_truth_force_tp1, ground_truth_graph.edge_index, preprocess_cfg)
    edge_attr_normalized_tp1 = normalizer.normalize(edge_features_tp1, edge_stats)

    # -------------- Final: create next-time–step input graph -----------------
    input_graph.x = x_node_features
    input_graph.edge_attr = edge_attr_normalized_tp1
    input_graph.x_pos_t = x_predict_tp1
    input_graph.x_pos_tm1_prev = x_t
    input_graph.x_vel_t = v_predict_tp1
    # only thing from ground truth graph, i.e. is not predicted
    input_graph.node_force = ground_truth_force_tp1
    return input_graph