import torch
from torch_geometric.data import Data

def do_rollout(model, test_loader, node_stats, edge_stats, target_stats, dt, device, skip_first,
               rollout_steps, normalizer, rest_minus_nonrest_pos):
    """
    Perform autoregressive rollout using the model.

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
        u = None

        for t, graph in enumerate(test_loader):
            if u is None:
                u = graph.rest_pos
            if t < skip_first:
                continue
            if t > rollout_steps:
                break
            if t == skip_first:
                input_graph = graph

            # Predict
            pred = model(input_graph.to(device))
            input_graph = input_graph.cpu()
            pred = pred.cpu()

            pred_denorm = normalizer.denormalize(pred, target_stats)
            von_mises_true = graph.y_von_mises

            a = pred_denorm[:, :3]
            von_mises = pred_denorm[:, 3]

            x = input_graph.x_pos_t
            v = input_graph.x_vel_t

            # Semi-implicit Euler integration
            boundary_condition = graph.x[:, 0]  # 1 = fixed, 0 = free

            v_new = v + a * dt
            x_new = x + v_new * dt

            # Enforce boundary conditions
            x_new[boundary_condition == 1] = x[boundary_condition == 1]
            v_new[boundary_condition == 1] = 0.0

            # Ground truth
            x_true = graph.x_next
            v_true = graph.v_next

            # Prepare input for next timestep
            input_graph = build_next_graph(simulation_graph=graph, x_new=x_new, v_new=v_new, rest_pos=u,
                                           node_stats=node_stats, edge_stats=edge_stats, normalizer=normalizer,
                                           rest_minus_nonrest_pos=rest_minus_nonrest_pos)

            # Store rollouts
            true_rollout.append(Data(x_pos_t=x_true,
                                     x_vel_t=v_true,
                                     von_mises=von_mises_true,
                                     x_element_connectivity=graph.x_element_connectivity))
            pred_rollout.append(Data(x_pos_t=x_new,
                                     x_vel_t=v_new,
                                     von_mises=von_mises,
                                     x_element_connectivity=graph.x_element_connectivity))

    return true_rollout, pred_rollout

def build_next_graph(simulation_graph, x_new, v_new, rest_pos, node_stats, edge_stats, normalizer,
                     rest_minus_nonrest_pos):
    """
    Builds a graph for building the graph of next timestep from predicted state (used in rollout).

    Args:
        simulation_graph: Data
            current step graph
        x_new: torch.Tensor
        v_new: torch.Tensor
        rest_pos: torch.Tensor
        node_stats: Stats
        edge_stats: Stats
        normalizer: Stats

    :return input_graph
    """
    input_graph = simulation_graph
    # ---------------------- Nodes ----------------------
    # ground truth
    boundary_condition = simulation_graph.x[:, 0]
    categorical_node_features = boundary_condition.unsqueeze(1)
    # FIXME: why next step?
    node_force = simulation_graph.node_force_next
    # concat into one node feature
    if rest_minus_nonrest_pos:
        x_minus_u = x_new - v_new
        valued_node_features = torch.concat([x_new, v_new, rest_pos, node_force, x_minus_u], dim=1)
    else:
        valued_node_features = torch.concat([x_new, v_new, rest_pos, node_force], dim=1)
    _x = torch.concat([categorical_node_features, normalizer.normalize(valued_node_features, node_stats)],
                      dim=1)
    _x[-1] = simulation_graph.x[-1]  # Keep force node features unchanged

    # --------------- Edges ----------------------
    # ground truth
    src, dst = simulation_graph.edge_index
    # prediction edge features
    xij = x_new[dst] - x_new[src]
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1)
    uij = rest_pos[dst] - rest_pos[src]
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1)
    # concat into one node feature
    if rest_minus_nonrest_pos:
        xij_minus_uij = xij - uij
        xij_minus_uij_norm = torch.norm(xij - uij, dim=1).unsqueeze(1)
        edge_features = torch.concat([xij, uij, xij_norm, uij_norm, xij_minus_uij, xij_minus_uij_norm], dim=1)
    else:
        edge_features = torch.concat([xij, uij, xij_norm, uij_norm], dim=1)
    _edge_attr_normalized = normalizer.normalize(edge_features, edge_stats)

    # -------------- Final -----------------
    input_graph.x = _x
    input_graph.edge_attr = _edge_attr_normalized
    input_graph.x_pos_t = x_new
    input_graph.x_vel_t = v_new
    input_graph.node_force = node_force
    return input_graph