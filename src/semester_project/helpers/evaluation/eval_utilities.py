def do_rollout(
        model: torch.nn.Module,
        test_loader: DataLoader,
        node_stats: Stats,
        edge_stats: Stats,
        target_stats: Stats,
        dt: float,
        device: torch.device,
        skip_first: int = 0,
        rollout_steps: int = 99999999,
):
    """Perform autoregressive rollout using the model.

    Args:
        model: Trained MeshGraphNet
        test_loader: DataLoader for test data
        node_stats, edge_stats, target_stats: Normalization statistics
        dt: Time step size
        device: Computation device
        skip_first: Number of initial steps to skip
        rollout_steps: Maximum number of steps to rollout

    Returns:
        true_rollout: List of ground truth graphs
        pred_rollout: List of predicted graphs
    """
    with torch.no_grad():
        true_rollout = []
        pred_rollout = []
        u = None

        for t, graph in enumerate(test_loader):
            if u is None:
                u = graph.u
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

            pred_denorm = denormalize(pred, target_stats)
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
            input_graph = build_mgn_graph_from_timestep(
                simulation_graph=graph,
                x_new=x_new,
                v_new=v_new,
                u=u,
                node_stats=node_stats,
                edge_stats=edge_stats,
                target_stats=target_stats,
            )

            # Store rollouts
            true_rollout.append(
                Data(
                    x_pos_t=x_true,
                    x_vel_t=v_true,
                    von_mises=von_mises_true,
                    x_element_connectivity=graph.x_element_connectivity,
                )
            )
            pred_rollout.append(
                Data(
                    x_pos_t=x_new,
                    x_vel_t=v_new,
                    von_mises=von_mises,
                    x_element_connectivity=graph.x_element_connectivity,
                )
            )
    return true_rollout, pred_rollout