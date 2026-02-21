def preprocess_data(
        data_dir: Path,
        split: str,
        dataset_size: str = "small",
        noise_scale: float = 0.0,
        dt: float = 0.08,
        recalc_velocities: bool = False,
        base_dir: Path = Path("./elastic-beam-3d"),
):
    """Preprocess simulation data into graph format.

    Args:
        data_dir: Directory containing raw simulation data (e.g., raw/single-direction-beam/small)
        split: 'train', 'val', or 'test'
        dataset_size: 'small', 'medium', or 'large'
        noise_scale: Amount of noise to add during training
        dt: Time step size
        recalc_velocities: Whether to recalculate velocities from positions
        base_dir: Base directory for elastic-beam-3d data
    """
    SEED = 42
    torch.manual_seed(SEED)

    # Input path: raw/single-direction-beam/{size}/{split}/graphs
    graphs_dir = data_dir / split / "graphs"
    filenames = list(graphs_dir.glob("*.pt"))

    out_graphs = []
    categorical_node_features = []
    valued_node_features = []
    edge_features = []
    traj_idx = []
    targets = []

    # Load stats if not training
    if split != "train":
        stats_path = base_dir / "processed" / dataset_size / "train" / "stats" / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Stats file not found at {stats_path}. "
                f"Please preprocess the training data first."
            )
        with open(stats_path, "r") as f:
            stats = json.load(f)
        node_stats = Stats.from_dict(stats["node"])
        edge_stats = Stats.from_dict(stats["edge"])
        target_stats = Stats.from_dict(stats["target"])

    for idx, filename in enumerate(filenames):
        print(f"Processing file: {filename.name} ({idx + 1}/{len(filenames)})")
        dataloader = torch.load(filename, weights_only=False)
        graphs = []
        for data in dataloader:
            graph = data.to("cpu")
            graphs.append(graph)
        whole_trajectory_batch = Batch.from_data_list(graphs)
        len_traj = len(graphs)

        # Extract positions over time
        x = whole_trajectory_batch.x.reshape(len_traj, -1, 3)
        v, a = calculate_kinematics(x, dt)

        # Add noise to training data
        if split == "train":
            noise = noise_scale * torch.randn_like(v)
            x = x + noise
            v = v - noise / dt  # adjust velocity accordingly

        v_next = v[1:]
        x_next = x[1:]

        u = x[0]  # Rest mesh position
        all_node_force = whole_trajectory_batch.x_node_force.reshape(len_traj, -1, 3)
        boundary_condition = graphs[0].x_bc  # 1 = fixed, 0 = free

        for t, graph in enumerate(graphs):
            if t > 48:  # Limit trajectory length
                break
            traj_idx.append(idx)
            _node_force = all_node_force[t]
            _x = x[t]
            _v = v[t]
            _a = a[t]

            _categorical_node_attr = boundary_condition
            _node_attr = build_valued_node_features(_x, _v, u, _node_force)
            _edge_attr = build_edge_features(_x, u, graph.edge_index)

            # Compute von Mises stress
            von_mises = torch.log(
                compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + 1e-6
            )

            if split != "train":
                # Use precomputed stats for validation
                edge_attr = normalize(_edge_attr, edge_stats)
                vector_node_attr = normalize(_node_attr, node_stats)
                categorical_node_attr = _categorical_node_attr
                node_attr = torch.concat(
                    [categorical_node_attr, vector_node_attr], dim=1
                )

                out_graphs.append(
                    Data(
                        x=node_attr,
                        edge_attr=edge_attr,
                        edge_index=graph.edge_index,
                        t=graph.time,
                        x_pos_t=x[t],
                        x_vel_t=v[t],
                        node_force=all_node_force[t],
                        node_force_next=all_node_force[t + 1],
                        x_next=x_next[t],
                        v_next=v_next[t],
                        y_acc_t=a[t],
                        y_von_mises=von_mises,
                        x_element_connectivity=graph.x_element_connectivity,
                        u=u,
                    )
                )
            else:
                # Accumulate for stats computation
                categorical_node_features.append(boundary_condition)
                valued_node_features.append(
                    build_valued_node_features(_x, _v, u, _node_force)
                )
                edge_features.append(build_edge_features(_x, u, graph.edge_index))
                out_graphs.append(
                    Data(
                        edge_index=graph.edge_index,
                        t=graph.time,
                        y_acc_t=_a,
                        x_pos_t=_x,
                        x_vel_t=_v,
                    )
                )
                targets.append(torch.concat([_a, von_mises], dim=1))

    if split == "train":
        # Compute and save statistics
        node_stats = compute_stats(valued_node_features)
        edge_stats = compute_stats(edge_features)
        target_stats = compute_stats(targets)

        stats = {
            "node": node_stats.to_dict(),
            "edge": edge_stats.to_dict(),
            "target": target_stats.to_dict(),
        }
        # Save to: processed/{size}/train/stats/stats.json
        stats_dir = base_dir / "processed" / dataset_size / "train" / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Stats saved to {stats_dir / 'stats.json'}")

        # Normalize and assign features
        for t in range(len(out_graphs)):
            _valued_node_features = normalize(valued_node_features[t], node_stats)
            _edge_features = normalize(edge_features[t], edge_stats)
            _targets = normalize(targets[t], target_stats)
            out_graphs[t].x = torch.concat(
                [categorical_node_features[t], _valued_node_features], dim=1
            )
            out_graphs[t].edge_attr = _edge_features
            out_graphs[t].y = _targets

    # Save preprocessed graphs to: processed/{size}/{split}/graphs/graph_XX.pt
    save_dir = base_dir / "processed" / dataset_size / split / "graphs"
    save_dir.mkdir(parents=True, exist_ok=True)
    for traj in set(traj_idx):
        print(f"Saving trajectory: {traj}")
        graphs = [graph for idx, graph in zip(traj_idx, out_graphs) if idx == traj]
        print(f"  Length: {len(graphs)} timesteps")
        file_path = save_dir / f"graph_{traj:02d}.pt"
        torch.save(graphs, file_path)

    print(f"Preprocessing complete for {split} split!")
    print(f"Saved to: {save_dir}")

if __name__ == "__main__":
    # Run preprocessing
    # Choose dataset size: 'small', 'medium', or 'large'
    DATASET_SIZE = "small"

    base_dir = Path("./elastic-beam-3d")
    raw_data_dir = base_dir / "raw" / "single-direction-beam" / DATASET_SIZE

    print(f"Preprocessing {DATASET_SIZE} dataset...")
    print("=" * 60)

    # Preprocess training data
    print("\n1. Preprocessing training data...")
    preprocess_data(
        data_dir=raw_data_dir,
        split="train",
        dataset_size=DATASET_SIZE,
        noise_scale=0.0003,
        recalc_velocities=True,
        base_dir=base_dir,
    )

    print("\n2. Preprocessing validation data...")
    preprocess_data(
        data_dir=raw_data_dir,
        split="val",
        dataset_size=DATASET_SIZE,
        recalc_velocities=True,
        base_dir=base_dir,
    )

    print("\n3. Preprocessing test data (if available)...")
    test_graphs_dir = raw_data_dir / "test" / "graphs"
    if test_graphs_dir.exists():
        preprocess_data(
            data_dir=raw_data_dir,
            split="test",
            dataset_size=DATASET_SIZE,
            recalc_velocities=True,
            base_dir=base_dir,
        )
    else:
        print("  No test data found, skipping...")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")