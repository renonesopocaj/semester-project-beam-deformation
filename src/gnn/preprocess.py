import torch
import json
import yaml
from src.gnn.helpers.misc.yamlobj import YamlObj
from src.gnn.helpers.data_utils.build_features import (calculate_kinematics, compute_von_mises_3d, build_edge_features,
                                                       build_valued_node_features)
from src.gnn.helpers.data_utils.stats import compute_stats, normalize, Stats
from torch_geometric.data import Data, Batch

EPS = 1e-6

def _save_preprocessed(split, traj_idx, out_graphs, base_dir, dataset_size):
    # Save preprocessed graphs to: processed/{size}/{split}/graphs/graph_XX.pt FIXME:    hardcoded and make function
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

def _save_stats(node_stats, edge_stats, target_stats, base_dir, dataset_size):
    stats = {"node": node_stats.to_dict(), "edge": edge_stats.to_dict(), "target": target_stats.to_dict()}
    # Save to: processed/{size}/train/stats/stats.json FIXME: hardcoded and make function
    stats_dir = base_dir / "processed" / dataset_size / "train" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to {stats_dir / 'stats.json'}")


def _add_noise_train(x, v, dt, noise_scale):
    # Add noise to training data
    noise = noise_scale * torch.randn_like(v)
    x = x + noise
    v = v - noise / dt  # adjust velocity accordingly
    # Obtain the next-time-step velocities and positions (same tensors but shifted of 1 timestep)
    return x, v

def _load_stats(base_dir, dataset_size):
    stats_path = base_dir / "processed" / dataset_size / "train" / "stats" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found at {stats_path}. Preprocess the training data first.")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    node_stats = Stats.from_dict(stats["node"])
    edge_stats = Stats.from_dict(stats["edge"])
    target_stats = Stats.from_dict(stats["target"])
    return node_stats, edge_stats, target_stats

def preprocess_data(data_dir, split, dataset_size, noise_scale, dt, recalc_velocities, base_dir, seed, device):
    """
    Preprocess simulation data into graph format. Note one file = one trajectory
    One trajectory = same graph that changes features over time (topology aka adjacency matrix doesn't change)
    Validation: 1. load files and stats
                2. for each file:
                    2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
                    2.b. normalize
                3. for each file: save
    Train: 1. load files
           2. for each file:
                2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
                     and add noise
                2.b. compute normalization stuff (mean + stddev) FIXME: CHECK IT'S ISOTROPIC
            3. for each file: save

    Args:
        data_dir: Path
            Directory containing raw simulation data (e.g., raw/single-direction-beam/small)
        split: str
            'train' | 'val' | 'test'
        dataset_size: str
            'small' | 'medium' | 'large'
        noise_scale: float
            Amount of noise to add during training
        dt: float
            Time step size
        recalc_velocities: bool
            Whether to recalculate velocities from positions
        base_dir: Path
            Base directory for elastic-beam-3d data
    """
    torch.manual_seed(seed)
    # Input path: raw/single-direction-beam/{size}/{split}/graphs
    # FIXME: hardcoded
    graphs_dir = data_dir / split / "graphs"
    filenames = list(graphs_dir.glob("*.pt"))
    out_graphs, categorical_node_features, valued_node_features, edge_features, traj_idx, targets = (
        [], [], [], [], [], [])

    # Load stats if validation
    if split != "train":
        node_stats, edge_stats, target_stats = _load_stats(base_dir, dataset_size)

    # Main loop: process all files, one file = one trajectory, idx_file = idx_file of the file
    for idx_file, filename in enumerate(filenames):
        print(f"Processing file: {filename.name} ({idx_file + 1}/{len(filenames)})")
        dataloader = torch.load(filename, weights_only=False)
        graphs = []
        for data in dataloader:
            graph = data.to(device)
            graphs.append(graph)
        whole_trajectory_batch = Batch.from_data_list(graphs)
        # Extract positions over time, NOTE: #graphs = len(graphs) = #timesteps
        # FIXME: assumption all graphs have same number of points? And where is .x attribute created?
        x = whole_trajectory_batch.x.reshape(len(graphs), -1, 3) # (#graphs, #points, 3) where 3 is (x,y,z)
        v, a = calculate_kinematics(x, dt) # v = (T, N, 3), a = (T-1, N, 3)
        if split == "train":
            x, v = _add_noise_train(x, v, dt, noise_scale)
        v_next = v[1:]
        x_next = x[1:]
        u = x[0] # Rest (first) mesh position
        # FIXME where is .x_node_force attribute created? and .x_bc?
        all_node_force = whole_trajectory_batch.x_node_force.reshape(len(graphs), -1, 3) # (#graphs, #points, 3)
        boundary_condition = graphs[0].x_bc  # 1 = fixed, 0 = free FIXME: WHERE IS THIS 1,0 ENCODED?

        # Inside loop: for each graph build its features
        for t, graph in enumerate(graphs):
            # FIXME: WHY LIMIT?
            if t > 48:  # Limit trajectory length
                break
            traj_idx.append(idx_file) # idx_file of the traj (of the file process)
            # create local (inside for) variables
            _node_force = all_node_force[t]
            _x, _v, _a = x[t], v[t], a[t]
            _categorical_node_attr = boundary_condition
            _node_attr = build_valued_node_features(_x, _v, u, _node_force)
            _edge_attr = build_edge_features(_x, u, graph.edge_index)
            _von_mises = torch.log(compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + EPS) # Compute von Mises stress FIXME: WHERE IS .edge_index, .y_sig_t FIXME: what is torch.log

            # Validation: normalize with stats already computed, and add to out_graphs as PyG data
            if split != "train":
                # Use precomputed stats for validation
                _edge_attr_norm = normalize(_edge_attr, edge_stats)
                _vector_node_attr_norm = normalize(_node_attr, node_stats)
                node_attr = torch.concat([_categorical_node_attr, _vector_node_attr_norm], dim=1)

                out_graphs.append(
                    Data(x=node_attr, edge_attr=_edge_attr_norm, edge_index=graph.edge_index, t=graph.time, x_pos_t=x[t],
                         x_vel_t=v[t], node_force=all_node_force[t], node_force_next=all_node_force[t + 1],
                         x_next=x_next[t], v_next=v_next[t], y_acc_t=a[t], y_von_mises=_von_mises,
                         x_element_connectivity=graph.x_element_connectivity, u=u)
                )
            else:
                # Train: accumulate to out_graphs for stats computation
                categorical_node_features.append(boundary_condition)
                valued_node_features.append(build_valued_node_features(_x, _v, u, _node_force))
                edge_features.append(build_edge_features(_x, u, graph.edge_index))
                out_graphs.append(Data(edge_index=graph.edge_index, t=graph.time, y_acc_t=_a, x_pos_t=_x, x_vel_t=_v))
                targets.append(torch.concat([_a, _von_mises], dim=1))

    # Normalize data and save stats if it's in training preprocessing
    if split == "train":
        # Compute statistics
        node_stats = compute_stats(valued_node_features)
        edge_stats = compute_stats(edge_features)
        target_stats = compute_stats(targets)
        # Save statistics
        _save_stats(node_stats, edge_stats, target_stats, base_dir, dataset_size)
        # Normalize and assign features
        for t in range(len(out_graphs)):
            _valued_node_features = normalize(valued_node_features[t], node_stats)
            _edge_features = normalize(edge_features[t], edge_stats)
            _targets = normalize(targets[t], target_stats)
            out_graphs[t].x = torch.concat([categorical_node_features[t], _valued_node_features], dim=1)
            out_graphs[t].edge_attr = _edge_features
            out_graphs[t].y = _targets

    _save_preprocessed(split, traj_idx, out_graphs, base_dir, dataset_size) # Save all processed data (in all cases)

if __name__ == "__main__":
    # Run preprocessing
    # Choose dataset size: 'small', 'medium', or 'large'
    YAML_PATH = "yamls/preprocessing.yaml"
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = YamlObj(cfg)
    # raw_data_dir = cfg.base_dir / "raw" / "single-direction-beam" / cfg.dataset_size
    print(f"Preprocessing {cfg.dataset_size} dataset..." + "=" * 60)
    # Preprocess training data
    print("\n1. Preprocessing training data...")
    preprocess_data(data_dir=raw_data_dir, split="train", dataset_size=cfg.dataset_size, noise_scale=0.0003,
                    recalc_velocities=True, base_dir=cfg.base_dir, dt=0.08, seed=cfg.seed, device=device)

    print("\n2. Preprocessing validation data...")
    preprocess_data(data_dir=raw_data_dir, split="val", dataset_size=cfg.dataset_size, recalc_velocities=True,
                    base_dir=cfg.base_dir, noise_scale=0.0, dt=0.08, seed=cfg.seed, device=device)

    print("\n3. Preprocessing test data (if available)...")
    # test_graphs_dir = raw_data_dir / "test" / "graphs"
    if test_graphs_dir.exists():
        preprocess_data(data_dir=raw_data_dir, split="test", dataset_size=cfg.dataset_size, recalc_velocities=True,
                        base_dir=cfg.base_dir, noise_scale = 0.0, dt = 0.08, seed=cfg.seed, device=device)
    else:
        print("  No test data found, skipping...")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")