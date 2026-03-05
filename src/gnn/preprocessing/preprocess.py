import torch
import json
import hydra
from omegaconf import OmegaConf
from src.gnn.helpers.misc.yamlobject import PreprocessYaml
from src.gnn.preprocessing.data_utils.build_features import (calculate_kinematics, compute_von_mises_3d, build_edge_features,
                                                             build_valued_node_features)
from src.gnn.preprocessing.data_utils.stats import compute_stats, normalize, Stats
from torch_geometric.data import Data, Batch
from pathlib import Path

EPS = 1e-6

def _save_preprocessed(split, traj_idx, out_graphs, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for traj in set(traj_idx):
        print(f"Saving trajectory: {traj}")
        graphs = [graph for idx, graph in zip(traj_idx, out_graphs) if idx == traj]
        print(f"  Length: {len(graphs)} timesteps")
        file_path = save_dir / f"graph_{traj:02d}.pt"
        torch.save(graphs, file_path)
    print(f"Preprocessing complete for {split} split! Saved to: {save_dir}")

def _save_stats(node_stats, edge_stats, target_stats, save_dir):
    stats = {"node": node_stats.to_dict(), "edge": edge_stats.to_dict(), "target": target_stats.to_dict()}
    stats_dir = Path(save_dir + "/stats")
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

def _load_stats(train_save_dir):
    """
    Args:
        train_save_dir: str or Path
            Path to the directory where training data was saved (contains stats/ subfolder).
        return: (Stats, Stats, Stats)
            node_stats, edge_stats, target_stats loaded from stats.json.
    """
    stats_path = Path(train_save_dir) / "stats" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found at {stats_path}. Preprocess the training data first.")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    node_stats = Stats.from_dict(stats["node"])
    edge_stats = Stats.from_dict(stats["edge"])
    target_stats = Stats.from_dict(stats["target"])
    return node_stats, edge_stats, target_stats

def preprocess_data(cfg):
    """
    Preprocess simulation data into graph format. Note one file = one trajectory
    One trajectory = same graph that changes features over time (topology aka adjacency matrix doesn't change)
    Validation: 1. load files and stats
                2. for each file:
                    2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
                    2.b. normalize with stats loaded in 1
                3. for each file: save
    Train: 1. load files
           2. for each file:
                2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
                     and add noise
                2.b. compute normalization stuff (mean + stddev)
           3. for each file: save

    Args:
        cfg: YamlObject
    """
    torch.manual_seed(cfg.seed)
    filenames = list(Path(cfg.graphs_dir).glob("*.pt"))
    out_graphs, categorical_node_features, valued_node_features, edge_features, traj_idx_list, targets = (
        [], [], [], [], [], [])

    # Load edge, node, target stats if validation to then normalize without recomputing
    if cfg.split != "train":
        node_stats, edge_stats, target_stats = _load_stats(cfg.train_save_dir)

    # Main loop: process all files, one file = one trajectory, idx_file = idx_file of the file
    for idx_file, filename in enumerate(filenames):
        print(f"Processing file: {filename.name} ({idx_file + 1}/{len(filenames)})")
        dataloader = torch.load(filename, weights_only=False)

        # Create list graphs_list_batch with all graphs in the file
        graphs_list_batch = []
        for data in dataloader:
            graph = data.to(cfg.device)
            graphs_list_batch.append(graph)
        # (For each file) From graphs_list (all graphs in batch), create a Batch object, and compute features
        # If you do batch.node_feature, you are creating a tensor which contains a tensor
        # (# graphs in batch, dim_feat_per_graph) where dim_feat_per_graph are the dimension of that node_feature for
        # one graph
        whole_trajectory_batch = Batch.from_data_list(graphs_list_batch)
        # Extract positions over time, NOTE: #graphs = len(graphs) = #timesteps .x attribute encodes positions
        x = whole_trajectory_batch.x.reshape(len(graphs_list_batch), -1, 3) # (#graphs, #points, 3) where 3 is (x,y,z)
        v, a = calculate_kinematics(x, cfg.dt) # v = (T, N, 3), a = (T-1, N, 3)
        if cfg.split == "train":
            x, v = _add_noise_train(x, v, cfg.dt, cfg.noise_scale_train)
        v_next = v[1:] # (T-1, N, 3)
        x_next = x[1:] # (T-1, N, 3)
        u = x[0] # Rest (first) mesh position (N, 3)
        # x_node_force is the force (scalar) in every point (x,y,z) of the graph at time t
        all_node_force = whole_trajectory_batch.x_node_force.reshape(len(graphs_list_batch), -1, 3) # (T, N, 3)
        # .x_bc contains 0,1 value with 1 = fixed, 0 = free
        boundary_condition = graphs_list_batch[0].x_bc

        # Inside loop: for each graph build its features
        for t, graph in enumerate(graphs_list_batch):
            # limit trajectory length to compute acceleration
            if t > cfg.limit_timesteps - 2:
                break
            traj_idx_list.append(idx_file) # idx_file of the traj (of the file process)
            # create local (inside for) variables
            _node_force = all_node_force[t]
            _x, _v, _a = x[t], v[t], a[t]
            _categorical_node_attr = boundary_condition
            # concatenate [position, velocity, rest_position, node_force]
            _node_attr = build_valued_node_features(_x, _v, u, _node_force)
            # Obtain edge features [rel_pos, rel_pos_norm, rest_rel_pos, rest_rel_pos_norm]
            # .edge_index contains mesh connectivity
            _edge_attr = build_edge_features(_x, u, graph.edge_index)
            # .y_sig_t contains the full stress tensor
            # took the log because it helps making the distribution normal-like (symmetrical)
            if cfg.mises_log:
                # # Output _von_mises is (N, 1) [graph.y_sig_t is (N, 9)]
                _von_mises = torch.log(compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + EPS)
            else:
                _von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)

            # Validation: normalize with stats already computed, and add to out_graphs as PyG data
            if cfg.split != "train":
                # Use precomputed stats for validation
                # FIXME: not isotropic
                if cfg.isotropic_norm:
                    raise NotImplementedError
                else:
                    _edge_attr_norm = normalize(_edge_attr, edge_stats)
                    _vector_node_attr_norm = normalize(_node_attr, node_stats)
                node_attr = torch.concat([_categorical_node_attr, _vector_node_attr_norm], dim=1)

                out_graphs.append(
                    Data(
                        # targets: current time acceleration and von mises stress
                        y_acc_t=a[t],
                        y_von_mises=_von_mises,
                        # time step of current graph (it was a global attribute on the graph)
                        t=graph.time,
                        # ==================================================================
                        #                       NODE features
                        # ==================================================================
                        # x = [is_bc, position, velocity, rest_position, node_force]
                        x=node_attr,
                        # FIXME: aren't all these they already in node_attr = [x, v, u, node_force]? why saving them twice?
                        x_pos_t=x[t],
                        x_vel_t=v[t],
                        node_force=all_node_force[t],
                        rest_x0=u,
                        # note: the force is like the actuator: you need the next-step
                        node_force_next=all_node_force[t + 1],
                        # note: the position x_next, velocity v_next of t + 1 can be given since they can be computed from
                        # x, v at time t and acceleration of t - 1 (previous step prediction)
                        x_next=x_next[t],
                        v_next=v_next[t],
                        # ==================================================================
                        #                           EDGE features
                        # ==================================================================
                        edge_attr=_edge_attr,
                        # features to encode mesh connectivity in two different ways
                        edge_index=graph.edge_index,
                        x_element_connectivity=graph.x_element_connectivity,
                    )
                )
            else: # If train:
                # accumulate all graphs to out_graphs for stats computation
                categorical_node_features.append(boundary_condition)
                valued_node_features.append(build_valued_node_features(_x, _v, u, _node_force))
                edge_features.append(build_edge_features(_x, u, graph.edge_index))
                # TODO: is this Data only used for stats computation? Why a different object?
                out_graphs.append(
                    Data(
                        edge_index=graph.edge_index,
                        t=graph.time,
                        y_acc_t=_a,
                        x_pos_t=_x,
                        x_vel_t=_v
                    )
                )
                targets.append(torch.concat([_a, _von_mises], dim=1))

    # Normalize data and save stats if it's in training preprocessing
    if cfg.split == "train":
        # Compute statistics
        node_stats = compute_stats(valued_node_features)
        edge_stats = compute_stats(edge_features)
        target_stats = compute_stats(targets)
        # Save statistics
        _save_stats(node_stats, edge_stats, target_stats, cfg.save_dir)
        # Normalize and assign features
        for t in range(len(out_graphs)):
            _valued_node_features = normalize(valued_node_features[t], node_stats)
            _edge_features = normalize(edge_features[t], edge_stats)
            _targets = normalize(targets[t], target_stats)
            out_graphs[t].x = torch.concat([categorical_node_features[t], _valued_node_features], dim=1)
            out_graphs[t].edge_attr = _edge_features
            out_graphs[t].y = _targets

    _save_preprocessed(cfg.split, traj_idx_list, out_graphs, cfg.save_dir) # Save all processed data (in all cases)

@hydra.main(version_base=None, config_path="../../../conf", config_name="preprocess")
def main(cfg):
    """
    Args:
        cfg: omegaconf.DictConfig
            Hydra-composed configuration. Run with --multirun to sweep over multiple values,
            e.g.: python -m src.gnn.preprocess --multirun split=train,val,test
    """
    cfg_obj = PreprocessYaml(cfg)
    print(f"Preprocessing {cfg_obj.dataset_size} dataset, split={cfg_obj.split}...\n" + "=" * 60)
    save_dir = Path(cfg_obj.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_dir / "config.yaml")
    preprocess_data(cfg=cfg_obj)
    print("\n" + "=" * 60 + "\nPreprocessing complete!")


if __name__ == "__main__":
    main()