"""
preprocess.py

Preprocess simulation data into graph format. Note one file = one trajectory.
One trajectory = same graph that changes features over time (topology aka adjacency matrix doesn't change)
Saves redundant features as pyg attributes to perform rollout more easily without having to use tensor slices.
Validation/test:
    1. load files and stats
    2. for each file:
        2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
        2.b. normalize with stats loaded in 1
    3. for each file: save

Train:
    1. load files
    2. for each file:
        2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
             and add noise
        2.b. compute normalization stuff (mean + stddev)
    3. for each file: save
"""


import torch
import json
import hydra
from omegaconf import OmegaConf, open_dict
from src.gnn.mesh_numerical_helpers.time_discretizer import TimeDiscretizer, create_time_discretizer
from src.gnn.misc_helpers.yamlobject import PreprocessYaml
from src.gnn.preprocessing.build_features import (compute_von_mises_3d, build_edge_features, build_node_features)
from src.gnn.preprocessing.normalizer import Normalizer, NormalizerFactory
from src.gnn.preprocessing.stats import load_stats
from src.gnn.preprocessing.stats import Stats
from torch_geometric.data import Data, Batch
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from src.gnn.mesh_numerical_helpers.compute_angle import (extract_surface_triangles_torch,
                                                          filter_fixed_surface_triangles, max_surface_angle)

EPS = 1e-6

class Preprocessor(ABC):
    node_stats: Stats
    edge_stats: Stats
    target_stats: Stats
    normalizer: Normalizer
    out_graphs: list[Data]
    categorical_node_features: list[torch.Tensor]
    valued_node_features: list[torch.Tensor] # each tensor is (N, 12)
    edge_features: list[torch.Tensor]
    traj_idx_list: list[int]
    targets: list[torch.Tensor]
    time_discretizer: TimeDiscretizer

    def __init__(self, cfg):
        (self.out_graphs, self.categorical_node_features, self.valued_node_features, self.edge_features,
         self.traj_idx_list, self.targets) = ([], [], [], [], [], [])
        self.theta_max = 0.0
        self.time_discretizer = create_time_discretizer(cfg.time_discretization_method)
        self.normalizer = NormalizerFactory.create_normalizer(cfg.normalizer_method, cfg.isotropic_normalizer)

    def _save_preprocessed(self, traj_idx, out_graphs, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for traj in tqdm(set(traj_idx), desc="Saving trajectories"):
            # print(f"Saving trajectory: {traj}")
            graphs = [graph for idx, graph in zip(traj_idx, out_graphs) if idx == traj]
            # print(f"\tLength: {len(graphs)} timesteps")
            file_path = save_dir / f"graph_{traj:02d}.pt"
            torch.save(graphs, file_path)

    def preprocess_data(self, cfg, graphs_dir, save_dir):
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
            graphs_dir: Path/str
                path to the directory containing the raw graphs
            save_dir: Path/str
                Path to save the preprocessed data
        """
        torch.manual_seed(cfg.seed)
        filenames = list(Path(graphs_dir).glob("*.pt"))
        split_idx_edge = 9
        self._load_stats(cfg)  # if train doesn't do anything since I have to compute them

        pbar = tqdm(enumerate(filenames), total=len(filenames), desc="Processing trajectories")
        # Main loop: process all files, one file = one trajectory, idx_file = index of the file
        for idx_file, filename in pbar:
            # pbar.set_description(f"{filename.name}")
            # print(f"Processing file: {filename.name} ({idx_file + 1}/{len(filenames)})")
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
            x_raw = whole_trajectory_batch.x.reshape(len(graphs_list_batch), -1, 3)  # (#graphs, #points, 3) where 3 is (x,y,z)
            x = x_raw.clone()
            v, a = self.time_discretizer.differentiate_position(x, cfg.dt)
            # v, a = calculate_kinematics(x, cfg.dt)  # v = (T, N, 3), a = (T-1, N, 3)
            rest_pos_x = x[0].clone()  # Rest (first) mesh position (N, 3)
            # Compute angle
            surface_left_triangles, surface_right_triangles = self._maybe_update_maxangle_train(x, graphs_list_batch,
                                                                                                rest_pos_x, cfg)
            v_next = v[1:]  # (T-1, N, 3)
            x_next = x[1:]  # (T-1, N, 3)
            # rest_pos_x = x[0]  # Rest (first) mesh position (N, 3) FIXME: does it make any difference?
            # x_node_force is the force (scalar) in every point (x,y,z) of the graph at time t
            all_node_force = whole_trajectory_batch.x_node_force.reshape(len(graphs_list_batch), -1, 3)  # (T, N, 3)
            # .x_bc contains 0,1 value with 1 = fixed, 0 = free
            boundary_condition = graphs_list_batch[0].x_bc

            # Inside loop: for each graph build its features
            for t, graph in enumerate(graphs_list_batch):
                # limit trajectory length to compute acceleration
                if t > cfg.limit_timesteps - 2:
                    break
                self.traj_idx_list.append(idx_file)  # idx_file of the traj (of the file process)
                # create local (inside for) variables
                _node_force = all_node_force[t]
                # All are (N, 3)
                _x, _v, _a = x[t], v[t], a[t]
                _categorical_node_attr = boundary_condition
                # concatenate [position, velocity, rest_position, node_force] and x_minus_u optionally (3)
                _node_attr = build_node_features(_x, _v, rest_pos_x, _node_force, cfg.enable_node_norms)
                # Obtain edge features [rel_pos, rel_pos_norm, rest_rel_pos, rest_rel_pos_norm]
                # .edge_index contains mesh connectivity
                _edge_attr = build_edge_features(_x, rest_pos_x, graph.edge_index, _v, cfg.velocity_edge)
                # .y_sig_t contains the full stress tensor
                # took the log because it helps making the distribution normal-like (symmetrical)
                # Output _von_mises is (N, 1) [graph.y_sig_t is (N, 9)]
                _von_mises_log = torch.log(compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + EPS)
                _von_mises_nolog = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
                if cfg.mises_log:
                    _von_mises = _von_mises_log
                else:
                    _von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)

                # accumulate all graphs to out_graphs for stats computation
                self.categorical_node_features.append(boundary_condition)
                self.valued_node_features.append(_node_attr.clone())
                self.edge_features.append(_edge_attr.clone())
                # Appends and creates PyG data object for both val and train
                # For val it's already normalized
                self._append_outgraphs(graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                                       all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log,
                                       surface_left_triangles, surface_right_triangles)

        # --- Open: All these do nothing if split NOT train ---
        self._compute_stats_train(cfg, split_idx_edge)  # if val doesn't do anything
        self._save_stats_train(self.node_stats, self.edge_stats, self.target_stats, save_dir)  # if val nothing
        self._save_auxiliary_train_artifacts(save_dir)
        self._normalize_assign_features_train()
        # Save all processed data (in all cases)
        self._save_preprocessed(self.traj_idx_list, self.out_graphs, save_dir)

    @abstractmethod
    def _compute_stats_train(self, cfg, split_idx_edge):
        raise NotImplementedError("compute_stats_train")

    @abstractmethod
    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        raise NotImplementedError("_save_stats")

    @abstractmethod
    def _normalize_assign_features_train(self):
        raise NotImplementedError("normalize_assign_features")

    @abstractmethod
    def _load_stats(self, cfg):
        raise NotImplementedError("_load_stats")

    @abstractmethod
    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log,
                          boundary_left_triangles, boundary_right_triangles):
        raise NotImplementedError("_normalize_append_outgraphs")

    @abstractmethod
    def _maybe_update_maxangle_train(self, x_raw, graphs_list_batch, rest_pos_x, cfg):
        raise NotImplementedError("_maybe_update_train_curvature_threshold")

    @abstractmethod
    def _save_auxiliary_train_artifacts(self, save_dir):
        raise NotImplementedError("_save_auxiliary_train_artifacts")


class TrainPreprocessor(Preprocessor):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        stats = {"node": node_stats.to_dict(), "edge": edge_stats.to_dict(), "target": target_stats.to_dict()}
        stats_dir = save_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Stats saved to {stats_dir / 'stats.json'}")

    def _save_auxiliary_train_artifacts(self, save_dir):
        stats_dir = save_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / "theta_max.json", "w") as f:
            json.dump({"theta_max": float(self.theta_max)}, f, indent=4)
        print(f"Theta max saved to {stats_dir / 'theta_max.json'}")

    def _compute_stats_train(self, cfg, split_idx_edge):
        print("\nComputing train stats...\n")
        # Compute statistics
        PLACEHOLDER_ALL_ISOTROPIC = self.valued_node_features[0].shape[1]
        self.node_stats = self.normalizer.compute_stats(self.valued_node_features, PLACEHOLDER_ALL_ISOTROPIC)
        # print(f"edge_features.shape = {self.edge_features[0].shape}")
        self.edge_stats = self.normalizer.compute_stats(self.edge_features, split_idx_edge)
        # isotropic acceleration iff normalizer is isotropic
        self.target_stats = self.normalizer.compute_stats(self.targets, 3) # FIXME: hardcoded 3 for now
        print("\nFinished computing train stats\n")

    def _load_stats(self, cfg):
        pass

    def _maybe_update_maxangle_train(self, x_raw, graphs_list_batch, rest_pos_x, cfg):
        boundary_condition = graphs_list_batch[0].x_bc
        # Obtain triangles for angle computation
        fixed_mask = boundary_condition.squeeze(1) > 0.5
        surface_left_triangles, surface_right_triangles = extract_surface_triangles_torch(
            graphs_list_batch[0].x_element_connectivity, rest_pos_x.device)
        surface_left_triangles, surface_right_triangles, _ = filter_fixed_surface_triangles(surface_left_triangles,
                                                                                            surface_right_triangles,
                                                                                            fixed_mask)
        max_timesteps = min(x_raw.shape[0], cfg.limit_timesteps)
        for timestep_idx in range(max_timesteps):
            current_theta_max = max_surface_angle(x_raw[timestep_idx], surface_left_triangles,
                                                  surface_right_triangles)
            self.theta_max = max(self.theta_max, float(current_theta_max.detach().item()))

        return surface_left_triangles, surface_right_triangles

    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log,
                          boundary_left_triangles, boundary_right_triangles):
        # Note: later for each data I will create the standard pyg - compatible attributes x, y, edge_attr actually
        # used for NNs
        self.out_graphs.append(Data(edge_index=graph.edge_index, # global index
                                    t=graph.time, # global t attribute
                                    y_acc_t=_a, # TARGET feature acceleration
                                    x_pos_t=_x, # FEATURE position
                                    x_vel_t=_v, # FEATURE velocity
                                    rest_pos=rest_pos_x, # rest position (for runtime noise injection)
                                    node_force=all_node_force[t],
                                    x_bc=_categorical_node_attr,
                                    boundary_left_tri_index=boundary_left_triangles,
                                    boundary_right_tri_index=boundary_right_triangles))
        self.targets.append(torch.concat([_a, _von_mises], dim=1))

    def _normalize_assign_features_train(self):
        # Normalize and assign features
        pbar = tqdm(range(len(self.out_graphs)), total=len(self.out_graphs), desc="Normalizing trajectories")
        # for t in range(len(self.out_graphs)):
        for t in pbar:
            _valued_node_features = self.normalizer.normalize(self.valued_node_features[t], self.node_stats)
            _edge_features = self.normalizer.normalize(self.edge_features[t], self.edge_stats)
            _targets = self.normalizer.normalize(self.targets[t], self.target_stats)
            # Creating standard PyG attributes x (feature), edge_attr, y (target)
            # x.shape = (N, 4, 3)
            self.out_graphs[t].x = torch.concat([self.categorical_node_features[t], _valued_node_features],
                                                dim=1)
            # edge_attr.shape = (N, 8, 3)
            self.out_graphs[t].edge_attr = _edge_features
            # target y.shape = (N, 4, 3)
            self.out_graphs[t].y = _targets
            # if t == 0:
            #     print(f"The shape of x is = {self.out_graphs[0].x.shape} edge_attr = "
            #           f"{self.out_graphs[t].edge_attr.shape} out_graphs[t].y = {self.out_graphs[t].y.shape}")


class ValPreprocessor(Preprocessor):
    save_dir_train: Path

    def __init__(self, cfg, save_dir_train):
        super().__init__(cfg)
        self.save_dir_train = save_dir_train

    def _compute_stats_train(self, cfg, split_idx_edge):
        pass

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        pass

    def _normalize_assign_features_train(self):
        pass

    def _load_stats(self, cfg):
        self.node_stats, self.edge_stats, self.target_stats = load_stats(self.save_dir_train)
        print(f"Stats:\n\tnode_stats.mean = \t{self.node_stats.mean}\n\tnode_stats.std =\t{self.node_stats.std}"
              f"edge_stats.mean =\t{self.edge_stats.mean}\n\tedge_stats.std =\t{self.edge_stats.std}"
              f"\n\ttarget_stats =\t{self.target_stats.mean}\n\ttarget_stats.std =\t{self.target_stats.std}")

    def _maybe_update_maxangle_train(self, x_raw, graphs_list_batch, rest_pos_x, cfg):
        boundary_condition = graphs_list_batch[0].x_bc
        # Obtain triangles for angle computation
        fixed_mask = boundary_condition.squeeze(1) > 0.5
        surface_left_triangles, surface_right_triangles = extract_surface_triangles_torch(
            graphs_list_batch[0].x_element_connectivity, rest_pos_x.device)
        surface_left_triangles, surface_right_triangles, _ = filter_fixed_surface_triangles(surface_left_triangles,
                                                                                            surface_right_triangles,
                                                                                            fixed_mask)
        return surface_left_triangles, surface_right_triangles

    def _save_auxiliary_train_artifacts(self, save_dir):
        pass

    def _normalize_before_appending(self, _edge_attr, _node_attr, _categorical_node_attr, _target):
        # Use precomputed stats for validation
        _numerical_node_attr_norm = self.normalizer.normalize(_node_attr, self.node_stats)
        edge_attr_norm = self.normalizer.normalize(_edge_attr, self.edge_stats)
        node_attr = torch.concat([_categorical_node_attr, _numerical_node_attr_norm], dim=1)
        target_norm = self.normalizer.normalize(_target, self.target_stats)
        return node_attr, edge_attr_norm, target_norm

    # Validation: normalize with stats already computed, and add to out_graphs as PyG data
    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log,
                          boundary_left_triangles, boundary_right_triangles):
        _target = torch.concat([_a, _von_mises], dim=1)
        node_attr_norm, edge_attr_norm, target_norm = self._normalize_before_appending(
            _edge_attr, _node_attr, _categorical_node_attr, _target)
        self.out_graphs.append(
            Data(# targets: current time acceleration and von mises stress
                y=target_norm,
                y_acc_t=_a,
                y_von_mises=_von_mises,
                von_mises_log = _von_mises_log,
                # (global attribute on the graph) time step of current graph
                t=graph.time,
                # ==================================================================
                #                       NODE features
                # ==================================================================
                # x = [is_bc, position, velocity, rest_position, node_force]
                x=node_attr_norm,  # this will be the NN input feature for training
                # note: the position x_next, velocity v_next of t + 1 can be given since they can be computed
                # from x, v at time t and acceleration of t - 1 (previous step prediction)
                # note: not used for NN input, used only for building next time step graph / ground truth
                x_next=x_next[t],
                v_next=v_next[t],
                node_force_next=all_node_force[t + 1],
                # all these are already in node_attr = [x, v, u, node_force], so saving them twice
                # is only due to rollout/other
                x_pos_t=_x,
                x_vel_t=_v,
                node_force=all_node_force[t],
                rest_pos=rest_pos_x,
                # ==================================================================
                #                           EDGE features
                # ==================================================================
                edge_attr=edge_attr_norm,
                # features to encode mesh connectivity in two different ways
                edge_index=graph.edge_index,
                x_element_connectivity=graph.x_element_connectivity,
                boundary_left_tri_index=boundary_left_triangles,
                boundary_right_tri_index=boundary_right_triangles)
        )

def create_preprocessor(split, cfg, save_dir_train):
    if split == "train":
        return TrainPreprocessor(cfg)
    elif split == "val":
        return ValPreprocessor(cfg, save_dir_train)
    elif split == "test":
        return ValPreprocessor(cfg, save_dir_train)
    else:
        raise NotImplementedError(f"{split} not implemented")

@hydra.main(version_base=None, config_path="../../conf/preprocess", config_name="preprocess")
def main(cfg):
    """
    Args:
        cfg: omegaconf.DictConfig
            Hydra-composed configuration. Run with --multirun to sweep over multiple values,
            e.g.: python -m src.gnn.preprocess --multirun split=train,val,test
    """
    cfg_obj = PreprocessYaml(cfg)
    print(f"Preprocessing {cfg_obj.dataset_size} dataset\n" + "=" * 60)
    print(f"\tSave dir of config.yaml = {cfg_obj.save_dir}")
    save_dir = Path(cfg_obj.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # OmegaConf.save(cfg, save_dir / "config.yaml")
    # =============== NAME ================
    isotropic_str = '_isotr' if cfg_obj.isotropic_normalizer else '_anisotr'
    miseslog_str = '_log' if cfg_obj.mises_log else '_nolog'
    normalizer_str = 'z' if cfg_obj.normalizer_method == 'z' else 'rob'
    processed_dir_name = f"{normalizer_str}{isotropic_str}{miseslog_str}"
    processed_dataset_dir = Path(cfg_obj.save_dir) / processed_dir_name / cfg_obj.dataset_size
    print(f"\tProcessed data dir {processed_dataset_dir}")
    processed_dataset_dir.mkdir(parents=True, exist_ok=True)
    # =============== NAME ================

    # =============== SPLITS ================
    # Where to save results
    save_dir_train = processed_dataset_dir / "train"
    save_dir_test = processed_dataset_dir / "test"
    save_dir_val = processed_dataset_dir / "val"
    # Raw paths
    graphs_dir_train = Path(cfg_obj.raw_data_dir) / "train" / "graphs"
    graphs_dir_test = Path(cfg_obj.raw_data_dir) / "test" / "graphs"
    graphs_dir_val = Path(cfg_obj.raw_data_dir) / "val" / "graphs"
    # =============== SPLITS ================

    with open_dict(cfg):
        cfg.node_channels = 16
        cfg.edge_channels = 12
        cfg.train_dir = save_dir_train.as_posix()
        cfg.test_dir = save_dir_test.as_posix()
        cfg.val_dir = save_dir_val.as_posix()
    OmegaConf.save(cfg, processed_dataset_dir / "config.yaml")

    # Create preprocessors object
    preprocessor_train = create_preprocessor("train", cfg_obj, None)
    preprocessor_val = create_preprocessor("val", cfg_obj, save_dir_train)
    preprocessor_test = create_preprocessor("test", cfg_obj, save_dir_train)
    print("\n" + "=" * 60 + "\nTrain preprocesing")
    preprocessor_train.preprocess_data(cfg_obj, graphs_dir_train, save_dir_train)
    print("\n" + "=" * 60 + "\nPreprocessing complete!"+ "\n" + "=" * 60)
    print("\n" + "=" * 60 + "\nTest preprocessing"+ "\n" + "=" * 60)
    preprocessor_test.preprocess_data(cfg_obj, graphs_dir_test, save_dir_test)
    print("\n" + "=" * 60 + "\nPreprocessing complete!"+ "\n" + "=" * 60)
    print("\n" + "=" * 60 + "\nVal preprocessing"+ "\n" + "=" * 60)
    preprocessor_val.preprocess_data(cfg_obj, graphs_dir_val, save_dir_val)
    print("\n" + "=" * 60 + "\nPreprocessing complete!" + "\n" + "=" * 60)


if __name__ == "__main__":
    main()
