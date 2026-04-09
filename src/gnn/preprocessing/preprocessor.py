"""
preprocessor.py

Offline preprocessing of data. Noise injection is done online, not offline, due to having to change it each epoch.
A lot of redundancy in Data(...) which are the PyG-formatted graphs, because it lets us do rollout and other stuff
later in training without having to slice tensors and "remembering" which feature is where.
One trajectory = same graph that changes features over time (topology aka adjacency matrix doesn't change)
Note one file = one trajectory.

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

from typing import Any
import torch
import json
from src.gnn.mesh_numerical_helpers.time_discretizer import TimeDiscretizer, create_time_discretizer
from src.gnn.preprocessing.build_features import (compute_von_mises_3d, make_scripted_feature_builders)
from src.gnn.preprocessing.normalizer import Normalizer, create_normalizer
from src.gnn.preprocessing.stats import load_stats
from src.gnn.preprocessing.stats import Stats
from torch_geometric.data import Data, Batch
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from src.gnn.mesh_numerical_helpers.compute_angle import (extract_surface_triangles_torch,
                                                          filter_fixed_surface_triangles, max_surface_angle)

EPS = 1e-6

def sort_filenames_by_traj_params(filenames, graphs_dir):
    """
    Sort raw simulation filenames by traj_idx from trajectory_params.json so that
    the preprocessing file order matches the FEM generation combo order,
    regardless of filename differences (e.g. different T/Nsteps for selected trajectories).

    Args:
        filenames: list of Path
            unsorted list of .pt file paths
        graphs_dir: Path or str
            directory containing the graph .pt files (trajectory_params.json is one level up)
    return: list of Path
        sorted filenames
    """
    traj_params_path = Path(graphs_dir).parent / "trajectory_params.json"
    if not traj_params_path.exists():
        return sorted(filenames)
    with open(traj_params_path) as f:
        traj_params = json.load(f)
    name_to_idx = {name: info["traj_idx"] for name, info in traj_params.items()}
    return sorted(filenames, key=lambda fn: name_to_idx.get(fn.name, 0))

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
    selected_indices: set
    selected_limit: int
    split_idx_node: int | None
    split_idx_edge: int | None
    _build_node_features_scripted: Any
    _build_edge_features_scripted: Any

    def __init__(self, cfg, selected_indices, selected_limit):
        (self.out_graphs, self.categorical_node_features, self.valued_node_features, self.edge_features,
         self.traj_idx_list, self.targets) = ([], [], [], [], [], [])
        self.theta_max = 0.0
        self.time_discretizer = create_time_discretizer(cfg.time_discretization_method)
        self.normalizer = create_normalizer(cfg.normalizer_method, cfg.isotropic_normalizer)
        self.selected_indices = set(selected_indices)
        self.selected_limit = int(selected_limit)
        self.split_idx_node = None
        self.split_idx_edge = None
        self._build_node_features_scripted = None
        self._build_edge_features_scripted = None

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
        self._build_node_features_scripted, self._build_edge_features_scripted = make_scripted_feature_builders(cfg)
        filenames = sort_filenames_by_traj_params(list(Path(graphs_dir).glob("*.pt")), graphs_dir)
        if not filenames:
            raise FileNotFoundError(f"No .pt files found in {graphs_dir}")
        self._load_stats(cfg)  # if train doesn't do anything since I have to compute them

        split_idx_node = None
        split_idx_edge = None
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
                                                                                                rest_pos_x, cfg,
                                                                                                idx_file)
            v_next = v[1:]  # (T-1, N, 3)
            x_next = x[1:]  # (T-1, N, 3)
            # rest_pos_x = x[0]  # Rest (first) mesh position (N, 3)
            # x_node_force is the force (scalar) in every point (x,y,z) of the graph at time t
            all_node_force = whole_trajectory_batch.x_node_force.reshape(len(graphs_list_batch), -1, 3)  # (T, N, 3)
            # .x_bc contains 0,1 value with 1 = fixed, 0 = free
            boundary_condition = graphs_list_batch[0].x_bc

            # Inside loop: for each graph build its features
            effective_limit = self.selected_limit if idx_file in self.selected_indices else cfg.limit_timesteps
            max_usable_timesteps = min(max(0, effective_limit - 1), a.shape[0])
            # Initialized past x position
            _x_pos_tm1_prev = x[0]
            for t, graph in enumerate(graphs_list_batch):
                if t >= max_usable_timesteps:
                    break
                self.traj_idx_list.append(idx_file)  # idx_file of the traj (of the file process)
                # create local (inside for) variables
                _node_force = all_node_force[t]
                # All are (N, 3)
                _x_t, _v_t, _a_t = x[t], v[t], a[t]
                _categorical_node_attr = boundary_condition
                # concatenate [position, velocity, rest_position, node_force] and x_minus_u optionally (3)
                _node_attr, split_idx_node = self._build_node_features_scripted(
                    _x_t, _v_t, _x_pos_tm1_prev, rest_pos_x, _node_force)
                # Obtain edge features [rel_pos, rel_pos_norm, rest_rel_pos, rest_rel_pos_norm]
                # .edge_index contains mesh connectivity
                _edge_attr, split_idx_edge = self._build_edge_features_scripted(
                    _x_t, _v_t, _x_pos_tm1_prev, rest_pos_x, _node_force, graph.edge_index)
                # .y_sig_t contains the full stress tensor
                # Output _von_mises_raw is (N, 1) [graph.y_sig_t is (N, 9)]
                _von_mises_raw = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
                # log transform helps making the distribution normal-like (symmetrical)
                _von_mises_log = torch.log(_von_mises_raw + EPS)
                _von_mises = _von_mises_log if cfg.mises_log else _von_mises_raw

                # accumulate all graphs to out_graphs for stats computation
                self.categorical_node_features.append(boundary_condition)
                self.valued_node_features.append(_node_attr.clone())
                self.edge_features.append(_edge_attr.clone())
                # Appends and creates PyG data object for both val and train
                # For val the features are normalized via the train stats before appending, for train I have yet
                # to compute the stats
                self._append_outgraphs(graph, _a_t, _x_t, _v_t, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                                       all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log,
                                       surface_left_triangles, surface_right_triangles, _x_pos_tm1_prev)
                _x_pos_tm1_prev = _x_t.clone()

        self.split_idx_node = split_idx_node
        self.split_idx_edge = split_idx_edge

        # --- Open: All these do nothing if split NOT train ---
        self._compute_stats_train(cfg, split_idx_edge, split_idx_node)  # if val doesn't do anything
        self._save_stats_train(self.node_stats, self.edge_stats, self.target_stats, save_dir)  # if val nothing
        self._save_auxiliary_train_artifacts(save_dir)
        self._normalize_assign_features_train()
        # Save all processed data (in all cases)
        self._save_preprocessed(self.traj_idx_list, self.out_graphs, save_dir)

    @abstractmethod
    def _compute_stats_train(self, cfg, split_idx_edge, split_idx_node):
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
                          boundary_left_triangles, boundary_right_triangles, x_pos_tm1_prev):
        raise NotImplementedError("_normalize_append_outgraphs")

    @abstractmethod
    def _maybe_update_maxangle_train(self, x_raw, graphs_list_batch, rest_pos_x, cfg, idx_file):
        raise NotImplementedError("_maybe_update_train_curvature_threshold")

    @abstractmethod
    def _save_auxiliary_train_artifacts(self, save_dir):
        raise NotImplementedError("_save_auxiliary_train_artifacts")


class TrainPreprocessor(Preprocessor):

    def __init__(self, cfg, selected_indices, selected_limit):
        super().__init__(cfg, selected_indices, selected_limit)

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
        layout_payload = {
            "split_idx_node": int(self.split_idx_node) if self.split_idx_node is not None else None,
            "split_idx_edge": int(self.split_idx_edge) if self.split_idx_edge is not None else None,
        }
        with open(stats_dir / "feature_layout.json", "w") as f:
            json.dump(layout_payload, f, indent=4)
        print(f"Feature layout saved to {stats_dir / 'feature_layout.json'}")

    def _compute_stats_train(self, cfg, split_idx_edge, split_idx_node):
        """
        Args
            cfg: PreprocessYaml
                contains config parameters
            split_idx_edge:
                index where to split the edge features  between isotropic (3 coordinates) and not isotropic ("1 coordinate")
            split_idx_node:
                index where to split the node features between isotropic (3 coordinates) and not isotropic ("1 coordinate")
        :return:
        """
        print("\nComputing train stats...\n")
        # Compute statistics
        self.node_stats = self.normalizer.compute_stats(self.valued_node_features, split_idx_node)
        # print(f"edge_features.shape = {self.edge_features[0].shape}")
        self.edge_stats = self.normalizer.compute_stats(self.edge_features, split_idx_edge)
        # isotropic acceleration iff normalizer is isotropic
        # FIXME: hardcoded target handling
        self.target_stats = self.normalizer.compute_stats(self.targets, 3)
        print("\nFinished computing train stats\n")

    def _load_stats(self, cfg):
        pass

    def _maybe_update_maxangle_train(self, x_raw, graphs_list_batch, rest_pos_x, cfg, idx_file):
        boundary_condition = graphs_list_batch[0].x_bc
        # Obtain triangles for angle computation
        fixed_mask = boundary_condition.squeeze(1) > 0.5
        surface_left_triangles, surface_right_triangles = extract_surface_triangles_torch(
            graphs_list_batch[0].x_element_connectivity, rest_pos_x.device)
        surface_left_triangles, surface_right_triangles, _ = filter_fixed_surface_triangles(surface_left_triangles,
                                                                                            surface_right_triangles,
                                                                                            fixed_mask)
        effective_limit = self.selected_limit if idx_file in self.selected_indices else cfg.limit_timesteps
        max_timesteps = min(x_raw.shape[0], effective_limit)
        for timestep_idx in range(max_timesteps):
            current_theta_max = max_surface_angle(x_raw[timestep_idx], surface_left_triangles,
                                                  surface_right_triangles)
            self.theta_max = max(self.theta_max, float(current_theta_max.detach().item()))

        return surface_left_triangles, surface_right_triangles

    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log, boundary_left_triangles,
                          boundary_right_triangles, x_pos_tm1_prev):
        # Note: later for each data I will create the standard pyg - compatible attributes x, y, edge_attr actually
        # used for NNs
        self.out_graphs.append(Data(y_acc_t=_a, # TARGET feature acceleration
                                    # Helpers for rollout or other tasks
                                    edge_index=graph.edge_index,  # global index
                                    t=graph.time,  # global t attribute
                                    # Features (input, x)
                                    x_pos_t=_x, # FEATURE position
                                    x_pos_tm1_prev=x_pos_tm1_prev,
                                    x_vel_t=_v, # FEATURE velocity
                                    rest_pos=rest_pos_x, # rest position (for runtime noise injection)
                                    node_force=all_node_force[t],
                                    x_bc=_categorical_node_attr,
                                    # Helper for curvature physics informed loss
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

    def __init__(self, cfg, save_dir_train, selected_indices, selected_limit):
        super().__init__(cfg, selected_indices, selected_limit)
        self.save_dir_train = save_dir_train

    def _compute_stats_train(self, cfg, split_idx_edge, split_idx_node):
        pass

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        pass

    def _normalize_assign_features_train(self):
        pass

    def _load_stats(self, cfg):
        self.node_stats, self.edge_stats, self.target_stats = load_stats(self.save_dir_train)
        print(f"Stats:\n\tnode_stats.mean = \t{self.node_stats.mean}\n\tnode_stats.std =\t{self.node_stats.std}"
              f"\n\tedge_stats.mean =\t{self.edge_stats.mean}\n\tedge_stats.std =\t{self.edge_stats.std}"
              f"\n\ttarget_stats =\t{self.target_stats.mean}\n\ttarget_stats.std =\t{self.target_stats.std}")

    def _maybe_update_maxangle_train(self, x_raw, graphs_list_batch, rest_pos_x, cfg, idx_file):
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
                          all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log, boundary_left_triangles,
                          boundary_right_triangles, x_pos_tm1_prev):
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
                x_pos_tm1_prev=x_pos_tm1_prev,
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
