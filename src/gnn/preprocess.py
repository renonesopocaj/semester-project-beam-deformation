import torch
import json
import hydra
from omegaconf import OmegaConf, open_dict
from src.gnn.helpers.misc.yamlobject import PreprocessYaml
from src.gnn.preprocessing.build_features import (calculate_kinematics, compute_von_mises_3d,
                                                  build_edge_features,
                                                  build_valued_node_features)
from src.gnn.preprocessing.normalizer import Normalizer, NormalizerFactory
from src.gnn.preprocessing.stats import load_stats
from src.gnn.preprocessing.stats import Stats
from src.gnn.preprocessing.noiser import Noiser, GaussianNoiser, NoNoiser
from torch_geometric.data import Data, Batch
from pathlib import Path
from abc import ABC, abstractmethod

EPS = 1e-6


class Preprocessor(ABC):
    noiser: Noiser
    node_stats: Stats
    edge_stats: Stats
    target_stats: Stats
    normalizer: Normalizer
    out_graphs: list[torch.Tensor]
    categorical_node_features: list[torch.Tensor]
    valued_node_features: list[torch.Tensor] # each tensor is (N, 12)
    edge_features: list[torch.Tensor]
    traj_idx_list: list[torch.Tensor]
    targets: list[torch.Tensor]

    def __init__(self):
        (self.out_graphs, self.categorical_node_features, self.valued_node_features, self.edge_features,
         self.traj_idx_list, self.targets) = ([], [], [], [], [], [])

    def _save_preprocessed(self, traj_idx, out_graphs, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for traj in set(traj_idx):
            print(f"Saving trajectory: {traj}")
            graphs = [graph for idx, graph in zip(traj_idx, out_graphs) if idx == traj]
            print(f"  Length: {len(graphs)} timesteps")
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
        """
        torch.manual_seed(cfg.seed)
        filenames = list(Path(graphs_dir).glob("*.pt"))
        split_idx_edge = 9 if cfg.rest_minus_nonrest_pos else 6
        self._load_stats(cfg)  # if train doesn't do anything since I have to compute them

        # Main loop: process all files, one file = one trajectory, idx_file = index of the file
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
            x = whole_trajectory_batch.x.reshape(len(graphs_list_batch), -1, 3)  # (#graphs, #points, 3) where 3 is (x,y,z)
            v, a = calculate_kinematics(x, cfg.dt)  # v = (T, N, 3), a = (T-1, N, 3)
            x, v = self.noiser.add_noise_train(x, v, cfg.dt)
            v_next = v[1:]  # (T-1, N, 3)
            x_next = x[1:]  # (T-1, N, 3)
            rest_pos_x = x[0]  # Rest (first) mesh position (N, 3)
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
                _node_attr = build_valued_node_features(_x, _v, rest_pos_x, _node_force, cfg.rest_minus_nonrest_pos)
                # Obtain edge features [rel_pos, rel_pos_norm, rest_rel_pos, rest_rel_pos_norm]
                # .edge_index contains mesh connectivity
                _edge_attr = build_edge_features(_x, rest_pos_x, graph.edge_index, cfg.rest_minus_nonrest_pos)
                # .y_sig_t contains the full stress tensor
                # took the log because it helps making the distribution normal-like (symmetrical)
                if cfg.mises_log:
                    # # Output _von_mises is (N, 1) [graph.y_sig_t is (N, 9)]
                    _von_mises = torch.log(compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + EPS)
                else:
                    _von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)

                # accumulate all graphs to out_graphs for stats computation
                self.categorical_node_features.append(boundary_condition)
                # concatenate _x, _v, rest_pos_x, _node_force and optionally x_minus_u
                self.valued_node_features.append(build_valued_node_features(_x, _v, rest_pos_x, _node_force,
                                                                            cfg.rest_minus_nonrest_pos))
                self.edge_features.append(build_edge_features(_x, rest_pos_x, graph.edge_index,
                                                              cfg.rest_minus_nonrest_pos))
                # Appends and creates PyG data object for both val and train
                # For val it's already normalized
                self._append_outgraphs(graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                                       all_node_force, rest_pos_x, _categorical_node_attr)

        # --- Open: All these do nothing if split NOT train ---
        self._compute_stats_train(cfg, split_idx_edge)  # if val doesn't do anything
        self._save_stats_train(self.node_stats, self.edge_stats, self.target_stats, save_dir)  # if val nothing
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
                          all_node_force, rest_pos_x, _categorical_node_attr):
        raise NotImplementedError("_normalize_append_outgraphs")

class TrainPreprocessor(Preprocessor):

    def __init__(self, cfg):
        super().__init__()
        self.noiser = GaussianNoiser(cfg.noise_scale_train)
        self.normalizer = NormalizerFactory.create_normalizer(cfg.normalizer_method, cfg.isotropic_normalizer)

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        stats = {"node": node_stats.to_dict(), "edge": edge_stats.to_dict(), "target": target_stats.to_dict()}
        stats_dir = save_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Stats saved to {stats_dir / 'stats.json'}")

    def _compute_stats_train(self, cfg, split_idx_edge):
        # Compute statistics
        PLACEHOLDER_ALL_ISOTROPIC = self.valued_node_features[0].shape[1]
        print(f"valued_node_features.shape = {self.valued_node_features[0].shape}")
        self.node_stats = self.normalizer.compute_stats(self.valued_node_features, PLACEHOLDER_ALL_ISOTROPIC)
        print(f"edge_features.shape = {self.edge_features[0].shape}")
        self.edge_stats = self.normalizer.compute_stats(self.edge_features, split_idx_edge)
        print(f"targets.shape = {self.targets[0].shape}")
        self.target_stats = self.normalizer.compute_stats(self.targets, 0)

    def _load_stats(self, cfg):
        pass

    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr):
        # Note: later for each data I will create the standard pyg - compatible attributes x, y, edge_attr actually
        # used for NNs
        self.out_graphs.append(
            Data(
                edge_index=graph.edge_index, # global index
                t=graph.time, # global t attribute
                y_acc_t=_a, # target feature acceleration
                x_pos_t=_x, # feature position
                x_vel_t=_v # feature velocity
            )
        )
        self.targets.append(torch.concat([_a, _von_mises], dim=1))

    def _normalize_assign_features_train(self):
        # Normalize and assign features
        for t in range(len(self.out_graphs)):
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
            if t == 0:
                print(f"The shape of x is = {self.out_graphs[0].x.shape} edge_attr = "
                      f"{self.out_graphs[t].edge_attr.shape} out_graphs[t].y = {self.out_graphs[t].y.shape}")


class ValPreprocessor(Preprocessor):
    save_dir_train: Path

    def __init__(self, cfg, save_dir_train):
        super().__init__()
        self.noiser = NoNoiser()
        self.normalizer = NormalizerFactory.create_normalizer(cfg.normalizer_method, cfg.isotropic_normalizer)
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

    def _normalize_before_appending(self, _edge_attr, _node_attr, _categorical_node_attr, _target):
        # Use precomputed stats for validation
        _numerical_node_attr_norm = self.normalizer.normalize(_node_attr, self.node_stats)
        edge_attr_norm = self.normalizer.normalize(_edge_attr, self.edge_stats)
        node_attr = torch.concat([_categorical_node_attr, _numerical_node_attr_norm], dim=1)
        target_norm = self.normalizer.normalize(_target, self.target_stats)
        return node_attr, edge_attr_norm, target_norm

    # Validation: normalize with stats already computed, and add to out_graphs as PyG data
    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr):
        _target = torch.concat([_a, _von_mises], dim=1)
        node_attr_norm, edge_attr_norm, target_norm = self._normalize_before_appending(
            _edge_attr, _node_attr, _categorical_node_attr, _target)
        self.out_graphs.append(
            Data(
                # targets: current time acceleration and von mises stress
                y=target_norm,
                y_acc_t=_a,
                y_von_mises=_von_mises,
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
                x_element_connectivity=graph.x_element_connectivity
            )
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
    save_dir = Path(cfg_obj.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_dir / "config.yaml")
    processed_dir_name = (f"{cfg_obj.normalizer_method}_{'isotropic' if cfg_obj.isotropic_normalizer is True else 'anisotropic'}_"
                   f"{'miseslog' if cfg_obj.mises_log is True else 'nolog'}")
    processed_dataset_dir = Path(cfg_obj.save_dir) / processed_dir_name / cfg_obj.dataset_size
    # note: this is hardcoded to be quicker, but I could have just saved the resulting vectors
    with open_dict(cfg):
        # note numerical feat resp 15 and 12, + 1 categorical
        cfg.node_channels = 16 if cfg_obj.rest_minus_nonrest_pos is True else 13
        cfg.edge_channels = 12 if cfg_obj.rest_minus_nonrest_pos is True else 8
    processed_dataset_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, processed_dataset_dir / "config.yaml")
    # Where to save results
    save_dir_train = processed_dataset_dir / "train"
    save_dir_test = processed_dataset_dir / "test"
    save_dir_val = processed_dataset_dir / "val"
    # Raw paths
    graphs_dir_train = Path(cfg_obj.raw_data_dir) / "train" / "graphs"
    graphs_dir_test = Path(cfg_obj.raw_data_dir) / "test" / "graphs"
    graphs_dir_val = Path(cfg_obj.raw_data_dir) / "val" / "graphs"
    # Create preprocessors object
    preprocessor_train = create_preprocessor("train", cfg_obj, None)
    preprocessor_val = create_preprocessor("val", cfg_obj, save_dir_train)
    preprocessor_test = create_preprocessor("test", cfg_obj, save_dir_train)
    print("\n" + "=" * 60 + "\ntrain")
    preprocessor_train.preprocess_data(cfg_obj, graphs_dir_train, save_dir_train)
    print("\n" + "=" * 60 + "\nPreprocessing complete!")
    print("\n" + "=" * 60 + "\ntest")
    preprocessor_test.preprocess_data(cfg_obj, graphs_dir_test, save_dir_test)
    print("\n" + "=" * 60 + "\nPreprocessing complete!")
    print("\n" + "=" * 60 + "\nval")
    preprocessor_val.preprocess_data(cfg_obj, graphs_dir_val, save_dir_val)
    print("\n" + "=" * 60 + "\nPreprocessing complete!")
    print("\n" + "=" * 60 + "\nPreprocessing complete!")


if __name__ == "__main__":
    main()
