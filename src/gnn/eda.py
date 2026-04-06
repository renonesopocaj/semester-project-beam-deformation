"""
eda.py
Main file to run in order to do Exploratory Data Analysis.
- The del command are there to avoid out of memory errors.
This script does the following:
1. Preprocess the data and save it in a format that is easy to use for EDA.
2. Plot the distributions of the features.
3. Plot uniformly sampled trajectories (10%, i.e. for 1000 samples 100)
- For train the normalized distributions are computed via train stats, for val and test also with train stats.
- To avoid code repetition, the preprocessing reuses the samefeature builders used in preprocess.py.
- All the possible normalization are done.
"""

import copy
import gc
import json
import math
import os
import shutil
from pathlib import Path

import hydra
from src.gnn.eda_helpers.eda_features_helpers import EdaObj, build_node_feature_names, build_edge_feature_names
from src.gnn.eda_helpers.eda_plot import plot_all_distributions
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from omegaconf import OmegaConf

from src.gnn.misc_helpers.yamlobject import PreprocessYaml, EdaYaml
from src.gnn.preprocess import (TrainPreprocessor, load_selected_trajectories,
                                _selected_indices_for_split, _selected_limit_from_json,
                                _sort_filenames_by_traj_params)
from src.gnn.preprocessing.build_features import (build_node_features, build_edge_features, compute_von_mises_3d)
from src.gnn.preprocessing.stats import Stats
from src.gnn.validation.plot_rollout import make_ground_truth_gif

EPS = 1e-6
DEVICE = "cpu"


def _to_cpu_clone(value):
    """
    Args:
        value: object
            Tensor-like or array-like value.
        return: object
            Detached CPU copy preserving the original container type when possible.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    return value


def _build_uniformly_spread_indices(total_count, percentage):
    """
    Args:
        total_count: int
            Total number of trajectories in the split.
        percentage: float
            Percentage of trajectories to sample.
    return: set[int]
        Uniformly spread trajectory indices.
    """
    if total_count <= 0:
        return set()
    if percentage <= 0.0:
        return set()
    n_selected = int(round(total_count * percentage / 100.0))
    n_selected = max(1, min(total_count, n_selected))
    if n_selected == total_count:
        return set(range(total_count))
    selected_indices = set()
    for idx in range(n_selected):
        selected_index = min(total_count - 1, int(math.floor((idx + 0.5) * total_count / n_selected)))
        selected_indices.add(selected_index)
    return selected_indices


def _build_gif_trajectory_indices(total_count, generate_gifs, GIF_SAMPLE_PERCENTAGE, GIF_GENERATION_MODE):
    """
    Args:
        total_count: int
            Total number of trajectories in the split.
        generate_gifs: bool
            Whether GIF generation is enabled for the current run.
    return: set[int]
        Trajectory indices whose GIFs should be generated.
    """
    if not generate_gifs:
        return set()
    if GIF_GENERATION_MODE == "ALL":
        return set(range(total_count))
    if GIF_GENERATION_MODE == "UNIFORMLY_SAMPLED":
        return _build_uniformly_spread_indices(total_count, GIF_SAMPLE_PERCENTAGE)
    raise ValueError(f"Unknown GIF_GENERATION_MODE: {GIF_GENERATION_MODE}")


def _create_variant_cfg(base_cfg, normalizer_method, isotropic_normalizer):
    """
    Args:
        base_cfg: PreprocessYaml
            Base preprocessing config.
        normalizer_method: str
            Normalizer method for the variant.
        isotropic_normalizer: bool
            Whether the normalizer pools xyz triplets isotropically.
    return: PreprocessYaml
        Independent config object for one EDA run.
    """
    variant_cfg = copy.deepcopy(base_cfg)
    variant_cfg.normalizer_method = normalizer_method
    variant_cfg.isotropic_normalizer = isotropic_normalizer
    return variant_cfg

class EdaPreprocessor(TrainPreprocessor):
    eda_obj: object
    artifact_dir: Path
    run_name: str
    gif_generation_mode: str
    gif_sample_percentage: float
    eda_split: str
    train_artifact_dir: Path
    traj_params: dict

    def __init__(self, cfg, artifact_dir, run_name, gif_generation_mode, gif_sample_percentage,
                 eda_split, train_artifact_dir, selected_indices, selected_limit):
        """
        Construct an EDA-oriented preprocessor: same stats machinery as TrainPreprocessor, plus paths and an EdaObj
        wired to feature names from the current config.

        1. Call the train preprocessor base initializer (normalizer factory, empty accumulators).
        2. Store artifact paths, GIF settings, split name, and train artifact root for stats reuse.
        3. Build EdaObj with node and edge feature name lists matching build_node_features / build_edge_features.

        Args:
            cfg: object
                Config object.
            artifact_dir: str or Path
                Root directory for EDA artifacts.
            run_name: str
                Name of the current normalization run.
            gif_generation_mode: str
                GIF generation mode (ALL or UNIFORMLY_SAMPLED).
            gif_sample_percentage: float
                Percentage of trajectories to sample for GIFs.
            eda_split: str
                Current EDA split (train, val, or test).
            train_artifact_dir: str or Path
                Root directory for train EDA artifacts (used to load/save stats).
            selected_indices: set[int]
                Trajectory indices that should use selected_limit.
            selected_limit: int
                Timestep limit for selected trajectories.
        """
        super().__init__(cfg, selected_indices, selected_limit)
        self.artifact_dir = Path(artifact_dir)
        self.run_name = run_name
        self.eda_obj = EdaObj(build_node_feature_names(cfg), build_edge_feature_names(cfg))
        self.gif_generation_mode = gif_generation_mode
        self.gif_sample_percentage = gif_sample_percentage
        self.eda_split = eda_split
        self.train_artifact_dir = Path(train_artifact_dir)
        self.traj_params = {}

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        pass

    def _save_preprocessed(self, traj_idx, out_graphs, save_dir):
        pass

    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log, boundary_left_triangles,
                          boundary_right_triangles, x_pos_tm1_prev):
        pass

    def _save_eda_stats(self):
        """
        Write self.node_stats, self.edge_stats, and self.target_stats to JSON under the train EDA artifact tree so val/test
        EDA runs can load identical normalization parameters.

        1. Ensure the directory train_artifact_dir / .eda_stats / run_name exists.
        2. Serialize the three Stats objects via to_dict into one JSON file named stats.json.
        3. Print the output path for logging.
        """
        stats_dir = self.train_artifact_dir / ".eda_stats" / self.run_name
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats = {"node": self.node_stats.to_dict(),
                 "edge": self.edge_stats.to_dict(),
                 "target": self.target_stats.to_dict()}
        with open(stats_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        print(f"EDA stats saved to {stats_dir / 'stats.json'}")

    def _load_eda_stats(self):
        """
        Read stats.json produced by a prior train-split EDA run and assign self.node_stats, self.edge_stats, and
        self.target_stats.

        1. Resolve stats.json under train_artifact_dir / .eda_stats / run_name.
        2. Raise a clear error if the file is missing.
        3. Parse JSON and rebuild Stats objects with Stats.from_dict.
        """
        stats_path = self.train_artifact_dir / ".eda_stats" / self.run_name / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"EDA stats not found at {stats_path}. "
                f"Run EDA on the train split first to generate them.")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.node_stats = Stats.from_dict(stats["node"])
        self.edge_stats = Stats.from_dict(stats["edge"])
        self.target_stats = Stats.from_dict(stats["target"])
        print(f"EDA stats loaded from {stats_path}")

    def preprocess_data(self, cfg, graphs_dir, generate_gifs):
        """
        Run EDA preprocessing for the current split: either collect raw feature lists, or cache per-trajectory tensors,
        fit or load normalization stats, then fill the EdaObj with optionally normalized values.

        1. Optionally load trajectory metadata from trajectory_params.json.
        2. If normalizer is disabled, stream each file through a trajectory feature bundle into EdaObj (raw mode).
        3. Otherwise write one disk cache file per trajectory, compute stats on train (or load saved stats), then
           re-read caches and append normalized tensors to EdaObj.

        Args:
            cfg: object
                Preprocessing config.
            graphs_dir: str or Path
                Directory containing raw trajectory `.pt` files.
            generate_gifs: bool
                True iff uniformly sampled ground-truth GIFs should be generated.
        """
        torch.manual_seed(cfg.seed)
        # Load trajectory_params.json (written by main_run_simulation) if available.
        params_json_path = Path(graphs_dir).parent / "trajectory_params.json"
        if params_json_path.exists():
            with open(params_json_path, "r") as f:
                self.traj_params = json.load(f)
        else:
            self.traj_params = {}
        filenames = _sort_filenames_by_traj_params(list(Path(graphs_dir).glob("*.pt")), graphs_dir)
        if cfg.normalizer_method.lower() == "none":
            self._collect_raw_eda(cfg, filenames, generate_gifs)
            return

        cache_dir = self.artifact_dir / ".eda_cache" / self.run_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache_files, split_idx_node, split_idx_edge = self._cache_trajectories(cfg, filenames, cache_dir)
            if self.eda_split == "train":
                self._compute_stats_from_cache(cache_files, split_idx_node, split_idx_edge)
                self._save_eda_stats()
            else:
                self._load_eda_stats()
            self._collect_normalized_eda(cache_files)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _collect_raw_eda(self, cfg, filenames, generate_gifs):
        """
        Build per-timestep node, edge, and target tensors for each trajectory and append them to EdaObj without
        normalization (used when normalizer_method is none).

        1. Decide which trajectory indices get GIF exports.
        2. For each raw `.pt` file, build a trajectory_feature_bundle and pass its timestep lists to EdaObj.
        3. Optionally render a ground-truth GIF and free memory after each file.

        Args:
            cfg: object
                Preprocessing config.
            filenames: list[Path]
                Trajectory files to process.
            generate_gifs: bool
                Whether to generate trajectory GIFs during the pass.
        """
        gif_indices = _build_gif_trajectory_indices(len(filenames), generate_gifs, self.gif_sample_percentage,
                                                    self.gif_generation_mode)
        pbar = tqdm(enumerate(filenames), total=len(filenames), desc=f"EDA {self.run_name}")
        for traj_idx, filename in pbar:
            keep_gif_graphs = traj_idx in gif_indices
            trajectory_feature_bundle, trajectory_graphs = self._build_trajectory_feature_bundle(
                cfg, filename, keep_gif_graphs, traj_idx)
            self._append_trajectory_feature_bundle_to_eda(trajectory_feature_bundle, normalize=False)
            if trajectory_graphs is not None:
                traj_metadata = self.traj_params.get(filename.name, None)
                self._generate_ground_truth_gif(trajectory_graphs, traj_idx, traj_metadata)
            del trajectory_feature_bundle
            del trajectory_graphs
            gc.collect()

    def _cache_trajectories(self, cfg, filenames, cache_dir):
        """
        For each raw trajectory file, build the same per-timestep tensors as training preprocessing would use, save them
        to a temporary `.pt` file, and return paths plus the node/edge isotropic split indices shared by all files.

        1. Iterate sorted trajectory files and call _build_trajectory_feature_bundle without GIF side data.
        2. torch.save each trajectory_feature_bundle under cache_dir with a stable index-based name.
        3. Verify split_idx_node and split_idx_edge match every trajectory; return paths and those indices.

        Args:
            cfg: object
                Preprocessing config.
            filenames: list[Path]
                Trajectory files to process.
            cache_dir: Path
                Directory used to store compact raw trajectory bundles.
            return: (list[Path], int, int)
                Cache file paths and split indices for node/edge xyz-triplet regions.
        """
        cache_files = []
        split_idx_node = None
        split_idx_edge = None
        pbar = tqdm(enumerate(filenames), total=len(filenames), desc=f"Caching {self.run_name}")
        for traj_idx, filename in pbar:
            trajectory_feature_bundle, _ = self._build_trajectory_feature_bundle(cfg, filename, False, traj_idx)
            cache_path = cache_dir / f"traj_{traj_idx:05d}.pt"
            torch.save(trajectory_feature_bundle, cache_path)
            cache_files.append(cache_path)
            if split_idx_node is None:
                split_idx_node = trajectory_feature_bundle["split_idx_node"]
            elif split_idx_node != trajectory_feature_bundle["split_idx_node"]:
                raise ValueError("Inconsistent node split_idx across cached trajectories.")
            if split_idx_edge is None:
                split_idx_edge = trajectory_feature_bundle["split_idx_edge"]
            elif split_idx_edge != trajectory_feature_bundle["split_idx_edge"]:
                raise ValueError("Inconsistent edge split_idx across cached trajectories.")
            del trajectory_feature_bundle
            gc.collect()
        if split_idx_node is None or split_idx_edge is None:
            raise ValueError("No trajectory cache produced, cannot infer split_idx.")
        return cache_files, split_idx_node, split_idx_edge

    def _load_family_from_cache(self, cache_files, bundle_field_name, start_column):
        """
        Load one tensor family (all timesteps, all trajectories) from the on-disk trajectory caches into a flat list.

        1. For each cache file, load the saved trajectory_feature_bundle dict.
        2. Take the list stored under bundle_field_name and extend the output with each timestep tensor.
        3. Optionally drop leading columns (e.g. skip boundary-condition column for node valued stats).

        Args:
            cache_files: list[Path]
                Cache files created by _cache_trajectories().
            bundle_field_name: str
                Key in the saved dict: node_inputs, edge_features, or targets.
            start_column: int or None
                Optional column offset used to skip the BC feature in node inputs.
            return: list[torch.Tensor]
                Family tensors loaded from cache.
        """
        tensor_list = []
        for cache_path in cache_files:
            cached_trajectory_bundle = torch.load(cache_path, map_location="cpu", weights_only=False)
            for tensor in cached_trajectory_bundle[bundle_field_name]:
                if start_column is None:
                    tensor_list.append(tensor)
                else:
                    tensor_list.append(tensor[:, start_column:])
            del cached_trajectory_bundle
        return tensor_list

    def _compute_stats_from_cache(self, cache_files, split_idx_node, split_idx_edge):
        """
        Fit self.node_stats, self.edge_stats, and self.target_stats from all cached timestep tensors using the current
        normalizer (including isotropic pooling boundaries).

        1. Load node_inputs from disk, strip the BC column, compute node_stats with split_idx_node.
        2. Load edge_features and compute edge_stats with split_idx_edge.
        3. Load targets and compute target_stats with TARGET_ACCELERATION_SPLIT_IDX for the acceleration xyz block.

        Args:
            cache_files: list[Path]
                Cache files created by _cache_trajectories().
            split_idx_node: int
                Number of node valued feature columns grouped as xyz triplets.
            split_idx_edge: int
                Number of edge feature columns grouped as xyz triplets.
            return: None
                Fills self.node_stats, self.edge_stats, self.target_stats.
        """
        print("\nComputing EDA stats from cached trajectories...\n")
        node_tensors = self._load_family_from_cache(cache_files, "node_inputs", 1)
        self.node_stats = self.normalizer.compute_stats(node_tensors, split_idx_node)
        del node_tensors
        gc.collect()

        edge_tensors = self._load_family_from_cache(cache_files, "edge_features", None)
        self.edge_stats = self.normalizer.compute_stats(edge_tensors, split_idx_edge)
        del edge_tensors
        gc.collect()

        target_tensors = self._load_family_from_cache(cache_files, "targets", None)
        self.target_stats = self.normalizer.compute_stats(target_tensors, 3)
        del target_tensors
        gc.collect()
        print("\nFinished computing EDA stats\n")

    def _collect_normalized_eda(self, cache_files):
        """
        Re-read every cached trajectory_feature_bundle, normalize tensors with train-loaded or train-fitted stats, and
        append each timestep to self.eda_obj.

        1. Load each cache file as a trajectory_feature_bundle dict.
        2. For each timestep triple (node_input, edge_features, target), normalize then call EdaObj.append_timestep.
        3. Drop the loaded bundle from memory after each file to limit RAM use.

        Args:
            cache_files: list[Path]
                Cache files created by _cache_trajectories().
            return: None
                Fills self.eda_obj from normalized cached trajectories.
        """
        pbar = tqdm(cache_files, total=len(cache_files), desc=f"Collecting {self.run_name}")
        for cache_path in pbar:
            cached_trajectory_bundle = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._append_trajectory_feature_bundle_to_eda(cached_trajectory_bundle, normalize=True)
            del cached_trajectory_bundle
            gc.collect()

    def _normalize_node_input(self, node_input):
        """
        Normalize only the valued node feature columns using self.node_stats, leaving the boundary-condition column
        unchanged.

        1. Split node_input into column 0 (BC) and columns 1: (valued features).
        2. Apply self.normalizer.normalize to the valued block with self.node_stats.
        3. Concatenate BC and normalized valued columns back to the original width.

        Args:
            node_input: torch.Tensor
                Raw node input tensor `[bc, valued_node_features]`.
            return: torch.Tensor
                Node input tensor with only the valued part normalized.
        """
        normalized_valued = self.normalizer.normalize(node_input[:, 1:], self.node_stats)
        return torch.concat([node_input[:, 0:1], normalized_valued], dim=1)

    def _append_trajectory_feature_bundle_to_eda(self, trajectory_feature_bundle, normalize):
        """
        Zip the per-timestep lists inside trajectory_feature_bundle and feed each step into EdaObj, optionally after
        normalizing with self.node_stats, self.edge_stats, and self.target_stats.

        1. Iterate aligned entries from node_inputs, edge_features, and targets.
        2. If normalize is true, normalize the valued node block, edges, and targets.
        3. Call append_timestep on self.eda_obj for each timestep.

        Args:
            trajectory_feature_bundle: dict
                Per-trajectory structure with keys node_inputs, edge_features, targets, split_idx_node, split_idx_edge
                (split indices are ignored here; only the three tensor lists are consumed).
            normalize: bool
                Whether to apply the current normalizer before appending to EdaObj.
            return: None
                Appends all timestep features to self.eda_obj.
        """
        for node_input, edge_features, target in zip(trajectory_feature_bundle["node_inputs"],
                                                     trajectory_feature_bundle["edge_features"],
                                                     trajectory_feature_bundle["targets"]):
            if normalize:
                node_input = self._normalize_node_input(node_input)
                edge_features = self.normalizer.normalize(edge_features, self.edge_stats)
                target = self.normalizer.normalize(target, self.target_stats)
            self.eda_obj.append_timestep(node_input, edge_features, target)

    def _build_trajectory_feature_bundle(self, cfg, filename, keep_gif_graphs, traj_idx):
        """
        Load one raw trajectory `.pt` file, compute kinematics and the same node/edge/target tensors as the training
        preprocessor uses for each usable timestep, and return them as lists plus feature-layout split indices.

        1. Batch all graphs in the file and differentiate positions to get velocity and acceleration.
        2. For each timestep within the effective limit, build node_input, edge_attr, and target tensors on CPU clones.
        3. Optionally collect minimal PyG Data objects for GIF rendering; return the dict and that list (or None).

        Args:
            cfg: object
                Preprocessing config.
            filename: Path
                Raw trajectory file.
            keep_gif_graphs: bool
                True iff the returned optional list should include minimal GIF graph objects.
            traj_idx: int
                Index of the trajectory in processing order.
            return: (dict, list or None)
                trajectory_feature_bundle dict and optional GIF graph list for the current trajectory.
        """
        dataloader = torch.load(filename, weights_only=False)
        graphs_list_batch = []
        for data in dataloader:
            graph = data.to(cfg.device)
            graphs_list_batch.append(graph)

        whole_trajectory_batch = Batch.from_data_list(graphs_list_batch)
        x = whole_trajectory_batch.x.reshape(len(graphs_list_batch), -1, 3)
        v, a = self.time_discretizer.differentiate_position(x, cfg.dt)
        rest_pos_x = x[0].clone()
        all_node_force = whole_trajectory_batch.x_node_force.reshape(len(graphs_list_batch), -1, 3)
        boundary_condition = graphs_list_batch[0].x_bc

        trajectory_feature_bundle = {"node_inputs": [], "edge_features": [], "targets": [],
                                     "split_idx_node": None, "split_idx_edge": None}
        trajectory_graphs = [] if keep_gif_graphs else None

        effective_limit = self.selected_limit if traj_idx in self.selected_indices else cfg.limit_timesteps
        max_usable_timesteps = min(max(0, effective_limit - 1), a.shape[0])
        _x_tm1_prev = x[0].clone()
        for t, graph in enumerate(graphs_list_batch):
            if t >= max_usable_timesteps:
                break
            node_force = all_node_force[t]
            x_t = x[t]
            v_t = v[t]
            a_t = a[t]
            valued_node_attr, split_idx_node = build_node_features(x_t, v_t, _x_tm1_prev, rest_pos_x, node_force,
                                                                   cfg)
            if trajectory_feature_bundle["split_idx_node"] is None:
                trajectory_feature_bundle["split_idx_node"] = split_idx_node
            node_input = torch.concat([boundary_condition, valued_node_attr], dim=1)
            edge_attr, split_idx_edge = build_edge_features(x_t, v_t, _x_tm1_prev, rest_pos_x, node_force,
                                                            graph.edge_index, cfg)
            if trajectory_feature_bundle["split_idx_edge"] is None:
                trajectory_feature_bundle["split_idx_edge"] = split_idx_edge
            von_mises_log = torch.log(compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + EPS)
            if cfg.mises_log:
                von_mises = von_mises_log
            else:
                von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
            target = torch.concat([a_t, von_mises], dim=1)

            trajectory_feature_bundle["node_inputs"].append(_to_cpu_clone(node_input))
            trajectory_feature_bundle["edge_features"].append(_to_cpu_clone(edge_attr))
            trajectory_feature_bundle["targets"].append(_to_cpu_clone(target))

            if keep_gif_graphs:
                trajectory_graphs.append(Data(x_pos_t=_to_cpu_clone(x_t),
                                              y_sig_t=_to_cpu_clone(graph.y_sig_t),
                                              x_element_connectivity=_to_cpu_clone(graph.x_element_connectivity)))
            # update for next iter
            _x_tm1_prev = x_t.clone()

        del dataloader; del graphs_list_batch; del whole_trajectory_batch; del x; del v; del a; del rest_pos_x
        del all_node_force; del boundary_condition; del _x_tm1_prev

        return trajectory_feature_bundle, trajectory_graphs

    def _generate_ground_truth_gif(self, trajectory_graphs, traj_idx, traj_metadata):
        """
        Attach von Mises stress to each minimal graph and write one animated GIF under the EDA artifact folder.

        1. For each graph in the list, set graph.von_mises from the stress tensor.
        2. Call make_ground_truth_gif with fixed scene dimensions and trajectory metadata in the filename context.
        3. Create the output directory if needed.

        Args:
            trajectory_graphs: list[Data]
                Raw trajectory graphs for one trajectory.
            traj_idx: int
                Current trajectory index.
            traj_metadata: dict or None
                Trajectory parameters from trajectory_params.json, or None if unavailable.
            return: None
                Writes one GIF to disk.
        """
        gif_dir = self.artifact_dir / "ground_truth_gifs"
        os.makedirs(gif_dir, exist_ok=True)
        for graph in trajectory_graphs:
            graph.von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
        gif_path = gif_dir / f"trajectory_{traj_idx:02d}.gif"
        make_ground_truth_gif(trajectory_graphs, L=1.0, W=0.1, D=0.04, gif_path=gif_path, fps=4, verbose=False,
                              split_name="eda", traj_idx=traj_idx, traj_metadata=traj_metadata)



@hydra.main(version_base=None, config_path="../../conf/preprocess", config_name="eda_train")
def main(cfg):
    """
    Args:
        cfg: object
            Hydra-composed configuration.
    """
    print("\n" + "=" * 80 + "\n\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n" + "=" * 80 + "\n\n")

    eda_cfg = EdaYaml(cfg)
    base_cfg = PreprocessYaml(cfg)

    eda_split = eda_cfg.eda_split
    if eda_split not in ["train", "val", "test"]:
        raise ValueError(f"Unknown eda_split: {eda_split}")

    run_metadata = {run["name"]: run for run in eda_cfg.normalization_runs}
    save_dir = Path(f"eda_artifacts/{str(base_cfg.dataset_type)}/{eda_split}")
    train_artifact_dir = Path(f"eda_artifacts/{str(base_cfg.dataset_type)}/train")
    print(f"Preprocessing size={base_cfg.dataset_size} dataset")
    print(f"\tSave dir of eda = {save_dir}\n")

    graphs_dir = Path(base_cfg.raw_data_dir) / eda_split / "graphs"
    eda_obj_dict = {}
    selected_trajs = load_selected_trajectories(base_cfg.extra_info_dir)
    selected_limit = _selected_limit_from_json(selected_trajs, base_cfg.limit_timesteps)
    selected_indices = _selected_indices_for_split(selected_trajs, eda_split)
    if eda_split == "train":
        selected_indices = set()

    if selected_trajs:
        print(f"\tSelected trajectories for {eda_split} ({len(selected_indices)}): {sorted(selected_indices)}")
        print(f"\tSelected trajectory limit_timesteps: {selected_limit}")

    for run in eda_cfg.normalization_runs:
        variant_cfg = _create_variant_cfg(base_cfg, run["normalizer_method"], run["isotropic_normalizer"])
        print("\n" + "=" * 60 + f"\n{eda_split}, {run['label'].lower()}\n" + "=" * 60)
        preprocessor = EdaPreprocessor(variant_cfg, save_dir, run["name"], eda_cfg.gif_generation_mode,
                                       eda_cfg.gif_sample_percentage, eda_split, train_artifact_dir,
                                       selected_indices, selected_limit)
        preprocessor.preprocess_data(variant_cfg, graphs_dir, run["generate_gifs"])
        eda_obj_dict[run["name"]] = preprocessor.eda_obj
        del preprocessor
        gc.collect()

    if eda_cfg.generate_distributions:
        print("\n\n" + "=" * 60 + "Starting to plot all distributions..." + "=" * 60 + "\n")
        plot_all_distributions(eda_obj_dict, save_dir, eda_cfg.scalar_run_names, eda_cfg.vector_run_names,
                               run_metadata, eda_cfg.scalar_feature_names, eda_cfg.run_colors)
    else:
        print("\n\nDistribution generation is disabled, skipping.\n")
    del eda_obj_dict
    gc.collect()


if __name__ == "__main__":
    main()
