"""
eda.py
Main file to run in order to do Exploratory Data Analysis.
The del command are there to avoid out of memory errors.
This script does the following:
1. Preprocess the data and save it in a format that is easy to use for EDA.
2. Plot the distributions of the features.
3. Plot uniformly sampled trajectories (10%, i.e. for 1000 samples 100)
For train the normalized distributions are computed via train stats, for val and test also with train stats.
To avoid code repetition, the preprocessing reuses the samefeature builders used in preprocess.py.
All the possible normalization are done.
"""

import copy
import gc
import json
import math
import os
import shutil
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from src.gnn.misc_helpers.yamlobject import PreprocessYaml, EdaYaml
from src.gnn.preprocess import TrainPreprocessor
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


def _get_variant_names_for_feature(attr_name, SCALAR_RUN_NAMES, VECTOR_RUN_NAMES, SCALAR_FEATURE_NAMES):
    """
    Args:
        attr_name: str
            EDA feature name.
        return: list[str]
            The normalization runs that should be plotted for that feature.
    """
    if attr_name in SCALAR_FEATURE_NAMES:
        return SCALAR_RUN_NAMES
    return VECTOR_RUN_NAMES


def _get_display_label(run_name, attr_name, RUN_METADATA, SCALAR_FEATURE_NAMES):
    """
    Args:
        run_name: str
            Normalization run name.
        attr_name: str
            EDA feature name.
        return: str
            Human-readable plot label.
    """
    if attr_name in SCALAR_FEATURE_NAMES:
        scalar_labels = {"none": "No norm", "z_isotr": "Z norm", "robust_isotr": "Robust norm"}
        return scalar_labels[run_name]
    return RUN_METADATA[run_name]["label"]


def plot_all_distributions(eda_obj_dict, save_dir, scalar_run_names, vector_run_names, run_metadata,
                           scalar_feature_names, run_colors):
    """
    Args:
        eda_obj_dict: dict[str, EdaObj]
            One EdaObj per normalization run.
        save_dir: str or Path
            Directory where the plots will be saved.
        scalar_run_names: list[str]
        vector_run_names: list[str]
        run_metadata: dict[str, dict]
        scalar_feature_names: list[str]
        run_colors: dict[str, str]
    """
    os.makedirs(save_dir, exist_ok=True)
    reference_eda_obj = eda_obj_dict["none"]
    pbar = tqdm(reference_eda_obj.__dict__, desc="Plotting distribution")
    for key in pbar:
        if len(getattr(reference_eda_obj, key)) == 0:
            continue
        pbar.set_postfix({"key": key})
        plot_distribution(key, eda_obj_dict, save_dir, scalar_run_names, vector_run_names, run_metadata,
                          scalar_feature_names, run_colors)


def plot_distribution(attr_name, eda_obj_dict, save_dir, scalar_run_names, vector_run_names, run_metadata,
                      scalar_feature_names, run_colors):
    """
    Plots non-normalized distribution on the left y-axis and the selected
    normalized distributions on the right y-axis (shared scale).

    Args:
        attr_name: str
            Attribute name (used for titles/labels).
        eda_obj_dict: dict[str, EdaObj]
            Mapping run_name -> EdaObj.
        save_dir: str or Path
            Directory where the plots will be saved.
        scalar_run_names: list[str]
        vector_run_names: list[str]
        run_metadata: dict[str, dict]
        scalar_feature_names: list[str]
        run_colors: dict[str, str]
    """
    run_names = _get_variant_names_for_feature(attr_name, scalar_run_names, vector_run_names, scalar_feature_names)
    chunks_list = []
    labels = []
    colors = []
    for run_name in run_names:
        if run_name not in eda_obj_dict:
            continue
        values = getattr(eda_obj_dict[run_name], attr_name)
        if len(values) == 0:
            continue
        chunks = []
        for tensor in values:
            chunks.append(tensor.reshape(-1, 1))
        merged_tensor = torch.cat(chunks, dim=0).to(torch.float32)
        chunks_list.append(merged_tensor.reshape(-1).detach().to(DEVICE).numpy())
        labels.append(_get_display_label(run_name, attr_name, run_metadata, scalar_feature_names))
        colors.append(run_colors[run_name])
        del chunks
        del values
        del merged_tensor

    if len(chunks_list) == 0:
        return

    raw_min = float(chunks_list[0].min())
    raw_max = float(chunks_list[0].max())

    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = ax_left.twinx()

    ax_left.hist(chunks_list[0], bins=200, density=True, alpha=0.3, color=colors[0], label=labels[0])
    ax_left.set_ylabel("Density (non-normalized)")

    for data, color, label in zip(chunks_list[1:], colors[1:], labels[1:]):
        ax_right.hist(data, bins=200, density=True, alpha=0.5, color=color,
                      label=label, histtype="step", linewidth=1.5)
    ax_right.set_ylabel("Density (normalized)")

    ax_left.set_xlabel(attr_name)
    fig.suptitle(f"{attr_name} distribution")
    ax_left.grid(True, alpha=0.3)
    ax_left.text(0.02, 0.98, f"Raw range: [{raw_min:.6g}, {raw_max:.6g}]",
                 transform=ax_left.transAxes, va="top", ha="left",
                 bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    fig.legend(handles_left + handles_right, labels_left + labels_right,
               loc="upper right", bbox_to_anchor=(0.98, 0.95))

    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"eda_{attr_name}.png", dpi=150)
    plt.close(fig)


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
                 eda_split, train_artifact_dir):
        """
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
        """
        super().__init__(cfg)
        self.artifact_dir = Path(artifact_dir)
        self.run_name = run_name
        self.eda_obj = EdaObj()
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
                          boundary_right_triangles):
        pass

    def _save_eda_stats(self):
        """
        Persist node/edge/target stats to disk so that val/test runs can reuse
        train-computed statistics.
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
        Load train-computed node/edge/target stats so that val/test splits
        are normalized with the same statistics used during training EDA.
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
        Process the dataset trajectory by trajectory to avoid accumulating the full
        rich EDA representation in memory.

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
        filenames = sorted(Path(graphs_dir).glob("*.pt"))
        if cfg.normalizer_method.lower() == "none":
            self._collect_raw_eda(cfg, filenames, generate_gifs)
            return

        cache_dir = self.artifact_dir / ".eda_cache" / self.run_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache_files = self._cache_trajectories(cfg, filenames, cache_dir)
            split_idx_edge = 9
            if self.eda_split == "train":
                self._compute_stats_from_cache(cache_files, split_idx_edge)
                self._save_eda_stats()
            else:
                self._load_eda_stats()
            self._collect_normalized_eda(cache_files)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _collect_raw_eda(self, cfg, filenames, generate_gifs):
        """
        Args:
            cfg: object
                Preprocessing config.
            filenames: list[Path]
                Trajectory files to process.
            generate_gifs: bool
                Whether to generate trajectory GIFs during the pass.
            return: None
                Fills self.eda_obj lazily.
        """
        gif_indices = _build_gif_trajectory_indices(len(filenames), generate_gifs, self.gif_sample_percentage,
                                                    self.gif_generation_mode)
        pbar = tqdm(enumerate(filenames), total=len(filenames), desc=f"EDA {self.run_name}")
        for traj_idx, filename in pbar:
            keep_gif_graphs = traj_idx in gif_indices
            payload, trajectory_graphs = self._build_trajectory_payload(cfg, filename, keep_gif_graphs)
            self._collect_payload(payload, normalize=False)
            if trajectory_graphs is not None:
                traj_metadata = self.traj_params.get(filename.name, None)
                self._generate_ground_truth_gif(trajectory_graphs, traj_idx, traj_metadata)
            del payload
            del trajectory_graphs
            gc.collect()

    def _cache_trajectories(self, cfg, filenames, cache_dir):
        """
        Args:
            cfg: object
                Preprocessing config.
            filenames: list[Path]
                Trajectory files to process.
            cache_dir: Path
                Directory used to store compact raw trajectory bundles.
            return: list[Path]
                Cache file paths.
        """
        cache_files = []
        pbar = tqdm(enumerate(filenames), total=len(filenames), desc=f"Caching {self.run_name}")
        for traj_idx, filename in pbar:
            payload, _ = self._build_trajectory_payload(cfg, filename, False)
            cache_path = cache_dir / f"traj_{traj_idx:05d}.pt"
            torch.save(payload, cache_path)
            cache_files.append(cache_path)
            del payload
            gc.collect()
        return cache_files

    def _load_family_from_cache(self, cache_files, payload_key, start_column):
        """
        Args:
            cache_files: list[Path]
                Cache files created by _cache_trajectories().
            payload_key: str
                One of node_inputs, edge_features, or targets.
            start_column: int or None
                Optional column offset used to skip the BC feature in node inputs.
            return: list[torch.Tensor]
                Family tensors loaded from cache.
        """
        tensor_list = []
        for cache_path in cache_files:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            for tensor in payload[payload_key]:
                if start_column is None:
                    tensor_list.append(tensor)
                else:
                    tensor_list.append(tensor[:, start_column:])
            del payload
        return tensor_list

    def _compute_stats_from_cache(self, cache_files, split_idx_edge):
        """
        Args:
            cache_files: list[Path]
                Cache files created by _cache_trajectories().
            split_idx_edge: int
                Number of edge feature columns grouped as xyz triplets.
            return: None
                Fills self.node_stats, self.edge_stats, self.target_stats.
        """
        print("\nComputing EDA stats from cached trajectories...\n")
        node_tensors = self._load_family_from_cache(cache_files, "node_inputs", 1)
        self.node_stats = self.normalizer.compute_stats(node_tensors, node_tensors[0].shape[1])
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
        Args:
            cache_files: list[Path]
                Cache files created by _cache_trajectories().
            return: None
                Fills self.eda_obj from normalized cached trajectories.
        """
        pbar = tqdm(cache_files, total=len(cache_files), desc=f"Collecting {self.run_name}")
        for cache_path in pbar:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._collect_payload(payload, normalize=True)
            del payload
            gc.collect()

    def _normalize_node_input(self, node_input):
        """
        Args:
            node_input: torch.Tensor
                Raw node input tensor `[bc, valued_node_features]`.
            return: torch.Tensor
                Node input tensor with only the valued part normalized.
        """
        normalized_valued = self.normalizer.normalize(node_input[:, 1:], self.node_stats)
        return torch.concat([node_input[:, 0:1], normalized_valued], dim=1)

    def _collect_payload(self, payload, normalize):
        """
        Args:
            payload: dict
                Compact trajectory bundle containing node_inputs, edge_features, and targets.
            normalize: bool
                Whether to apply the current normalizer before appending to EdaObj.
            return: None
                Appends all timestep features in the payload to self.eda_obj.
        """
        for node_input, edge_features, target in zip(payload["node_inputs"], payload["edge_features"],
                                                     payload["targets"]):
            if normalize:
                node_input = self._normalize_node_input(node_input)
                edge_features = self.normalizer.normalize(edge_features, self.edge_stats)
                target = self.normalizer.normalize(target, self.target_stats)
            self.eda_obj.append_timestep(node_input, edge_features, target)

    def _build_trajectory_payload(self, cfg, filename, keep_gif_graphs):
        """
        Args:
            cfg: object
                Preprocessing config.
            filename: Path
                Raw trajectory file.
            keep_gif_graphs: bool
                True iff the returned payload should include minimal GIF graph objects.
            return: (dict, list or None)
                Compact payload and optional GIF graph list for the current trajectory.
        """
        dataloader = torch.load(filename, weights_only=False)
        graphs_list_batch = []
        for data in dataloader:
            graph = data.to(cfg.device)
            graphs_list_batch.append(graph)

        whole_trajectory_batch = Batch.from_data_list(graphs_list_batch)
        x = whole_trajectory_batch.x.reshape(len(graphs_list_batch), -1, 3)
        rest_pos_x = x[0].clone()
        all_node_force = whole_trajectory_batch.x_node_force.reshape(len(graphs_list_batch), -1, 3)
        boundary_condition = graphs_list_batch[0].x_bc

        payload = {"node_inputs": [], "edge_features": [], "targets": []}
        trajectory_graphs = [] if keep_gif_graphs else None

        for t, graph in enumerate(graphs_list_batch):
            if t > cfg.limit_timesteps - 2:
                break
            node_force = all_node_force[t]
            x_t = x[t]
            v_t = v[t]
            a_t = a[t]
            valued_node_attr = build_node_features(x_t, v_t, rest_pos_x, node_force, cfg.enable_node_norms,
                                                   cfg.enable_stretch_node)
            node_input = torch.concat([boundary_condition, valued_node_attr], dim=1)
            edge_attr = build_edge_features(x_t, rest_pos_x, graph.edge_index, v_t, node_force, cfg.velocity_edge,
                                            cfg.enable_force_edge, cfg.enable_stretch_edge, cfg.enable_rotation_angle)
            von_mises_log = torch.log(compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + EPS)
            if cfg.mises_log:
                von_mises = von_mises_log
            else:
                von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
            target = torch.concat([a_t, von_mises], dim=1)

            payload["node_inputs"].append(_to_cpu_clone(node_input))
            payload["edge_features"].append(_to_cpu_clone(edge_attr))
            payload["targets"].append(_to_cpu_clone(target))

            if keep_gif_graphs:
                trajectory_graphs.append(Data(x_pos_t=_to_cpu_clone(x_t),
                                              y_sig_t=_to_cpu_clone(graph.y_sig_t),
                                              x_element_connectivity=_to_cpu_clone(graph.x_element_connectivity)))

        del dataloader; del graphs_list_batch; del whole_trajectory_batch; del x; del v; del a; del rest_pos_x
        del all_node_force; del boundary_condition

        return payload, trajectory_graphs

    def _generate_ground_truth_gif(self, trajectory_graphs, traj_idx, traj_metadata):
        """
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


class EdaObj:
    acceleration_x: list
    acceleration_y: list
    acceleration_z: list
    von_mises: list
    velocity_x: list
    velocity_y: list
    velocity_z: list
    position_x: list
    position_y: list
    position_z: list
    edge_relative_pos_x: list
    edge_relative_pos_y: list
    edge_relative_pos_z: list
    edge_relative_pos_norm: list
    edge_rest_relative_pos_x: list
    edge_rest_relative_pos_y: list
    edge_rest_relative_pos_z: list
    edge_rest_relative_pos_norm: list
    force_x: list
    force_y: list
    force_z: list
    edge_displacement_pos_x: list
    edge_displacement_pos_y: list
    edge_displacement_pos_z: list
    edge_displacement_norm: list
    rest_pos_x: list
    rest_pos_y: list
    rest_pos_z: list
    displacement_pos_x: list
    displacement_pos_y: list
    displacement_pos_z: list

    def __init__(self):
        (self.acceleration_x, self.acceleration_y, self.acceleration_z, self.von_mises,
         self.velocity_x, self.velocity_y, self.velocity_z,
         self.position_x, self.position_y, self.position_z,
         self.displacement_pos_x, self.displacement_pos_y, self.displacement_pos_z,
         self.force_x, self.force_y, self.force_z,
         self.rest_pos_x, self.rest_pos_y, self.rest_pos_z,
         self.edge_relative_pos_x, self.edge_relative_pos_y, self.edge_relative_pos_z, self.edge_relative_pos_norm,
         self.edge_rest_relative_pos_x, self.edge_rest_relative_pos_y, self.edge_rest_relative_pos_z,
         self.edge_rest_relative_pos_norm,
         self.edge_displacement_pos_x, self.edge_displacement_pos_y, self.edge_displacement_pos_z,
         self.edge_displacement_norm) = \
            ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [])

    def append_timestep(self, node_features, edge_features, target):
        """
        Args:
            node_features: torch.Tensor
                Node input tensor `[bc, valued_node_features]`.
            edge_features: torch.Tensor
                Edge feature tensor.
            target: torch.Tensor
                Target tensor `[acc_x, acc_y, acc_z, von_mises]`.
            return: None
                Appends the current timestep to the internal EDA lists.
        """
        self.acceleration_x.append(target[:, 0])
        self.acceleration_y.append(target[:, 1])
        self.acceleration_z.append(target[:, 2])
        self.von_mises.append(target[:, 3])

        self.position_x.append(node_features[:, 1])
        self.position_y.append(node_features[:, 2])
        self.position_z.append(node_features[:, 3])
        self.velocity_x.append(node_features[:, 4])
        self.velocity_y.append(node_features[:, 5])
        self.velocity_z.append(node_features[:, 6])
        self.rest_pos_x.append(node_features[:, 7])
        self.rest_pos_y.append(node_features[:, 8])
        self.rest_pos_z.append(node_features[:, 9])
        self.force_x.append(node_features[:, 10])
        self.force_y.append(node_features[:, 11])
        self.force_z.append(node_features[:, 12])

        self.displacement_pos_x.append(node_features[:, 13])
        self.displacement_pos_y.append(node_features[:, 14])
        self.displacement_pos_z.append(node_features[:, 15])

        self.edge_relative_pos_x.append(edge_features[:, 0])
        self.edge_relative_pos_y.append(edge_features[:, 1])
        self.edge_relative_pos_z.append(edge_features[:, 2])
        self.edge_rest_relative_pos_x.append(edge_features[:, 3])
        self.edge_rest_relative_pos_y.append(edge_features[:, 4])
        self.edge_rest_relative_pos_z.append(edge_features[:, 5])

        self.edge_displacement_pos_x.append(edge_features[:, 6])
        self.edge_displacement_pos_y.append(edge_features[:, 7])
        self.edge_displacement_pos_z.append(edge_features[:, 8])
        self.edge_relative_pos_norm.append(edge_features[:, 9])
        self.edge_rest_relative_pos_norm.append(edge_features[:, 10])
        self.edge_displacement_norm.append(edge_features[:, 11])


@hydra.main(version_base=None, config_path="../../conf/preprocess", config_name="eda_test")
def main(cfg):
    """
    Args:
        cfg: object
            Hydra-composed configuration.
    """
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

    for run in eda_cfg.normalization_runs:
        variant_cfg = _create_variant_cfg(base_cfg, run["normalizer_method"], run["isotropic_normalizer"])
        print("\n" + "=" * 60 + f"\n{eda_split}, {run['label'].lower()}\n" + "=" * 60)
        preprocessor = EdaPreprocessor(variant_cfg, save_dir, run["name"], eda_cfg.gif_generation_mode,
                                       eda_cfg.gif_sample_percentage, eda_split, train_artifact_dir)
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
