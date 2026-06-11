"""
attention_transolver_visualization.py

Builds attention-interpretability GIFs for a trained Transolver checkpoint.

Run:
    python -m src.gnn.scripts.attention_transolver_visualization
    python -m src.gnn.scripts.attention_transolver_visualization checkpoint.source=local \
        checkpoint.local_checkpoint_path=/abs/path/to/best.ckpt
"""

from pathlib import Path

import hydra
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.gnn.dataloaders.lazytimestepdataset import LazyTimeStepDataset
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import extract_surface_faces
from src.gnn.mesh_numerical_helpers.time_discretizer import create_time_discretizer
from src.gnn.misc_helpers.misc import to_numpy
from src.gnn.misc_helpers.yamlobject import PreprocessYaml, TrainYaml
from src.gnn.plots.plot_rollout import _finite_coordinate_bounds_from_stack, update_surface_collection
from src.gnn.preprocessing.build_features import make_scripted_feature_builders
from src.gnn.preprocessing.normalizer import create_normalizer
from src.gnn.preprocessing.stats import load_stats
from src.gnn.train import create_model
from src.gnn.training.online_edge_augmentation import (
    OnlineRandomEdgeAugmenter,
    SPLIT_ROLLOUT_TEST,
    SPLIT_ROLLOUT_VAL,
    fold_online_seed,
    num_graphs_in_pyg_batch,
)
from src.gnn.training.rollout import _build_next_graph
from src.gnn.training.target_strategy import create_target_strategy

matplotlib.use("Agg")
plt.ioff()

EPS = 1e-8
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path):
    """
    Read one YAML file.

    Args:
        path: str or Path
            YAML file to load.
    return: dict
        Parsed YAML payload.
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _resolve_dataset_dir(dataset_path):
    """
    Resolve a processed dataset directory that contains train/val/test/config.yaml.

    Args:
        dataset_path: str
            Either the size-resolved processed directory or its parent root.
    return: Path
        Dataset directory ready for loading.
    """
    base_path = Path(dataset_path).expanduser().resolve()
    if (base_path / "train").exists() and (base_path / "config.yaml").exists():
        return base_path
    medium_path = base_path / "medium"
    if (medium_path / "train").exists() and (medium_path / "config.yaml").exists():
        return medium_path
    raise FileNotFoundError(
        "Could not resolve a processed dataset directory from '%s'. "
        "Point dataset_path either to the size-resolved folder or to a root containing a medium/ child."
        % base_path
    )


def _find_nested_value(payload, key):
    """
    Recursively find one key in a nested mapping.

    Args:
        payload: dict
            Nested mapping.
        key: str
            Key to search for.
    return: object
        Found value, or None when absent.
    """
    if not isinstance(payload, dict):
        return None
    for current_key, current_value in payload.items():
        if current_key == key:
            return current_value
        nested_result = _find_nested_value(current_value, key)
        if nested_result is not None:
            return nested_result
    return None


def _hydrate_legacy_preprocess_dict(dataset_preprocess_dict):
    """
    Backfill feature-toggle keys that older processed datasets did not persist in config.yaml.

    Args:
        dataset_preprocess_dict: dict
            Parsed dataset config.yaml payload.
    return: dict
        Updated config payload with the missing feature keys filled from the named feature preset.
    """
    required_feature_keys = [
        "enable_velocity_node",
        "enable_displacement_node",
        "enable_force_node",
        "enable_rest_pos_node",
    ]
    missing_keys = [key for key in required_feature_keys if _find_nested_value(dataset_preprocess_dict, key) is None]
    if len(missing_keys) == 0:
        dataset_cfg = OmegaConf.create(dataset_preprocess_dict)
    else:
        feature_name = _find_nested_value(dataset_preprocess_dict, "name_features")
        if feature_name is None:
            raise KeyError(
                "Dataset config is missing keys %s and does not expose name_features, so the legacy feature preset "
                "cannot be recovered." % missing_keys
            )
        feature_preset_path = REPO_ROOT / "conf" / "preprocess" / "features" / ("%s.yaml" % feature_name)
        if not feature_preset_path.exists():
            raise FileNotFoundError(
                "Dataset config is missing keys %s and the feature preset '%s' was not found at %s."
                % (missing_keys, feature_name, feature_preset_path)
            )
        feature_preset_dict = _load_yaml(feature_preset_path)
        dataset_cfg = OmegaConf.create(dataset_preprocess_dict)
        for key in missing_keys:
            value = feature_preset_dict.get(key)
            if value is None:
                raise KeyError(
                    "Feature preset '%s' does not define required key '%s'." % (feature_name, key)
                )
            OmegaConf.update(dataset_cfg, key, value, merge=True)
    if _find_nested_value(dataset_preprocess_dict, "edge_aug_name") is None:
        OmegaConf.update(dataset_cfg, "edge_aug_name", "", merge=True)
    return OmegaConf.to_container(dataset_cfg, resolve=True)


def _resolve_local_training_files(local_checkpoint_path, local_train_config_path, local_preprocess_config_path):
    """
    Resolve local checkpoint/config files, inferring sibling YAMLs from saved_models layout when omitted.

    Args:
        local_checkpoint_path: str or None
        local_train_config_path: str or None
        local_preprocess_config_path: str or None
    return: tuple[Path, Path, Path | None]
        Checkpoint path, train.yaml path, optional preprocess.yaml path.
    """
    if local_checkpoint_path is None:
        raise ValueError("checkpoint.local_checkpoint_path is required when checkpoint.source=local.")
    checkpoint_path = Path(local_checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint_path)
    model_dir = checkpoint_path.parent.parent
    train_cfg_path = Path(local_train_config_path).expanduser().resolve() if local_train_config_path is not None else model_dir / "train.yaml"
    preprocess_cfg_path = (
        Path(local_preprocess_config_path).expanduser().resolve()
        if local_preprocess_config_path is not None else model_dir / "preprocess.yaml"
    )
    if not train_cfg_path.exists():
        raise FileNotFoundError("Train config not found: %s" % train_cfg_path)
    if not preprocess_cfg_path.exists():
        preprocess_cfg_path = None
    return checkpoint_path, train_cfg_path, preprocess_cfg_path


def _download_wandb_run_file(run, file_name, download_dir):
    """
    Download one run file through the W&B Public API.

    Args:
        run: wandb.apis.public.Run
            Resolved W&B run handle.
        file_name: str
            File path inside the run.
        download_dir: Path
            Root directory for downloads.
    return: Path | None
        Downloaded file path, or None if the run does not expose that file.
    """
    run_file = run.file(file_name)
    if run_file is None:
        return None
    downloaded = run_file.download(root=str(download_dir), replace=True)
    return Path(downloaded.name).resolve()


def _resolve_wandb_training_files(cfg_checkpoint):
    """
    Download checkpoint/config files from one W&B run, with optional local config fallbacks.

    Args:
        cfg_checkpoint: DictConfig
            Checkpoint config subtree.
    return: tuple[Path, Path, Path | None]
        Local paths to checkpoint, train.yaml, and optional preprocess.yaml.
    """
    import wandb

    if cfg_checkpoint.wandb_run_path is None:
        raise ValueError("checkpoint.wandb_run_path is required when checkpoint.source=wandb_run_file.")
    download_dir = Path(cfg_checkpoint.download_dir).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(str(cfg_checkpoint.wandb_run_path))

    checkpoint_path = _download_wandb_run_file(run, str(cfg_checkpoint.checkpoint_file), download_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Run '%s' does not expose file '%s'." % (cfg_checkpoint.wandb_run_path, cfg_checkpoint.checkpoint_file)
        )
    train_cfg_path = _download_wandb_run_file(run, str(cfg_checkpoint.train_config_file), download_dir)
    if train_cfg_path is None:
        if cfg_checkpoint.local_train_config_path is None:
            raise FileNotFoundError(
                "Run '%s' does not expose '%s', and checkpoint.local_train_config_path was not provided."
                % (cfg_checkpoint.wandb_run_path, cfg_checkpoint.train_config_file)
            )
        train_cfg_path = Path(cfg_checkpoint.local_train_config_path).expanduser().resolve()
    preprocess_cfg_path = _download_wandb_run_file(run, str(cfg_checkpoint.preprocess_config_file), download_dir)
    if preprocess_cfg_path is None and cfg_checkpoint.local_preprocess_config_path is not None:
        preprocess_cfg_path = Path(cfg_checkpoint.local_preprocess_config_path).expanduser().resolve()
    return checkpoint_path, train_cfg_path, preprocess_cfg_path


def _resolve_checkpoint_files(cfg_checkpoint):
    """
    Resolve checkpoint/train/preprocess files from either W&B or the local filesystem.

    Args:
        cfg_checkpoint: DictConfig
            Checkpoint config subtree.
    return: tuple[Path, Path, Path | None]
        Local paths to checkpoint, train.yaml, and optional preprocess.yaml.
    """
    source = str(cfg_checkpoint.source)
    if source == "local":
        return _resolve_local_training_files(
            cfg_checkpoint.local_checkpoint_path,
            cfg_checkpoint.local_train_config_path,
            cfg_checkpoint.local_preprocess_config_path,
        )
    if source == "wandb_run_file":
        return _resolve_wandb_training_files(cfg_checkpoint)
    raise ValueError("Unsupported checkpoint.source '%s'." % source)


def _compare_checkpoint_and_dataset_preprocess(checkpoint_preprocess_dict, dataset_preprocess_dict):
    """
    Fail fast when the dataset preprocessing contract differs from the checkpoint one.

    Args:
        checkpoint_preprocess_dict: dict | None
            Preprocess config saved next to the checkpoint, when available.
        dataset_preprocess_dict: dict
            Preprocess config stored in the dataset directory.
    return: None
        Raises ValueError on incompatible feature layouts.
    """
    if checkpoint_preprocess_dict is None:
        print("Checkpoint preprocess.yaml not available; assuming dataset config matches the trained feature layout.")
        return
    keys_to_match = [
        "target_mode",
        "node_channels",
        "edge_channels",
        "normalizer_method",
        "isotropic_normalizer",
        "time_discretization_method",
        "enable_past_history",
    ]
    mismatches = []
    for key in keys_to_match:
        if checkpoint_preprocess_dict.get(key) != dataset_preprocess_dict.get(key):
            mismatches.append((key, checkpoint_preprocess_dict.get(key), dataset_preprocess_dict.get(key)))
    if len(mismatches) > 0:
        lines = [
            "Checkpoint preprocess config does not match the requested dataset config.",
            "Mismatches:",
        ]
        for key, checkpoint_value, dataset_value in mismatches:
            lines.append("  %s: checkpoint=%r dataset=%r" % (key, checkpoint_value, dataset_value))
        raise ValueError("\n".join(lines))


def _normalize_model_state_dict(raw_state_dict):
    """
    Strip Lightning and torch.compile prefixes from a checkpoint state dict.

    Args:
        raw_state_dict: dict
            Raw checkpoint state mapping.
    return: dict
        Pure model state dict compatible with the naked torch.nn.Module.
    """
    normalized = {}
    for key, value in raw_state_dict.items():
        clean_key = str(key)
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model."):]
        while clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]
        normalized[clean_key] = value
    return normalized


def _load_model_weights(model, checkpoint_path, device):
    """
    Load model weights from a Lightning checkpoint.

    Args:
        model: torch.nn.Module
            Model instance to populate.
        checkpoint_path: Path
            Lightning checkpoint path.
        device: torch.device
            Load device.
    return: None
        Raises RuntimeError if the state dict is incompatible.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    normalized_state_dict = _normalize_model_state_dict(raw_state_dict)
    stats_keys = ["node_stats.mean", "node_stats.std", "edge_stats.mean", "edge_stats.std", "target_stats.mean", "target_stats.std"]
    normalized_state_dict = {k: v for k, v in normalized_state_dict.items() if k not in stats_keys}
    load_result = model.load_state_dict(normalized_state_dict, strict=True)
    if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
        raise RuntimeError(
            "Checkpoint load mismatch. Missing=%s Unexpected=%s"
            % (load_result.missing_keys, load_result.unexpected_keys)
        )


def _effective_node_attention_matrix(attention_state, head_index):
    """
    Recover an effective node-to-node interaction matrix from slice routing and slice attention.

    Rows are receiver nodes i, columns are source nodes j. The construction follows the actual
    Physics-Attention computation: source nodes route into slices, slice tokens attend to other
    slices, then receiver nodes deslice from the updated slice tokens.

    Args:
        attention_state: dict
            Cache produced by one PhysicsAttentionIrregularMesh forward pass.
        head_index: int or None
            Specific head to select; None averages all heads then row-normalizes.
    return: torch.Tensor
        Tensor of shape (N, N), approximately row-stochastic.
    """
    slice_weights = attention_state["slice_weights"]
    slice_norm = attention_state["slice_norm"]
    slice_attention = attention_state["slice_attention"]
    if int(slice_weights.shape[0]) != 1:
        raise ValueError("Attention visualization expects rollout batch_size=1, got %d." % int(slice_weights.shape[0]))
    sender_mass = slice_weights / (slice_norm[:, :, None, :] + EPS)
    matrix_per_head = torch.einsum("bhiq,bhqk,bhjk->bhij", slice_weights, slice_attention, sender_mass)[0]
    if head_index is None:
        matrix = matrix_per_head.mean(dim=0)
    else:
        matrix = matrix_per_head[int(head_index)]
    row_sum = matrix.sum(dim=1, keepdim=True)
    matrix = matrix / torch.clamp(row_sum, min=EPS)
    return matrix


def _all_heads_effective_node_attention_matrices(attention_state):
    """
    Compute per-head row-normalised effective node-to-node interaction matrices.

    Args:
        attention_state: dict
            Cache produced by one PhysicsAttentionIrregularMesh forward pass with keys
            slice_weights (1, H, N, G), slice_norm (1, H, G), slice_attention (1, H, G, G).
    return: np.ndarray
        Array of shape (H, N, N) with each head independently row-normalised.
    """
    slice_weights = attention_state["slice_weights"]
    slice_norm = attention_state["slice_norm"]
    slice_attention = attention_state["slice_attention"]
    if int(slice_weights.shape[0]) != 1:
        raise ValueError("Attention visualization expects rollout batch_size=1, got %d." % int(slice_weights.shape[0]))
    sender_mass = slice_weights / (slice_norm[:, :, None, :] + EPS)
    matrix_per_head = torch.einsum("bhiq,bhqk,bhjk->bhij", slice_weights, slice_attention, sender_mass)[0]  # H N N
    row_sums = matrix_per_head.sum(dim=-1, keepdim=True)
    matrix_per_head = matrix_per_head / torch.clamp(row_sums, min=EPS)
    return to_numpy(matrix_per_head)


def _num_model_blocks(model):
    """
    Return the number of Transolver blocks in a TransolverNet backbone.

    Args:
        model: TransolverNet
    return: int
    """
    return len(model.model.blocks)


def _num_model_heads(model):
    """
    Return the number of attention heads in the first Transolver block.

    Args:
        model: TransolverNet
    return: int
    """
    return model.model.blocks[0].Attn.heads


def _resolve_visualization_plan(block_cfg, head_cfg, num_blocks, num_heads):
    """
    Build the ordered list of (block_idx, head_idx_or_None, label) tuples to visualize.

    Six modes emerge from the 3x2 combination of block and head selectors:
      - block=int,  head=null  →  1 entry  (single block, heads averaged)
      - block="all", head=null  →  L entries (all blocks, each head-averaged)
      - block="all", head="all" →  L*H entries (all blocks, every head)
      - block=int,  head="all" →  H entries  (single block, every head)
      - block="avg", head=null  →  1 entry  (all blocks averaged, heads averaged)
      - block="avg", head="all" →  H entries (all blocks averaged, every head separately)

    Args:
        block_cfg: int or str
            Integer block index (negative indexing supported), the string "all" (one entry per block),
            or the string "avg" (average across all blocks).
        head_cfg: None or str
            None to average all heads; the string "all" to visualize each head separately.
        num_blocks: int
        num_heads: int
    return: list[tuple[int | str, int | None, str]]
        Each entry is (block_idx_or_"avg", head_idx_or_None, label).
    """
    if str(block_cfg) == "avg":
        block_indices = ["avg"]
    elif str(block_cfg) == "all":
        block_indices = list(range(num_blocks))
    else:
        bi = int(block_cfg)
        if bi < 0:
            bi = num_blocks + bi
        if bi < 0 or bi >= num_blocks:
            raise IndexError(
                "attention_block_index=%d is out of range for %d blocks." % (int(block_cfg), num_blocks)
            )
        block_indices = [bi]

    if head_cfg is not None and str(head_cfg) == "all":
        head_indices = list(range(num_heads))
    else:
        head_indices = [None]

    plan = []
    for block_idx in block_indices:
        for head_idx in head_indices:
            if block_idx == "avg":
                if head_idx is None:
                    label = "blockavg_headavg"
                else:
                    label = "blockavg_head%02d" % head_idx
            else:
                if head_idx is None:
                    label = "block%02d_headavg" % block_idx
                else:
                    label = "block%02d_head%02d" % (block_idx, head_idx)
            plan.append((block_idx, head_idx, label))
    return plan


def _compute_combo_matrices(per_step_states, block_idx, head_idx_or_none):
    """
    Extract the sequence of N x N attention matrices for one (block, head) combination.

    Args:
        per_step_states: list[list[dict]]
            Cached attention states: outer list indexed by timestep, inner by block index.
            Each dict contains CPU tensors slice_weights, slice_norm, slice_attention.
        block_idx: int or str
            Index into the inner block list, or "avg" to average across all blocks.
        head_idx_or_none: int or None
            Head to extract. None averages all heads then row-normalises.
    return: list[np.ndarray]
        One (N, N) row-normalised matrix per timestep.
    """
    matrices = []

    if str(block_idx) == "avg":
        for step_states in per_step_states:
            per_block_matrices = []
            for state in step_states:
                if head_idx_or_none is None:
                    matrix = _tensor_frame_to_numpy(_effective_node_attention_matrix(state, None))
                else:
                    matrix = _all_heads_effective_node_attention_matrices(state)[int(head_idx_or_none)]
                per_block_matrices.append(matrix)
            averaged_matrix = np.mean(per_block_matrices, axis=0)
            matrices.append(averaged_matrix)
    else:
        for step_states in per_step_states:
            state = step_states[block_idx]
            if head_idx_or_none is None:
                matrix = _tensor_frame_to_numpy(_effective_node_attention_matrix(state, None))
            else:
                matrix = _all_heads_effective_node_attention_matrices(state)[int(head_idx_or_none)]
            matrices.append(matrix)

    return matrices


def _tensor_frame_to_numpy(frame_tensor):
    """
    Convert a rollout tensor frame to a CPU numpy array.

    Args:
        frame_tensor: torch.Tensor
            Tensor to convert.
    return: np.ndarray
        Detached CPU array.
    """
    return to_numpy(frame_tensor).copy()


def _render_figure_to_rgb(fig):
    """
    Convert a Matplotlib figure to an RGB array.

    Args:
        fig: matplotlib.figure.Figure
            Figure already drawn.
    return: np.ndarray
        H x W x 3 uint8 frame.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _build_mesh_collection(ax, faces, coordinates, facecolor, alpha, poly_collection):
    """
    Draw or update one constant-color triangular surface.

    Args:
        ax: Axes3D
            Target axis.
        faces: np.ndarray
            Boundary triangles.
        coordinates: np.ndarray
            Node positions.
        facecolor: tuple
            RGBA color.
        alpha: float
            Surface transparency.
        poly_collection: Poly3DCollection or None
            Existing mesh artist.
    return: Poly3DCollection
        Updated mesh artist.
    """
    polygons = coordinates[faces]
    if poly_collection is None:
        poly_collection = Poly3DCollection(polygons, facecolors=facecolor, edgecolors="black", linewidths=0.15)
        poly_collection.set_alpha(alpha)
        ax.add_collection3d(poly_collection)
        return poly_collection
    poly_collection.set_verts(polygons)
    poly_collection.set_facecolor(facecolor)
    poly_collection.set_alpha(alpha)
    return poly_collection


def _configure_3d_axis(ax, coordinate_min, coordinate_max, coordinate_span, padding, title):
    """
    Apply consistent 3D axis limits and camera settings.

    Args:
        ax: Axes3D
        coordinate_min: np.ndarray
        coordinate_max: np.ndarray
        coordinate_span: np.ndarray
        padding: np.ndarray
        title: str
    return: None
    """
    ax.set_xlim(coordinate_min[0] - padding[0], coordinate_max[0] + padding[0])
    ax.set_ylim(coordinate_min[1] - padding[1], coordinate_max[1] + padding[1])
    ax.set_zlim(coordinate_min[2] - padding[2], coordinate_max[2] + padding[2])
    ax.set_box_aspect(tuple(coordinate_span.tolist()))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title)


def _make_selected_node_attention_gif(output_path, positions, faces, matrices, receiver_node_idx, fps, tqdm_desc):
    """
    Build one GIF where the ground-truth mesh is colored by source-node attention into one receiver node.

    Args:
        output_path: Path
            Output GIF path.
        positions: list[np.ndarray]
            Ground-truth node coordinates for each timestep.
        faces: np.ndarray
            Surface triangles.
        matrices: list[np.ndarray]
            Effective attention matrices, rows=receiver nodes, cols=source nodes.
        receiver_node_idx: int
            Node whose incoming attention row is visualized.
        fps: int
            GIF frame rate.
        tqdm_desc: str
            Description label shown in the frame-level progress bar.
    return: None
    """
    all_coordinates = np.concatenate(positions, axis=0)
    coordinate_min, coordinate_max, coordinate_span, padding = _finite_coordinate_bounds_from_stack(all_coordinates)
    node_scores = [matrix[receiver_node_idx] for matrix in matrices]
    vmax = max(float(np.max(scores)) for scores in node_scores)
    norm = colors.Normalize(vmin=0.0, vmax=max(vmax, EPS))
    cmap = matplotlib.colormaps["inferno"]
    frames = []

    for step_idx, (coords, scores) in tqdm(
        enumerate(zip(positions, node_scores)),
        desc=tqdm_desc,
        total=len(positions),
        leave=False,
    ):
        fig = plt.figure(figsize=(7.5, 6.5), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        surface = update_surface_collection(ax, faces, coords, scores, cmap, norm, None)
        ax.scatter(
            coords[receiver_node_idx, 0],
            coords[receiver_node_idx, 1],
            coords[receiver_node_idx, 2],
            c="cyan",
            s=36,
            edgecolors="black",
            linewidths=0.5,
            depthshade=False,
        )
        ax.text(
            coords[receiver_node_idx, 0],
            coords[receiver_node_idx, 1],
            coords[receiver_node_idx, 2],
            " i=%d" % receiver_node_idx,
            fontsize=9,
        )
        _configure_3d_axis(
            ax,
            coordinate_min,
            coordinate_max,
            coordinate_span,
            padding,
            "Incoming Attention on Ground-Truth Mesh\nreceiver i=%d | step %d / %d"
            % (receiver_node_idx, step_idx, len(positions) - 1),
        )
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.08, label="effective attention from source node j")
        fig.tight_layout()
        frames.append(_render_figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps)


def _make_attention_matrix_gif(output_path, positions, faces, matrices, fps, tqdm_desc):
    """
    Build one GIF with the node-to-node attention matrix on top and the numbered ground-truth mesh below.

    Args:
        output_path: Path
            Output GIF path.
        positions: list[np.ndarray]
            Ground-truth node coordinates for each timestep.
        faces: np.ndarray
            Surface triangles.
        matrices: list[np.ndarray]
            Effective attention matrices, rows=receiver nodes, cols=source nodes.
        fps: int
            GIF frame rate.
        tqdm_desc: str
            Description label shown in the frame-level progress bar.
    return: None
    """
    TOP_N_NODES = 50
    all_coordinates = np.concatenate(positions, axis=0)
    coordinate_min, coordinate_max, coordinate_span, padding = _finite_coordinate_bounds_from_stack(all_coordinates)
    vmax = max(float(np.max(matrix)) for matrix in matrices)
    norm = colors.Normalize(vmin=0.0, vmax=max(vmax, EPS))
    # Global normalization for top-node coloring: column sums = total attention each node receives as source.
    all_col_sums = [matrix.sum(axis=0) for matrix in matrices]
    node_cmap = matplotlib.colormaps["plasma"]
    global_col_vmax = max(float(np.max(cs)) for cs in all_col_sums)
    node_norm = colors.Normalize(vmin=0.0, vmax=max(global_col_vmax, EPS))
    frames = []
    cmap = matplotlib.colormaps["magma"]

    for step_idx, (coords, matrix, col_sums) in tqdm(
        enumerate(zip(positions, matrices, all_col_sums)),
        desc=tqdm_desc,
        total=len(positions),
        leave=False,
    ):
        fig = plt.figure(figsize=(14.0, 16.0), dpi=150)
        ax_matrix = fig.add_subplot(211)
        im = ax_matrix.imshow(matrix, cmap=cmap, norm=norm, origin="lower", aspect="auto", interpolation="nearest")
        ax_matrix.set_title(
            "Effective Node Attention Matrix\nrows = receiver node i, columns = source node j | step %d / %d"
            % (step_idx, len(positions) - 1),
            fontsize=12,
            fontweight="bold",
        )
        ax_matrix.set_xlabel("source node j", fontsize=11)
        ax_matrix.set_ylabel("receiver node i", fontsize=11)
        ax_matrix.tick_params(labelsize=9)
        fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.02, label="effective attention")

        ax_mesh = fig.add_subplot(212, projection="3d")
        _build_mesh_collection(
            ax_mesh,
            faces,
            coords,
            facecolor=(0.76, 0.78, 0.81, 0.12),
            alpha=0.12,
            poly_collection=None,
        )

        # All nodes as small grey background dots.
        ax_mesh.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="grey", s=4, depthshade=False, alpha=0.4, zorder=1)

        # Top-50 nodes by column sum colored by their attention weight.
        top_indices = np.argsort(col_sums)[-TOP_N_NODES:]
        top_scores = col_sums[top_indices]
        top_coords = coords[top_indices]
        top_face_colors = node_cmap(node_norm(top_scores))
        ax_mesh.scatter(
            top_coords[:, 0], top_coords[:, 1], top_coords[:, 2],
            c=top_face_colors, s=28, depthshade=False, zorder=2, edgecolors="none",
        )
        for i, node_idx in enumerate(top_indices):
            x_coord, y_coord, z_coord = coords[node_idx]
            ax_mesh.text(x_coord, y_coord, z_coord, " %d" % node_idx, fontsize=5.0, zorder=3)

        node_mappable = plt.cm.ScalarMappable(norm=node_norm, cmap=node_cmap)
        node_mappable.set_array([])
        fig.colorbar(node_mappable, ax=ax_mesh, fraction=0.025, pad=0.08, label="column-sum attention (source importance)")

        _configure_3d_axis(
            ax_mesh,
            coordinate_min,
            coordinate_max,
            coordinate_span,
            padding,
            "Top-%d source nodes by total attention received | step %d / %d" % (TOP_N_NODES, step_idx, len(positions) - 1),
        )
        fig.tight_layout()
        frames.append(_render_figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps)


def _run_attention_rollout(model, traj_loader, train_cfg, preprocess_cfg, node_stats, edge_stats, target_stats,
                           target_strategy, device, traj_idx, split_name):
    """
    Run an autoregressive rollout while caching Physics-Attention states for all blocks at every step.

    Args:
        model: TransolverNet
            Attention-capture-enabled model.
        traj_loader: DataLoader
            Single-trajectory rollout loader.
        train_cfg: TrainYaml
            Resolved training config.
        preprocess_cfg: PreprocessYaml
            Dataset preprocess config.
        node_stats: Stats
        edge_stats: Stats
        target_stats: Stats
        target_strategy: TargetStrategy
        device: torch.device
        traj_idx: int
            Trajectory index for deterministic online-edge seeding.
        split_name: str
            Dataset split name.
    return: tuple[list[np.ndarray], list[list[dict]], object]
        Ground-truth positions (T x (N, 3)),
        per_step_states (T x L dicts with CPU tensors slice_weights/slice_norm/slice_attention),
        and element connectivity from the first timestep.
    """
    node_stats.to(device)
    edge_stats.to(device)
    target_stats.to(device)
    model.eval()
    model.set_attention_capture(True)
    normalizer = create_normalizer(preprocess_cfg.normalizer_method, preprocess_cfg.isotropic_normalizer)
    time_discretizer = create_time_discretizer(preprocess_cfg.time_discretization_method)
    node_feature_builder, edge_feature_builder = make_scripted_feature_builders(preprocess_cfg)
    rollout_tag = SPLIT_ROLLOUT_TEST if split_name == "test" else SPLIT_ROLLOUT_VAL
    online_augmenter = None
    if float(train_cfg.random_online) > 0.0:
        online_augmenter = OnlineRandomEdgeAugmenter(edge_feature_builder, normalizer, edge_stats)

    per_step_states = []
    true_positions = []
    connectivity = None
    rest_pos = None
    input_graph = None

    with torch.no_grad():
        time_discretizer.reset()
        for step_idx, graph_ground_truth in tqdm(
            enumerate(traj_loader), desc="Rollout", total=len(traj_loader), leave=True
        ):
            graph_ground_truth = graph_ground_truth.to(device, non_blocking=True)
            if rest_pos is None:
                rest_pos = graph_ground_truth.rest_pos
            if input_graph is None:
                input_graph = graph_ground_truth

            if online_augmenter is not None:
                num_graphs = num_graphs_in_pyg_batch(input_graph)
                seeds = []
                for graph_idx in range(num_graphs):
                    seeds.append(fold_online_seed(train_cfg.seed, rollout_tag, traj_idx, step_idx, graph_idx))
                model_in = online_augmenter.apply(input_graph, float(train_cfg.random_online), seeds)
            else:
                model_in = input_graph

            pred = model(model_in.to(device))

            # Cache all-block attention states on CPU for later multi-combo extraction.
            attention_states = model.get_last_attention_states()
            per_step_states.append([
                {k: v.cpu() for k, v in state.items()}
                for state in attention_states
            ])

            pred_denorm = normalizer.denormalize(pred, target_stats)
            acceleration_pred = pred_denorm[:, 0:3]
            rollout_fields = target_strategy.rollout_fields_from_pred_denorm(pred_denorm)
            _ = rollout_fields["vm"]
            x_true = graph_ground_truth.x_next
            v_true = graph_ground_truth.v_next
            true_positions.append(_tensor_frame_to_numpy(x_true))
            if connectivity is None:
                connectivity = graph_ground_truth.x_element_connectivity

            x_t = input_graph.x_pos_t
            v_t = input_graph.x_vel_t
            boundary_condition = graph_ground_truth.x[:, 0]
            x_predict_tp1, v_predict_tp1 = time_discretizer.integrate_acceleration(acceleration_pred, v_t,
                                                                                    preprocess_cfg.dt, x_t)
            x_predict_tp1[boundary_condition == 1] = x_t[boundary_condition == 1]
            v_predict_tp1[boundary_condition == 1] = 0.0
            input_graph = _build_next_graph(
                ground_truth_graph=graph_ground_truth,
                x_predict_tp1=x_predict_tp1,
                v_predict_tp1=v_predict_tp1,
                x_t=x_t,
                rest_pos=rest_pos,
                node_stats=node_stats,
                edge_stats=edge_stats,
                normalizer=normalizer,
                node_feature_builder=node_feature_builder,
                edge_feature_builder=edge_feature_builder,
            )
    model.set_attention_capture(False)
    return true_positions, per_step_states, connectivity


def _load_model_and_dataset_base(cfg):
    """
    Load model and dataset, resolving files and validating configs.

    Args:
        cfg: DictConfig
            Script config.
    return: tuple[TransolverNet, LazyTimeStepDataset, torch.device, TrainYaml, PreprocessYaml, str, str]
        Model, dataset, device, train config, preprocess config, checkpoint_tag, split_name.
    """
    dataset_dir = _resolve_dataset_dir(cfg.dataset_path)
    checkpoint_path, train_cfg_path, checkpoint_preprocess_cfg_path = _resolve_checkpoint_files(cfg.checkpoint)
    train_cfg_dict = _load_yaml(train_cfg_path)
    dataset_preprocess_dict = _hydrate_legacy_preprocess_dict(_load_yaml(dataset_dir / "config.yaml"))
    checkpoint_preprocess_dict = None
    if checkpoint_preprocess_cfg_path is not None:
        checkpoint_preprocess_dict = _load_yaml(checkpoint_preprocess_cfg_path)
    _compare_checkpoint_and_dataset_preprocess(checkpoint_preprocess_dict, dataset_preprocess_dict)

    train_cfg = TrainYaml(train_cfg_dict)
    preprocess_cfg = PreprocessYaml(dataset_preprocess_dict)
    if str(train_cfg.model_name).lower() != "transolver":
        raise ValueError("This script only supports Transolver checkpoints, got model_name=%s." % train_cfg.model_name)
    train_target_mode = _find_nested_value(train_cfg_dict, "target_mode")
    if str(train_target_mode) != str(preprocess_cfg.target_mode):
        raise ValueError(
            "Train config target_mode=%s does not match dataset target_mode=%s."
            % (train_target_mode, preprocess_cfg.target_mode)
        )

    split_name = str(cfg.split)
    split_dir = dataset_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError("Split directory not found: %s" % split_dir)
    dataset = LazyTimeStepDataset(sample_dir=str(split_dir), device="cpu", cache_size=1)

    if str(cfg.device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(cfg.device))

    graph_template = dataset[dataset.trajectory_boundaries[0][0]]
    stress_out_dim = 6 if preprocess_cfg.target_mode == "full_stress" else 1
    model = create_model(train_cfg, preprocess_cfg, device, graph_template, stress_out_dim)
    _load_model_weights(model, checkpoint_path, device)
    model.to(device)

    checkpoint_tag = "%s_%s" % (checkpoint_path.parent.parent.name, checkpoint_path.stem)
    return model, dataset, device, train_cfg, preprocess_cfg, checkpoint_tag, split_name


def _prepare_trajectory(dataset, checkpoint_tag, split_name, output_root, traj_idx, max_timesteps):
    """
    Validate and prepare a single trajectory for rollout.

    Args:
        dataset: LazyTimeStepDataset
            Dataset instance.
        checkpoint_tag: str
            Tag derived from checkpoint path.
        split_name: str
            Dataset split name.
        output_root: str or Path
            Root output directory.
        traj_idx: int
            Trajectory index to validate and load.
        max_timesteps: int or None
            Optional cap on number of timesteps.
    return: tuple[DataLoader, Path]
        DataLoader for the trajectory and output directory.
    """
    if traj_idx < 0 or traj_idx >= dataset.num_trajectories:
        raise IndexError(
            "trajectory_idx=%d is out of range for split '%s' with %d trajectories."
            % (traj_idx, split_name, dataset.num_trajectories)
        )
    start, end = dataset.trajectory_boundaries[traj_idx]
    if max_timesteps is not None:
        end = min(end, start + max_timesteps)
    traj_loader = DataLoader(
        torch.utils.data.Subset(dataset, range(start, end)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    output_dir = Path(output_root).expanduser().resolve() / checkpoint_tag / split_name / ("traj_%03d" % traj_idx)
    output_dir.mkdir(parents=True, exist_ok=True)
    return traj_loader, output_dir


@hydra.main(version_base=None, config_path="../../../conf", config_name="attention_transolver_visualization")
def main(cfg):
    """
    Entry point for Transolver attention-visualization generation.

    Supports four visualization modes driven by attention_block_index and attention_head_index:
      - block=int,   head=null  → 1 matrix GIF + R node GIFs   (heads averaged, single block)
      - block="all", head=null  → L matrix GIFs + L*R node GIFs (heads averaged, all blocks)
      - block="all", head="all" → L*H matrix GIFs + L*H*R node GIFs
      - block=int,   head="all" → H matrix GIFs + H*R node GIFs (all heads, single block)

    Args:
        cfg: DictConfig
            Hydra-composed script configuration.
    return: None
        Writes GIFs under cfg.output_root.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))
    model, dataset, device, train_cfg, preprocess_cfg, checkpoint_tag, split_name = _load_model_and_dataset_base(cfg)
    dataset_dir = _resolve_dataset_dir(cfg.dataset_path)

    num_blocks = _num_model_blocks(model)
    num_heads = _num_model_heads(model)
    plan = _resolve_visualization_plan(cfg.attention_block_index, cfg.attention_head_index, num_blocks, num_heads)

    train_stats_dir = dataset_dir / "train"
    node_stats, edge_stats, target_stats = load_stats(train_stats_dir)
    target_strategy = create_target_strategy(preprocess_cfg.target_mode, train_cfg)

    for traj_idx in list(cfg.trajectory_idx):
        traj_loader, output_dir = _prepare_trajectory(
            dataset=dataset,
            checkpoint_tag=checkpoint_tag,
            split_name=split_name,
            output_root=cfg.output_root,
            traj_idx=traj_idx,
            max_timesteps=cfg.max_timesteps,
        )

        positions, per_step_states, connectivity = _run_attention_rollout(
            model=model,
            traj_loader=traj_loader,
            train_cfg=train_cfg,
            preprocess_cfg=preprocess_cfg,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            target_strategy=target_strategy,
            device=device,
            traj_idx=traj_idx,
            split_name=split_name,
        )
        if len(positions) == 0:
            raise RuntimeError("The selected rollout produced zero frames.")

        faces = extract_surface_faces(connectivity)
        num_nodes = int(positions[0].shape[0])
        for receiver_node_idx in cfg.reference_node_indices:
            if receiver_node_idx < 0 or receiver_node_idx >= num_nodes:
                raise IndexError(
                    "reference_node_indices contains %d, but the trajectory has %d nodes."
                    % (receiver_node_idx, num_nodes)
                )

        overall_att_dir = output_dir / "overall_att_matrix"
        overall_att_dir.mkdir(parents=True, exist_ok=True)

        for block_idx, head_idx, label in tqdm(plan, desc="Combos", leave=True):
            matrices = _compute_combo_matrices(per_step_states, block_idx, head_idx)

            matrix_gif_path = overall_att_dir / ("attention_matrix_%s.gif" % label)
            _make_attention_matrix_gif(
                output_path=matrix_gif_path,
                positions=positions,
                faces=faces,
                matrices=matrices,
                fps=int(cfg.fps),
                tqdm_desc="Matrix %s" % label,
            )
            print("Wrote %s" % matrix_gif_path)

            for receiver_node_idx in tqdm(cfg.reference_node_indices, desc="Ref nodes", leave=False):
                node_dir = output_dir / ("node_%03d" % int(receiver_node_idx))
                node_dir.mkdir(parents=True, exist_ok=True)
                node_gif_path = node_dir / (
                    "selected_node_%03d_%s_attention_mesh.gif" % (int(receiver_node_idx), label)
                )
                _make_selected_node_attention_gif(
                    output_path=node_gif_path,
                    positions=positions,
                    faces=faces,
                    matrices=matrices,
                    receiver_node_idx=int(receiver_node_idx),
                    fps=int(cfg.fps),
                    tqdm_desc="Node %d %s" % (receiver_node_idx, label),
                )
                print("Wrote %s" % node_gif_path)


if __name__ == "__main__":
    main()
