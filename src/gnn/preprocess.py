"""
preprocess.py

Preprocess simulation data into graph format. Note one file = one trajectory. Main access point.
"""


import shutil
from typing import Any

import torch
import json
import hydra
from omegaconf import OmegaConf, open_dict
from src.gnn.misc_helpers.yamlobject import PreprocessYaml
from src.gnn.preprocessing.build_features import (probe_feature_layout)
from src.gnn.preprocessing.preprocessor import TrainPreprocessor, ValPreprocessor
from pathlib import Path


def load_selected_trajectories(extra_info_dir):
    """
    Load selected_trajectories.json from the extra_info directory.

    Args:
        extra_info_dir: str or Path or None
            path to the extra_info directory
    return: dict
        parsed selected_trajectories.json content, or empty dict if not found
    """
    if extra_info_dir is None:
        return {}
    path = Path(extra_info_dir) / "selected_trajectories.json"
    if not path.exists():
        raise FileNotFoundError(f"Selected trajectories file not found at {path}. "
                                "Add selected_trajectories.json next to the raw dataset extra_info "
                                "(same directory as preprocess extra_info_dir / raw extra_info).")
    with open(path) as f:
        return json.load(f)


def create_dir_name(cfg_obj):
    isotropic_str = '_isotr' if cfg_obj.isotropic_normalizer else ''
    miseslog_str = '_log' if cfg_obj.mises_log else ''
    normalizer_str = 'z' if cfg_obj.normalizer_method == 'z' else 'rob'
    enable_absolute_pos_str = '_abs' if cfg_obj.enable_absolute_pos else '_noabs'
    enable_past_pos_displ_str = '_xtm1' if cfg_obj.enable_past_pos_displ else '_noxtm1'
    enable_node_norms_str = '_normnode' if cfg_obj.enable_node_norms else '_nonormnode'
    enable_edge_velocity_str = '_edgevel' if cfg_obj.enable_edge_velocity else '_noedgevel'
    enable_force_edge_str = '_edgeforce' if cfg_obj.enable_force_edge else '_noedgeforce'
    enable_stretch_edge_str = '_stretch' if cfg_obj.enable_stretch_edge else '_nostretch'
    enable_rotation_angle_str = '_rot' if cfg_obj.enable_rotation_angle else '_norot'
    time_discr_str = "_centr" if cfg_obj.time_discretization_method == "second_order_central" else "_newmark"
    processed_dir_name = (f"{normalizer_str}{isotropic_str}{time_discr_str}{miseslog_str}{enable_absolute_pos_str}"
                          f"{enable_past_pos_displ_str}{enable_node_norms_str}{enable_edge_velocity_str}"
                          f"{enable_force_edge_str}{enable_stretch_edge_str}{enable_rotation_angle_str}")
    return processed_dir_name

def selected_indices_for_split(selected_trajs, split):
    """
    Return the set of trajectory indices that have longer rollouts for the given split.

    Args:
        selected_trajs: dict
            content of selected_trajectories.json
        split: str
            one of "train", "val", "test"
    return: set of int
        trajectory indices with longer rollout
    """
    if split == "val":
        return set(selected_trajs.get("selected_validation_trajectories", []))
    elif split == "test":
        return set(selected_trajs.get("selected_test_trajectories", []))
    return set()


def selected_limit_from_json(selected_trajs, default_limit):
    """
    Return the selected timestep limit from the JSON, or the default if absent.

    Args:
        selected_trajs: dict
            content of selected_trajectories.json
        default_limit: int
            fallback limit_timesteps
    return: int
        timestep limit for selected trajectories
    """
    timestep_limit = int(selected_trajs.get("selected_num_steps", selected_trajs["num_steps"]))
    return timestep_limit

def _copy_trajectory_params_json(raw_graphs_dir, processed_split_dir):
    """
    Copy trajectory_params.json from the raw split directory (parent of graphs/) into the processed split directory.

    Args:
        raw_graphs_dir: str or Path
            Path to raw split graphs folder, e.g. .../test/graphs.
        processed_split_dir: str or Path
            Processed split output directory, e.g. .../large/test.
    return: None
        Copies file when source exists; otherwise no-op.
    """
    src = Path(raw_graphs_dir).parent / "trajectory_params.json"
    if not src.exists():
        return
    dst = Path(processed_split_dir) / "trajectory_params.json"
    shutil.copy2(src, dst)
    print(f"\tCopied trajectory_params.json -> {dst}")


def create_preprocessor(split, cfg, save_dir_train, selected_indices, selected_limit):
    """
    Args:
        split: str
            one of "train", "val", "test"
        cfg: PreprocessYaml
            preprocessing config
        save_dir_train: Path or None
            path to the train preprocessed dir (for loading stats in val/test)
        selected_indices: set of int
            trajectory indices that have longer rollouts for this split
        selected_limit: int
            timestep limit for selected trajectories
    return: Preprocessor
        preprocessor instance for the given split
    """
    if split == "train":
        return TrainPreprocessor(cfg, selected_indices, selected_limit)
    elif split == "val":
        return ValPreprocessor(cfg, save_dir_train, selected_indices, selected_limit)
    elif split == "test":
        return ValPreprocessor(cfg, save_dir_train, selected_indices, selected_limit)
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
    print("\n" + "=" * 80 + "\n\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n" + "=" * 80 + "\n\n")
    cfg_obj = PreprocessYaml(cfg)
    print(f"Preprocessing {cfg_obj.dataset_size} dataset\n" + "=" * 60)
    print(f"\tSave dir of config.yaml = {cfg_obj.save_dir}")
    save_dir = Path(cfg_obj.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # OmegaConf.save(cfg, save_dir / "config.yaml")
    # =============================================== NAME ============================
    processed_dir_name = create_dir_name(cfg_obj)
    processed_dataset_dir = Path(cfg_obj.save_dir) / processed_dir_name / cfg_obj.dataset_size
    print(f"\tProcessed data dir {processed_dataset_dir}")
    processed_dataset_dir.mkdir(parents=True, exist_ok=True)
    # ===================================================================

    # =============== SPLITS ================
    # Where to save results
    save_dir_train = processed_dataset_dir / "train"
    save_dir_test = processed_dataset_dir / "test"
    save_dir_val = processed_dataset_dir / "val"
    # Raw paths
    graphs_dir_train = Path(cfg_obj.raw_data_dir) / "train" / "graphs"
    graphs_dir_test = Path(cfg_obj.raw_data_dir) / "test" / "graphs"
    graphs_dir_val = Path(cfg_obj.raw_data_dir) / "val" / "graphs"
    # ===============================

    _layout_device = torch.device(cfg_obj.device)
    _node_ch, _edge_ch, _, _ = probe_feature_layout(cfg_obj, _layout_device)
    print(f"\tDerived from build_features: node_channels={_node_ch}, edge_channels={_edge_ch}")
    extra_info_dest = processed_dataset_dir / "extra_info"
    with open_dict(cfg):
        cfg.node_channels = _node_ch
        cfg.edge_channels = _edge_ch
        cfg.train_dir = save_dir_train.as_posix()
        cfg.test_dir = save_dir_test.as_posix()
        cfg.val_dir = save_dir_val.as_posix()
        cfg.extra_info_dir = extra_info_dest.as_posix()
    OmegaConf.save(cfg, processed_dataset_dir / "config.yaml")

    # Copy extra_info JSONs into the processed dataset
    if cfg_obj.extra_info_dir is not None:
        raw_extra = Path(cfg_obj.extra_info_dir)
        extra_info_dest.mkdir(parents=True, exist_ok=True)
        for fname in ("selected_trajectories.json", "test_ood_features.json"):
            src = raw_extra / fname
            if src.exists():
                shutil.copy2(src, extra_info_dest / fname)
                print(f"\tCopied {fname} -> {extra_info_dest / fname}")

    # Load selected trajectory info for longer rollouts
    selected_trajs = load_selected_trajectories(cfg_obj.extra_info_dir)
    selected_limit = selected_limit_from_json(selected_trajs, cfg_obj.limit_timesteps)
    selected_val = selected_indices_for_split(selected_trajs, "val")
    selected_test = selected_indices_for_split(selected_trajs, "test")
    selected_train = set()
    if selected_trajs:
        print(f"\tSelected validation trajectories ({len(selected_val)}): {sorted(selected_val)}")
        print(f"\tSelected test trajectories ({len(selected_test)}): {sorted(selected_test)}")
        print(f"\tSelected trajectory limit_timesteps: {selected_limit}")

    # Create preprocessors object
    preprocessor_train = create_preprocessor("train", cfg_obj, None, selected_train, selected_limit)
    preprocessor_val = create_preprocessor("val", cfg_obj, save_dir_train, selected_val, selected_limit)
    preprocessor_test = create_preprocessor("test", cfg_obj, save_dir_train, selected_test, selected_limit)
    print("\n" + "=" * 60 + "\nTrain preprocessing")
    preprocessor_train.preprocess_data(cfg_obj, graphs_dir_train, save_dir_train)
    print("\n" + "=" * 60 + "\nPreprocessing complete!"+ "\n" + "=" * 60)
    print("\n" + "=" * 60 + "\nTest preprocessing"+ "\n" + "=" * 60)
    preprocessor_test.preprocess_data(cfg_obj, graphs_dir_test, save_dir_test)
    _copy_trajectory_params_json(graphs_dir_test, save_dir_test)
    print("\n" + "=" * 60 + "\nPreprocessing complete!"+ "\n" + "=" * 60)
    print("\n" + "=" * 60 + "\nVal preprocessing"+ "\n" + "=" * 60)
    preprocessor_val.preprocess_data(cfg_obj, graphs_dir_val, save_dir_val)
    _copy_trajectory_params_json(graphs_dir_val, save_dir_val)
    print("\n" + "=" * 60 + "\nPreprocessing complete!" + "\n" + "=" * 60)


if __name__ == "__main__":
    main()
