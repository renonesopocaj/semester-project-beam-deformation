"""
eda.py

Main file to run in order to do Exploratory Data Analysis.
- The del command are there to avoid out of memory errors.
This script does the following:
1. Preprocess the data and save it in a format that is easy to use for EDA.
2. Plot the distributions of the features.
3. Plot uniformly sampled trajectories (10%, i.e. for 1000 samples 100)
- For train the normalized distributions are computed via train stats, for val and test also with train stats.
- All the possible normalization are done.
"""

import copy
import gc
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.gnn.eda_helpers.eda_features_helpers import build_scalar_feature_names
from src.gnn.eda_helpers.eda_plot import plot_all_distributions
from src.gnn.eda_helpers.eda_preprocessor import EdaPreprocessor
from src.gnn.misc_helpers.yamlobject import PreprocessYaml, EdaYaml
from src.gnn.preprocess import (load_selected_trajectories, selected_indices_for_split, selected_limit_from_json)

DEVICE = "cpu"


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




@hydra.main(version_base=None, config_path="../../conf/preprocess", config_name="eda_train_easiest_rotated")
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
    selected_limit = selected_limit_from_json(selected_trajs, base_cfg.limit_timesteps)
    selected_indices = selected_indices_for_split(selected_trajs, eda_split)
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
        # Generate gifs is inside here
        preprocessor.preprocess_data(variant_cfg, graphs_dir, run["generate_gifs"])
        eda_obj_dict[run["name"]] = preprocessor.eda_obj
        del preprocessor
        gc.collect()

    if eda_cfg.generate_distributions:
        print("\n\n" + "=" * 60 + "Starting to plot all distributions..." + "=" * 60 + "\n")
        plot_all_distributions(eda_obj_dict, save_dir, eda_cfg.scalar_run_names, eda_cfg.vector_run_names,
                               run_metadata, build_scalar_feature_names(base_cfg), eda_cfg.run_colors)
    else:
        print("\n\nDistribution generation is disabled, skipping.\n")
    del eda_obj_dict
    gc.collect()


if __name__ == "__main__":
    main()
