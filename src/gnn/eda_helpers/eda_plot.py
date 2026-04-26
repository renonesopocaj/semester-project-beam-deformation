import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# FIXME: use a better device mechanism
DEVICE = "cpu"

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
    pbar = tqdm(reference_eda_obj.feature_names, desc="Plotting distribution")
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
