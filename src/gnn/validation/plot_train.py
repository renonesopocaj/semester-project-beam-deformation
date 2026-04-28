"""
plot_train.py
Contains functions for plotting training history specific stuff.
"""

import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def _save_dual_axis_loss_rollout(train_epochs, train_loss, val_epochs, val_loss, rollout_error, output_path):
    """
    Args:
        train_epochs: np.ndarray
            Epoch indices for training values.
        train_loss: np.ndarray
            Training loss values.
        val_epochs: np.ndarray
            Epoch indices for validation values.
        val_loss: np.ndarray
            Validation loss values.
        rollout_error: np.ndarray
            Validation rollout error values (same length as val_epochs).
        output_path: Path
            Destination image path.
    """
    fig, ax_left = plt.subplots(figsize=(10, 6))
    _draw_dual_axis_loss_rollout(ax_left, train_epochs, train_loss, val_epochs, val_loss, rollout_error)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _draw_dual_axis_loss_rollout(ax_left, train_epochs, train_loss, val_epochs, val_loss, rollout_error):
    """
    Args:
        ax_left: matplotlib.axes.Axes
            Primary axis (loss scale).
        train_epochs: np.ndarray
            Epoch indices for training values.
        train_loss: np.ndarray
            Training loss values.
        val_epochs: np.ndarray
            Epoch indices for validation values.
        val_loss: np.ndarray
            Validation loss values.
        rollout_error: np.ndarray
            Validation rollout error values.
    """
    line_train, = ax_left.plot(train_epochs, train_loss, label="Train Loss", linewidth=2.0)
    line_val = None
    if len(val_loss) > 0:
        line_val, = ax_left.plot(val_epochs, val_loss, label="Val Loss", linewidth=2.0, marker="o")
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Loss")
    ax_left.grid(True, alpha=0.3)

    lines = [line_train]
    if line_val is not None:
        lines.append(line_val)

    if len(rollout_error) > 0:
        ax_right = ax_left.twinx()
        line_rollout, = ax_right.plot(val_epochs, rollout_error, label="Val Rollout Error",
                                      linewidth=2.0, marker="s", color="tab:green", linestyle="--")
        ax_right.set_ylabel("Rollout Error")
        lines.append(line_rollout)

    ax_left.set_title("Loss & Rollout Error vs Epoch\n[training]")
    ax_left.legend(handles=lines, loc="best")


def _save_lr_curve(epochs, lr_values, output_path):
    """
    Args:
        epochs: np.ndarray
            Epoch indices.
        lr_values: np.ndarray
            Learning rate per epoch.
        output_path: Path
            Destination image path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_lr_curve(ax, epochs, lr_values)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _draw_lr_curve(ax, epochs, lr_values):
    """
    Args:
        ax: matplotlib.axes.Axes
            Target axis.
        epochs: np.ndarray
            Epoch indices.
        lr_values: np.ndarray
            Learning rate per epoch.
    """
    ax.plot(epochs, lr_values, linewidth=2.0, color="tab:purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate vs Epoch\n[training]")
    ax.grid(True, alpha=0.3)


def plot_training_history(trainer):
    """
    Args:
        trainer: Trainer
            Trainer instance holding the epoch histories.
    """

    model_dir = Path(trainer.model_dir)
    train_epochs = np.arange(len(trainer.train_epoch_loss_history))
    val_epochs = np.asarray(trainer.val_epochs_history)
    rollout_error = np.asarray(trainer.val_rollout_error_history)

    _save_dual_axis_loss_rollout(train_epochs, np.asarray(trainer.train_epoch_loss_history),
                                 val_epochs, np.asarray(trainer.val_epoch_loss_history),
                                 rollout_error, model_dir / "loss_rollout_vs_epoch.png")

    _save_epoch_curve(train_epochs, np.asarray(trainer.train_epoch_relative_mae_history), val_epochs,
                      np.asarray(trainer.val_epoch_relative_mae_history),
                      "Relative MAE vs Epoch", "Relative MAE",
                      model_dir / "relative_mae_vs_epoch.png")

    has_lr = len(trainer.lr_history) > 0
    if has_lr:
        _save_lr_curve(train_epochs, np.asarray(trainer.lr_history), model_dir / "lr_vs_epoch.png")

    summary_figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_dual_axis_loss_rollout(axes[0], train_epochs, np.asarray(trainer.train_epoch_loss_history),
                                 val_epochs, np.asarray(trainer.val_epoch_loss_history), rollout_error)
    _draw_epoch_curve(axes[1], train_epochs, np.asarray(trainer.train_epoch_relative_mae_history), val_epochs,
                      np.asarray(trainer.val_epoch_relative_mae_history),
                      "Relative MAE vs Epoch", "Relative MAE")
    if has_lr:
        _draw_lr_curve(axes[2], train_epochs, np.asarray(trainer.lr_history))
    else:
        axes[2].axis("off")
    summary_figure.tight_layout()
    summary_figure.savefig(model_dir / "training_history.png")
    plt.close(summary_figure)

    print(f"Total training steps: {len(trainer.train_history)}")
    print(f"Final training loss: {trainer.train_history[-1]:.6f}")
    if hasattr(trainer, "rollout_all_step_error"):
        print(f"Final rollout error: {trainer.rollout_all_step_error:.6f}")

def _draw_epoch_curve(ax, train_epochs, train_values, val_epochs, val_values, title, ylabel, train_label="Train",
                      val_label="Validation"):
    """
    Args:
        ax: matplotlib.axes.Axes
            Target axis.
        train_epochs: np.ndarray
            Epoch indices for training values.
        train_values: np.ndarray
            Training metric values.
        val_epochs: np.ndarray
            Epoch indices for validation values.
        val_values: np.ndarray
            Validation metric values.
        title: str
            Plot title.
        ylabel: str
            Plot y-axis label.
    """
    ax.plot(train_epochs, train_values, label=train_label, linewidth=2.0)
    if len(val_values) > 0:
        ax.plot(val_epochs, val_values, label=val_label, linewidth=2.0, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n[training]")
    ax.grid(True, alpha=0.3)
    ax.legend()


_MAX_GLOBAL_SAMPLES = 2000000


def _finite_parameter_flat_numpy(param):
    """
    Args:
        param: torch.nn.Parameter
            One model parameter tensor.
    return: np.ndarray
        1D float64 array of finite values (CPU), possibly empty.
    """
    arr = param.detach().float().cpu().numpy().ravel()
    if arr.size == 0:
        return arr.astype(np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.array([], dtype=np.float64)
    return arr[mask].astype(np.float64)


def _subsample_uniform_1d(values, max_count, seed):
    """
    Args:
        values: np.ndarray
            1D array.
        max_count: int
            Maximum number of samples to return.
        seed: int
            RNG seed for reproducibility when subsampling is needed.
    return: np.ndarray
        Either values unchanged or max_count values chosen without replacement when len(values) > max_count.
    """
    n = int(values.size)
    if n <= max_count:
        return values
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_count, replace=False)
    return values[idx]


def _collect_global_and_prefix_samples(model):
    """
    Builds a pooled sample of all parameter values (capped) and per–top-level-module concatenations.

    Args:
        model: torch.nn.Module
            Network whose named_parameters are read.
    return: (np.ndarray, dict)
        Global subsampled 1D values, and mapping top_level_prefix -> 1D concatenated finite values per prefix.
    """
    prefix_chunks = defaultdict(list)
    concat_parts = []
    for name, param in model.named_parameters():
        flat = _finite_parameter_flat_numpy(param)
        if flat.size == 0:
            continue
        concat_parts.append(flat)
        top = name.split(".", 1)[0]
        prefix_chunks[top].append(flat)
    if len(concat_parts) == 0:
        return np.array([], dtype=np.float64), {}
    pooled = np.concatenate(concat_parts)
    pooled = _subsample_uniform_1d(pooled, _MAX_GLOBAL_SAMPLES, seed=0)
    per_prefix = {}
    for top, parts in prefix_chunks.items():
        merged = np.concatenate(parts)
        seed = zlib.adler32(top.encode("utf-8")) & 0x7FFFFFFF
        per_prefix[top] = _subsample_uniform_1d(merged, _MAX_GLOBAL_SAMPLES, seed)
    return pooled, per_prefix


def plot_final_parameter_distributions(model, output_path):
    """
    Saves a figure with one histogram over all learned parameters (subsampled if huge) and one histogram
    per top-level submodule prefix (first segment of each parameter name).

    Args:
        model: torch.nn.Module
            Trained model to visualize.
        output_path: Path
            PNG path; parent directories are created if missing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_vals, per_prefix = _collect_global_and_prefix_samples(model)
    prefix_names = sorted(per_prefix.keys())
    n_extra = len(prefix_names)
    n_panels = 1 + n_extra
    if n_panels == 1 and global_vals.size == 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No parameters to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle("Learned parameter distributions [after training]", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    ncols = 3
    nrows = int(np.ceil(n_panels / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for j in range(len(axes_flat)):
        if j >= n_panels:
            axes_flat[j].set_axis_off()
    ax0 = axes_flat[0]
    if global_vals.size > 0:
        ax0.hist(global_vals, bins=80, density=True, color="steelblue", alpha=0.85)
    ax0.set_title("All parameters (pooled)")
    ax0.set_xlabel("Value")
    ax0.set_ylabel("Density")
    ax0.grid(True, alpha=0.3)
    for i, pname in enumerate(prefix_names):
        ax = axes_flat[1 + i]
        vals = per_prefix[pname]
        if vals.size > 0:
            ax.hist(vals, bins=60, density=True, color="coral", alpha=0.85)
        ax.set_title(pname[:48] + ("…" if len(pname) > 48 else ""))
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Learned parameter distributions [after training]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)


def _save_epoch_curve(train_epochs, train_values, val_epochs, val_values, title, ylabel, output_path,
                      train_label="Train", val_label="Validation"):
    """
    Args:
        train_epochs: np.ndarray
            Epoch indices for training values.
        train_values: np.ndarray
            Training metric values.
        val_epochs: np.ndarray
            Epoch indices for validation values.
        val_values: np.ndarray
            Validation metric values.
        title: str
            Plot title.
        ylabel: str
            Plot y-axis label.
        output_path: Path
            Destination image path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_epoch_curve(ax, train_epochs, train_values, val_epochs, val_values, title, ylabel, train_label, val_label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)