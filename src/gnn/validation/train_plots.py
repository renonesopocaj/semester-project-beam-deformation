from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
        return: None
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
        return: None
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

    ax_left.set_title("Loss & Rollout Error vs Epoch")
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
        return: None
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
        return: None
    """
    ax.plot(epochs, lr_values, linewidth=2.0, color="tab:purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate vs Epoch")
    ax.grid(True, alpha=0.3)


def plot_training_history(trainer):
    """
    Args:
        trainer: Trainer
            Trainer instance holding the epoch histories.
        return: None
            Saves end-of-training history plots.
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
        return: None
            Draws a single epoch curve on the axis.
    """
    ax.plot(train_epochs, train_values, label=train_label, linewidth=2.0)
    if len(val_values) > 0:
        ax.plot(val_epochs, val_values, label=val_label, linewidth=2.0, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

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
        return: None
            Saves the figure to disk.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_epoch_curve(ax, train_epochs, train_values, val_epochs, val_values, title, ylabel, train_label, val_label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)