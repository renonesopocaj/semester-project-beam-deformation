from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from src.gnn.helpers.plots.mesh import compute_coordinate_bounds, extract_surface_faces, to_numpy, update_surface_collection


TARGET_FIELDS = [
    ("acc_x", "Acceleration X"),
    ("acc_y", "Acceleration Y"),
    ("acc_z", "Acceleration Z"),
    ("von_mises", "von Mises"),
]


def _extract_scalar(graph, field_name):
    """
    Args:
        graph: Data
            Rollout graph containing nodewise prediction or target values.
        field_name: str
            One of `acc_x`, `acc_y`, `acc_z`, `von_mises`.
        return: np.ndarray
            Flattened scalar values for all nodes.
    """
    if field_name == "von_mises":
        return to_numpy(graph.von_mises).reshape(-1)

    component_index = {"acc_x": 0, "acc_y": 1, "acc_z": 2}[field_name]
    return to_numpy(graph.acceleration)[:, component_index].reshape(-1)


def _compute_error(true_values, pred_values, error_mode):
    """
    Args:
        true_values: np.ndarray
            Target scalar values.
        pred_values: np.ndarray
            Predicted scalar values.
        error_mode: str
            One of `absolute`, `relative`, `signed`.
        return: np.ndarray
            Nodewise error values for plotting.
    """
    signed_error = pred_values - true_values
    if error_mode == "absolute":
        return np.abs(signed_error)
    if error_mode == "relative":
        denominator = np.maximum(np.abs(true_values), 1e-12)
        return np.abs(signed_error) / denominator
    if error_mode == "signed":
        return signed_error
    raise ValueError(f"Unsupported error mode: {error_mode}")


def _compute_limits(all_values, error_mode):
    """
    Args:
        all_values: np.ndarray
            Values aggregated across the full rollout.
        error_mode: str
            One of `absolute`, `relative`, `signed`.
        return: tuple
            `(vmin, vmax)` for a stable color scale.
    """
    if error_mode == "signed":
        max_abs = float(np.max(np.abs(all_values)))
        if max_abs == 0.0:
            max_abs = 1e-12
        return -max_abs, max_abs

    vmax = float(np.max(all_values))
    if vmax == 0.0:
        vmax = 1e-12
    return 0.0, vmax


def _draw_epoch_curve(ax, train_epochs, train_values, val_epochs, val_values, title, ylabel):
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
    ax.plot(train_epochs, train_values, label="Train", linewidth=2.0)
    if len(val_values) > 0:
        ax.plot(val_epochs, val_values, label="Validation", linewidth=2.0, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _save_epoch_curve(train_epochs, train_values, val_epochs, val_values, title, ylabel, output_path):
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
    _draw_epoch_curve(ax, train_epochs, train_values, val_epochs, val_values, title, ylabel)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _build_rollout_summary(true_rollout, pred_rollout):
    """
    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        return: dict
            Cached rollout data shared by all validation plots.
    """
    coordinates_per_step = []
    for true_graph, pred_graph in zip(true_rollout, pred_rollout):
        coordinates_per_step.append(to_numpy(true_graph.x_pos_t))

    coordinate_min, coordinate_max, coordinate_span, padding = compute_coordinate_bounds(coordinates_per_step)
    summary = {
        "coordinates_per_step": coordinates_per_step,
        "faces": extract_surface_faces(true_rollout[0].x_element_connectivity),
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "coordinate_span": coordinate_span,
        "padding": padding,
        "fields": {},
    }

    for field_name, field_label in TARGET_FIELDS:
        target_per_step = []
        prediction_per_step = []
        signed_errors_per_step = []
        absolute_errors_per_step = []
        relative_errors_per_step = []
        edge_error_i = []
        edge_error_j = []

        for true_graph, pred_graph in zip(true_rollout, pred_rollout):
            target_values = _extract_scalar(true_graph, field_name)
            prediction_values = _extract_scalar(pred_graph, field_name)
            signed_errors = _compute_error(target_values, prediction_values, "signed")
            absolute_errors = np.abs(signed_errors)
            relative_errors = _compute_error(target_values, prediction_values, "relative")
            edge_index = to_numpy(true_graph.edge_index).astype(np.int64)

            target_per_step.append(target_values)
            prediction_per_step.append(prediction_values)
            signed_errors_per_step.append(signed_errors)
            absolute_errors_per_step.append(absolute_errors)
            relative_errors_per_step.append(relative_errors)
            edge_error_i.append(signed_errors[edge_index[0]])
            edge_error_j.append(signed_errors[edge_index[1]])

        errors_by_mode = {
            "absolute": absolute_errors_per_step,
            "relative": relative_errors_per_step,
            "signed": signed_errors_per_step,
        }
        summary["fields"][field_name] = {
            "label": field_label,
            "target_values": np.concatenate(target_per_step),
            "prediction_values": np.concatenate(prediction_per_step),
            "edge_error_i": np.concatenate(edge_error_i),
            "edge_error_j": np.concatenate(edge_error_j),
            "errors_by_mode": errors_by_mode,
            "limits_by_mode": {
                error_mode: _compute_limits(np.concatenate(error_values), error_mode)
                for error_mode, error_values in errors_by_mode.items()
            },
        }

    return summary


def _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path):
    """
    Args:
        rollout_summary: dict
            Cached rollout data shared by all validation plots.
        field_name: str
            Scalar field identifier.
        error_mode: str
            One of `absolute`, `relative`, `signed`.
        output_path: Path
            Destination GIF path.
        return: None
            Saves the animated error mesh to disk.
    """
    field_summary = rollout_summary["fields"][field_name]
    field_label = field_summary["label"]
    coordinates_per_step = rollout_summary["coordinates_per_step"]
    error_per_step = field_summary["errors_by_mode"][error_mode]
    faces = rollout_summary["faces"]
    coordinate_min = rollout_summary["coordinate_min"]
    coordinate_max = rollout_summary["coordinate_max"]
    coordinate_span = rollout_summary["coordinate_span"]
    padding = rollout_summary["padding"]
    vmin, vmax = field_summary["limits_by_mode"][error_mode]
    cmap = plt.cm.coolwarm if error_mode == "signed" else plt.cm.inferno
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.12)
    colorbar.set_label(f"{field_label} {error_mode} error")
    ax.set_xlim(coordinate_min[0] - padding[0], coordinate_max[0] + padding[0])
    ax.set_ylim(coordinate_min[1] - padding[1], coordinate_max[1] + padding[1])
    ax.set_zlim(coordinate_min[2] - padding[2], coordinate_max[2] + padding[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(tuple(coordinate_span + padding))
    ax.view_init(elev=30, azim=45)

    poly_collection = None
    frames = []
    for timestep, (coordinates, error_values) in enumerate(zip(coordinates_per_step, error_per_step)):
        poly_collection = update_surface_collection(ax, faces, coordinates, error_values, cmap, norm, poly_collection)
        ax.set_title(f"{field_label} {error_mode.title()} Error, step {timestep}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=4, loop=0)


def _plot_prediction_target_scatter(field_summary, output_path):
    """
    Args:
        field_summary: dict
            Cached values for one scalar field.
        output_path: Path
            Destination image path.
        return: None
            Saves the scatter plot to disk.
    """
    target_values = field_summary["target_values"]
    prediction_values = field_summary["prediction_values"]
    min_value = min(target_values.min(), prediction_values.min())
    max_value = max(target_values.max(), prediction_values.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(target_values, prediction_values, s=10, alpha=0.15)
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1.5)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{field_summary['label']} Prediction vs Target")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_edge_error_correlation(field_summary, output_path):
    """
    Args:
        field_summary: dict
            Cached values for one scalar field.
        output_path: Path
            Destination image path.
        return: None
            Saves the edge error correlation plot to disk.
    """
    error_i = field_summary["edge_error_i"]
    error_j = field_summary["edge_error_j"]
    if len(error_i) == 0 or np.std(error_i) == 0.0 or np.std(error_j) == 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(error_i, error_j)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    hexbin = ax.hexbin(error_i, error_j, gridsize=60, cmap="viridis", mincnt=1)
    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Edge count")
    ax.set_xlabel("error_i")
    ax.set_ylabel("error_j")
    ax.set_title(f"{field_summary['label']} Edge Error Correlation (corr={correlation:.4f})")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_validation_diagnostics(true_rollout, pred_rollout, output_dir, epoch):
    """
    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Directory where diagnostic artifacts will be saved.
        epoch: int
            Validation epoch identifier.
        return: None
            Saves all requested validation plots.
    """
    output_dir = Path(output_dir)
    rollout_summary = _build_rollout_summary(true_rollout, pred_rollout)
    for field_name, field_label in TARGET_FIELDS:
        for error_mode in ["absolute", "relative", "signed"]:
            output_path = output_dir / f"Epoch{epoch}_{field_name}_{error_mode}_error.gif"
            _render_error_mesh_gif(rollout_summary, field_name, error_mode, output_path)

        scatter_path = output_dir / f"Epoch{epoch}_{field_name}_prediction_vs_target.png"
        correlation_path = output_dir / f"Epoch{epoch}_{field_name}_edge_error_correlation.png"
        field_summary = rollout_summary["fields"][field_name]
        _plot_prediction_target_scatter(field_summary, scatter_path)
        _plot_edge_error_correlation(field_summary, correlation_path)


def plot_training_history(trainer):
    """
    Args:
        trainer: Trainer
            Trainer instance holding the epoch histories.
        return: None
            Saves end-of-training history plots.
    """
    print("\nTraining complete!")
    print(f"Models saved to: {trainer.model_dir}")

    model_dir = Path(trainer.model_dir)
    train_epochs = np.arange(len(trainer.train_epoch_loss_history))
    val_epochs = np.asarray(trainer.val_epochs_history)

    _save_epoch_curve(train_epochs, np.asarray(trainer.train_epoch_loss_history), val_epochs,
                      np.asarray(trainer.val_epoch_loss_history), "Loss vs Epoch", "Loss",
                      model_dir / "loss_vs_epoch.png")
    _save_epoch_curve(train_epochs, np.asarray(trainer.train_epoch_mae_history), val_epochs,
                      np.asarray(trainer.val_epoch_mae_history), "MAE vs Epoch", "MAE",
                      model_dir / "mae_vs_epoch.png")

    summary_figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_epoch_curve(axes[0], train_epochs, np.asarray(trainer.train_epoch_loss_history), val_epochs,
                      np.asarray(trainer.val_epoch_loss_history), "Loss vs Epoch", "Loss")
    _draw_epoch_curve(axes[1], train_epochs, np.asarray(trainer.train_epoch_mae_history), val_epochs,
                      np.asarray(trainer.val_epoch_mae_history), "MAE vs Epoch", "MAE")
    summary_figure.tight_layout()
    summary_figure.savefig(model_dir / "training_history.png")
    plt.close(summary_figure)

    print(f"Total training steps: {len(trainer.train_history)}")
    print(f"Final training loss: {trainer.train_history[-1]:.6f}")
    if hasattr(trainer, "rollout_all_step_error"):
        print(f"Final rollout error: {trainer.rollout_all_step_error:.6f}")