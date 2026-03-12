from pathlib import Path
from src.gnn.validation.plot_rollout import (_build_rollout_summary, _plot_prediction_target_scatter, TARGET_FIELDS,
                                             _plot_edge_error_correlation)
import torch
import torch.nn.functional as F

def plot_validation_diagnostics_norollout(true_rollout, pred_rollout, output_dir, filename_prefix, split_name):
    """
    Plot
    1. scatter plots: prediction vs target, where every p[oint is one node
    2. edge error correlation:

    Args:
        true_rollout: list
            Ground-truth rollout graphs.
        pred_rollout: list
            Predicted rollout graphs.
        output_dir: Path
            Directory where diagnostic artifacts will be saved.
        filename_prefix: str
            Prefix used for saved artifact names.
        return: None
            Saves all requested validation plots.
    """
    output_dir = Path(output_dir)
    rollout_summary = _build_rollout_summary(true_rollout, pred_rollout)
    num_timesteps = len(true_rollout)
    for field_name, field_label in TARGET_FIELDS:
        scatter_path = output_dir / f"{filename_prefix}_{field_name}_prediction_vs_target.png"
        correlation_path = output_dir / f"{filename_prefix}_{field_name}_edge_error_correlation.png"
        field_summary = rollout_summary["fields"][field_name]
        _plot_prediction_target_scatter(field_summary, scatter_path, split_name, num_timesteps)
        _plot_edge_error_correlation(field_summary, correlation_path, split_name, num_timesteps)

def compute_mean_relative_mae(pred, target):
    return torch.mean(compute_relative_mae(pred, target))

def compute_relative_mae(pred, target):
    return torch.abs(pred - target) / torch.clamp(torch.abs(target), min=1e-12)

def loss_factory(loss_type):
    """
    Maps a loss type string to the corresponding PyTorch loss function.

    Args:
        loss_type: str
            Name of the loss function.
        return: callable
            PyTorch loss function.
    """
    if loss_type == "mse":
        return F.mse_loss
    if loss_type == "mae":
        return F.l1_loss
    if loss_type == "huber":
        return F.huber_loss
    raise NotImplementedError("Unsupported loss function")