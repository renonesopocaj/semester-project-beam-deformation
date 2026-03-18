"""
misc.py
Contains miscellaneous helper functions.
"""

from pathlib import Path
from src.gnn.validation.plot_rollout import (build_rollout_summary, plot_prediction_target_scatter, TARGET_FIELDS,
                                             plot_edge_error_correlation)
import torch
import torch.nn.functional as F

def plot_validation_diagnostics_norollout(true_rollout, pred_rollout, output_dir, filename_prefix, split_name):
    """
    Plot FIXME: they are actually from rolllout
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
    print("Printing validation diagnostics norollout...")
    output_dir = Path(output_dir)
    rollout_summary = build_rollout_summary(true_rollout, pred_rollout)
    num_timesteps = len(true_rollout)
    for field_name, field_label in TARGET_FIELDS:
        scatter_path = output_dir / f"{filename_prefix}_{field_name}_prediction_vs_target.png"
        correlation_path = output_dir / f"{filename_prefix}_{field_name}_edge_error_correlation.png"
        field_summary = rollout_summary["fields"][field_name]
        plot_prediction_target_scatter(field_summary, scatter_path, split_name, num_timesteps)
        plot_edge_error_correlation(field_summary, correlation_path, split_name, num_timesteps)

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


    # def _collect_batch_metrics_norollout(self, pred, target, prefix):
    #     """
    #     Used in 1) training step 2) validation step 3) test step.
    #     Computes per-component and aggregate loss/MAE metrics for a single batch.
    #     - loss: overall loss
    #     - acceleration: loss & relative mae
    #     - stress: loss & relative mae
    #     - acc_x_loss: & relative mae
    #     - acc_y_loss: & relative mae
    #     - acc_z_loss: & relative mae
    #     - von_mises_loss: & relative mae
    #
    #     Args:
    #         pred: torch.Tensor
    #             Model prediction with shape `(num_nodes, 4)`.
    #         target: torch.Tensor
    #             Ground-truth tensor with shape `(num_nodes, 4)`.
    #         prefix: str
    #             Metric namespace prefix, for example `train`, `val`, or `test`.
    #     """
    #     metrics = {f"{prefix}/loss": self.loss_fn(pred, target),
    #                f"{prefix}/acc_all_loss": self.loss_fn(pred[:, 0:3].contiguous(), target[:, 0:3].contiguous()),
    #                f"{prefix}/stress_loss": self.loss_fn(pred[:, 3].contiguous(), target[:, 3].contiguous()),
    #                f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
    #                f"{prefix}/acc_all_rel_mae": compute_mean_relative_mae(
    #                    pred[:, 0:3].contiguous(), target[:, 0:3].contiguous()
    #                ),
    #                f"{prefix}/acc_x_rel_mae": compute_mean_relative_mae(
    #                    pred[:, 0].contiguous(), target[:, 0].contiguous()
    #                ),
    #                f"{prefix}/acc_y_rel_mae": compute_mean_relative_mae(
    #                    pred[:, 1].contiguous(), target[:, 1].contiguous()
    #                ),
    #                f"{prefix}/acc_z_rel_mae": compute_mean_relative_mae(
    #                    pred[:, 2].contiguous(), target[:, 2].contiguous()
    #                ),
    #                f"{prefix}/stress_rel_mae": compute_mean_relative_mae(
    #                    pred[:, 3].contiguous(), target[:, 3].contiguous()
    #                )}
    #     return metrics