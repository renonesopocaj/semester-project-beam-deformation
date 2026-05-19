"""
wandb_loggers.py

Contains helper functions for logging to W&B.
"""

from pathlib import Path
import wandb

def wandb_final_eval_metric_prefix(eval_split_name):
    """
    W&B key prefix for final evaluation scalars (trainer.test on val or test split).

    Args:
        eval_split_name: str
            name of the split
    return: str
        final_validation, final_test, or eval_split_name if unknown.
    """
    if eval_split_name == "validation":
        return "final_validation"
    if eval_split_name == "test":
        return "final_test"
    return eval_split_name

def log_all_wandb_media(overall_plot_paths, per_traj_in_depth_paths, eval_split_name, experiment,
                        has_wandb_logger, epoch_number):
    """
    Single function that logs all media to W&B.
    1. Logs overall/summary plots first,
    2. then per-trajectory in-depth plots for selected trajectories.

    Args:
        overall_plot_paths: dict
            Nested dict {"images": {name: Path}, "videos": {name: Path}} from overall_plot.
        per_traj_in_depth_paths: dict[int, dict]
            Mapping of traj_idx to nested dicts {"images": {...}, "videos": {...}} from in_depth.
        has_wandb_logger: bool
            checks if the wandb logger is enabled.
        eval_split_name: str
            name of the split
        experiment: ????
            run object
        epoch_number: int
            Epoch value included in the payload for the custom W&B x-axis.
    """
    # if not self._has_wandb_logger():
    if not has_wandb_logger:
        return
    # experiment = logger.experiment
    # Naming
    if eval_split_name == "validation":
        overall_root, traj_root = "final_validation", "final_validation_traj"
    elif eval_split_name == "test":
        overall_root, traj_root = "final_test", "final_test_traj"
    else:
        overall_root, traj_root = eval_split_name, f"{eval_split_name}_traj"
    # 1. Log overall/summary plots (run_summary is excluded: logged separately as train/run_summary)
    wandb_logs = {}
    for name, path in overall_plot_paths.get("images", {}).items():
        if name == "run_summary":
            continue
        if Path(path).exists():
            wandb_logs[f"{overall_root}/overall/{name}"] = wandb.Image(str(path))
    for name, path in overall_plot_paths.get("videos", {}).items():
        if Path(path).exists():
            wandb_logs[f"{overall_root}/overall/{name}"] = wandb.Video(str(path), format="gif")
    # 2. Log per-trajectory in-depth plots (one W&B folder per trajectory index)
    for traj_idx, traj_paths in per_traj_in_depth_paths.items():
        traj_prefix = f"{traj_root}/{traj_idx}"
        for name, path in traj_paths.get("images", {}).items():
            if Path(path).exists():
                wandb_logs[f"{traj_prefix}/{name}"] = wandb.Image(str(path))
        for name, path in traj_paths.get("videos", {}).items():
            if Path(path).exists():
                wandb_logs[f"{traj_prefix}/{name}"] = wandb.Video(str(path), format="gif")
    if len(wandb_logs) == 0:
        return
    wandb_logs["epoch"] = epoch_number
    experiment.log(wandb_logs, commit=True)

def define_wandb_metrics(has_wandb_logger, is_global_zero, experiment, residual_list):
    """
    Define all W&B metrics with epoch as the x-axis and appropriate summaries.
    Should be called once at the start of training, only on rank 0.

    Args:
        has_wandb_logger: bool
        is_global_zero: bool
        experiment: wandb run object
        residual_list: set or list of str
            Active residual names (e.g. {"curvature", "volume"}). Only metrics for
            enabled residuals are registered so the W&B dashboard stays clean.
    """
    # if not self._has_wandb_logger() or not self.trainer.is_global_zero:
    #     return
    # experiment = self.logger.experiment
    if not has_wandb_logger or not is_global_zero:
        return
    experiment.define_metric("epoch")
    experiment.define_metric("*", step_metric="epoch")
    # Training metrics
    experiment.define_metric("train/loss", summary="min")
    experiment.define_metric("train/run_summary", step_metric="epoch")
    experiment.define_metric("train/supervised_loss", summary="min")
    experiment.define_metric("train/curvature_loss", summary="min")
    experiment.define_metric("train/acc_all_loss", summary="min")
    experiment.define_metric("train/stress_loss", summary="min")
    experiment.define_metric("train/acc_all_rel_mae", summary="min")
    experiment.define_metric("train/acc_x_rel_mae", summary="min")
    experiment.define_metric("train/acc_y_rel_mae", summary="min")
    experiment.define_metric("train/acc_z_rel_mae", summary="min")
    experiment.define_metric("train/stress_rel_mae", summary="min")
    experiment.define_metric("train/lr", summary="last")
    if "curvature" in residual_list:
        experiment.define_metric("train/curvature_residual", summary="min")
    if "volume" in residual_list:
        experiment.define_metric("train/volume_residual", summary="min")

    # Validation metrics
    experiment.define_metric("val/loss", summary="min") # FIXME: currently not implemented, might renitroduce
    experiment.define_metric("val/overall_relative_mae", summary="min")
    experiment.define_metric("val/acc_all_loss", summary="min")
    experiment.define_metric("val/stress_loss", summary="min")
    experiment.define_metric("val/acc_all_rel_mae", summary="min")
    experiment.define_metric("val/acc_x_rel_mae", summary="min") # FIXME: currently not implemented, might renitroduce
    experiment.define_metric("val/acc_y_rel_mae", summary="min") # FIXME: currently not implemented, might renitroduce
    experiment.define_metric("val/acc_z_rel_mae", summary="min") # FIXME: currently not implemented, might renitroduce
    experiment.define_metric("val/stress_rel_mae", summary="min")
    experiment.define_metric("val/rollout_pos_mae", summary="min")

    # Final evaluation metrics (trainer.test on val or test)
    for prefix in ["final_test", "final_validation"]:
        experiment.define_metric(f"{prefix}/loss", summary="min")
        experiment.define_metric(f"{prefix}/overall_relative_mae", summary="min")
        experiment.define_metric(f"{prefix}/acc_all_rel_mae", summary="min")
        experiment.define_metric(f"{prefix}/stress_rel_mae", summary="min")
        experiment.define_metric(f"{prefix}/rollout_pos_mae", summary="min")
        experiment.define_metric(f"{prefix}/rollout_volume_mae", summary="min")