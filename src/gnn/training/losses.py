"""
losses.py

Contains loss functions and residuals.
Losses: mae, huber, mse
Residuals: angle_residual, volume TODO, energy???
"""

import torch
import torch.nn.functional as F
from src.gnn.mesh_numerical_helpers.compute_angle import compute_surface_angles_torch

EPS = 1e-6
TARGET_ACC_DIM = 3

def loss_factory(loss_type):
    """
    Maps a loss type string to the corresponding PyTorch loss function.

    Args:
        loss_type: str
            Name of the loss function.
        residual_list: list[str]
            list of name of residuals
    return: callable
        PyTorch loss function.
    """
    # for residual in residual_list: make a single callable function as a sum of the other residuals
    if loss_type == "mse":
        return F.mse_loss
    if loss_type == "mae":
        return F.l1_loss
    if loss_type == "huber":
        return F.huber_loss
    raise NotImplementedError("Unsupported loss function")

def volume_loss(pred, target):
    # FIXME: implement volume loss/residual: the difference between the "rest" and "non-rest" volume (scalars)
    #       has to be minimized
    return


def angle_residual(pred, batch, target_stats, dt, theta_max, time_discretizer):
    """
    Computes Relu(theta - theta_max) for each angle between surface triangles.

    Args:
        pred: torch.Tensor
            Model prediction with shape `(num_nodes, 4)`.
        batch: Data
            Batched PyG graph with cached boundary topology.
        target_stats: Stats or StatsModule
            Target normalization statistics.
        dt: float
            Simulation time step.
        theta_max: float
            Maximum allowed angle from the train set.
        time_discretizer: TimeDiscretizer
            Time integration strategy used for position reconstruction.
    return: torch.Tensor
        Mean thresholded curvature penalty.
    """
    predicted_next_positions = _reconstruct_predicted_next_positions(pred, batch, target_stats, dt, time_discretizer)
    triangle_angles = compute_surface_angles_torch(predicted_next_positions, batch.boundary_left_tri_index,
                                                   batch.boundary_right_tri_index)
    return torch.mean(torch.relu(triangle_angles - theta_max))


def compute_stress_loss_in_physical_space(pred, target, target_stats, loss_fn):
    """
    Denormalizes log-stress predictions and targets, exponentiates them back to physical
    stress space, and computes the loss there.

    Args:
        pred: torch.Tensor
            Model output of shape (num_nodes, 4). Last channel is normalized log-stress.
        target: torch.Tensor
            Normalized target of shape (num_nodes, 4). Last channel is normalized log-stress.
        target_stats: object
            Stats container with mean and std tensors of shape (4,) or compatible.
        loss_fn: callable
            Regression loss such as mse, mae, or huber.
    return: torch.Tensor
        Stress loss in physical stress space.
    """
    pred_log_norm = pred[:, 3:4]
    target_log_norm = target[:, 3:4]

    stress_mean = target_stats.mean[3:4]
    stress_std = target_stats.std[3:4]

    pred_log = pred_log_norm * stress_std + stress_mean
    target_log = target_log_norm * stress_std + stress_mean

    pred_stress = torch.exp(pred_log)
    target_stress = torch.exp(target_log)

    return loss_fn(pred_stress, target_stress)


def compute_total_loss_exp_mises(pred, target, target_stats, loss_fn, stress_weight):
    """
    Computes total loss when exp_mises is enabled: acceleration loss in normalized space
    plus stress loss in physical (exponentiated) space.

    Args:
        pred: torch.Tensor
            Model output of shape (num_nodes, 4).
        target: torch.Tensor
            Normalized target of shape (num_nodes, 4).
        target_stats: object
            Stats container with mean and std tensors.
        loss_fn: callable
            Regression loss.
        stress_weight: float
            Weight applied to stress loss.
    return: torch.Tensor
        Combined loss.
    """
    acc_loss = loss_fn(pred[:, 0:3], target[:, 0:3])
    stress_loss = compute_stress_loss_in_physical_space(pred, target, target_stats, loss_fn)
    return acc_loss + stress_weight * stress_loss


def _reconstruct_predicted_next_positions(pred, batch, target_stats, dt, time_discretizer):
    """
    Denormalizes predicted acceleration and reconstructs predicted next positions
    using the same time integration scheme that was used for preprocessing.
    Helper for angle_residual.

    Args:
        pred: torch.Tensor
            Model prediction with shape `(num_nodes, 4)`.
        batch: Data
            Batched PyG graph containing current position and velocity.
        target_stats: Stats or StatsModule
            Target normalization statistics.
        dt: float
            Simulation time step.
        time_discretizer: TimeDiscretizer
            Time integration strategy matching the preprocessing scheme.
    return: torch.Tensor
        Predicted next positions with shape `(num_nodes, 3)`.
    """
    pred_acc = pred[:, :TARGET_ACC_DIM]
    pred_acc_denorm = pred_acc * (target_stats.std[:TARGET_ACC_DIM] + EPS) + target_stats.mean[:TARGET_ACC_DIM]
    x_tp1, _ = time_discretizer.integrate_acceleration(pred_acc_denorm, batch.x_vel_t, dt, batch.x_pos_t)
    return x_tp1