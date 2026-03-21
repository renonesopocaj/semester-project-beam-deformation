import torch
import torch.nn.functional as F
from src.gnn.mesh_numerical_helpers.compute_angle import compute_boundary_dihedral_angles_torch

EPS = 1e-6
TARGET_ACC_DIM = 3

def loss_factory(loss_type, residual_list):
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

def reconstruct_predicted_next_positions(pred, batch, target_stats, dt):
    """
    Args:
        pred: torch.Tensor
            Model prediction with shape `(num_nodes, 4)`.
        batch: Data
            Batched PyG graph containing current position and velocity.
        target_stats: Stats or StatsModule
            Target normalization statistics.
        dt: float
            Simulation time step.
    return: torch.Tensor
        Predicted next positions with shape `(num_nodes, 3)`.
    """
    pred_acc = pred[:, :TARGET_ACC_DIM]
    pred_acc_denorm = pred_acc * (target_stats.std[:TARGET_ACC_DIM] + EPS) + target_stats.mean[:TARGET_ACC_DIM]
    return batch.x_pos_t + batch.x_vel_t * dt + pred_acc_denorm * (dt ** 2)


def compute_thresholded_curvature_loss(pred, batch, target_stats, dt, theta_max):
    """
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
            Maximum allowed dihedral angle from the train set.
    return: torch.Tensor
        Mean thresholded curvature penalty.
    """
    predicted_next_positions = reconstruct_predicted_next_positions(pred, batch, target_stats, dt)
    dihedral_angles = compute_boundary_dihedral_angles_torch(predicted_next_positions, batch.boundary_left_tri_index,
                                                             batch.boundary_right_tri_index, EPS)
    return torch.mean(torch.relu(dihedral_angles - theta_max))
