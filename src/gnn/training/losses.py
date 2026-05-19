"""
losses.py

Contains loss functions and residuals.
Losses: mae, huber, mse
Residuals: angle_residual, volume TODO, energy???
"""

import torch
import torch.nn.functional as F
from src.gnn.mesh_numerical_helpers.compute_angle import compute_surface_angles_torch
from src.gnn.mesh_numerical_helpers.compute_volume import batched_total_absolute_volumes
from src.gnn.mesh_numerical_helpers.misc_numerical_helpers import to_numpy_tetra_connectivity
from src.gnn.preprocessing.build_features import compute_von_mises_from_sig6

EPS = 1e-6
TARGET_ACC_DIM = 3


def _to_tetra_connectivity_tensor(connectivity, device):
    """
    Normalizes tetra connectivity to a torch.LongTensor on the requested device.

    Args:
        connectivity: object
            Connectivity payload from a PyG batch; may be tensor-like or nested python lists.
        device: torch.device
            Destination device.
    return: torch.Tensor
        Tensor with shape (num_tets, 4) and dtype long.
    """
    cells_np = to_numpy_tetra_connectivity(connectivity)
    cells = torch.as_tensor(cells_np, dtype=torch.long, device=device)
    if cells.numel() == 0:
        return cells.reshape(0, 4)
    if cells.dim() == 1 and int(cells.size(0)) == 4:
        cells = cells.reshape(1, 4)
    if cells.dim() != 2 or int(cells.size(1)) != 4:
        raise ValueError(f"Expected tetra connectivity shape (num_tets, 4), got {tuple(cells.shape)}.")
    return cells


def _to_pointer_tensor(pointer_like, pointer_name, device):
    """
    Converts a pointer-like object to a contiguous 1D long tensor.

    Args:
        pointer_like: object
            Pointer payload, for example batch.ptr or a slice vector from batch._slice_dict.
        pointer_name: str
            Name used in error messages.
        device: torch.device
            Destination device.
    return: torch.Tensor
        1D long tensor pointer.
    """
    pointer = torch.as_tensor(pointer_like, dtype=torch.long, device=device).reshape(-1).contiguous()
    if pointer.numel() == 0:
        raise ValueError(f"{pointer_name} is empty.")
    return pointer


def _batched_connectivity_entries(connectivity):
    """
    Splits a connectivity payload into per-graph entries when needed.

    Args:
        connectivity: object
            Batch connectivity payload.
    return: list
        One connectivity entry per graph when representable, otherwise a single-entry list.
    """
    if torch.is_tensor(connectivity):
        if connectivity.dim() <= 2:
            return [connectivity]
        return [connectivity[idx] for idx in range(int(connectivity.size(0)))]
    if isinstance(connectivity, (list, tuple)):
        return list(connectivity)
    return [connectivity]


def _prepare_tetra_connectivity_and_slices(batch, device):
    """
    Builds flattened tetra connectivity and matching per-graph slice pointers.

    Args:
        batch: Data
            Batched PyG graph containing x_element_connectivity.
        device: torch.device
            Destination device.
    return: (torch.Tensor, torch.Tensor)
        Flattened tetra connectivity `(total_tets, 4)` and tetra-slice pointer `(num_graphs + 1,)`.
    """
    num_graphs = int(torch.as_tensor(batch.ptr).numel()) - 1
    entries = _batched_connectivity_entries(batch.x_element_connectivity)
    if len(entries) == 1 and num_graphs > 1:
        cells = _to_tetra_connectivity_tensor(entries[0], device)
        tet_slices = _to_pointer_tensor(batch._slice_dict["x_element_connectivity"],
                                        "batch._slice_dict['x_element_connectivity']",
                                        device)
        return cells, tet_slices
    if len(entries) != num_graphs:
        raise ValueError(
            f"Volume residual expected {num_graphs} connectivity entries, got {len(entries)}."
        )
    all_cells = []
    tet_offsets = [0]
    for entry in entries:
        cells = _to_tetra_connectivity_tensor(entry, device)
        all_cells.append(cells)
        tet_offsets.append(tet_offsets[-1] + int(cells.size(0)))
    if len(all_cells) == 0:
        flat_cells = torch.zeros((0, 4), dtype=torch.long, device=device)
    else:
        flat_cells = torch.cat(all_cells, dim=0)
    tet_slices = torch.as_tensor(tet_offsets, dtype=torch.long, device=device)
    return flat_cells, tet_slices

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

def volume_residual(pred, target, batch, target_stats, dt, time_discretizer, loss_fn):
    """
    Computes residual between true and predicted next-step mesh volume for each graph in batch.

    Args:
        pred: torch.Tensor
            Model prediction with shape `(num_nodes, 4)`.
        target: torch.Tensor
            Ground-truth target with shape `(num_nodes, 4)`.
        batch: Data
            Batched PyG graph containing x_pos_t, x_vel_t and x_element_connectivity.
        target_stats: Stats or StatsModule
            Target normalization statistics.
        dt: float
            Simulation time step.
        time_discretizer: TimeDiscretizer
            Time integration strategy used for position reconstruction.
        loss_fn: callable
            Element-wise scalar loss (mse/mae/huber) used for volume matching.
    return: torch.Tensor
        Scalar differentiable residual on per-graph total volume.
    """
    if not hasattr(batch, "x_element_connectivity"):
        raise ValueError("Volume residual requested but batch has no x_element_connectivity. "
                         "Regenerate preprocessed training data with tetra connectivity enabled.")
    if not hasattr(batch, "_slice_dict"):
        raise ValueError("Volume residual requires batch._slice_dict.")
    pred_next_positions = _reconstruct_predicted_next_positions(pred, batch, target_stats, dt, time_discretizer)
    true_next_positions = _reconstruct_predicted_next_positions(target, batch, target_stats, dt, time_discretizer)
    batch_ptr = _to_pointer_tensor(batch.ptr, "batch.ptr", pred.device)
    tet_connectivity, tet_slices = _prepare_tetra_connectivity_and_slices(batch, pred.device)
    if int(batch_ptr.numel()) != int(tet_slices.numel()):
        raise ValueError("Volume residual requires matching lengths for node ptr and tetra ptr vectors.")
    pred_total_volume = batched_total_absolute_volumes(pred_next_positions, tet_connectivity, batch_ptr, tet_slices)
    true_total_volume = batched_total_absolute_volumes(true_next_positions, tet_connectivity, batch_ptr, tet_slices)
    return loss_fn(pred_total_volume, true_total_volume)


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


def full_stress_components_loss(pred, target, loss_fn, stress_weight):
    """
    Component-wise loss for the full-stress target layout `y = [acc(3), sig6(6)]`.

    Args:
        pred: torch.Tensor
            Model output of shape (N, 9).
        target: torch.Tensor
            Normalized target of shape (N, 9).
        loss_fn: callable
            Elementwise regression loss (mse/mae/huber).
        stress_weight: float
            Multiplier applied to the average loss over the 6 sig channels.
    return: torch.Tensor
        Combined scalar loss `acc_loss + stress_weight * stress_loss`.
    """
    acc_loss = loss_fn(pred[:, 0:3], target[:, 0:3])
    stress_loss = loss_fn(pred[:, 3:9], target[:, 3:9])
    return acc_loss + stress_weight * stress_loss


def compute_vm_regularizer_loss(pred, target_stats, target_vm_physical, loss_fn):
    """
    Differentiable von Mises regularizer: denormalize `pred[:, 3:9]` to physical sig6, compute the
    closed-form vm, and compare against the supplied physical-space ground-truth vm.

    Args:
        pred: torch.Tensor
            Model output of shape (N, 9).
        target_stats: object
            Target normalization stats; stress channels share a single mean/std in full_stress mode.
        target_vm_physical: torch.Tensor
            Ground-truth vm in physical units, shape (N,) or (N, 1) (will be reshaped to (N,)).
        loss_fn: callable
            Regression loss applied to the (predicted vm, target vm) pair.
    return: torch.Tensor
        Scalar loss tensor; the whole chain is differentiable in `pred[:, 3:9]`.
    """
    stress_slice = slice(3, 9)
    sig6_norm = pred[:, stress_slice]
    stress_mean = target_stats.mean[stress_slice]
    stress_std = target_stats.std[stress_slice]
    sig6_phys = sig6_norm * (stress_std + EPS) + stress_mean
    pred_vm = compute_von_mises_from_sig6(sig6_phys)
    return loss_fn(pred_vm, target_vm_physical.reshape(-1))


def full_stress_components_plus_vm_loss(pred, target, target_stats, loss_fn, stress_weight,
                                        target_vm_physical, vm_weight):
    """
    Component-wise full-stress loss plus an optional von Mises regularizer that backpropagates through
    the closed-form vm formula. The two losses are combined linearly with `vm_weight`.

    Args:
        pred: torch.Tensor
            Model output of shape (N, 9).
        target: torch.Tensor
            Normalized target of shape (N, 9).
        target_stats: object
            Target normalization stats.
        loss_fn: callable
            Regression loss.
        stress_weight: float
            Weight applied to the sig6 component loss.
        target_vm_physical: torch.Tensor
            Ground-truth vm in physical units.
        vm_weight: float
            Weight applied to the vm regularizer.
    return: torch.Tensor
        Combined scalar loss.
    """
    base = full_stress_components_loss(pred, target, loss_fn, stress_weight)
    vm_reg = compute_vm_regularizer_loss(pred, target_stats, target_vm_physical, loss_fn)
    return base + vm_weight * vm_reg


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