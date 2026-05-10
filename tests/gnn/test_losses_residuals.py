"""
Tests that all currently exported residuals in losses.py are covered by explicit tests and
that each residual supports differentiable backward pass through predictions.
"""

from types import SimpleNamespace
import torch
import torch.nn.functional as F

from src.gnn.mesh_numerical_helpers.time_discretizer import create_time_discretizer
from src.gnn.preprocessing.stats import Stats
from src.gnn.training import losses as losses_module
from src.gnn.training.losses import angle_residual, volume_residual


def _target_stats():
    return Stats(mean=torch.zeros(4, dtype=torch.float32), std=torch.ones(4, dtype=torch.float32))


def test_all_residual_functions_are_explicitly_covered():
    """
    Informal specification: every public residual function in losses.py must be tracked by tests,
    so adding a new residual requires adding a dedicated unit test for its value and backward behavior.
    """
    discovered = sorted(
        name for name in dir(losses_module) if name.endswith("_residual") and not name.startswith("_")
    )
    covered = sorted(["angle_residual", "volume_residual"])
    assert discovered == covered


def test_angle_residual_returns_finite_scalar_and_backpropagates():
    """
    Informal specification: angle residual must produce a finite scalar penalty and allow autograd to
    propagate gradients to predicted accelerations, so the curvature term can shape optimization updates.
    """
    batch = SimpleNamespace(
        x_pos_t=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        x_vel_t=torch.zeros((4, 3), dtype=torch.float32),
        boundary_left_tri_index=torch.tensor([[0], [1], [2]], dtype=torch.long),
        boundary_right_tri_index=torch.tensor([[1], [0], [3]], dtype=torch.long),
    )
    pred = torch.tensor(
        [[0.15, -0.10, 0.05, 0.0], [0.07, 0.03, -0.08, 0.0], [-0.04, 0.06, 0.02, 0.0], [0.05, -0.02, 0.04, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    residual_value = angle_residual(
        pred,
        batch,
        _target_stats(),
        dt=0.1,
        theta_max=0.0,
        time_discretizer=create_time_discretizer("second_order_central"),
    )
    assert residual_value.ndim == 0
    assert torch.isfinite(residual_value)
    residual_value.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()


def test_volume_residual_returns_finite_scalar_and_backpropagates_with_list_connectivity():
    """
    Informal specification: volume residual must accept batched tetra connectivity after collation, including
    list-like connectivity payloads, and remain differentiable so optimizer steps can use the volume penalty.
    """
    x_pos_t = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.2],
        ],
        dtype=torch.float32,
    )
    batch = SimpleNamespace(
        x_pos_t=x_pos_t,
        x_vel_t=torch.zeros((8, 3), dtype=torch.float32),
        x_element_connectivity=[[0, 1, 2, 3], [0, 1, 2, 3]],
        ptr=[0, 4, 8],
        _slice_dict={"x_element_connectivity": [0, 1, 2]},
    )
    pred = torch.tensor(
        [
            [0.10, -0.02, 0.04, 0.0],
            [0.03, 0.01, -0.06, 0.0],
            [-0.02, 0.05, 0.02, 0.0],
            [0.00, -0.04, 0.03, 0.0],
            [0.04, 0.02, -0.01, 0.0],
            [0.01, -0.03, 0.02, 0.0],
            [-0.05, 0.03, 0.00, 0.0],
            [0.02, 0.01, 0.04, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.zeros_like(pred)
    residual_value = volume_residual(
        pred,
        target,
        batch,
        _target_stats(),
        dt=0.1,
        time_discretizer=create_time_discretizer("second_order_central"),
        loss_fn=F.mse_loss,
    )
    assert residual_value.ndim == 0
    assert torch.isfinite(residual_value)
    residual_value.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()


def test_volume_residual_backpropagates_with_high_rank_batched_connectivity_tensor():
    """
    Informal specification: volume residual must support batched connectivity payloads where each graph carries
    an extra nesting dimension (e.g. batch collation yielding shape `(B, 1, T, 4)`), and still allow backward pass.
    """
    batch = SimpleNamespace(
        x_pos_t=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        x_vel_t=torch.zeros((8, 3), dtype=torch.float32),
        x_element_connectivity=torch.tensor(
            [[[[0, 1, 2, 3]]], [[[0, 1, 2, 3]]]],
            dtype=torch.long,
        ),
        ptr=torch.tensor([0, 4, 8], dtype=torch.long),
        _slice_dict={"x_element_connectivity": torch.tensor([0, 1, 2], dtype=torch.long)},
    )
    pred = torch.tensor(
        [
            [0.08, -0.01, 0.03, 0.0],
            [0.02, 0.02, -0.04, 0.0],
            [-0.03, 0.04, 0.01, 0.0],
            [0.01, -0.02, 0.02, 0.0],
            [0.04, 0.01, -0.02, 0.0],
            [0.01, -0.01, 0.01, 0.0],
            [-0.02, 0.02, 0.00, 0.0],
            [0.01, 0.00, 0.03, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.zeros_like(pred)
    residual_value = volume_residual(
        pred,
        target,
        batch,
        _target_stats(),
        dt=0.1,
        time_discretizer=create_time_discretizer("second_order_central"),
        loss_fn=F.mse_loss,
    )
    assert residual_value.ndim == 0
    assert torch.isfinite(residual_value)
    residual_value.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()
