"""
target_strategy.py

Strategy pattern encapsulating the differences between the von Mises (legacy) and full stress (sig6)
supervised-target layouts. The strategy owns:

- Output dimensionality of the stress head and the slice that maps prediction tensors to "stress channels".
- The total loss (component losses plus optional von Mises regularizer for full_stress).
- The per-step (`step_metrics`) and per-trajectory no-rollout metrics, including new `fullstress_*` metrics in
  full_stress mode while keeping `stress_*` semantics aligned with the legacy vm pipeline.
- Per-metric element-count weights used by Lightning for proper epoch aggregation.
- The fields stored on predicted rollout graphs (always `von_mises`; additionally `sig6` in full_stress mode).
- The normalizer's `isotropic_groups` for the supervised target tensor (acceleration always pooled across xyz).

All non-trivial mode branching lives here; trainer/rollout/plot code simply delegates to `target_strategy`.
"""

from abc import ABC, abstractmethod

import torch

from src.gnn.misc_helpers.misc import (
    absolute_mae_sum_and_count,
    compute_mean_absolute_mae,
    compute_mean_relative_mae,
    relative_mae_sum_and_count,
)
from src.gnn.preprocessing.build_features import compute_von_mises_from_sig6
from src.gnn.training.losses import (
    compute_total_loss_exp_mises,
    full_stress_components_loss,
    full_stress_components_plus_vm_loss,
)

EPS = 1e-12


def _denormalize_channels(x_norm, target_stats, channel_slice):
    """
    Affine denormalization of a contiguous channel slice using the stored target stats.

    Args:
        x_norm: torch.Tensor
            Normalized tensor with shape (N, F) where F covers at least `channel_slice`.
        target_stats: object
            Stats container exposing `.mean` and `.std` tensors aligned with the target layout.
        channel_slice: slice
            Slice describing which channels of the target the values in `x_norm` correspond to.
    return: torch.Tensor
        Tensor in the original physical units.
    """
    mean = target_stats.mean[channel_slice]
    std = target_stats.std[channel_slice]
    return x_norm * (std + 1e-8) + mean


def _physical_vm_from_data(data, exp_mises):
    """
    Recover the physical (raw) per-node von Mises tensor from the ground-truth aux fields on a PyG Data.

    `data.y_von_mises` stores the (possibly log-transformed when preprocess.mises_log=True) vm; the trainer
    flag `exp_mises` mirrors that convention and lets us undo the log to obtain raw vm whenever requested.

    Args:
        data: torch_geometric.data.Data or Batch
            Carries `y_von_mises` (N, 1) aux field saved by the preprocessor.
        exp_mises: bool
            True when the prediction (and thus the saved aux) is in log space; we then exponentiate to
            recover physical vm.
    return: torch.Tensor
        Physical-space vm tensor of shape (N, 1).
    """
    if exp_mises:
        return torch.exp(data.y_von_mises)
    return data.y_von_mises


class TargetStrategy(ABC):
    """
    Abstract base class for the supervised-target strategies. Subclasses live in this file.

    Subclasses must implement the read-only properties `target_dim`, `stress_out_dim`, `stress_slice`,
    and a small set of methods that the trainer / rollout / plotting modules dispatch to.
    """

    target_mode: str
    exp_mises: bool
    stress_weight: float
    full_stress_loss_mode: str
    full_stress_vm_weight: float

    def __init__(self, target_mode, exp_mises, stress_weight, full_stress_loss_mode, full_stress_vm_weight):
        """
        Initialize the strategy from the per-mode flags resolved on the training side.

        Args:
            target_mode: str
                "von_mises" or "full_stress".
            exp_mises: bool
                Legacy flag controlling whether the (von_mises mode) stress channel is interpreted in log space
                and exponentiated for the physical-space loss.
            stress_weight: float
                Multiplier applied to the stress component of the combined loss.
            full_stress_loss_mode: str
                "components" or "components_plus_vm" (full_stress mode only).
            full_stress_vm_weight: float
                Multiplier on the optional von Mises regularizer loss (full_stress mode only).
        """
        self.target_mode = target_mode
        self.exp_mises = bool(exp_mises)
        self.stress_weight = float(stress_weight)
        self.full_stress_loss_mode = str(full_stress_loss_mode)
        self.full_stress_vm_weight = float(full_stress_vm_weight)

    @property
    @abstractmethod
    def target_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def stress_out_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def stress_slice(self):
        raise NotImplementedError

    @abstractmethod
    def compute_total_loss(self, pred, target, target_stats, loss_fn, batch):
        """
        Compute the supervised loss (acceleration + stress component(s) and possibly a vm regularizer).
        Physics-informed residuals are added on the trainer side outside this method.

        Args:
            pred: torch.Tensor
                Model output of shape (N, target_dim).
            target: torch.Tensor
                Normalized target of shape (N, target_dim).
            target_stats: object
                Target normalization stats.
            loss_fn: callable
                Elementwise loss function.
            batch: torch_geometric.data.Batch
                Current training batch (carries aux fields like `y_von_mises` for the regularizer).
        return: torch.Tensor
            Scalar loss tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def step_metrics(self, pred, target, prefix, target_stats, batch):
        """
        Return the validation/test step metric dict (mode-aware).

        Args:
            pred: torch.Tensor
            target: torch.Tensor
            prefix: str
                Metric namespace prefix.
            target_stats: object
                Target normalization stats.
            batch: torch_geometric.data.Batch
                Current batch (used for vm-derived metrics in full_stress mode).
        return: dict
            Metric name to scalar tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def noroll_metric_weights(self, pred, metric_names):
        """
        Per-metric element-count weights for Lightning's epoch aggregation.

        Args:
            pred: torch.Tensor
            metric_names: list[str]
        return: dict
            metric_name -> int weight.
        """
        raise NotImplementedError

    @abstractmethod
    def per_traj_noroll_init(self):
        """
        Initialize accumulators for the per-trajectory no-rollout metrics.

        return: dict
            Mutable accumulator dict consumed by `per_traj_noroll_accumulate` and finalized later.
        """
        raise NotImplementedError

    @abstractmethod
    def per_traj_noroll_accumulate(self, state, pred, target, batch, target_stats):
        """
        Update accumulators with one batch worth of predictions/targets.

        Args:
            state: dict
                Accumulator dict from `per_traj_noroll_init`.
            pred: torch.Tensor
            target: torch.Tensor
            batch: torch_geometric.data.Batch
            target_stats: object
        return: None
            Mutates `state` in place.
        """
        raise NotImplementedError

    @abstractmethod
    def per_traj_noroll_finalize(self, state, out_dict):
        """
        Convert accumulators into per-trajectory mean values appended to `out_dict` (one entry per traj).

        Args:
            state: dict
                Accumulator dict from `per_traj_noroll_init`.
            out_dict: dict
                Output dict mapping metric_name -> list of floats.
        return: None
            Mutates `out_dict` in place by appending one float per trajectory to each list.
        """
        raise NotImplementedError

    @abstractmethod
    def rollout_fields_from_pred_denorm(self, pred_denorm):
        """
        Extract the per-step rollout fields (vm always, sig6 in full_stress mode) from the denormalized
        prediction tensor.

        Args:
            pred_denorm: torch.Tensor
                Denormalized prediction with shape (N, target_dim).
        return: dict
            {"vm": Tensor(N,), "sig6": Tensor(N,6) or None}.
        """
        raise NotImplementedError

    def target_isotropic_groups(self):
        """
        Normalization groups for the target tensor (used during preprocessing). Acceleration is always
        pooled across (x, y, z); stress channels are either an anisotropic scalar (vm) or all sig6
        components pooled together (full_stress).

        return: list of (int, int)
            (start, length) groups consumed by Normalizer.compute_stats.
        """
        if self.target_mode == "full_stress":
            return [(0, 3), (3, 6)]
        return [(0, 3), (3, 1)]


class VonMisesStrategy(TargetStrategy):
    """Legacy strategy: y = [acc(3), vm(1)] of shape (N, 4)."""

    @property
    def target_dim(self):
        return 4

    @property
    def stress_out_dim(self):
        return 1

    @property
    def stress_slice(self):
        return slice(3, 4)

    def compute_total_loss(self, pred, target, target_stats, loss_fn, batch):
        """
        Plain `loss_fn(pred, target)` unless `exp_mises=True`, in which case the stress channel is
        recovered to physical space via the legacy helper.
        """
        if self.exp_mises:
            return compute_total_loss_exp_mises(pred, target, target_stats, loss_fn, self.stress_weight)
        return loss_fn(pred, target)

    def step_metrics(self, pred, target, prefix, target_stats, batch):
        """Match the legacy behavior exactly so existing dashboards keep working."""
        pred_acc = pred[:, 0:3].contiguous()
        target_acc = target[:, 0:3].contiguous()
        pred_stress = pred[:, 3].contiguous()
        target_stress = target[:, 3].contiguous()
        metrics = {
            f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
            f"{prefix}/overall_abs_mae": compute_mean_absolute_mae(pred, target),
            f"{prefix}/acc_all_rel_mae": compute_mean_relative_mae(pred_acc, target_acc),
            f"{prefix}/acc_all_abs_mae": compute_mean_absolute_mae(pred_acc, target_acc),
            f"{prefix}/stress_rel_mae": compute_mean_relative_mae(pred_stress, target_stress),
            f"{prefix}/stress_abs_mae": compute_mean_absolute_mae(pred_stress, target_stress),
        }
        return metrics

    def noroll_metric_weights(self, pred, metric_names):
        weights = {}
        for name in metric_names:
            if name.endswith("overall_relative_mae") or name.endswith("overall_abs_mae"):
                weights[name] = int(pred.numel())
            elif name.endswith("acc_all_rel_mae") or name.endswith("acc_all_abs_mae") or name.endswith("acc_all_loss"):
                weights[name] = int(pred[:, 0:3].numel())
            elif name.endswith("stress_rel_mae") or name.endswith("stress_abs_mae") or name.endswith("stress_loss"):
                weights[name] = int(pred[:, 3].numel())
            else:
                raise ValueError("Unknown metric for node-weighted logging: %s" % name)
        return weights

    def per_traj_noroll_init(self):
        return {"acc_rel": [0.0, 0.0], "stress_rel": [0.0, 0.0], "acc_abs": [0.0, 0.0],
                "stress_abs": [0.0, 0.0], "overall_abs": [0.0, 0.0]}

    def per_traj_noroll_accumulate(self, state, pred, target, batch, target_stats):
        pred_acc = pred[:, 0:3].contiguous()
        target_acc = target[:, 0:3].contiguous()
        pred_stress = pred[:, 3].contiguous()
        target_stress = target[:, 3].contiguous()
        s_a, c_a = relative_mae_sum_and_count(pred_acc, target_acc)
        s_s, c_s = relative_mae_sum_and_count(pred_stress, target_stress)
        sa_a, ca_a = absolute_mae_sum_and_count(pred_acc, target_acc)
        sa_s, ca_s = absolute_mae_sum_and_count(pred_stress, target_stress)
        sa_o, ca_o = absolute_mae_sum_and_count(pred, target)
        state["acc_rel"][0] += float(s_a.detach().item()); state["acc_rel"][1] += float(c_a.detach().item())
        state["stress_rel"][0] += float(s_s.detach().item()); state["stress_rel"][1] += float(c_s.detach().item())
        state["acc_abs"][0] += float(sa_a.detach().item()); state["acc_abs"][1] += float(ca_a.detach().item())
        state["stress_abs"][0] += float(sa_s.detach().item()); state["stress_abs"][1] += float(ca_s.detach().item())
        state["overall_abs"][0] += float(sa_o.detach().item()); state["overall_abs"][1] += float(ca_o.detach().item())

    def per_traj_noroll_finalize(self, state, out_dict):
        def _div(pair):
            return pair[0] / pair[1] if pair[1] > 0 else 0.0
        out_dict.setdefault("acc_rel_mae", []).append(_div(state["acc_rel"]))
        out_dict.setdefault("stress_rel_mae", []).append(_div(state["stress_rel"]))
        out_dict.setdefault("acc_abs_mae", []).append(_div(state["acc_abs"]))
        out_dict.setdefault("stress_abs_mae", []).append(_div(state["stress_abs"]))
        out_dict.setdefault("overall_abs_mae", []).append(_div(state["overall_abs"]))

    def rollout_fields_from_pred_denorm(self, pred_denorm):
        vm = pred_denorm[:, 3]
        if self.exp_mises:
            vm = torch.exp(vm)
        return {"vm": vm, "sig6": None}


class FullStressStrategy(TargetStrategy):
    """Full-stress strategy: y = [acc(3), sig6(6)] of shape (N, 9). vm is derived analytically when needed."""

    @property
    def target_dim(self):
        return 9

    @property
    def stress_out_dim(self):
        return 6

    @property
    def stress_slice(self):
        return slice(3, 9)

    def compute_total_loss(self, pred, target, target_stats, loss_fn, batch):
        """
        Either (a) plain weighted component losses, or (b) plus a differentiable von Mises regularizer
        derived from `pred[:, 3:9]` and compared against the physical ground-truth vm (from `batch.y_von_mises`,
        respecting the `exp_mises` log convention).
        """
        if self.full_stress_loss_mode == "components":
            return full_stress_components_loss(pred, target, loss_fn, self.stress_weight)
        if self.full_stress_loss_mode == "components_plus_vm":
            target_vm_physical = _physical_vm_from_data(batch, self.exp_mises).reshape(-1)
            return full_stress_components_plus_vm_loss(pred, target, target_stats, loss_fn, self.stress_weight,
                                                       target_vm_physical, self.full_stress_vm_weight)
        raise ValueError(f"Unknown full_stress_loss_mode '{self.full_stress_loss_mode}'.")

    def _compute_pred_vm_physical(self, pred, target_stats):
        """
        Reconstruct physical vm from the 6-component prediction. Used both for metrics and for the regularizer
        path; kept here so the trainer never sees sig6 indexing details.

        Args:
            pred: torch.Tensor
                Normalized model output of shape (N, 9).
            target_stats: object
                Target normalization stats; the stress channels share a single stat in full_stress mode.
        return: torch.Tensor
            Physical vm tensor of shape (N,).
        """
        sig6_pred_denorm = _denormalize_channels(pred[:, 3:9], target_stats, slice(3, 9))
        return compute_von_mises_from_sig6(sig6_pred_denorm)

    def step_metrics(self, pred, target, prefix, target_stats, batch):
        """
        Provides:
        - `overall_*` / `acc_all_*`: identical conventions to vm mode (normalized space).
        - `stress_*` (vm-derived): physical vm comparison so existing dashboards keep working.
        - `fullstress_*`: new MAE over all 6 sig channels (normalized-space, mirroring vm mode `stress_*`).
        """
        pred_acc = pred[:, 0:3].contiguous()
        target_acc = target[:, 0:3].contiguous()
        pred_sig6 = pred[:, 3:9].contiguous()
        target_sig6 = target[:, 3:9].contiguous()

        pred_vm_physical = self._compute_pred_vm_physical(pred, target_stats)
        target_vm_physical = _physical_vm_from_data(batch, self.exp_mises).reshape(-1)

        metrics = {
            f"{prefix}/overall_relative_mae": compute_mean_relative_mae(pred, target),
            f"{prefix}/overall_abs_mae": compute_mean_absolute_mae(pred, target),
            f"{prefix}/acc_all_rel_mae": compute_mean_relative_mae(pred_acc, target_acc),
            f"{prefix}/acc_all_abs_mae": compute_mean_absolute_mae(pred_acc, target_acc),
            f"{prefix}/stress_rel_mae": compute_mean_relative_mae(pred_vm_physical, target_vm_physical),
            f"{prefix}/stress_abs_mae": compute_mean_absolute_mae(pred_vm_physical, target_vm_physical),
            f"{prefix}/fullstress_rel_mae": compute_mean_relative_mae(pred_sig6, target_sig6),
            f"{prefix}/fullstress_abs_mae": compute_mean_absolute_mae(pred_sig6, target_sig6),
        }
        return metrics

    def noroll_metric_weights(self, pred, metric_names):
        weights = {}
        sig6_numel = int(pred[:, 3:9].numel())
        # vm metrics aggregate over N nodes (the vm tensor is shape (N,)).
        vm_numel = int(pred.shape[0])
        for name in metric_names:
            if name.endswith("overall_relative_mae") or name.endswith("overall_abs_mae"):
                weights[name] = int(pred.numel())
            elif name.endswith("acc_all_rel_mae") or name.endswith("acc_all_abs_mae") or name.endswith("acc_all_loss"):
                weights[name] = int(pred[:, 0:3].numel())
            elif name.endswith("fullstress_rel_mae") or name.endswith("fullstress_abs_mae") or name.endswith("fullstress_loss"):
                weights[name] = sig6_numel
            elif name.endswith("stress_rel_mae") or name.endswith("stress_abs_mae") or name.endswith("stress_loss"):
                weights[name] = vm_numel
            else:
                raise ValueError("Unknown metric for node-weighted logging: %s" % name)
        return weights

    def per_traj_noroll_init(self):
        return {"acc_rel": [0.0, 0.0], "stress_rel": [0.0, 0.0], "acc_abs": [0.0, 0.0],
                "stress_abs": [0.0, 0.0], "overall_abs": [0.0, 0.0],
                "fullstress_rel": [0.0, 0.0], "fullstress_abs": [0.0, 0.0]}

    def per_traj_noroll_accumulate(self, state, pred, target, batch, target_stats):
        pred_acc = pred[:, 0:3].contiguous()
        target_acc = target[:, 0:3].contiguous()
        pred_sig6 = pred[:, 3:9].contiguous()
        target_sig6 = target[:, 3:9].contiguous()
        pred_vm_physical = self._compute_pred_vm_physical(pred, target_stats)
        target_vm_physical = _physical_vm_from_data(batch, self.exp_mises).reshape(-1)
        s_a, c_a = relative_mae_sum_and_count(pred_acc, target_acc)
        sa_a, ca_a = absolute_mae_sum_and_count(pred_acc, target_acc)
        s_s, c_s = relative_mae_sum_and_count(pred_vm_physical, target_vm_physical)
        sa_s, ca_s = absolute_mae_sum_and_count(pred_vm_physical, target_vm_physical)
        s_fs, c_fs = relative_mae_sum_and_count(pred_sig6, target_sig6)
        sa_fs, ca_fs = absolute_mae_sum_and_count(pred_sig6, target_sig6)
        sa_o, ca_o = absolute_mae_sum_and_count(pred, target)
        state["acc_rel"][0] += float(s_a.detach().item()); state["acc_rel"][1] += float(c_a.detach().item())
        state["stress_rel"][0] += float(s_s.detach().item()); state["stress_rel"][1] += float(c_s.detach().item())
        state["acc_abs"][0] += float(sa_a.detach().item()); state["acc_abs"][1] += float(ca_a.detach().item())
        state["stress_abs"][0] += float(sa_s.detach().item()); state["stress_abs"][1] += float(ca_s.detach().item())
        state["overall_abs"][0] += float(sa_o.detach().item()); state["overall_abs"][1] += float(ca_o.detach().item())
        state["fullstress_rel"][0] += float(s_fs.detach().item()); state["fullstress_rel"][1] += float(c_fs.detach().item())
        state["fullstress_abs"][0] += float(sa_fs.detach().item()); state["fullstress_abs"][1] += float(ca_fs.detach().item())

    def per_traj_noroll_finalize(self, state, out_dict):
        def _div(pair):
            return pair[0] / pair[1] if pair[1] > 0 else 0.0
        out_dict.setdefault("acc_rel_mae", []).append(_div(state["acc_rel"]))
        out_dict.setdefault("stress_rel_mae", []).append(_div(state["stress_rel"]))
        out_dict.setdefault("acc_abs_mae", []).append(_div(state["acc_abs"]))
        out_dict.setdefault("stress_abs_mae", []).append(_div(state["stress_abs"]))
        out_dict.setdefault("overall_abs_mae", []).append(_div(state["overall_abs"]))
        out_dict.setdefault("fullstress_rel_mae", []).append(_div(state["fullstress_rel"]))
        out_dict.setdefault("fullstress_abs_mae", []).append(_div(state["fullstress_abs"]))

    def rollout_fields_from_pred_denorm(self, pred_denorm):
        sig6 = pred_denorm[:, 3:9]
        vm = compute_von_mises_from_sig6(sig6)
        return {"vm": vm, "sig6": sig6}


def create_target_strategy(target_mode, train_cfg):
    """
    Factory that instantiates the appropriate strategy for the resolved target mode.

    Args:
        target_mode: str
            "von_mises" or "full_stress".
        train_cfg: object
            Training config exposing `exp_mises`, `stress_weight`, and (for full_stress) `full_stress_loss_mode`
            and `full_stress_vm_weight`.
    return: TargetStrategy
        Concrete strategy instance.
    """
    exp_mises = bool(getattr(train_cfg, "exp_mises", False))
    stress_weight = float(getattr(train_cfg, "stress_weight", 1.0))
    full_stress_loss_mode = str(getattr(train_cfg, "full_stress_loss_mode", "components"))
    full_stress_vm_weight = float(getattr(train_cfg, "full_stress_vm_weight", 0.0))
    if target_mode == "von_mises":
        return VonMisesStrategy(target_mode, exp_mises, stress_weight, full_stress_loss_mode, full_stress_vm_weight)
    if target_mode == "full_stress":
        return FullStressStrategy(target_mode, exp_mises, stress_weight, full_stress_loss_mode, full_stress_vm_weight)
    raise ValueError(f"Unknown target_mode '{target_mode}'.")
