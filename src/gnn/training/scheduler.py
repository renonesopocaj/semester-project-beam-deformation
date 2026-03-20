"""
scheduler.py
Learning rate schedulers
"""

import math
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from abc import ABC, abstractmethod
from typing import Any

class SchedulerWrapper(ABC):
    """
    Wrapper around a PyTorch LR scheduler that provides a uniform interface
    for all three strategies (step, plateau, cosine) and exposes early-stopping
    and pre-restart-save signals.
    """
    scheduler: Any # scheduler torch object
    save: bool # to save checkpoint of model
    stop: bool # to stop training

    def __init__(self, scheduler):
        """
        Args:
            scheduler: torch LR scheduler instance
        """
        self.scheduler = scheduler
        self.save = False
        self.stop = False

    @abstractmethod
    def step(self, metric, epoch):
        """
        Step the underlying LR scheduler and update internal state flags.

        Args:
            metric: float
                Validation/training metric used for plateau scheduling and
                early-stopping decisions.
            epoch: int
                Current epoch index (0-based).
        """
        raise NotImplementedError("step in SchedulerWrapper abstract class not implemented")

    def get_last_lr(self):
        """
        Returns:
            list[float]: Last computed learning rate(s).
        """
        return self.scheduler.get_last_lr()

    def state_dict(self):
        """
        Create dict to save class state

        Args:
            return: dict
                Serializable snapshot of the wrapper and the inner scheduler.
        """
        return {
            "inner": self.scheduler.state_dict(),
            "save": self.save,
            "stop": self.stop,
        }

    def load_state_dict(self, state):
        """
        Create a scheduler object from a dict-checkpoint (restoring state of the scheduler when it saved the checkpoint)

        Args:
            state: dict
                State previously returned by state_dict().
            return: None
        """
        self.scheduler.load_state_dict(state["inner"])
        self.save = state["save"]
        self.stop = state["stop"]

class PlateauEarlyStoppingWrapper(SchedulerWrapper):
    best_metric: float
    no_improve_count: int
    no_improve_count: int
    plateau_threshold: float
    early_stopping_patience: float

    def __init__(self, scheduler, early_stopping_patience, plateau_threshold):
        """
        Args:
            scheduler: torch LR scheduler instance
            scheduler_type: str
                "step", "plateau", or "cosine"
            early_stopping_patience: int
                Number of epochs with no improvement before triggering stop.
                Only meaningful for "plateau".
            plateau_threshold: float
                Minimum relative improvement to reset early-stopping counter.
                Only meaningful for "plateau".
        """
        super().__init__(scheduler)
        self.early_stopping_patience = early_stopping_patience
        self.plateau_threshold = plateau_threshold
        self.best_metric = float("inf")
        self.no_improve_count = 0

    def step(self, metric, epoch):
        self.scheduler.step(metric)
        if not math.isfinite(metric):
            return
        if self.best_metric == float("inf"):
            self.best_metric = metric
            self.no_improve_count = 0
            return
        # Update early-stopping counter
        relative_improvement = (self.best_metric - metric) / (abs(self.best_metric) + 1e-9)
        if relative_improvement > self.plateau_threshold:
            self.best_metric = metric
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        if self.no_improve_count >= self.early_stopping_patience:
            self.stop = True

    def get_last_lr(self):
        return [pg["lr"] for pg in self.scheduler.optimizer.param_groups]

    def state_dict(self):
        """
        Create dict to save class state

        Args:
            return: dict
                Serializable snapshot including early-stopping counters.
        """
        base = super().state_dict()
        base["best_metric"] = self.best_metric
        base["no_improve_count"] = self.no_improve_count
        return base

    def load_state_dict(self, state):
        """
        Create a scheduler object from a dict-checkpoint (restoring state of the scheduler when it saved the checkpoint)

        Args:
            state: dict
                State previously returned by state_dict().
            return: None
        """
        super().load_state_dict(state)
        self.best_metric = state["best_metric"]
        self.no_improve_count = state["no_improve_count"]

class CosineWrapper(SchedulerWrapper):

    def __init__(self, scheduler):
        super().__init__(scheduler)

    def step(self, metric, epoch):
        # Detect upcoming restart before stepping: if T_cur is at the last
        # position of the current cycle, the next step will reset to T_cur=0.
        if self.scheduler.T_cur == self.scheduler.T_i - 1:
            self.save = True
        self.scheduler.step(epoch + 1)

class StepWrapper(SchedulerWrapper):

    def __init__(self, scheduler):
        super().__init__(scheduler)

    def step(self, metric, epoch):
        self.scheduler.step()

def scheduler_factory(optimizer, scheduler_cfg):
    """
    Build and return a SchedulerWrapper from the scheduler config dict, or None
    if no scheduling is requested.

    Args:
        optimizer: torch.optim.Optimizer
            The optimizer whose LR will be scheduled.
        scheduler_cfg: dict or None
            The `lr_scheduler` sub-dict from the training YAML.  Expected keys:
                type: str
                    scheduler type: "step" | "plateau" | "cosine"
                step: dict
                    step_size, gamma
                plateau: dict
                    contains factor, patience, min_lr, early_stopping_patience, threshold
                cosine: dict
                    contains T_0, T_mult, eta_min, save_before_restart (bool)
        return: SchedulerWrapper or None
    """
    if scheduler_cfg["type"] == "step":
        inner = StepLR(optimizer, step_size=scheduler_cfg["step_size"], gamma=scheduler_cfg["gamma"])
        return StepWrapper(inner)
    elif scheduler_cfg["type"] == "plateau":
        inner = ReduceLROnPlateau(optimizer, mode="min", factor=scheduler_cfg["factor"],
                                  patience=scheduler_cfg["patience"],
                                  threshold=scheduler_cfg["threshold"], min_lr=scheduler_cfg["min_lr"])
        return PlateauEarlyStoppingWrapper(inner,
                                           early_stopping_patience=scheduler_cfg["early_stopping_patience"],
                                           plateau_threshold=scheduler_cfg["threshold"])
    elif scheduler_cfg["type"] == "cosine":
        inner = CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_cfg["T_0"], T_mult=scheduler_cfg["T_mult"],
                                            eta_min=scheduler_cfg["eta_min"])
        return CosineWrapper(inner)
    else:
        raise ValueError(f"Unknown lr_scheduler type: '{scheduler_cfg['type']}'. Choose from "
                         f"'step' | 'plateau' | 'cosine' | 'none'")
