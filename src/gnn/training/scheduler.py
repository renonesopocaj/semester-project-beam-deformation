import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from abc import ABC, abstractmethod

class SchedulerWrapper(ABC):
    """
    Wrapper around a PyTorch LR scheduler that provides a uniform interface
    for all three strategies (step, plateau, cosine) and exposes early-stopping
    and pre-restart-save signals.

    Attributes:
        scheduler_type: str
            One of "step", "plateau", "cosine".
        scheduler: torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau
        early_stopping_patience: int
            Epochs without improvement before setting the stop flag (plateau only).
        plateau_threshold: float
            Minimum relative change considered as improvement (plateau only).
        best_metric: float
            Best observed metric value seen so far (plateau only).
        no_improve_count: int
            Consecutive epochs without improvement (plateau only).
        stop: bool
            Whether the early-stopping criterion has been met.
        save: bool
            Whether the model should be saved before the next cosine restart.
    """

    def __init__(self, scheduler):
        """
        Args:
            scheduler: torch LR scheduler instance
        """
        self.scheduler = scheduler

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

class PlateauEarlyStoppingWrapper(SchedulerWrapper):

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
        self.stop = False
        self.save = False

    def step(self, metric, epoch):
        self.scheduler.step(metric)
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
    if scheduler_cfg.type == "step":
        step_cfg = scheduler_cfg["step"]
        inner = StepLR(optimizer, step_size=step_cfg["step_size"], gamma=step_cfg["gamma"])
        return StepWrapper(inner)
    elif scheduler_cfg.type == "plateau":
        p_cfg = scheduler_cfg["plateau"]
        inner = ReduceLROnPlateau(optimizer, mode="min", factor=p_cfg["factor"], patience=p_cfg["patience"],
                                  threshold=p_cfg["threshold"], min_lr=p_cfg["min_lr"])
        return PlateauEarlyStoppingWrapper(inner,
                                           early_stopping_patience=p_cfg["early_stopping_patience"],
                                           plateau_threshold=p_cfg["threshold"])
    elif scheduler_cfg.type == "cosine":
        c_cfg = scheduler_cfg["cosine"]
        inner = CosineAnnealingWarmRestarts(optimizer, T_0=c_cfg["T_0"], T_mult=c_cfg["T_mult"],
                                            eta_min=c_cfg["eta_min"])
        return CosineWrapper(inner)
    else:
        raise ValueError(f"Unknown lr_scheduler type: '{scheduler_cfg.type}'. Choose from "
                         f"'step' | 'plateau' | 'cosine'")
