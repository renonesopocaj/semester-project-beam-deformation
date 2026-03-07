import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts


class SchedulerWrapper:
    """
    Thin wrapper around a PyTorch LR scheduler that provides a uniform interface
    for all three strategies (step, plateau, cosine) and exposes early-stopping
    and pre-restart-save signals.

    Attributes:
        scheduler_type: str
            One of "step", "plateau", "cosine".
        _scheduler: torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau
        _early_stopping_patience: int
            Epochs without improvement before setting the stop flag (plateau only).
        _plateau_threshold: float
            Minimum relative change considered as improvement (plateau only).
        _best_metric: float
            Best observed metric value seen so far (plateau only).
        _no_improve_count: int
            Consecutive epochs without improvement (plateau only).
        _stop: bool
            Whether the early-stopping criterion has been met.
        _save: bool
            Whether the model should be saved before the next cosine restart.
    """

    def __init__(self, scheduler, scheduler_type, early_stopping_patience, plateau_threshold):
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
        self._scheduler = scheduler
        self.scheduler_type = scheduler_type
        self._early_stopping_patience = early_stopping_patience
        self._plateau_threshold = plateau_threshold
        self._best_metric = float("inf")
        self._no_improve_count = 0
        self._stop = False
        self._save = False

    def step(self, metric, epoch):
        """
        Step the underlying LR scheduler and update internal state flags.

        For "plateau": passes `metric` to ReduceLROnPlateau and updates the
        early-stopping counter.
        For "cosine": detects whether the next epoch will trigger a warm restart
        (T_cur == T_i - 1) and sets the save flag accordingly.
        For "step": straightforward step with no metric.

        Args:
            metric: float
                Validation/training metric used for plateau scheduling and
                early-stopping decisions.
            epoch: int
                Current epoch index (0-based).
        """
        if self.scheduler_type == "plateau":
            self._scheduler.step(metric)
            # Update early-stopping counter
            relative_improvement = (self._best_metric - metric) / (abs(self._best_metric) + 1e-9)
            if relative_improvement > self._plateau_threshold:
                self._best_metric = metric
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            if self._no_improve_count >= self._early_stopping_patience:
                self._stop = True

        elif self.scheduler_type == "cosine":
            # Detect upcoming restart before stepping: if T_cur is at the last
            # position of the current cycle, the next step will reset to T_cur=0.
            if self._scheduler.T_cur == self._scheduler.T_i - 1:
                self._save = True
            self._scheduler.step(epoch + 1)

        else:  # "step"
            self._scheduler.step()

    def should_stop(self):
        """
        Returns:
            bool: True if early stopping should terminate training.
        """
        return self._stop

    def should_save(self):
        """
        Returns:
            bool: True if the model should be saved at this epoch (pre-restart checkpoint).
        """
        return self._save

    def reset_save_flag(self):
        """Reset the pre-restart save flag after the checkpoint has been written."""
        self._save = False

    def get_last_lr(self):
        """
        Returns:
            list[float]: Last computed learning rate(s).
        """
        if self.scheduler_type == "plateau":
            return [pg["lr"] for pg in self._scheduler.optimizer.param_groups]
        return self._scheduler.get_last_lr()


def scheduler_factory(optimizer, scheduler_cfg, total_epochs):
    """
    Build and return a SchedulerWrapper from the scheduler config dict, or None
    if no scheduling is requested.

    Args:
        optimizer: torch.optim.Optimizer
            The optimizer whose LR will be scheduled.
        scheduler_cfg: dict or None
            The `lr_scheduler` sub-dict from the training YAML.  Expected keys:
                type: str — "step" | "plateau" | "cosine" | "none"
                step: dict  — step_size, gamma
                plateau: dict — factor, patience, min_lr, early_stopping_patience, threshold
                cosine: dict  — T_0, T_mult, eta_min, save_before_restart (bool)
        total_epochs: int
            Total number of training epochs (used as a fallback for cosine).
        return: SchedulerWrapper or None
    """
    if scheduler_cfg is None:
        return None

    sched_type = scheduler_cfg.get("type", "none")

    if sched_type == "none":
        return None

    if sched_type == "step":
        step_cfg = scheduler_cfg["step"]
        inner = StepLR(
            optimizer,
            step_size=step_cfg["step_size"],
            gamma=step_cfg["gamma"],
        )
        return SchedulerWrapper(inner, "step", early_stopping_patience=0, plateau_threshold=0.0)

    if sched_type == "plateau":
        p_cfg = scheduler_cfg["plateau"]
        inner = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=p_cfg["factor"],
            patience=p_cfg["patience"],
            threshold=p_cfg["threshold"],
            min_lr=p_cfg["min_lr"],
        )
        return SchedulerWrapper(
            inner,
            "plateau",
            early_stopping_patience=p_cfg["early_stopping_patience"],
            plateau_threshold=p_cfg["threshold"],
        )

    if sched_type == "cosine":
        c_cfg = scheduler_cfg["cosine"]
        inner = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=c_cfg["T_0"],
            T_mult=c_cfg["T_mult"],
            eta_min=c_cfg["eta_min"],
        )
        return SchedulerWrapper(inner, "cosine", early_stopping_patience=0, plateau_threshold=0.0)

    raise ValueError(f"Unknown lr_scheduler type: '{sched_type}'. Choose from 'step', 'plateau', 'cosine', 'none'.")
