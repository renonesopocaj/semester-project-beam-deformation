import json
import random
from pathlib import Path
from types import SimpleNamespace
from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
import numpy as np
import torch
import wandb

from src.gnn.training.trainer import LitModule


def model_builder(model_cfg):
    """
    Args:
        model_cfg: dict
        return: MeshGraphNet
    """
    return MeshGraphNet(node_channels=model_cfg["node_channels"], edge_channels=model_cfg["edge_channels"],
                        latent_size=model_cfg["latent_size"], num_msgs=model_cfg["num_msgs"])


def optimizer_builder(model, optimizer_cfg):
    """
    Args:
        model: torch.nn.Module
        optimizer_cfg: dict
        return: torch.optim.Adam
    """
    return torch.optim.Adam(model.parameters(), lr=optimizer_cfg["lr"])

def capture_rng_state():
    """
    Args:
        return: dict
            Python, NumPy, torch CPU, and (optionally) torch CUDA RNG state,
            stored in a format compatible with torch.save / torch.load.
    """
    np_state = np.random.get_state()
    rng_state = {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def restore_rng_state(rng_state):
    """
    Args:
        rng_state: dict
            RNG state previously produced by capture_rng_state.
        return: None
    """
    random.setstate(rng_state["python"])
    np_tuple = rng_state["numpy"]
    np.random.set_state((np_tuple[0], np.array(np_tuple[1], dtype=np.uint32), np_tuple[2], np_tuple[3], np_tuple[4]))
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def trainer_state_to_dict(trainer, epoch, model_cfg, optimizer_cfg, scheduler_cfg):
    """
    Args:
        trainer: Trainer
            Trainer instance to serialise.
        epoch: int
            Last completed epoch.
        model_cfg: dict
            JSON-serialisable model construction config.
        optimizer_cfg: dict
            JSON-serialisable optimizer construction config.
        scheduler_cfg: dict or None
            JSON-serialisable scheduler construction config.
        return: dict
            JSON-serialisable trainer state.
    """
    run_id = trainer.run.id if trainer.run is not None else None

    return {
        "training_id": trainer.training_id,
        "loss_type": trainer.loss_type,
        "use_wandb": trainer.use_wandb,
        "wandb_run_id": run_id,
        "preprocess_cfg": {
            "normalizer_method": trainer.normalizer_method,
            "isotropic_normalizer": trainer.isotropic_normalizer,
            "displacement": trainer.rest_minus_nonrest_pos,
        },
        "epoch": epoch,
        "model_cfg": model_cfg,
        "optimizer_cfg": optimizer_cfg,
        "scheduler_cfg": scheduler_cfg,
        "history": {
            "train_history": trainer.train_history,
            "train_acc_history": trainer.train_acc_history,
            "train_stress_history": trainer.train_stress_history,
            "train_epoch_loss_history": trainer.train_epoch_loss_history,
            "train_epoch_relative_mae_history": trainer.train_epoch_relative_mae_history,
            "val_epoch_loss_history": trainer.val_epoch_loss_history,
            "val_epoch_relative_mae_history": trainer.val_epoch_relative_mae_history,
            "val_epochs_history": trainer.val_epochs_history,
            "val_rollout_error_history": trainer.val_rollout_error_history,
            "lr_history": trainer.lr_history,
            "train_component_loss_history": trainer.train_component_loss_history,
            "train_component_mae_history": trainer.train_component_mae_history,
            "val_component_loss_history": trainer.val_component_loss_history,
            "val_component_mae_history": trainer.val_component_mae_history,
        },
        "inflight_epoch_state": {
            "epoch_losses": trainer.epoch_losses,
            "train_relative_mae_losses": trainer.train_relative_mae_losses,
            "train_acc_loss": trainer.train_acc_loss,
            "train_stress_loss": trainer.train_stress_loss,
            "epoch_component_loss": trainer.epoch_component_loss,
            "epoch_component_mae": trainer.epoch_component_mae,
        },
    }


def apply_trainer_state(trainer, trainer_state):
    """
    Args:
        trainer: Trainer
            Trainer instance to mutate.
        trainer_state: dict
            Deserialised trainer state.
        return: None
    """
    history = trainer_state["history"]
    trainer.train_history = history["train_history"]
    trainer.train_acc_history = history["train_acc_history"]
    trainer.train_stress_history = history["train_stress_history"]
    trainer.train_epoch_loss_history = history["train_epoch_loss_history"]
    trainer.train_epoch_relative_mae_history = history.get("train_epoch_relative_mae_history", [])
    trainer.val_epoch_loss_history = history["val_epoch_loss_history"]
    trainer.val_epoch_relative_mae_history = history.get("val_epoch_relative_mae_history", [])
    trainer.val_epochs_history = history["val_epochs_history"]
    trainer.val_rollout_error_history = history["val_rollout_error_history"]
    trainer.lr_history = history.get("lr_history", [])
    trainer.train_component_loss_history = history["train_component_loss_history"]
    trainer.train_component_mae_history = history["train_component_mae_history"]
    trainer.val_component_loss_history = history["val_component_loss_history"]
    trainer.val_component_mae_history = history["val_component_mae_history"]

    inflight = trainer_state["inflight_epoch_state"]
    trainer.epoch_losses = inflight["epoch_losses"]
    trainer.train_relative_mae_losses = inflight.get("train_relative_mae_losses", [])
    trainer.train_acc_loss = inflight["train_acc_loss"]
    trainer.train_stress_loss = inflight["train_stress_loss"]
    trainer.epoch_component_loss = inflight["epoch_component_loss"]
    trainer.epoch_component_mae = inflight["epoch_component_mae"]


def create_wandb_run(use_wandb, wandb_run_id):
    """
    Args:
        use_wandb: bool
            Whether W&B logging is enabled.
        wandb_run_id: str or None
            Existing run ID to resume, or None for a new run.
        return: wandb.sdk.wandb_run.Run
    """
    if not use_wandb:
        return wandb.init(mode="disabled")
    if wandb_run_id is None:
        return wandb.init()
    return wandb.init(id=wandb_run_id, resume="must")


def save_checkpoint_bundle(trainer, checkpoint_dir, epoch, model_cfg, optimizer_cfg, scheduler_cfg):
    """
    Write a full checkpoint bundle (torch_state.pt + trainer_state.json).
    The checkpoint is taken at a stable boundary (end of epoch, after
    optimizer/scheduler state has been fully updated).

    Args:
        trainer: Trainer
            Trainer to serialise.
        checkpoint_dir: str or pathlib.Path
            Output checkpoint directory.
        epoch: int
            Last completed epoch.
        model_cfg: dict
            JSON-serialisable model construction config.
        optimizer_cfg: dict
            JSON-serialisable optimizer construction config.
        scheduler_cfg: dict or None
            JSON-serialisable scheduler construction config.
        return: pathlib.Path
            Directory containing the checkpoint bundle.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch_state = {
        "epoch": epoch,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        "rng_state": capture_rng_state(),
        "loss": trainer.loss,
        "rollout_all_step_error": trainer.rollout_all_step_error,
        "model_training_mode": trainer.model.training,
    }

    trainer_state = trainer_state_to_dict(trainer, epoch, model_cfg, optimizer_cfg, scheduler_cfg)
    torch.save(torch_state, checkpoint_dir / "torch_state.pt")

    with open(checkpoint_dir / "trainer_state.json", "w", encoding="utf-8") as fh:
        json.dump(trainer_state, fh, indent=2)

    print(f"Checkpoint saved to {checkpoint_dir}")
    return checkpoint_dir


def build_trainer_from_checkpoint(checkpoint_dir, model_builder, optimizer_builder, scheduler_builder, device,
                                  model_root):
    """
    Fully reconstruct a Trainer (and scheduler) from a previously saved
    checkpoint bundle.  This is the inverse of save_checkpoint_bundle.

    Args:
        checkpoint_dir: str or pathlib.Path
            Directory containing torch_state.pt and trainer_state.json.
        model_builder: callable
            model_builder(model_cfg) -> torch.nn.Module
        optimizer_builder: callable
            optimizer_builder(model, optimizer_cfg) -> torch.optim.Optimizer
        scheduler_builder: callable or None
            scheduler_builder(optimizer, scheduler_cfg) -> SchedulerWrapper
        device: torch.device
            Target device.
        model_root: str or pathlib.Path
            Root directory for Trainer.model_dir.
        return: (Trainer, int)
            Restored trainer and the next epoch to start from.
    """
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "trainer_state.json", "r", encoding="utf-8") as fh:
        trainer_state = json.load(fh)

    torch_state = torch.load(checkpoint_dir / "torch_state.pt", map_location=device, weights_only=False)

    preprocess_cfg = SimpleNamespace(**trainer_state["preprocess_cfg"])

    model = model_builder(trainer_state["model_cfg"])
    model = model.to(device)
    optimizer = optimizer_builder(model, trainer_state["optimizer_cfg"])

    scheduler = None
    if scheduler_builder is not None and trainer_state["scheduler_cfg"] is not None:
        scheduler = scheduler_builder(optimizer, trainer_state["scheduler_cfg"])

    run = create_wandb_run(trainer_state["use_wandb"], trainer_state["wandb_run_id"])

    trainer = LitModule(
        model=model, optimizer=optimizer, device=device,
        loss_type=trainer_state["loss_type"], use_wandb=trainer_state["use_wandb"],
        preprocess_cfg=preprocess_cfg, scheduler=scheduler,
        training_id=trainer_state["training_id"], run=run, model_root=model_root,
    )

    trainer.model.load_state_dict(torch_state["model_state_dict"], strict=True)
    trainer.optimizer.load_state_dict(torch_state["optimizer_state_dict"])

    if scheduler is not None and torch_state["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(torch_state["scheduler_state_dict"])

    trainer.loss = torch_state["loss"]
    trainer.rollout_all_step_error = torch_state["rollout_all_step_error"]

    apply_trainer_state(trainer, trainer_state)
    restore_rng_state(torch_state["rng_state"])

    if torch_state["model_training_mode"]:
        trainer.model.train()
    else:
        trainer.model.eval()

    next_epoch = int(torch_state["epoch"]) + 1
    return trainer, next_epoch


def load_trainer_for_eval(checkpoint_path, preprocess_cfg, device, model_root, model, loss_type, training_id):
    """
    Load a minimal Trainer from a checkpoint for evaluation only.
    Restores the model weights and the preprocessing config needed by
    final_test.  Optimizer, scheduler, training history, and RNG state are
    NOT restored.

    Args:
        checkpoint_path: str or pathlib.Path
            Directory containing torch_state.pt and trainer_state.json.
        device: torch.device
            Target device for the model.
        model_root: str or pathlib.Path
            Root directory passed to Trainer for building model_dir.
        return: Trainer
            Trainer with model loaded and set to eval mode.
    """
    torch_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.load_state_dict(torch_state["model_state_dict"], strict=True)
    model.eval()

    run = wandb.init(mode="disabled")

    trainer = LitModule(
        model=model, optimizer=None, device=device,
        loss_type=loss_type, use_wandb=False,
        preprocess_cfg=preprocess_cfg, scheduler=None,
        training_id=training_id, run=run, model_root=model_root,
    )
    return trainer
