import os
import time
import yaml
import torch
import wandb
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.training.trainer import Trainer
from src.gnn.preprocessing.stats import load_stats
from src.gnn.training.scheduler import scheduler_factory
from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
from src.gnn.helpers.yamlobject import TrainYaml, PreprocessYaml
from src.gnn.helpers.train_plots import plot_training_history
from src.gnn.training.checkpointing import (save_checkpoint_bundle, build_trainer_from_checkpoint, model_builder,
                                            optimizer_builder)

matplotlib.use("Agg")
plt.ioff()


def _build_config_dicts(cfg, preprocess_cfg):
    """
    Args:
        cfg: TrainYaml
            Parsed training configuration.
        preprocess_cfg: PreprocessYaml
            Parsed preprocessing configuration.
        return: (dict, dict, dict or None)
            model_cfg, optimizer_cfg, scheduler_cfg -- all JSON-serialisable.
    """
    model_cfg = {"node_channels": preprocess_cfg.node_channels, "edge_channels": preprocess_cfg.edge_channels,
                 "latent_size": cfg.latent_dim, "num_msgs": cfg.num_messages}
    optimizer_cfg = {"initial_lr": cfg.initial_lr}
    scheduler_cfg = cfg.scheduler_cfg
    return model_cfg, optimizer_cfg, scheduler_cfg

def train(cfg, preprocess_cfg, device, train_dict, preprocess_dict):
    """
    Note: need dict version of train_cfg and preprocess_cfg in order to save them with OmegaConf.

    Args:
        cfg: TrainYaml
            Parsed training configuration.
        preprocess_cfg: PreprocessYaml
            Parsed preprocessing configuration.
        device: torch.device
        train_dict: dict
            dict of cfg
        preprocess_dict: dict
            dict of preprocess_cfg

    return: (Trainer, Stats, Stats, Stats, float, dict, dict, dict)
    """
    model_root = Path(os.getcwd()) / "saved_models"
    model_cfg, optimizer_cfg, scheduler_cfg = _build_config_dicts(cfg, preprocess_cfg)

    print("Train dataset:")
    train_dataset = InMemoryTimeStepDataset(sample_dir=cfg.train_dir, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    if cfg.enable_val:
        print("Validation dataset:")
        validation_dataset = InMemoryTimeStepDataset(sample_dir=cfg.val_dir, device=device)
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    print(f"Loading stats from {cfg.train_dir}...")
    node_stats, edge_stats, target_stats = load_stats(cfg.train_dir)
    dt = preprocess_cfg.dt

    print(f"\tNode features: mean shape = {node_stats.mean.shape}, std shape = {node_stats.std.shape}")
    print(f"\tEdge features: mean shape = {edge_stats.mean.shape}, std shape = {edge_stats.std.shape}")
    print(f"\tTarget features: mean shape = {target_stats.mean.shape}, std shape = {target_stats.std.shape}")

    start_epoch = 0

    # Init model, either resuming a previous train or from scratch
    if cfg.resume_from is not None:
        # Resume previous training
        print(f"Resuming training from checkpoint: {cfg.resume_from}")
        trainer, start_epoch = build_trainer_from_checkpoint(
            checkpoint_dir=cfg.resume_from, model_builder=model_builder,
            optimizer_builder=optimizer_builder, scheduler_builder=scheduler_factory,
            device=device, model_root=model_root,
        )
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"Total parameters: {total_params:,}")
    else:
        # Init new training
        torch.manual_seed(cfg.seed)
        model = MeshGraphNet(node_channels=preprocess_cfg.node_channels,
                             edge_channels=preprocess_cfg.edge_channels,
                             latent_size=cfg.latent_dim,
                             num_msgs=cfg.num_messages)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr)
        scheduler = scheduler_factory(optimizer, cfg.scheduler_cfg)

        training_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        run = wandb.init() if cfg.use_wandb else wandb.init(mode="disabled")

        trainer = Trainer(model=model, optimizer=optimizer, device=device, loss_type=cfg.loss_type,
                          use_wandb=cfg.use_wandb, preprocess_cfg=preprocess_cfg, scheduler=scheduler,
                          training_id=training_id, run=run, model_root=model_root)

        OmegaConf.save(train_dict, trainer.model_dir / "train.yaml")
        OmegaConf.save(preprocess_dict, trainer.model_dir / "preprocess.yaml")

    sched_type = cfg.scheduler_cfg["type"]
    print(f"Training ID: {trainer.training_id}\n\tLearning rate: {cfg.initial_lr}\n\tLR scheduler: {sched_type}\n\t"
          f"Total epochs: {cfg.total_epochs}\n\tStart epoch: {start_epoch}\n\t"
          f"Validation interval: {cfg.validation_interval}\n\t"
          f"Loss type: {cfg.loss_type}\n\tUsing wandb: {cfg.use_wandb}\n\tModel directory: {trainer.model_dir}")

    # Main training and validation loop
    epoch_bar = tqdm(range(start_epoch, cfg.total_epochs), desc="Epochs")
    for epoch in epoch_bar:
        # For all batches
        for batch in tqdm(train_dataloader, desc="Batches", leave=False):
            # One train step + collect necessary metrics on training set
            batch = batch.to(device)
            trainer.train(graph=batch)

        # Log non-rollout metrics such as relative mae and loss at the end of each epoch.
        # Also update scheduler.
        trainer.epoch_end_log_metrics_scheduler(numepoch=epoch)

        # Scheduler-triggered save (e.g. before cosine warm-restart)
        if trainer.scheduler.save:
            print(f"Epoch {epoch}: saving checkpoint before cosine warm-restart.")
            ckpt_dir = trainer.model_dir / f"checkpoint_epoch_{epoch}"
            save_checkpoint_bundle(trainer, ckpt_dir, epoch, model_cfg, optimizer_cfg, scheduler_cfg)
            trainer.scheduler.save = False

        # Periodic checkpoint
        if cfg.checkpoint_interval > 0 and (epoch + 1) % cfg.checkpoint_interval == 0:
            ckpt_dir = trainer.model_dir / f"checkpoint_epoch_{epoch}"
            save_checkpoint_bundle(trainer, ckpt_dir, epoch, model_cfg, optimizer_cfg, scheduler_cfg)

        # Validation phase
        if epoch % cfg.validation_interval == 0 and cfg.enable_val:
            trainer.validation(validation_loader=validation_dataloader, node_stats=node_stats, edge_stats=edge_stats,
                               target_stats=target_stats, dt=dt, epoch=epoch)
            # tell me if it's correct
            epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}",
                                   "rollout_error": f"{trainer.rollout_all_step_error:.6f}",
                                   "learning rate": f"{trainer.scheduler}",
                                   "error": f"{trainer.val_epoch_relative_mae_history[-1]}"
                                   })
            # print(f"\nEpoch {epoch}: Train Loss = {trainer.loss:.6f}, "
            #       f"Validation Rollout Error = {trainer.rollout_all_step_error:.6f}")
        else:
            epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}"})

    return (trainer, node_stats, edge_stats, target_stats, dt, model_cfg, optimizer_cfg, scheduler_cfg,
            validation_dataloader)


@hydra.main(version_base=None, config_path="../../conf/train", config_name="train")
def main(cfg):
    """
    Args:
        cfg: omegaconf.DictConfig
            Hydra-composed configuration. Run with --multirun to sweep over multiple values,
            e.g.: python -m src.gnn.train --multirun training.lr=1e-3,5e-4,1e-4
    """
    # Open configs
    train_cfg = TrainYaml(cfg)
    with open(Path(train_cfg.dataset_dir) / "config.yaml", "r") as file:
        preprocess_dict = yaml.safe_load(file)

    preprocess_cfg = PreprocessYaml(preprocess_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train, validation and log validation metrics
    trainer, node_stats, edge_stats, target_stats, dt, model_cfg, optimizer_cfg, scheduler_cfg, validation_dataloader \
        = train(train_cfg, preprocess_cfg, device, cfg, preprocess_dict)

    # Plot loss, lr, metric on test set
    plot_training_history(trainer)

    # Test model on a new trajectory (test set)
    print("\nTest dataset:")
    test_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.test_dir, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    final_rollout_error = trainer.final_test(test_dataloader=test_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="test")
    print(f"Final evaluation rollout error: {final_rollout_error:.6f}")

    # Test model on a new trajectory (validation set)
    print("\nValidation dataset:")
    final_rollout_error = trainer.final_test(test_dataloader=validation_dataloader, node_stats=node_stats,
                                             edge_stats=edge_stats, target_stats=target_stats, dt=dt,
                                             split_name="validation")
    print(f"\nFinal evaluation rollout error: {final_rollout_error:.6f}")

    # Test model from an already seen trajectory (took from training set)
    # FIXME: training graph have stuff like rest_pos which is not set in preprocess and is set only for validation
    # FIXME: add it for only the first graph -- if it's ok, maybe it's not and I need it for all the graphs in the
    #  trajectory. Worst case see how heavier it is to add it for all of the graphs.
    # print("Train-sample dataset (seen trajectory):")
    # train_files = sorted([f for f in os.listdir(train_cfg.train_dir) if f.startswith("graph_") and f.endswith(".pt")])
    # train_sample_data = torch.load(os.path.join(train_cfg.train_dir, train_files[0]), map_location=device, weights_only=False)
    # train_sample_dataloader = DataLoader(train_sample_data, batch_size=1, shuffle=False)
    # train_rollout_error = trainer.final_test(test_dataloader=train_sample_dataloader, node_stats=node_stats,
    #                                          edge_stats=edge_stats, target_stats=target_stats, dt=dt,
    #                                          split_name="sanity")
    # print(f"\nTrain-sample rollout error: {train_rollout_error:.6f}")

    # Final checkpoint
    final_ckpt_dir = trainer.model_dir / f"checkpoint_epoch_{train_cfg.total_epochs - 1}"
    save_checkpoint_bundle(trainer, final_ckpt_dir, train_cfg.total_epochs - 1,
                           model_cfg, optimizer_cfg, scheduler_cfg)


if __name__ == "__main__":
    main()
