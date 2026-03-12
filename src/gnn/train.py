import os
import time
import yaml
from pathlib import Path

import hydra
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
from src.gnn.misc_helpers.yamlobject import PreprocessYaml, TrainYaml
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.preprocessing.stats import load_stats
from src.gnn.training.scheduler import scheduler_factory
from src.gnn.training.trainer import LitModule

matplotlib.use("Agg")
plt.ioff()

def create_model(train_cfg, preprocess_cfg, device):
    if train_cfg.model_name == "MGN":
        return MeshGraphNet(node_channels=preprocess_cfg.node_channels, edge_channels=preprocess_cfg.edge_channels,
                            latent_size=train_cfg.latent_dim,
                            num_msgs=train_cfg.num_messages)
    elif train_cfg.model_name == "clofnet":
        raise NotImplementedError("CLOFNet not implemented")
    else:
        raise NotImplementedError(f"Model type {train_cfg.model_type} not implemented")

def train(cfg, preprocess_cfg, device, train_dict, preprocess_dict):
    """
    Args:
        cfg: TrainYaml
            Parsed training configuration.
        preprocess_cfg: PreprocessYaml
            Parsed preprocessing configuration.
        device: torch.device
            Target device.
        train_dict: dict
            Hydra/OmegaConf training config.
        preprocess_dict: dict
            Preprocess config.
        return: tuple
            Lightning module, Lightning trainer, stats, dt, and validation loader.
    """
    model_root = Path(os.getcwd()) / "saved_models"
    model_root.mkdir(parents=True, exist_ok=True)

    # ========================= Dataset stuff ===================
    validation_dataloader = None
    print("Train dataset:")
    train_dataset = InMemoryTimeStepDataset(sample_dir=cfg.train_dir, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                                  num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    if cfg.enable_val:
        print("Validation dataset:")
        validation_dataset = InMemoryTimeStepDataset(sample_dir=cfg.val_dir, device=device)
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False,
                                           num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    print(f"Loading stats from {cfg.train_dir}...")
    node_stats, edge_stats, target_stats = load_stats(cfg.train_dir)
    dt = preprocess_cfg.dt
    print(f"\tNode features: mean shape = {node_stats.mean.shape}, std shape = {node_stats.std.shape}")
    print(f"\tEdge features: mean shape = {edge_stats.mean.shape}, std shape = {edge_stats.std.shape}")
    print(f"\tTarget features: mean shape = {target_stats.mean.shape}, std shape = {target_stats.std.shape}")

    # ========================= Model and Lightning wrapper =================
    torch.manual_seed(cfg.seed)
    model = create_model(cfg, preprocess_cfg, device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"\tTotal parameters: {total_params:,}")
    # Set optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr)
    scheduler = scheduler_factory(optimizer, cfg.scheduler_cfg)
    training_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Set Lightning Wrapper of Model.
    lightning_module = LitModule(model=model, optimizer=optimizer, scheduler=scheduler, loss_type=cfg.loss_type,
                                 preprocess_cfg=preprocess_cfg, training_id=training_id, model_root=model_root)
    lightning_module.set_rollout_context(validation_loader=validation_dataloader, node_stats=node_stats,
                                         edge_stats=edge_stats, target_stats=target_stats, dt=dt)
    # Save configs via hydra
    OmegaConf.save(train_dict, lightning_module.model_dir / "train.yaml")
    OmegaConf.save(preprocess_dict, lightning_module.model_dir / "preprocess.yaml")

    # =================== Wandb logger ===================
    if cfg.use_wandb:
        logger = WandbLogger(project="meshgraphnet", name=training_id, save_dir=str(lightning_module.model_dir),
                             log_model=False)
    else:
        logger = False

    # ============ Set checkpointing rules (note: also scheduler is saved properly) =============
    callbacks = []
    checkpoint_str_dir = str(lightning_module.model_dir / "checkpoints")
    # ALlocate a callback for checkpoints of model every time I get the "best model"
    if cfg.enable_val:
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_str_dir, filename="best",
                                         monitor="val/loss", mode="min",
                                         save_top_k=1))
    # periodic save every cfg.checkpoint_interval epoch
    if cfg.checkpoint_interval > 0:
        # save_top_k = -1 --> do not take into account the loss, save anyway
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_str_dir, filename="epoch-{epoch:04d}",
                                         every_n_epochs=cfg.checkpoint_interval, save_top_k=-1))

    # ======================= Lightning Trainer ===================
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if cfg.training_type == "sanity":
        fit_trainer = L.Trainer(max_epochs=cfg.total_epochs, accelerator=accelerator, devices=1, logger=logger,
                                callbacks=callbacks, default_root_dir=str(lightning_module.model_dir),
                                check_val_every_n_epoch=cfg.validation_interval, log_every_n_steps=1,
                                enable_checkpointing=len(callbacks) > 0,
                                limit_train_batches=5, limit_val_batches=5, limit_test_batches=5)
    elif cfg.training_type == "slurm":
        raise NotImplementedError("Slurm training not implemented")
    elif cfg.training_type == "default":
        fit_trainer = L.Trainer(max_epochs=cfg.total_epochs, accelerator=accelerator, devices=1, logger=logger,
                                callbacks=callbacks, default_root_dir=str(lightning_module.model_dir),
                                check_val_every_n_epoch=cfg.validation_interval, log_every_n_steps=1,
                                enable_checkpointing=len(callbacks) > 0)
    else:
        raise NotImplementedError("Default")

    print(f"Training ID: {lightning_module.training_id}\n\t Initial learning rate: {cfg.initial_lr}\n\t"
          f"LR scheduler: {cfg.scheduler_cfg['type']}\n\t Total epochs: {cfg.total_epochs}\n\t"
          f"Validation interval: {cfg.validation_interval}\n\t Loss type: {cfg.loss_type}\n\t"
          f"Using wandb: {cfg.use_wandb}\n\t Model directory: {lightning_module.model_dir}")

    # ================== Train ==================
    fit_trainer.fit(model=lightning_module, train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader, ckpt_path=cfg.resume_from)

    return lightning_module, fit_trainer, node_stats, edge_stats, target_stats, dt, validation_dataloader


@hydra.main(version_base=None, config_path="../../conf/train", config_name="train")
def main(cfg):
    """
    Args:
        cfg: omegaconf.DictConfig
            Hydra-composed configuration.
        return: None
            Runs train and final evaluation.
    """
    train_cfg = TrainYaml(cfg)

    with open(Path(train_cfg.dataset_dir) / "config.yaml", "r") as file:
        preprocess_dict = yaml.safe_load(file)

    preprocess_cfg = PreprocessYaml(preprocess_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =================== Train ===================
    lightning_module, fit_trainer, node_stats, edge_stats, target_stats, dt, validation_dataloader = \
        train(train_cfg, preprocess_cfg, device, cfg, preprocess_dict)

    # =================== Final validation ===================
    if validation_dataloader is not None:
        print("=" * 60 + "\nValidation dataset plots and evaluation...\n" + "=" * 60 + "\n")
        lightning_module.set_test_context(test_loader=validation_dataloader, split_name="validation", log_media=True)
        fit_trainer.test(model=lightning_module, dataloaders=validation_dataloader, ckpt_path=None)

    # =================== Testing ===================
    print("=" * 60 + "\nTest dataset plots and evaluation...\n" + "=" * 60 + "\n")
    # Test dataset
    test_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.test_dir, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=train_cfg.num_workers, 
                                 persistent_workers=train_cfg.num_workers > 0)
    # Test
    lightning_module.set_test_context(test_loader=test_dataloader, split_name="test", log_media=True)
    fit_trainer.test(model=lightning_module, dataloaders=test_dataloader, ckpt_path=None)

if __name__ == "__main__":
    main()