"""
train.py
Main training script.
Final test/validation media may include scatter plots with R^2 from scikit-learn (see validation/plot_rollout.py).
"""

import os
import time
import yaml
import logging
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
import matplotlib
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
from src.gnn.misc_helpers.yamlobject import PreprocessYaml, TrainYaml, cfg_for_wandb, train_cfg_yaml_for_wandb
from src.gnn.preprocessing.inmemorytimestepdataset import InMemoryTimeStepDataset, InMemoryNoisedDataset
from src.gnn.preprocessing.stats import load_stats, load_theta_max, assert_stats_finite
from src.gnn.preprocess import load_selected_trajectories
from src.gnn.training.scheduler import scheduler_factory
from src.gnn.training.trainer import LitModule
from src.gnn.misc_helpers.custom_pbar import ShortMetricProgressBar
from src.gnn.training.optimizers import optimizer_factory

matplotlib.use("Agg")
plt.ioff()
THETA_MAX_MARGIN = 1e-6


def resolve_trainer_precision(accelerator, configured_precision):
    """
    Maps YAML precision to a value safe for the chosen Lightning accelerator.

    Args:
        accelerator: str
            Lightning accelerator, e.g. "cpu" or "gpu".
        configured_precision: str
            Value from train config (e.g. bf16-mixed, 16-mixed, 32-true).
        return: str
            Precision string passed to L.Trainer; mixed modes downgrade to 32-true on CPU.
    """
    if accelerator == "cpu" and configured_precision in ("16-mixed", "bf16-mixed"):
        return "32-true"
    return configured_precision


def maybe_compile_model(model, train_cfg):
    """
    Optionally wraps the model with torch.compile and enables graph-break logs.
    On CUDA, uses mode reduce-overhead (Inductor CUDA graphs / lower launch overhead);
    on CPU, uses mode default.

    Args:
        model: torch.nn.Module
            Instantiated model to compile.
        train_cfg: TrainYaml
            Training configuration containing the torch_compile flag.
        return: torch.nn.Module
            Original model if compilation is disabled, otherwise compiled model.
    """
    if not train_cfg.torch_compile:
        return model
    # Enable verbose Dynamo diagnostics so graph breaks are reported with details.
    torch._dynamo.config.verbose = True
    torch._logging.set_logs(graph_breaks=True, dynamo=logging.INFO)
    compile_mode = "reduce-overhead" if torch.cuda.is_available() else "default"
    return torch.compile(model, mode=compile_mode, fullgraph=True)

def create_model(train_cfg, preprocess_cfg, device):
    if train_cfg.model_name == "MGN":
        # node_channels, edge_channels, encoder_latent_size, decoder_latent_size, processor_latent_size,
        # num_msgs, num_layers_graphnetblock, layer_norm_enable_encoder, layer_norm_enable_decoder,
        # layer_norm_enable_processor
        return MeshGraphNet(node_channels=preprocess_cfg.node_channels, edge_channels=preprocess_cfg.edge_channels,
                            encoder_latent_size=train_cfg.latent_dim, decoder_latent_size=train_cfg.latent_dim,
                            processor_latent_size=train_cfg.latent_dim,
                            num_msgs=train_cfg.num_messages,
                            num_layers_graphnetblock=train_cfg.num_layers_graphnetblock,
                            layer_norm_enable_encoder=train_cfg.layer_norm_enable_encoder,
                            layer_norm_enable_decoder=train_cfg.layer_norm_enable_decoder,
                            layer_norm_enable_processor=train_cfg.layer_norm_enable_processor)
    elif train_cfg.model_name == "clofnet":
        raise NotImplementedError("CLOFNet not implemented")
    else:
        raise NotImplementedError(f"Model type {train_cfg.model_name} not implemented")

def train(train_cfg, preprocess_cfg, device, train_dict, preprocess_dict):
    """
    Args:
        train_cfg: TrainYaml
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
        Lightning module, Lightning trainer, stats, dt, validation dataset, and validation dataloader.
    """
    model_root = Path(os.getcwd()) / "saved_models"
    model_root.mkdir(parents=True, exist_ok=True)

    # ========================= Stats (loaded first, needed by noised dataset) ===================
    print(f"Loading stats from {train_cfg.train_dir}...")
    node_stats, edge_stats, target_stats = load_stats(train_cfg.train_dir)
    assert_stats_finite(node_stats, edge_stats, target_stats)
    dt = preprocess_cfg.dt
    print(f"\tNode features: mean shape = {node_stats.mean.shape}, std shape = {node_stats.std.shape}")
    print(f"\tEdge features: mean shape = {edge_stats.mean.shape}, std shape = {edge_stats.std.shape}")
    print(f"\tTarget features: mean shape = {target_stats.mean.shape}, std shape = {target_stats.std.shape}")
    theta_max = None
    if "curvature" in train_cfg.residual_list:
        theta_max = load_theta_max(train_cfg.train_dir) + THETA_MAX_MARGIN
        print(f"\tLoaded theta_max threshold = {theta_max}")

    # ========================= Dataset stuff ===================
    validation_dataset = None
    validation_dataloader = None
    print("Train dataset:")
    # device = "cpu" hardcoded for performance reason
    if train_cfg.noise_scale_train > 0:
        print(f"\tNoise injection enabled: scale={train_cfg.noise_scale_train}, gamma={train_cfg.noise_gamma}")
        train_dataset = InMemoryNoisedDataset(
            sample_dir=train_cfg.train_dir, device="cpu",
            noise_scale=train_cfg.noise_scale_train, gamma=train_cfg.noise_gamma,
            dt=dt, node_stats=node_stats, edge_stats=edge_stats, target_stats=target_stats,
            base_seed=train_cfg.noise_seed, preprocess_cfg=preprocess_cfg,
            noise_also_prev=train_cfg.noise_also_prev,
            noise_prev_scale=train_cfg.noise_prev_scale_train)
    else:
        train_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.train_dir, device="cpu")
    # PyTorch's docs say it is generally not recommended to return CUDA tensors from multi-process data loading, and
    # recommend CPU tensors plus pin_memory=True for faster transfer instead.
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                                  num_workers=train_cfg.num_workers, persistent_workers=train_cfg.num_workers > 0,
                                  pin_memory=train_cfg.pin_memory)
    if train_cfg.enable_val:
        print("Validation dataset:")
        # device = "cpu" hardcoded for performance reason
        validation_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.val_dir, device="cpu")
        # PyTorch's docs say it is generally not recommended to return CUDA tensors from multi-process data loading, and
        # recommend CPU tensors plus pin_memory=True for faster transfer instead.
        # NOTE: batch size does not influence rollout since we use a separate dataloader (internal)
        val_batch_size = train_cfg.batch_size * 2
        validation_dataloader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=False,
                                           num_workers=train_cfg.num_workers, persistent_workers=train_cfg.num_workers > 0,
                                           pin_memory=train_cfg.pin_memory)

    # ========================= Model and Lightning wrapper =================
    #torch.manual_seed(cfg.seed)
    model = create_model(train_cfg, preprocess_cfg, device)
    model = maybe_compile_model(model, train_cfg)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"\tTotal parameters: {total_params:,}")
    # Set optimizer and scheduler
    optimizer = optimizer_factory(model, train_cfg)
    # optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.initial_lr)
    scheduler = scheduler_factory(optimizer, train_cfg.scheduler_cfg)
    training_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Set Lightning Wrapper of Model.
    wandb_train_cfg_yaml = train_cfg_yaml_for_wandb(train_dict) if train_cfg.use_wandb else ""
    lightning_module = LitModule(model=model, optimizer=optimizer, scheduler=scheduler, loss_type=train_cfg.loss_type,
                                 preprocess_cfg=preprocess_cfg, training_id=training_id, model_root=model_root,
                                 len_train_dataloader=len(train_dataloader),
                                 enable_wandb_watch=(not train_cfg.torch_compile) and train_cfg.enable_wandb_watch,
                                 train_cfg=train_cfg, theta_max=theta_max, wandb_train_cfg_yaml=wandb_train_cfg_yaml)
    lightning_module.set_rollout_context(validation_dataset=validation_dataset, node_stats=node_stats,
                                         edge_stats=edge_stats, target_stats=target_stats, dt=dt)
    # Save configs via hydra
    OmegaConf.save(train_dict, lightning_module.model_dir / "train.yaml", resolve=True)
    OmegaConf.save(preprocess_dict, lightning_module.model_dir / "preprocess.yaml", resolve=True)

    # =================== Wandb logger ===================
    if train_cfg.use_wandb:
        # One important detail for your snippet: in Lightning, WandbLogger(log_model=True) or
        # WandbLogger(log_model="all") cannot be combined with offline=True; that configuration raises a
        # MisconfigurationException.
        logger = WandbLogger(project="beam_deformation", name=training_id, save_dir=str(lightning_module.model_dir),
                             log_model=False, offline=True, config=cfg_for_wandb(train_dict))
    else:
        logger = False

    # ============ Set checkpointing rules (note: also scheduler is saved properly) =============
    callbacks = []
    checkpoint_str_dir = str(lightning_module.model_dir / "checkpoints")
    # ALlocate a callback for checkpoints of model every time I get the "best model"
    if train_cfg.enable_val:
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_str_dir, filename="best",
                                         monitor="val/overall_relative_mae", mode="min",
                                         save_top_k=1))
    # periodic save every cfg.checkpoint_interval epoch
    if train_cfg.checkpoint_interval > 0:
        # save_top_k = -1 --> do not take into account the loss, save anyway
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_str_dir, filename="epoch-{epoch:04d}",
                                         every_n_epochs=train_cfg.checkpoint_interval, save_top_k=-1))

    callbacks.append(ShortMetricProgressBar(refresh_rate=1, process_position=0, leave=False))

    # ======================= Lightning Trainer ===================
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer_precision = resolve_trainer_precision(accelerator, train_cfg.precision)
    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    if train_cfg.profiling:
        profiler = SimpleProfiler(dirpath=str(lightning_module.model_dir / "profiles"), filename="simple_profiler")
    else:
        profiler = None
    if train_cfg.training_type == "sanity":
        fit_trainer = L.Trainer(max_epochs=train_cfg.total_epochs, logger=logger,
                                callbacks=callbacks, default_root_dir=str(lightning_module.model_dir),
                                check_val_every_n_epoch=train_cfg.validation_interval, log_every_n_steps=1,
                                enable_checkpointing=len(callbacks) > 0,
                                limit_train_batches=5, limit_val_batches=5, limit_test_batches=5,
                                accelerator=accelerator, devices=train_cfg.devices,
                                precision=trainer_precision,
                                profiler=profiler)
    elif train_cfg.training_type == "slurm":
        if not torch.cuda.is_available():
            raise RuntimeError("SLURM GPU training requested, but CUDA is not available.")
        fit_trainer = L.Trainer(max_epochs=train_cfg.total_epochs, logger=logger,
                                callbacks=callbacks, default_root_dir=str(lightning_module.model_dir),
                                check_val_every_n_epoch=train_cfg.validation_interval, log_every_n_steps=1,
                                enable_checkpointing=len(callbacks) > 0,
                                accelerator=accelerator,
                                 devices=train_cfg.devices, num_nodes=1,
                                 # strategy="ddp",
                                precision=trainer_precision,
                                profiler=profiler) # device stuff
    elif train_cfg.training_type == "default":
        fit_trainer = L.Trainer(max_epochs=train_cfg.total_epochs, logger=logger,
                                callbacks=callbacks, default_root_dir=str(lightning_module.model_dir),
                                check_val_every_n_epoch=train_cfg.validation_interval, log_every_n_steps=1,
                                enable_checkpointing=len(callbacks) > 0,
                                accelerator=accelerator,
                                devices=train_cfg.devices,
                                precision=trainer_precision,
                                profiler=profiler)
    else:
        raise NotImplementedError("Default")

    print(f"Training ID: {lightning_module.training_id}\n\t Initial learning rate: {train_cfg.initial_lr}\n\t"
          f"LR scheduler: {train_cfg.scheduler_cfg['type']}\n\t Total epochs: {train_cfg.total_epochs}\n\t"
          f"Validation interval: {train_cfg.validation_interval}\n\t Loss type: {train_cfg.loss_type}\n\t"
          f"Using wandb: {train_cfg.use_wandb}\n\t Model directory: {lightning_module.model_dir}")

    # ================== Train ==================
    fit_trainer.fit(model=lightning_module, train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader, ckpt_path=train_cfg.resume_from)

    return lightning_module, fit_trainer, node_stats, edge_stats, target_stats, dt, validation_dataset, validation_dataloader


@hydra.main(version_base=None, config_path="../../conf/train", config_name="train")
def main(cfg):
    print("\n"+ "=" * 80 + "\n\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n" + "=" * 80 + "\n\n")

    train_cfg = TrainYaml(cfg)
    seed_everything(train_cfg.seed, workers=True)  # Recommended by Lightning docs

    with open(Path(train_cfg.dataset_dir) / "config.yaml", "r") as file:
        preprocess_dict = yaml.safe_load(file)

    preprocess_cfg = PreprocessYaml(preprocess_dict)
    # Resolve extra_info_dir: prefer train config, fall back to preprocess config
    if train_cfg.extra_info_dir is None and preprocess_cfg.extra_info_dir is not None:
        train_cfg.extra_info_dir = preprocess_cfg.extra_info_dir
    if train_cfg.extra_info_dir is not None:
        print(f"Extra info dir: {train_cfg.extra_info_dir}")

    # Resolve selected trajectory indices: YAML overrides take priority, otherwise read from JSON
    selected_trajs = load_selected_trajectories(train_cfg.extra_info_dir)
    if train_cfg.selected_validation_trajectories is None:
        train_cfg.selected_validation_trajectories = selected_trajs.get("selected_validation_trajectories", [])
    if train_cfg.selected_test_trajectories is None:
        train_cfg.selected_test_trajectories = selected_trajs.get("selected_test_trajectories", [])
    if train_cfg.selected_validation_trajectories:
        print(f"Selected validation trajectories: {train_cfg.selected_validation_trajectories}")
    if train_cfg.selected_test_trajectories:
        print(f"Selected test trajectories: {train_cfg.selected_test_trajectories}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and train_cfg.training_type == "slurm":
        raise RuntimeError("CUDA is not available.")
    print(f"Using device: {device}")

    # =================== Train + validation ===================
    lightning_module, fit_trainer, node_stats, edge_stats, target_stats, dt, validation_dataset, validation_dataloader = \
        train(train_cfg, preprocess_cfg, device, cfg, preprocess_dict)

    # =================== Final validation ===================
    if validation_dataset is not None:
        print("=" * 60 + "\nValidation dataset plots and evaluation...\n" + "=" * 60 + "\n")
        lightning_module.set_test_context(test_dataset=validation_dataset, split_name="validation")
        fit_trainer.test(model=lightning_module, dataloaders=validation_dataloader, ckpt_path=None)

    # =================== Testing ===================
    print("=" * 60 + "\nTest dataset plots and evaluation...\n" + "=" * 60 + "\n")
    # device = "cpu" hardcoded for performance reason
    test_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.test_dir, device="cpu")
    # pin memory = True for faster transfer instead.
    # We don't care about batch size since we're really just doing it in the end AND we're doing mostly rolllout
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=train_cfg.num_workers,
                                 persistent_workers=train_cfg.num_workers > 0,
                                 pin_memory=train_cfg.pin_memory)
    lightning_module.set_test_context(test_dataset=test_dataset, split_name="test")
    fit_trainer.test(model=lightning_module, dataloaders=test_dataloader, ckpt_path=None)

if __name__ == "__main__":
    main()
