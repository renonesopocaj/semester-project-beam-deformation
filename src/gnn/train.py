"""
train.py

Main training script.
Final test/validation media may include scatter plots with R^2 from scikit-learn (see validation/plot_rollout.py).
"""

import json
import os
import time
import yaml
import logging
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.profilers import SimpleProfiler
import matplotlib
import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from src.gnn.models.gunet.gunet import GUNet
from src.gnn.models.mag.mag import MAGNet
from src.gnn.models.meshgraphnet.meshgraphnet import MeshGraphNet
from src.gnn.models.transolver.transolvernet import TransolverNet
from src.gnn.models.clofnet.clofnet import (ClofNetDirectAcceleration, ClofNetExtraLayer, ClofNetRecurrent,
                                            ClofNetSeparateMessages)
from src.gnn.preprocessing.build_features import probe_feature_layout
from src.gnn.misc_helpers.yamlobject import PreprocessYaml, TrainYaml, cfg_for_wandb, train_cfg_yaml_for_wandb
from src.gnn.misc_helpers.misc import (human_readable_storage, module_param_and_buffer_storage_bytes,
                                       pyg_data_tensor_storage_bytes)
from src.gnn.dataloaders.inmemorytimestepdataset import InMemoryTimeStepDataset, InMemoryNoisedDataset
from src.gnn.preprocessing.stats import load_stats, load_theta_max, assert_stats_finite
from src.gnn.preprocess import load_selected_trajectories
from src.gnn.training.scheduler import scheduler_factory
from src.gnn.training.trainer import LitModule
from src.gnn.misc_helpers.custom_pbar import ShortMetricProgressBar
from src.gnn.training.optimizers import optimizer_factory

matplotlib.use("Agg")
plt.ioff()
THETA_MAX_MARGIN = 1e-6
MAX_SELECTED_TIMESTEPS = 400


def _load_traj_metadata(split_dir, extra_info_dir):
    """
    Build a map traj_idx -> metadata dict for in-depth plot subtitles.

    Prefers processed split_dir/trajectory_params.json (fx, fy, fz, L, W, D, ood_features).
    If missing, falls back to extra_info/test_ood_features.json for the test split only (ood_features only).

    Args:
        split_dir: str or Path
            Processed split directory, e.g. .../large/test.
        extra_info_dir: str or Path or None
            Processed dataset extra_info directory, or None.
    return: dict
        Keys are int trajectory indices; values are dicts suitable for _format_traj_metadata_text.
    """
    split_path = Path(split_dir)
    params_path = split_path / "trajectory_params.json"
    if params_path.exists():
        with open(params_path) as fh:
            raw_params = json.load(fh)
        by_idx = {}
        for _fname, entry in raw_params.items():
            by_idx[int(entry["traj_idx"])] = entry
        return by_idx
    if extra_info_dir is None:
        return {}
    if split_path.name != "test":
        return {}
    ood_path = Path(extra_info_dir) / "test_ood_features.json"
    if not ood_path.exists():
        return {}
    with open(ood_path) as fh:
        ood_raw = json.load(fh)
    by_idx = {}
    for key_str, ood_list in ood_raw.items():
        by_idx[int(key_str)] = {"ood_features": ood_list}
    return by_idx


def _selected_eval_rollout_limit(selected_trajs, fallback_limit):
    """
    Resolve the selected trajectory rollout length used in final evaluation and cap it to MAX_SELECTED_TIMESTEPS.

    Args:
        selected_trajs: dict
            Parsed selected_trajectories.json.
        fallback_limit: int
            Fallback value when selected trajectory step information is not present.
    return: int
        Max number of rollout steps for selected trajectories in final evaluation.
    """
    raw_steps = selected_trajs.get("selected_num_steps", selected_trajs.get("num_steps", fallback_limit))
    return min(int(raw_steps), MAX_SELECTED_TIMESTEPS)


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


def maybe_compile_model(model, train_cfg, protect_from_oom):
    """
    Optionally wraps the model with torch.compile and enables graph-break logs.
    On CUDA, uses mode reduce-overhead (Inductor CUDA graphs / lower launch overhead);
    on CPU, uses mode default.

    Args:
        model: torch.nn.Module
            Instantiated model to compile.
        train_cfg: TrainYaml
            Training configuration containing the torch_compile flag.
        protect_from_oom: bool
            When True, compile without reduce-overhead mode on CUDA to lower peak VRAM.
        return: torch.nn.Module
            Original model if compilation is disabled, otherwise compiled model.
    """
    if not train_cfg.torch_compile:
        return model
    # Enable verbose Dynamo diagnostics so graph breaks are reported with details.
    torch._dynamo.config.verbose = True
    torch._logging.set_logs(graph_breaks=True, dynamo=logging.INFO)

    if protect_from_oom:
        # Important for your OOM case:
        # torch._inductor.config.triton.cudagraphs = False
        # Safer than "reduce-overhead" for memory.
        compiled_model = torch.compile(model, fullgraph=True)
    else:
        compile_mode = "reduce-overhead" if torch.cuda.is_available() else "default"
        compiled_model = torch.compile(model, mode=compile_mode, fullgraph=True)
    return compiled_model

def create_model(train_cfg, preprocess_cfg, device, graph_template):
    if train_cfg.model_name == "MGN":
        # node_channels, edge_channels, encoder_latent_size, decoder_latent_size, processor_latent_size,
        # num_msgs, num_layers_graphnetblock, layer_norm_enable_encoder, layer_norm_enable_decoder,
        # layer_norm_enable_processor
        return MeshGraphNet(node_channels=preprocess_cfg.node_channels, edge_channels=preprocess_cfg.edge_channels,
                            encoder_latent_size=train_cfg.hidden_processor_dim, encoder_decoder_latent_size=train_cfg.hidden_processor_dim,
                            processor_latent_size=train_cfg.hidden_processor_dim, num_msgs=train_cfg.num_messages,
                            num_layers_graphnetblock=train_cfg.num_layers_graphnetblock,
                            layer_norm_enable_encoder=train_cfg.layer_norm_enable_encoder,
                            layer_norm_enable_decoder=train_cfg.layer_norm_enable_decoder,
                            layer_norm_enable_processor=train_cfg.layer_norm_enable_processor,
                            num_layers_heads=train_cfg.num_layers_heads,
                            num_layers_encoder=train_cfg.num_layers_encoder, dropout_encoder=train_cfg.dropout_encoder,
                            dropout_head=train_cfg.dropout_head, dropout_processor=train_cfg.dropout_processor,
                            activation_checkpointing=train_cfg.activation_checkpointing)

    elif train_cfg.model_name == "clofnet":
        _, _, split_idx_node, split_idx_edge = probe_feature_layout(preprocess_cfg, device)
        act_name = str(train_cfg.clofnet_activation).lower()
        if act_name == "silu":
            def clofnet_act_fn_factory():
                return torch.nn.SiLU()
        elif act_name == "leakyrelu":
            neg_slope = float(train_cfg.leakyrelu_negative_slope)
            def clofnet_act_fn_factory():
                return torch.nn.LeakyReLU(negative_slope=neg_slope)
        elif act_name == "relu":
            def clofnet_act_fn_factory():
                return torch.nn.ReLU()
        else:
            raise ValueError("Unknown clofnet_activation: %s" % train_cfg.clofnet_activation)
        in_dim_node_embed = int(preprocess_cfg.node_channels) - int(split_idx_node)
        in_dim_edge_scalars = int(preprocess_cfg.edge_channels) - int(split_idx_edge)
        common_kwargs = {
            "in_dim_node_embed": in_dim_node_embed,
            "in_dim_edge_scalars": in_dim_edge_scalars,
            "hidden_dim": int(train_cfg.hidden_dim),
            "fuse_hid_dim": int(train_cfg.embedding_edge_dim),
            "act_fn_factory": clofnet_act_fn_factory,
            "n_layers": int(train_cfg.n_layers),
            "coords_weight": float(train_cfg.coords_weight),
            "recurrent": bool(train_cfg.recurrent),
            "enable_norm_diff": bool(train_cfg.enable_norm_diff),
            "tanh": bool(train_cfg.tanh),
            "attention_enable": bool(train_cfg.attention_enable),
            "clamp_enable": bool(train_cfg.clamp_enable),
            "vel_mlp_dim_out": int(train_cfg.out_coord_mlp_dim),
            "split_idx_node": int(split_idx_node),
            "split_idx_edge": int(split_idx_edge),
            "node_channels": int(preprocess_cfg.node_channels),
            "edge_channels": int(preprocess_cfg.edge_channels),
        }
        variant = str(train_cfg.clofnet_variant).lower()
        if variant == "recurrent":
            return ClofNetRecurrent(**common_kwargs)
        if variant == "separate":
            return ClofNetSeparateMessages(**common_kwargs)
        if variant == "extra_layer":
            return ClofNetExtraLayer(**common_kwargs)
        if variant in {"direct", "version5", "v5"}:
            return ClofNetDirectAcceleration(**common_kwargs)
        raise ValueError("Unknown clofnet_variant: %s" % train_cfg.clofnet_variant)
    elif train_cfg.model_name == "gunet":
        return GUNet(in_channels=preprocess_cfg.node_channels, hidden_channels=train_cfg.hidden_channels,
                     layer_norm=train_cfg.layer_norm, depth=train_cfg.depth,
                     pool_ratios=train_cfg.pool_ratios, sum_res=train_cfg.sum_res,
                     enable_embedder=train_cfg.enable_embedder, encoder_layers=train_cfg.encoder_layers,
                     decoder_layers=train_cfg.decoder_layers, dropout_encoder=train_cfg.dropout_encoder,
                     dropout_head=train_cfg.dropout_head)
    elif train_cfg.model_name == "magnet":
        return MAGNet(preprocess_cfg.node_channels, 4, train_cfg.hidden_channels, train_cfg.num_poolings,
                      train_cfg.layers_per_level, train_cfg.window_size, train_cfg.pool_seed, train_cfg.pool_aggr,
                      train_cfg.activation, graph_template)
    elif train_cfg.model_name == "transolver":
        return TransolverNet(
            graph_template,
            preprocess_cfg.node_channels,
            train_cfg.transolver_n_layers,
            train_cfg.transolver_n_hidden,
            train_cfg.transolver_dropout,
            train_cfg.transolver_n_head,
            train_cfg.transolver_time_input,
            train_cfg.transolver_act,
            train_cfg.transolver_mlp_ratio,
            train_cfg.transolver_slice_num,
            train_cfg.transolver_ref,
            train_cfg.transolver_unified_pos,
        )
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
    if "volume" in train_cfg.residual_list:
        print(f"\tVolume residual enabled with weight = {train_cfg.volume_loss_weight}")

    # ========================= Dataset stuff ===================
    validation_dataset = None
    validation_dataloader = None
    print("Train dataset:")
    # device = "cpu" hardcoded for performance reason
    if train_cfg.noise_scale_train > 0:
        print(f"\tNoise injection enabled: scale={train_cfg.noise_scale_train}, gamma={train_cfg.noise_gamma}")
        # Per-epoch noise draws: LitModule.on_train_epoch_start calls set_noise_epoch on this dataset.
        train_dataset = InMemoryNoisedDataset(
            sample_dir=train_cfg.train_dir, device="cpu",
            noise_scale=train_cfg.noise_scale_train, gamma=train_cfg.noise_gamma,
            dt=dt, node_stats=node_stats, edge_stats=edge_stats, target_stats=target_stats,
            base_seed=train_cfg.noise_seed, preprocess_cfg=preprocess_cfg,
            noise_also_prev=train_cfg.noise_also_prev,
            noise_prev_scale=train_cfg.noise_prev_scale_train,
            noise_feature_update_mode=train_cfg.noise_feature_update_mode,
            noise_per_epoch=train_cfg.noise_per_epoch)
    else:
        train_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.train_dir, device="cpu")
    # PyTorch's docs say it is generally not recommended to return CUDA tensors from multi-process data loading, and
    # recommend CPU tensors plus pin_memory=True for faster transfer instead.
    # With torch.compile, drop the last partial batch so every step sees identical batched tensor shapes (avoids
    # CUDA-graph / inductor issues on the smaller last batch under DDP).
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                                  num_workers=train_cfg.num_workers, persistent_workers=train_cfg.num_workers > 0,
                                  pin_memory=train_cfg.pin_memory,
                                  #drop_last=train_cfg.torch_compile
                                  )
    peek_batch = next(iter(train_dataloader))
    peek_batch_bytes = pyg_data_tensor_storage_bytes(peek_batch)
    peek_num_graphs = int(peek_batch.num_graphs)

    print(f"\n\nOne training batch (collated, tensor storage only): {human_readable_storage(peek_batch_bytes)} "
          f"({peek_batch_bytes} bytes), num_graphs={peek_num_graphs}, config batch_size={train_cfg.batch_size}")
    print("(Excludes activations, gradients, optimizer state, and allocator overhead; GPU VRAM is typically larger.)\n\n")
    if train_cfg.enable_val:
        print("Validation dataset:")
        # device = "cpu" hardcoded for performance reason
        validation_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.val_dir, device="cpu")
        # PyTorch's docs say it is generally not recommended to return CUDA tensors from multi-process data loading, and
        # recommend CPU tensors plus pin_memory=True for faster transfer instead.
        # NOTE: batch size does not influence rollout since we use a separate dataloader (internal)
        val_batch_size = train_cfg.batch_size
        validation_dataloader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=False,
                                           num_workers=train_cfg.num_workers, persistent_workers=train_cfg.num_workers > 0,
                                           pin_memory=train_cfg.pin_memory)

    # ========================= Model and Lightning wrapper =================
    #torch.manual_seed(cfg.seed)
    model_eager = create_model(train_cfg, preprocess_cfg, device, peek_batch)
    del peek_batch
    eager_model_bytes = module_param_and_buffer_storage_bytes(model_eager)
    print("\n\nModel info:")
    print(f"\tModel params+buffers before torch.compile: {human_readable_storage(eager_model_bytes)} "
          f"({eager_model_bytes} bytes)")
    model = maybe_compile_model(model_eager, train_cfg, train_cfg.protect_from_oom)
    wrapped_model_bytes = module_param_and_buffer_storage_bytes(model)
    print(f"\tModel params+buffers after torch.compile: {human_readable_storage(wrapped_model_bytes)} "
          f"({wrapped_model_bytes} bytes) "
          "(same tensor storage; torch.compile may still increase peak VRAM for kernels/caching.)")
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
        wb_run_display_name = training_id if train_cfg.wandb_run_name is None else train_cfg.wandb_run_name
        wb_offline_flag = train_cfg.wandb_offline
        if train_cfg.wandb_group is not None:
            logger = WandbLogger(project=train_cfg.wandb_project, name=wb_run_display_name, group=train_cfg.wandb_group,
                                 save_dir=str(lightning_module.model_dir), log_model=False, offline=wb_offline_flag,
                                 config=cfg_for_wandb(train_dict))
        else:
            logger = WandbLogger(project=train_cfg.wandb_project, name=wb_run_display_name,
                                 save_dir=str(lightning_module.model_dir), log_model=False, offline=wb_offline_flag,
                                 config=cfg_for_wandb(train_dict))
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
    elif train_cfg.training_type == "slurm" and train_cfg.devices != 1:
        if not torch.cuda.is_available():
            raise RuntimeError("SLURM GPU training requested, but CUDA is not available.")
        fit_trainer = L.Trainer(max_epochs=train_cfg.total_epochs, logger=logger,
                                callbacks=callbacks, default_root_dir=str(lightning_module.model_dir),
                                check_val_every_n_epoch=train_cfg.validation_interval, log_every_n_steps=1,
                                enable_checkpointing=len(callbacks) > 0,
                                accelerator=accelerator,
                                devices=train_cfg.devices, num_nodes=1,
                                strategy="ddp",
                                # strategy=DDPStrategy(find_unused_parameters=False, start_method="fork"),
                                precision=trainer_precision,
                                profiler=profiler) # device stuff
    elif train_cfg.training_type == "slurm" and train_cfg.devices == 1:
        if not torch.cuda.is_available():
            raise RuntimeError("SLURM GPU training requested, but CUDA is not available.")
        fit_trainer = L.Trainer(max_epochs=train_cfg.total_epochs, logger=logger,
                                callbacks=callbacks, default_root_dir=str(lightning_module.model_dir),
                                check_val_every_n_epoch=train_cfg.validation_interval, log_every_n_steps=1,
                                enable_checkpointing=len(callbacks) > 0,
                                accelerator=accelerator,
                                devices=train_cfg.devices, num_nodes=1,
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
        raise NotImplementedError(f"training_type={train_cfg.training_type}")

    print(f"Training ID: {lightning_module.training_id}\n\t Initial learning rate: {train_cfg.initial_lr}\n\t"
          f"LR scheduler: {train_cfg.scheduler_cfg['type']}\n\t Total epochs: {train_cfg.total_epochs}\n\t"
          f"Validation interval: {train_cfg.validation_interval}\n\t Loss type: {train_cfg.loss_type}\n\t"
          f"Using wandb: {train_cfg.use_wandb}\n\t Model directory: {lightning_module.model_dir}")

    # ================== Train ==================
    fit_trainer.fit(model=lightning_module, train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader, ckpt_path=train_cfg.resume_from)

    return (lightning_module, fit_trainer, node_stats, edge_stats, target_stats, dt, validation_dataset,
            validation_dataloader)


def run_train_pipeline(cfg):
    """
    Same flow as CLI main(...) but returns metrics when called from hyperparameter tooling (Optuna).

    Args:
        cfg: omegaconf.DictConfig

    return: tuple
        (best_val_relative_mae, final_metrics_plain_dict, wandb_run_id)
        First element is checkpoint-best val/overall_relative_mae from ModelCheckpoint;
        Second maps keys like final_validation/acc_all_rel_mae ... to floats after val+test phases;
        Third is W&B run id when use_wandb else None.
    """
    # Print train cfg
    print("\n"+ "=" * 80 + "\n\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n" + "=" * 80 + "\n\n")

    train_cfg = TrainYaml(cfg)
    seed_everything(train_cfg.seed, workers=True)  # Recommended by Lightning docs

    with open(Path(train_cfg.dataset_dir) / "config.yaml", "r") as file:
        preprocess_dict = yaml.safe_load(file)

    # Print preprocess cfg
    print("\n" + "=" * 80 + "\n\n" + OmegaConf.to_yaml(preprocess_dict, resolve=True) + "\n" + "=" * 80 + "\n\n")

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
    train_cfg.selected_eval_rollout_limit = _selected_eval_rollout_limit(selected_trajs,
                                                                         fallback_limit=MAX_SELECTED_TIMESTEPS)

    if train_cfg.selected_validation_trajectories:
        print(f"Selected validation trajectories: {train_cfg.selected_validation_trajectories}")
    if train_cfg.selected_test_trajectories:
        print(f"Selected test trajectories: {train_cfg.selected_test_trajectories}")
    print(f"Selected trajectory final-eval rollout limit: {train_cfg.selected_eval_rollout_limit}")

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
        val_traj_metadata = _load_traj_metadata(train_cfg.val_dir, train_cfg.extra_info_dir)
        lightning_module.set_test_context(test_dataset=validation_dataset, split_name="validation",
                                          traj_metadata=val_traj_metadata)
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
    test_traj_metadata = _load_traj_metadata(train_cfg.test_dir, train_cfg.extra_info_dir)
    lightning_module.set_test_context(test_dataset=test_dataset, split_name="test",
                                       traj_metadata=test_traj_metadata)
    fit_trainer.test(model=lightning_module, dataloaders=test_dataloader, ckpt_path=None)

    best_val_relative_mae = float("nan")
    if train_cfg.enable_val:
        for chk in fit_trainer.checkpoint_callbacks:
            monitored = getattr(chk, "monitor", "")
            if monitored == "val/overall_relative_mae":
                best_score_tensor = chk.best_model_score
                if best_score_tensor is not None:
                    best_val_relative_mae = float(best_score_tensor.detach().item())
                break

    final_metrics_plain = {}
    for key, value in fit_trainer.callback_metrics.items():
        if not (str(key).startswith("final_validation/") or str(key).startswith("final_test/")):
            continue
        if hasattr(value, "detach"):
            final_metrics_plain[str(key)] = float(value.detach().item())
        else:
            final_metrics_plain[str(key)] = float(value)

    wandb_run_id = None
    if train_cfg.use_wandb:
        wandb_run_id = str(lightning_module.logger.experiment.id)
        wandb.finish()

    return best_val_relative_mae, final_metrics_plain, wandb_run_id


@hydra.main(version_base=None, config_path="../../conf/train", config_name="train")
def main(cfg):
    run_train_pipeline(cfg)

if __name__ == "__main__":
    main()