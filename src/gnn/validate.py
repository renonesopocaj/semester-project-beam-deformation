"""
validate.py
Standalone evaluation script. Loads a Lightning checkpoint from a completed
training run and runs the same final evaluation path (test_step + on_test_epoch_end)
that train.py uses, producing rollout metrics and diagnostic plots.

Configuration is read from conf/validate/validate.yaml.
"""

from pathlib import Path

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml
from torch_geometric.loader import DataLoader

from src.gnn.misc_helpers.yamlobject import PreprocessYaml, TrainYaml
from src.gnn.dataloaders.inmemorytimestepdataset import InMemoryTimeStepDataset
from src.gnn.preprocessing.stats import load_stats, load_theta_max
from src.gnn.train import create_model, resolve_trainer_precision, _load_traj_metadata
from src.gnn.training.scheduler import scheduler_factory
from src.gnn.training.trainer import LitModule

matplotlib.use("Agg")
plt.ioff()
THETA_MAX_MARGIN = 1e-6


def _load_split(split_name, train_cfg):
    """
    Args:
        split_name: str
            "validation" or "test".
        train_cfg: TrainYaml
            Training config with dataset directory paths.
        return: (InMemoryTimeStepDataset, DataLoader)
    """
    split_dir = train_cfg.val_dir if split_name == "validation" else train_cfg.test_dir
    dataset = InMemoryTimeStepDataset(sample_dir=split_dir, device="cpu")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=train_cfg.num_workers,
                            persistent_workers=train_cfg.num_workers > 0,
                            pin_memory=True)
    return dataset, dataloader


def _resolve_in_depth(in_depth_cfg, num_trajectories):
    """
    Resolves the in_depth config value to a concrete list of trajectory indices.

    Args:
        in_depth_cfg: str or list
            "all" or a list of int trajectory indices.
        num_trajectories: int
            Total number of trajectories in the dataset.
        return: list[int]
            Resolved list of trajectory indices.
    """
    if in_depth_cfg == "all":
        return list(range(num_trajectories))
    return list(in_depth_cfg)


def main():
    with open("conf/validate/validate.yaml", "r") as fh:
        cfg = yaml.safe_load(fh)

    training_id = cfg["training_id"]
    model_root = Path(cfg.get("model_root", "saved_models"))
    checkpoint_name = cfg.get("checkpoint_name", "best")
    in_depth_test = cfg.get("in_depth_test")
    in_depth_val = cfg.get("in_depth_val")

    # Determine which splits to evaluate based on non-null fields
    splits_to_run = []
    if in_depth_val is not None:
        splits_to_run.append(("validation", in_depth_val))
    if in_depth_test is not None:
        splits_to_run.append(("test", in_depth_test))
    if not splits_to_run:
        print("No splits configured. Set in_depth_test or in_depth_val in conf/validate/validate.yaml.")
        return

    model_dir = model_root / training_id
    checkpoint_path = model_dir / "checkpoints" / f"{checkpoint_name}.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load the configs that were saved during training
    with open(model_dir / "train.yaml", "r") as fh:
        train_cfg = TrainYaml(yaml.safe_load(fh))
    with open(model_dir / "preprocess.yaml", "r") as fh:
        preprocess_cfg = PreprocessYaml(yaml.safe_load(fh))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training ID: {training_id}")
    print(f"Checkpoint: {checkpoint_path}")

    # Build model, optimizer, scheduler (optimizer/scheduler needed by LitModule constructor
    # but won't be stepped during eval; their state is overwritten by the checkpoint)
    template_dataset = InMemoryTimeStepDataset(sample_dir=train_cfg.train_dir, device="cpu")
    graph_template = template_dataset[0]
    target_mode = getattr(preprocess_cfg, "target_mode", "von_mises")
    stress_out_dim = 6 if target_mode == "full_stress" else 1
    model = create_model(train_cfg, preprocess_cfg, device, graph_template, stress_out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.initial_lr)
    scheduler = scheduler_factory(optimizer, train_cfg.scheduler_cfg)

    theta_max = None
    if "curvature" in train_cfg.residual_list:
        theta_max = load_theta_max(train_cfg.train_dir) + THETA_MAX_MARGIN

    lightning_module = LitModule(
        model=model, optimizer=optimizer, scheduler=scheduler,
        loss_type=train_cfg.loss_type, preprocess_cfg=preprocess_cfg,
        training_id=training_id, model_root=model_root,
        len_train_dataloader=1, enable_wandb_watch=False,
        train_cfg=train_cfg, theta_max=theta_max, wandb_train_cfg_yaml="")

    # Write standalone eval outputs to a separate subdirectory
    lightning_module.model_evaluation_dir = model_dir / "standalone_evaluation"
    lightning_module.model_evaluation_dir.mkdir(parents=True, exist_ok=True)
    # Standalone eval always produces plots
    lightning_module.log_test_media = True

    # Load stats and register StatsModule submodules (must happen before ckpt_path loading)
    node_stats, edge_stats, target_stats = load_stats(train_cfg.train_dir)
    dt = preprocess_cfg.dt
    lightning_module.set_rollout_context(
        validation_dataset=None, node_stats=node_stats,
        edge_stats=edge_stats, target_stats=target_stats, dt=dt)

    # Lightweight eval trainer (no logger, single device)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    eval_precision = resolve_trainer_precision(accelerator, train_cfg.precision)
    eval_trainer = L.Trainer(logger=False, accelerator=accelerator, devices=1, precision=eval_precision)

    for split_name, in_depth_cfg in splits_to_run:
        print("\n" + "=" * 60)
        print(f"Evaluating split: {split_name}")
        print("=" * 60 + "\n")
        dataset, dataloader = _load_split(split_name, train_cfg)
        split_dir = train_cfg.val_dir if split_name == "validation" else train_cfg.test_dir

        # Resolve "all" to the full trajectory range now that we know num_trajectories
        selected = _resolve_in_depth(in_depth_cfg, dataset.num_trajectories)
        if split_name == "test":
            lightning_module.selected_test_trajectories = selected
        else:
            lightning_module.selected_validation_trajectories = selected

        traj_metadata = _load_traj_metadata(split_dir, train_cfg.extra_info_dir)
        lightning_module.set_test_context(test_dataset=dataset, split_name=split_name,
                                          traj_metadata=traj_metadata)
        eval_trainer.test(model=lightning_module, dataloaders=dataloader,
                          ckpt_path=str(checkpoint_path))


if __name__ == "__main__":
    main()
