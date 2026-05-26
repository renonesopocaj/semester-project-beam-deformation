# Inductive Biases and Learning Strategies for Learning Dynamics of Continuous Matter

This repository contains two main pipelines:

1. `src/fem_generation/`: generate raw beam deformation trajectories with FEniCS and save them as PyTorch Geometric graphs.
2. `src/gnn/`: preprocess those trajectories, train graph neural networks, and evaluate them with rollout diagnostics.

## End-to-end workflow

1. Generate raw FEM trajectories — see [FEM generation (documentation TODO)](#fem-generation-documentation-todo).
2. Preprocess into graph datasets: `python -m src.gnn.preprocess` (configured by `conf/preprocess/preprocess.yaml`).
3. Optional EDA: `python -m src.gnn.eda` (pick an `eda_*.yaml` config name under `conf/preprocess/`).
4. Train a model: `python -m src.gnn.train` (configured by `conf/train/train.yaml`).
5. Evaluate a saved checkpoint: fill in `conf/validate/validate.yaml` then run `python -m src.gnn.validate`.

Optional branches:
- **Hyperparameter sweeps**: `python -m src.gnn.hyperparam_opt -cn <study>` where `<study>` is any YAML name from `conf/hyperparam/`.
- **DeformingPlate comparison**: preprocess the external dataset with `src/gnn/preprocess_deforming_plate.py`, then run `src/gnn/run_dataset_comparison.py` to compute cross-dataset locality and frequency metrics.

## How to run

### Environment

The active environment specification lives in `environments/env_pip.yml` (conda env name: `semesterproject`). Older snapshots are archived under `environments/deprecated/`.

```bash
conda env create -f environments/env_pip.yml
conda activate semesterproject
```

### CLI examples

```bash
# Preprocess a dataset (override Hydra compose groups on the command line)
python -m src.gnn.preprocess dataset_path=biax_longer size=medium features=all

# Train with a named model preset
python -m src.gnn.train model=mgn_deep_midmsg \
    data.dataset_dir=dataset/processed/biax_longer/z_isotr_centr_nonlin/medium

# Run a hyperparameter sweep study
python -m src.gnn.hyperparam_opt -cn mgn_sweep

# Standalone evaluation (edit conf/validate/validate.yaml first)
python -m src.gnn.validate
```

### SLURM wrappers

All `run_*.sh` scripts are SLURM job scripts for the EPFL IZAR cluster:

| Script | Purpose |
|--------|---------|
| `run_fem_gen.sh` | FEM trajectory generation (multirun over geometry × force axes × size) |
| `run_preprocessing.sh` | Batch preprocessing multirun |
| `run_1gpu.sh` | Single-GPU training job |
| `run_2gpu.sh` | Two-GPU training job |
| `run_mgn_sweep.sh` | MeshGraphNet architecture sweep |
| `run_mgn_largedata_sweep.sh` | MGN sweep on the large dataset split |
| `run_force_geom_sweep.sh` | Force axes / geometry sweep |
| `run_feat_sweep.sh` | Feature ablation sweep |
| `run_noise_sweep.sh` | Noise injection hyperparameter sweep |
| `run_residual_sweep.sh` | Physics residual loss sweep |
| `run_loss_mgn_sweep.sh` | Loss function sweep |
| `run_gunet_sweep.sh` | Graph U-Net architecture sweep |
| `run_transolver_sweep.sh` | Transolver architecture sweep |
| `run_monitor_metric_sweep.sh` | Validation monitor metric sweep |

## Repository tree

```text
semester-project/
├── conf/
│   ├── fem_generation/         ← FEM Hydra compose (config.yaml + force_axes/, geometry/, size/, resolution/, others/)
│   ├── hyperparam/             ← one YAML per sweep study (18 studies)
│   ├── preprocess/             ← preprocess.yaml + size/, dataset_path/, features/, normalizer/, eda_*.yaml
│   ├── train/                  ← train.yaml + model/, computational/, devices/, data/,
│   │                              loss/, noise/, lr_scheduler/, residuals/, validation/, training/
│   └── validate/
│       └── validate.yaml
├── environments/
│   ├── env_pip.yml             ← active environment spec (semesterproject)
│   └── deprecated/             ← archived conda/pip snapshots
├── src/
│   ├── fem_generation/         ← FEM solver (details TBD — see §FEM TODO below)
│   └── gnn/
│       ├── preprocess.py
│       ├── train.py
│       ├── validate.py
│       ├── hyperparam_opt.py
│       ├── eda.py
│       ├── preprocess_deforming_plate.py
│       ├── run_dataset_comparison.py
│       ├── dataloaders/
│       ├── preprocessing/
│       ├── models/
│       │   ├── meshgraphnet/
│       │   ├── clofnet/        ← includes before_adaptation/ and unused/ (archived variants)
│       │   ├── gunet/
│       │   ├── mag/
│       │   ├── transolver/
│       │   ├── gatv2/
│       │   └── gatmgn/
│       ├── training/
│       ├── plots/
│       ├── mesh_numerical_helpers/
│       ├── eda_helpers/
│       ├── comparison_defplate/
│       └── misc_helpers/
├── tests/
│   ├── fem_generation/
│   └── gnn/                    ← 20+ test modules (see §Tests)
├── tests_artifacts/            ← runtime test outputs (not committed)
├── run_*.sh                    ← SLURM job scripts
└── .gitignore
```

Runtime-generated directories (`dataset/`, `saved_models/`, `hyperparam_outputs/`, `hydra/`, `wandb/`) are not part of the committed tree.

## What each part does

### `conf/fem_generation/`

Hydra compose tree for FEM trajectory generation. The entry config `config.yaml` composes four groups: `force_axes` (`single`, `biax`, `triax`), `geometry` (`longer`, `allgeoms`), `size` (`small`, `medium`, `large`), and `resolution`. Legacy monolithic configs are kept under `others/`. Full documentation deferred — see [FEM generation (documentation TODO)](#fem-generation-documentation-todo).

### `conf/preprocess/`

Hydra tree for `src/gnn/preprocess.py`. The compose root is `preprocess.yaml`; its defaults are:

```
size:          small
dataset_path:  biax_longer
normalizer:    z_isotropic
features:      all
```

Top-level fields include `target_mode` (`von_mises` | `full_stress`), `time_discretization_method` (`second_order_central` | `newmark`), and an `edge_augmentation` block for offline topology augmentation.

#### `conf/preprocess/dataset_path/`

Presets follow the naming convention `{force_axes}_{geometry}` (e.g. `biax_longer`, `triax_allgeoms`, `single_allgeoms`). `slurm_*` variants point to scratch paths on the IZAR cluster. Presets under `old/` are legacy configs from an earlier `single_direction_beam` / `rotated_beam` naming scheme.

#### `conf/preprocess/features/`

Feature toggle matrices. `all.yaml` is the primary preset enabling every node and edge feature. Notable alternatives:

- `nonlin.yaml`: removes all linear-relationship edge quantities (velocities, forces, displacements on edges) so the model must learn them from message passing.
- `onlyedge.yaml`, `noabspos_yesedge.yaml`: partial ablations.
- `old*/` subdirectory: historical feature sets kept for reference.

#### `conf/preprocess/normalizer/`

Four presets from a 2×2 matrix:

| | Isotropic | Anisotropic |
|-|-----------|-------------|
| Z-score | `z_isotropic.yaml` | `z_anisotropic.yaml` |
| Robust | `robust_isotropic.yaml` | `robust_anisotropic.yaml` |

The feature set and normalizer choice are typically encoded in the processed dataset directory name (e.g. `z_isotr_centr_nonlin`).

### `conf/train/`

Hydra tree for `src/gnn/train.py`. The compose root is `train.yaml` (or `train_sweep_defaults.yaml` when used by `hyperparam_opt.py`). A `sanity.yaml` shortcut exists for quick iteration.

#### `conf/train/model/`

52 presets covering 7 model families:

| Model name | Preset prefix | Notes |
|------------|--------------|-------|
| `MGN` | `mgn_*` | MeshGraphNet; many size/depth variants |
| `clofnet` | `clofnet_*` | CLOFNet; variant selected by `clofnet_variant` key (`recurrent`, `direct`, `extra_layer`, `separate`) |
| `gunet` | `gunet_*` | Graph U-Net with pooling ratio sweeps |
| `magnet` | `magnet*` | MAGNet graph U-Net with clique pooling |
| `transolver` | `transolver_*` | Attention-based operator; slice/ref/head counts vary |
| `gatv2` | `gatv2_*` | GATv2 multi-head attention |
| `gatmgn` | `gatmgn_*` | Hybrid GAT + MeshGraphNet |

#### `conf/train/loss/`

- `default.yaml`: MSE, `exp_mises: false`, `stress_weight: 1.0`.
- `mae.yaml`: switches to L1 loss.
- `exp_mises_mse.yaml` / `exp_mises_mae.yaml`: apply `exp` inverse before computing stress loss.
- `full_stress_components.yaml` / `full_stress_components_plus_vm.yaml`: loss variants for the 6-component full-stress target mode.

#### `conf/train/noise/`

- `default.yaml`: small fixed Gaussian noise on positions (`noise_scale_train: 3e-4`).
- `no_noise`: zero noise.
- `noise_per_epoch_noprev` / `noise_also_prev`: per-epoch resampling variants.

#### `conf/train/residuals/`

Physics residual loss terms appended to the main prediction loss. `default.yaml` has an empty `residual_list`; `volume.yaml` and `all.yaml` add geometric/volume loss terms.

#### `conf/train/devices/`

- `one_gpu.yaml`: single-GPU training.
- `two_gpu.yaml`: two-GPU DDP training.

#### `conf/train/computational/`

Controls batch size, precision, `torch_compile`, number of workers, and gradient accumulation. SLURM-specific variants (e.g. `in_memory_slurm_1gpu`, `in_memory_slurm_2gpu`, `in_memory_slurm_1gpu_bf16`) pre-set appropriate values for the cluster.

#### `conf/train/lr_scheduler/`

- `step.yaml` / `step_longer.yaml`: step decay.
- `plateau.yaml` / `plateau_long.yaml` / `plateau_long_lower.yaml`: reduce-on-plateau variants.
- `cosine.yaml`: cosine annealing.

#### `conf/train/training/`

- `default.yaml`: 60 epochs, checkpoint every 10.
- `slurm.yaml`: 300 epochs, checkpoint every 20.
- `sanity.yaml`: very short run for smoke-testing.

#### `conf/train/validation/`

- `default.yaml`: validation every epoch; `val_monitor_metric: overall_abs_mae`; rollout-on-validation disabled by default; test media logging enabled.
- `monitor_rel_mae.yaml`: switches monitor to relative MAE.

#### `conf/train/data/`

- `default.yaml`: points to `dataset/processed/biax_longer/z_isotr_centr_nonlin/medium` with the standard `train/`, `val/`, `test/` split layout.

### `conf/hyperparam/`

18 study files, each describing one or more sequential sweep stages. Pass the study name with `-cn` on the command line:

`mgn_sweep`, `mgn_second_sweep`, `force_geom_sweeps`, `feature_sweeps`, `noise_sweeps`, `noise_sweeps_gamma0.3`, `noise_sweeps_gamma0.6`, `residuals_sweep`, `loss_mgn_sweep`, `graph_unet_sweeps`, `clofnet_sweep_L8rec`, `clofnet_sweep_L8recframesourceu`, `clofnet_sweep_direct`, `transolver_sweep`, `integration_scheme_sweep`, `normalization_sweep`, `monitor_metric_sweep`, `sanity_sweep`.

Each study YAML specifies `train_compose_config` (typically `train_sweep_defaults`), `optimization_metric` (`overall_abs_mae` or `overall_relative_mae`), and `pareto_x_metric` / `pareto_y_metric` pairs drawn from `final_test/*` W&B keys.

### `conf/validate/`

`validate.yaml` is filled in before running `src/gnn/validate.py`:

```yaml
training_id: "FILL_IN"       # name of the run folder under saved_models/
model_root: "saved_models"
checkpoint_name: "best"
in_depth_test: null           # list of trajectory indices, "all", or null to skip
in_depth_val: null
```

### `src/gnn/`

#### Entry scripts

- `preprocess.py`: Hydra entrypoint; converts raw graph trajectories (produced by FEM generation) into normalized train/val/test graph datasets. Reads `conf/preprocess/preprocess.yaml`.
- `train.py`: Hydra + Lightning entrypoint; builds datasets, instantiates the model selected by `model_name` config key, runs training, and performs final evaluation on val and test splits with `final_validation/*` / `final_test/*` W&B logging.
- `validate.py`: standalone evaluation script; loads a saved checkpoint by `training_id`, re-runs the same evaluation path (rollout metrics, diagnostic plots) as `train.py`'s final phase.
- `hyperparam_opt.py`: Optuna-free sequential grid-sweep driver; reads a study YAML from `conf/hyperparam/`, runs each stage, and carries the winning option overrides into the next stage. Artifacts go to `hyperparam_outputs/<study_name>/`.
- `eda.py`: exploratory data analysis entrypoint; loads processed datasets and produces feature distribution and normalization diagnostic plots.
- `preprocess_deforming_plate.py`: preprocesses the external DeformingPlate dataset into the same graph format.
- `run_dataset_comparison.py`: orchestrates the biax_longer vs. DeformingPlate comparison (locality + frequency metrics) and writes `summary.csv` and plots.

#### `dataloaders/`

- `inmemorytimestepdataset.py`: `InMemoryTimeStepDataset` — loads all preprocessed graph timesteps into RAM for fast training; `InMemoryNoisedDataset` — wraps the above to inject per-node Gaussian noise at load time.
- `lazytimestepdataset.py`: `LazyTimeStepDataset` / `LazyNoisedDataset` — on-demand loading with LRU file cache; drop-in alternative for large datasets that do not fit in memory.

#### `preprocessing/`

- `build_features.py`: constructs all node and edge features (positions, velocities, displacements, forces, norms, stretch, rotation angle) and targets (acceleration, von Mises, full-stress 6-vector) according to the feature config toggles.
- `preprocessor.py`: orchestrates the full preprocessing pipeline for a single dataset split.
- `normalizer.py`: normalization classes and factory; supports z-score and robust normalizers in isotropic (grouped) and anisotropic (per-channel) modes.
- `stats.py`: computes, serializes, and loads feature statistics (`stats.json`).
- `noiser.py`: train-time noise injection utilities.
- `edge_augmentation.py`: offline graph topology augmentation (k-hop, random, wall-edge variants).

#### `models/`

- `models_helpers.py`: shared MLP builder (`make_mlp`) and related utilities.
- `meshgraphnet/meshgraphnet.py`: MeshGraphNet — encoder–processor–decoder with residual message-passing blocks and optional activation checkpointing.
- `clofnet/clofnet.py`: CLOFNet variants (`ClofNetRecurrent`, `ClofNetDirectAcceleration`, `ClofNetExtraLayer`); equivariant message passing with local coordinate frames. Key sub-modules: `scalarizer_localframe.py`, `clof_mp_blocks.py`, `clof_aggregators.py`, `clof_helpers.py`. `before_adaptation/` holds the pre-project adaptation snapshots; `unused/` holds discarded experimental variants.
- `gunet/gunet.py`: Graph U-Net with configurable depth, pooling ratios, and skip connections.
- `mag/mag.py`: MAGNet — fixed-topology PyTorch graph U-Net with MAg pooling, clique pooling, unpooling, and skip concatenations.
- `transolver/transolvernet.py`: Transolver — attention-based neural operator adapted for irregular meshes; uses physics-informed attention with configurable slice count and reference token count.
- `gatv2/gatv2.py`: GATv2Net — multi-head GATv2 attention processor with optional per-layer residuals.
- `gatmgn/gatmgn.py`: GatMGNNet — hybrid model combining GAT attention with MeshGraphNet-style message passing.

#### `training/`

- `trainer.py`: Lightning module; defines training step, validation step, test step, rollout evaluation, and all W&B / disk media logging.
- `rollout.py`: autoregressive rollout loop; calls the model at each timestep, integrates acceleration to position/velocity with the configured time discretization scheme, and enforces boundary conditions on fixed nodes.
- `losses.py`: loss functions for acceleration and stress targets, including position-reconstruction losses that use the same integration scheme as preprocessing.
- `target_strategy.py`: strategy pattern selecting between `von_mises` (4-dim output) and `full_stress` (9-dim output) target modes.
- `scheduler.py`: learning-rate scheduler wrappers and factory.
- `optimizers.py`: optimizer factory (Adam, fused Adam).
- `online_edge_augmentation.py`: per-batch online random edge augmentation (controlled by `random_online` in `train.yaml`).
- `wandb_loggers.py`: W&B logging helpers for images and GIFs.
- `hyperparam_opt_helpers.py`: Pareto plots, per-trial JSON serialization, and the post-sweep W&B summary run.
- `pareto_metric_config.py`: configuration objects for Pareto metric axis pairs.

#### `plots/`

- `plot_rollout.py`: rollout GIF and frame generation — deformation sequences, error heatmaps, volume-over-time curves.
- `plot_train.py`: end-of-training history plots — loss, relative MAE, rollout MAE, learning-rate curves.
- `plot_overall.py`: aggregate diagnostic plots across multiple trajectories or splits.
- `plot_diagnostic.py`: lower-level mesh diagnostic plots (stress fields, spectral analysis, curvature).

#### `mesh_numerical_helpers/`

- `compute_volume.py`: mesh volume computation from a PyG graph.
- `compute_angle.py`: dihedral / face angle computation.
- `time_discretizer.py`: numerical time integration classes (`SecondOrderCentral`, `Newmark`) implementing the same schemes used in preprocessing and rollout.
- `misc_numerical_helpers.py`: surface extraction, graph distances, spectral utilities.

#### `eda_helpers/`

- `eda_preprocessor.py`: computes per-split feature statistics for EDA.
- `eda_features_helpers.py`: low-level feature extraction helpers for EDA.
- `eda_plot.py`: EDA plotting utilities (histograms, scatter, normalization comparisons).

#### `comparison_defplate/`

Cross-dataset comparison suite for `biax_longer` vs. the external DeformingPlate benchmark:
- `runner.py`: full orchestration (loads trajectories, computes all metrics, writes `summary.csv` and plots).
- `metrics_base.py`, `locality_metrics.py`, `frequency_metrics.py`: metric definitions.
- `trajectory.py`: trajectory data containers and signal extraction.
- `loaders.py`: dataset loaders for both dataset families.
- `graph_ops.py`, `plotting.py`: utilities.

#### `misc_helpers/`

- `yamlobject.py`: `PreprocessYaml`, `TrainYaml` — flattened attribute-style wrappers around Hydra configs.
- `misc.py`: generic helper functions (loss utilities, relative error, small plotting wrappers).
- `compute_r2.py`: R² coefficient of determination helper.
- `custom_pbar.py`: custom progress-bar utilities.
- `pyg_inspector.py`: utilities for inspecting PyTorch Geometric data objects.

## Learning task summary

This project learns the dynamics of deformable 3D beams from finite-element simulations. The data generation pipeline solves transient beam deformation with FEniCS, converts each timestep into a graph, and trains a GNN to predict the next-step mechanical response at every mesh node.

### Context

Each sample is one timestep of a beam mesh represented as a graph with fixed topology over time. Three dataset families exist, determined by which force axes are active during simulation:

- `single` (`force_axes=single`): only `f_y` is non-zero.
- `biax` (`force_axes=biax`): `f_y` and `f_z` vary independently.
- `triax` (`force_axes=triax`): all three force components vary.

Each family further splits by geometry: `longer` (one aspect ratio) and `allgeoms` (multiple widths/lengths). The active default training dataset is `biax_longer`.

The split logic is physically meaningful:
- `train`: Cartesian-product parameter combinations inside the configured ranges.
- `val`: in-distribution values inside the training ranges but not on the exact training grid.
- `test`: out-of-distribution combinations where at least one parameter (force components, beam length, beam width) lies outside the training range.

### Input features

All features are controlled by the `conf/preprocess/features/*.yaml` toggle matrix; the actual channel counts for a given run are recorded in the `config.yaml` saved alongside the processed dataset. The `all.yaml` preset (used by default) enables:

- **Node features**: current position `x_t` (3), velocity `v_t` (3), rest position `x_rest` (3), applied nodal force (3), displacement `x_t − x_rest` (3), boundary-condition flag (1 categorical), and optional per-node norms.
- **Edge features**: relative position `x_j − x_i` (3), rest relative position (3), relative displacement (3), their norms, stretch ratio (1), rotation angle (1), and edge-projected velocity and force vectors.

A leaner alternative is `nonlin.yaml`, which strips all linear-relationship edge quantities and forces the model to infer them through message passing.

### Target features

The target dimensionality depends on `target_mode` in `preprocess.yaml`:

| `target_mode` | Output shape | Content |
|---------------|-------------|---------|
| `von_mises` (default) | `(N, 4)` | acceleration `(a_x, a_y, a_z)` + von Mises stress |
| `full_stress` | `(N, 9)` | acceleration `(a_x, a_y, a_z)` + 6-component stress tensor |

With `mises_log: true` (default in all current presets), the von Mises target is stored as `log(σ_vm + ε)` before normalization. During rollout, predicted acceleration is denormalized and integrated via the configured scheme (default: second-order central differences) to produce the next positions and velocities.

### Models

| `model_name` | Family | Config group |
|-------------|--------|-------------|
| `MGN` | MeshGraphNet (encoder–processor–decoder) | `conf/train/model/mgn_*` |
| `clofnet` | CLOFNet (equivariant, local frames) | `conf/train/model/clofnet_*` |
| `gunet` | Graph U-Net | `conf/train/model/gunet_*` |
| `magnet` | MAGNet (clique pooling U-Net) | `conf/train/model/magnet*` |
| `transolver` | Transolver (attention operator) | `conf/train/model/transolver_*` |
| `gatv2` | GATv2 | `conf/train/model/gatv2_*` |
| `gatmgn` | GAT + MGN hybrid | `conf/train/model/gatmgn_*` |

The default training preset uses:
- model: `MGN` (`mgn_smallest`: 8 message-passing blocks, latent dim 128)
- loss: MSE on normalized `[acceleration, stress]`
- optimizer: Adam, initial LR `1e-3`

### Metrics

**Training:**
- Optimized objective: `train/loss` (MSE by default on normalized targets).

**Validation during training:**
- Main monitored metric: `val/overall_abs_mae` (default) or `val/overall_relative_mae`.
- Best-checkpoint selection uses the same metric.

**Final evaluation** (Lightning `Trainer.test` after training):
- Val split metrics logged under `final_validation/*`; test split under `final_test/*`.
- Scalar metrics: `overall_relative_mae`, `acc_all_rel_mae`, `stress_rel_mae`, `rollout_pos_mae`, `rollout_volume_mae`.
- `final_validation/rollout_pos_mae_all_trajectories` aggregates rollout MAE over all validation trajectories.

**Weights & Biases media layout.** When `log_test_media: true` (default in `conf/train/validation/default.yaml`):

- Per-epoch validation scalars: `val/*`.
- Final validation (val split): summary media under `final_validation/overall/…`; per selected trajectory under `final_validation_traj/<idx>/…`.
- Final test: summary under `final_test/overall/…`; per selected trajectory under `final_test_traj/<idx>/…`.

On disk, plots live under `saved_models/<run_id>/model_evaluation/validation/` or `…/test/`.

## Hyperparameter optimization

### Sweep mechanics

Each study YAML under `conf/hyperparam/` defines a `sweeps:` list. `hyperparam_opt.py` runs each stage as a sequential grid; after each stage the winning option's `overrides` are prepended to subsequent stages. `fixed_overrides` within a stage apply only to that stage and are not carried forward.

The compose root for each trial is set by `train_compose_config` (default: `train_sweep_defaults`). The study tracks `optimization_metric` (`overall_abs_mae` or `overall_relative_mae`) and optionally produces Pareto plots for pairs defined in `pareto_x_metric` / `pareto_y_metric`, drawn from `final_test/*` W&B keys.

### Output artifacts

One invocation of `python -m src.gnn.hyperparam_opt` produces:

- `saved_models/<trial_timestamp>/` per trial: `train.yaml`, `preprocess.yaml`, `checkpoints/`, `training/`, `model_evaluation/`, and offline `wandb/`.
- `hyperparam_outputs/<study_name>/` (timestamped at driver start): `all_trials.json`, per-stage `sweep_<name>_overall_mae.png`, Pareto plots `pareto_<i>.png`, and `summary_abs_mae.png` / `summary_rel_mae.png`.
- `hydra/outputs/hyperparam/<date>/<time>/` from the driver's Hydra run.
- W&B offline runs: one per trial under `saved_models/<trial_timestamp>/wandb/`, plus one `<study_name>__summary` run per set of summary plots under `./wandb/`.

## Tests

Tests live in `tests/gnn/` (20+ modules) and `tests/fem_generation/`. Run with `pytest` from the repo root.

**Config composition:**
```shell
# Verify a training config composes and interpolates without error
python -m pytest tests/gnn/test_config_train.py
python -m pytest tests/gnn/test_config_hyperparam_opt.py --cn loss_mgn_sweep
```

**Model forward passes and parameter counts:**
```shell
python -m pytest tests/gnn/test_model_param_count.py
python -m pytest tests/gnn/test_gunet_forward.py
python -m pytest tests/gnn/test_magnet_forward.py
python -m pytest tests/gnn/test_scalarization_clofnet.py
python -m pytest tests/gnn/test_scalarization_clofnet_whole.py
```

**Training logic:**
```shell
python -m pytest tests/gnn/test_losses_residuals.py
python -m pytest tests/gnn/test_do_rollout_sanity.py
python -m pytest tests/gnn/test_online_edge_seed.py
python -m pytest tests/gnn/test_recurrent_m_ij_init.py
python -m pytest tests/gnn/test_make_mlp_layer_norm.py
```

**Preprocessing:**
```shell
python -m pytest tests/gnn/test_normalizer_grouping.py
python -m pytest tests/gnn/test_feature_layout_pipeline_consistency.py
```

**Hyperparameter sweep utilities:**
```shell
python -m pytest tests/gnn/test_hyperparam_pareto_metrics.py
python -m pytest tests/gnn/test_summary_metric_plots.py
```

FEM generation tests live in `tests/fem_generation/` and will be documented alongside the FEM module.

## FEM generation (documentation TODO)

Raw trajectory generation lives in `src/fem_generation/` with Hydra configs under `conf/fem_generation/`. Entry point: `python -m src.fem_generation.main_run_fem` (see `run_fem_gen.sh`). This README will be extended with solver details, config axes, and dataset naming in a follow-up pass.

## Notes

- Runtime-generated outputs (`dataset/`, `saved_models/`, `hyperparam_outputs/`, `hydra/`, `wandb/`) are not part of the committed tree.
- `tests_artifacts/` holds output files produced by tests locally; it is not committed.
- W&B may add automatic groups such as `charts/` and `system/` on top of the logged keys above.
