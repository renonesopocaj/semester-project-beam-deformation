# Inductive Biases and Learning Strategies for Learning Dynamics of Continuous Matter

This repository contains two main pipelines:

1. `src/fem_generation/`: generate raw beam deformation trajectories with FEniCS and save them as PyTorch Geometric graphs.
2. `src/gnn/`: preprocess those trajectories, train graph neural networks, and evaluate them with rollout diagnostics.

## Repository tree

Omitting all `conf` sub-sub-directories:

```text
semester-project/
├── .git/
├── conf/
│   ├── fem_generation/
│   ├── preprocess/
│   └── train/
├── old_venv/
├── src/
│   ├── fem_generation/
│   │   ├── __init__.py
│   │   ├── beam_fea_solver.py
│   │   ├── fem2torch_helpers.py
│   │   ├── main_run_simulation.py
│   │   └── visualization_fea.py
│   └── gnn/
│       ├── misc_helpers/
│       │   ├── __init__.py
│       │   ├── custom_pbar.py
│       │   ├── misc.py
│       │   ├── pyg_inspector.py
│       │   └── yamlobject.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── models_helpers.py
│       │   ├── clofnet/
│       │   │   ├── __init__.py
│       │   │   ├── clof.py
│       │   │   ├── gcl.py
│       │   │   └── layers.py
│       │   ├── gunet/
│       │   │   └── __init__.py
│       │   ├── mag/
│       │   │   └── __init__.py
│       │   └── meshgraphnet/
│       │       ├── __init__.py
│       │       └── meshgraphnet.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── build_features.py
│       │   ├── inmemorytimestepdataset.py
│       │   ├── noiser.py
│       │   ├── normalizer.py
│       │   └── stats.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── checkpointing.py
│       │   ├── rollout.py
│       │   ├── scheduler.py
│       │   └── trainer.py
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── mesh_build_numerical.py
│       │   ├── plot_rollout.py
│       │   └── train_plots.py
│       ├── eda.py
│       ├── preprocess.py
│       ├── train.py
│       └── validate.py
├── .gitignore
├── env_nodep_pip.yml
├── environment.yml
├── environment_nodep.yml
├── explicit-spec.txt
└── README.md
```

## What each part does

### Top level

- `conf/`: Hydra configuration tree for FEM generation, preprocessing, and training.
- `old_venv/`: archived environment files kept for reference.
- `src/`: project source code.
- `.gitignore`: Git ignore rules.
- `env_nodep_pip.yml`: lighter environment specification with some pip-installed extras.
- `environment.yml`: fuller Conda environment specification for the project.
- `environment_nodep.yml`: lighter Conda environment specification without a large dependency lock.
- `explicit-spec.txt`: exported package lock / environment snapshot.
- `README.md`: repository overview and structure documentation.

### `conf/fem_generation/`

- `config_original.yaml`: FEM generation config for the single-direction/original-force dataset.
- `config_rotated.yaml`: FEM generation config for the rotated-force dataset where force components vary independently.

### `conf/preprocess/`

- `preprocess.yaml`: main Hydra composition for preprocessing raw FEM trajectories into model-ready graph datasets.
- `eda.yaml`: Hydra composition for exploratory data analysis runs.

#### `conf/preprocess/dataset/`

- `small.yaml`: small dataset-size preset.
- `medium.yaml`: medium dataset-size preset.
- `large.yaml`: large dataset-size preset.

#### `conf/preprocess/features/`

- `default.yaml`: default feature-selection preset.
- `displacement.yaml`: enables displacement-related node and edge features.

#### `conf/preprocess/normalizer/`

- `z_anisotropic.yaml`: z-score normalization with per-channel anisotropic stats.
- `z_isotropic.yaml`: z-score normalization with grouped/isotropic stats.
- `robust_anisotropic.yaml`: robust normalization with per-channel anisotropic stats.
- `robust_isotropic.yaml`: robust normalization with grouped/isotropic stats.

#### `conf/preprocess/numerical/`

- `default.yaml`: default numerical-target settings, including the usual von Mises handling.
- `no_mises_log.yaml`: disables log-transform handling for von Mises targets.

#### `conf/preprocess/paths/`

- `single_direction_beam.yaml`: path preset for the original beam dataset family.
- `rotated_beam.yaml`: path preset for the rotated-force dataset family.

### `conf/train/`

- `train.yaml`: top-level training Hydra config that composes all training sub-configs.

#### `conf/train/computational/`

- `default.yaml`: default hardware/runtime settings.
- `gpu.yaml`: GPU-oriented runtime settings.

#### `conf/train/data/`

- `default.yaml`: points training to a processed dataset and its train/val/test folders.

#### `conf/train/lr_scheduler/`

- `step.yaml`: step learning-rate scheduler preset.
- `plateau.yaml`: reduce-on-plateau scheduler preset.
- `cosine.yaml`: cosine scheduler preset.

#### `conf/train/model/`

- `meshgraphnet.yaml`: standard MeshGraphNet architecture preset.
- `meshgraphnet_large.yaml`: larger MeshGraphNet preset.

#### `conf/train/training/`

- `training.yaml`: default training-mode settings.
- `sanity.yaml`: short sanity-run training settings.
- `slurm.yaml`: SLURM/distributed-oriented training settings.

#### `conf/train/validation/`

- `validation.yaml`: validation/evaluation behavior settings used by training.

### `old_venv/`

- `environment.yml`: older archived environment file.
- `kevin/freeze.txt`: archived pip freeze snapshot.

### `src/fem_generation/`

- `__init__.py`: package marker for FEM generation utilities.
- `beam_fea_solver.py`: core 3D beam FEA solver in FEniCS; builds the mesh, applies boundary conditions and traction, advances time, and saves per-timestep graph snapshots.
- `fem2torch_helpers.py`: converts FEniCS fields and mesh information into PyTorch / PyTorch Geometric-friendly arrays and tensors.
- `main_run_simulation.py`: reads FEM generation configs, builds train/val/test parameter combinations, deduplicates them, and launches simulations.
- `visualization_fea.py`: plotting and GIF utilities for raw FEM trajectories and diagnostic checks.

### `src/gnn/`

- `eda.py`: exploratory data analysis entrypoint for processed graph features and normalization comparisons.
- `preprocess.py`: main preprocessing script that converts raw graph trajectories into normalized train/val/test graph samples.
- `train.py`: main training entrypoint using Hydra and Lightning; builds datasets, model, optimizer, trainer, and final evaluation.
- `validate.py`: standalone saved-model evaluation / validation script.

### `src/gnn/misc_helpers/`

- `__init__.py`: package marker for miscellaneous helpers.
- `custom_pbar.py`: custom progress-bar utilities.
- `misc.py`: generic helper functions, including losses, relative errors, and some rollout diagnostic plotting wrappers.
- `pyg_inspector.py`: helper utilities for inspecting PyTorch Geometric objects.
- `yamlobject.py`: small config-wrapper classes that flatten Hydra/YAML configs into attribute-style Python objects.

### `src/gnn/models/`

- `__init__.py`: package marker for model definitions.
- `models_helpers.py`: shared neural-network building helpers, such as reusable MLP builders.

#### `src/gnn/models/meshgraphnet/`

- `__init__.py`: package marker.
- `meshgraphnet.py`: MeshGraphNet implementation used by the current training pipeline.

#### `src/gnn/models/clofnet/`

- `__init__.py`: package marker.
- `clof.py`: main CLOFNet model definition.
- `gcl.py`: graph-convolution / interaction-layer logic for CLOFNet.
- `layers.py`: lower-level neural layers used by CLOFNet.

#### `src/gnn/models/gunet/`

- `__init__.py`: placeholder package marker for a future GUNet implementation.

#### `src/gnn/models/mag/`

- `__init__.py`: exports `MAGNet`.
- `mag.py`: fixed-topology PyTorch MAgNET graph U-Net with MAg, clique pooling, unpooling, and skip concatenations.

### `src/gnn/preprocessing/`

- `__init__.py`: package marker.
- `build_features.py`: constructs node, edge, kinematic, and target features such as velocity, acceleration, and von Mises stress.
- `inmemorytimestepdataset.py`: dataset wrapper that loads processed graph timesteps into memory.
- `noiser.py`: train-time noise injection utilities.
- `normalizer.py`: normalization logic and factory classes.
- `stats.py`: feature-statistics computation, serialization, and loading utilities.

### `src/gnn/training/`

- `__init__.py`: package marker.
- `checkpointing.py`: helper utilities related to model checkpoint management.
- `rollout.py`: autoregressive rollout logic for evaluating the trained model over time.
- `scheduler.py`: scheduler wrappers and factory logic.
- `trainer.py`: Lightning module implementation; defines training, validation, testing, rollout evaluation, and media logging.

### `src/gnn/validation/`

- `__init__.py`: package marker.
- `mesh_build_numerical.py`: numerical mesh utilities used by validation plots, such as surface extraction, volume, curvature, graph distances, and spectral analysis.
- `plot_rollout.py`: rollout plotting and GIF generation for deformation, error heatmaps, volume, and mesh diagnostics.
- `train_plots.py`: end-of-training history plots such as loss, relative MAE, rollout error, and learning-rate curves.

## End-to-end workflow

1. Generate raw FEM trajectories with `src/fem_generation/main_run_simulation.py`.
2. Convert them into processed graph datasets with `src/gnn/preprocess.py`.
3. Inspect distributions or artifacts with `src/gnn/eda.py` if needed.
4. Train a model with `src/gnn/train.py`.
5. Evaluate saved checkpoints and produce rollout diagnostics with `src/gnn/validate.py` or the final evaluation phase inside `src/gnn/train.py`.

## Notes

- The repository currently has no committed `tests/`, `scripts/`, or `notebooks/` directories.
- Runtime-generated outputs such as processed datasets, saved models, Hydra outputs, and GIF/plot artifacts are not part of the committed tree above.

## Learning Task Summary

This project learns the dynamics of deformable 3D beams from finite-element simulations. The data generation pipeline solves transient beam deformation with FEniCS, converts each timestep into a graph, and then trains a graph neural network to predict the next-step mechanical response at every mesh node.

### Context

- Each sample is one timestep of a beam mesh represented as a graph whose topology is fixed over time.
- The repository supports two dataset families:
- `single_direction_beam`: force varies only along `y` during training.
- `rotated_beam`: force components along `x`, `y`, and `z` all vary during training.
- The current committed training preset points to the processed `single_direction_beam` dataset, specifically `dataset/processed/single_direction_beam/z_isotr_log_displace/large`.
- The split logic is physically meaningful:
- `train`: Cartesian-product combinations inside the configured ranges.
- `val`: in-distribution values that stay inside the training ranges but do not lie on the exact training grid.
- `test`: out-of-distribution combinations where at least one among `force_x`, `force_y`, `force_z`, beam length, and beam width lies outside the training range.

### Input Features

After preprocessing, each graph node receives one categorical feature and a set of valued physical features.

- Node categorical feature:
- boundary-condition flag (`1` fixed, `0` free)
- Node valued features:
- current position `x_t` (3)
- current velocity `v_t` (3)
- rest position `x_rest` (3)
- applied nodal force at the current timestep (3)
- optional displacement `x_t - x_rest` (3)
- With the default preprocessing preset (`features: displacement`), this gives `16` node channels total: `1 + 15`.

- Edge features:
- current relative position `x_j - x_i` (3)
- rest relative position `x_rest,j - x_rest,i` (3)
- optional relative displacement term `(x_j - x_i) - (x_rest,j - x_rest,i)` (3)
- norms of the relative vectors above
- With displacement enabled, this gives `12` edge channels.

### Target Features

The model performs node-wise prediction.

- Predicted quantities:
- acceleration at the current timestep `(a_x, a_y, a_z)`
- von Mises stress (scalar)
- In the default preprocessing configuration, the stress target is stored as `log(von_mises + eps)` before normalization (`mises_log: true`).
- During rollout, the predicted acceleration is denormalized and integrated with a semi-implicit Euler update to produce the next positions and velocities.

### How It Is Predicted

The active training pipeline currently uses `MeshGraphNet`.

- Encoder:
- one MLP for node features and one MLP for edge features
- Processor:
- a stack of residual message-passing blocks
- each block updates edge embeddings from sender, receiver, and edge states, then updates node embeddings from aggregated incoming messages
- Decoder:
- one head predicts 3D acceleration
- one head predicts scalar von Mises stress

The default training preset uses:

- model: `MeshGraphNet`
- latent size: `128`
- message-passing blocks: `8`
- loss: `mse`

Other architectures exist in the codebase (`CLOFNet`, `GUNet`, `MAGNet`); `train.py` instantiates the model selected by Hydra `model_name` (e.g. `MGN`, `gunet`, `magnet`).

### Metrics

The repository tracks different metrics for training, validation, and final evaluation.

- Training:
- optimized objective is `train/loss`
- by default this is `MSE` on the normalized target `[acceleration, stress]`

- Validation during training:
- main monitored metric is `val/overall_relative_mae`
- best-checkpoint selection also uses `val/overall_relative_mae`
- optional `val/acc_all_loss` and `val/stress_loss` exist in the code but are disabled by default in the current config
- rollout validation metric `val/final_roll_err_pos` exists, but rollout-on-validation is disabled by default in the current config

- Final validation and test evaluation (Lightning `Trainer.test` after training):
- One-step batch metrics and rollout summary `final_roll_err_pos` are logged to Weights & Biases under `final_validation/*` when the val split is evaluated and under `final_test/*` when the test split is evaluated. This keeps them separate from per-epoch training validation metrics under `val/*`.
- Metric names mirror the training-style layout: `overall_relative_mae` (all four target channels), `acc_all_rel_mae`, `stress_rel_mae`, and `final_roll_err_pos` (mean absolute position error at the last rollout timestep).

**Weights & Biases media layout.** When test-time media logging is enabled (`log_test_media` in training config), images and GIFs use the following key prefixes so each in-depth trajectory has its own folder in the W&B UI:

- Training: `train/*` scalars; per-epoch validation: `val/*` (unchanged).
- Final validation (val split, end of `train.py`): overall summary media under `final_validation/overall/…`; per selected trajectory under `final_validation_traj/<trajectory_index>/…`.
- Test split: overall summary under `final_test/overall/…`; per selected trajectory under `final_test_traj/<trajectory_index>/…`.
- W&B may also add automatic groups such as `charts/` and `system/`.

On disk, plots still live under `saved_models/<run_id>/model_evaluation/validation/` or `…/test/`; only the W&B log keys use the `final_*` namespaces above.

In short, the task is to learn a graph-based surrogate model that maps the current beam state and forcing to the next-step nodal acceleration and stress, and then to assess both one-step prediction quality and long-horizon rollout quality.
