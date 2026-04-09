---
description: 
alwaysApply: true
---
Conda env: "fenicsproject".
Paths below are relative to `src/`.
The idea of this project is to predict, using GNNs and physics informed techniques, stress at time t (von Mises or full stress tensor) and acceleration at time t of a deforming beam which initially has a deformation via a traction and then moves until going at rest position. Acceleration at time t contains information about x[t+1] position at time t+1, so then integrating I actually am able to compute the next-step position, which is the real final target. We have the full pipeline, from fem dataset generation in src/fem_generation to preprocessing and then training and validation.
I will run also on a slurm cluster, with the dataset on a folder /scratch and the repo and saved_models etc on another folder /home.
An example of a train config is the following: 
```
computational:
  num_workers: 0
  batch_size: 256
  precision: 32-true
  torch_compile: true
  profiling: false
  pin_memory: true
  rollout_batch_size: 1
  fused_adam: true
model:
  model_name: MGN
  num_messages: 8
  latent_dim: 512
  num_layers_graphnetblock: 2
  layer_norm_enable_encoder: true
  layer_norm_enable_processor: true
  layer_norm_enable_decoder: false
training:
  training_type: slurm
  total_epochs: 100
  resume_from: null
  checkpoint_interval: 20
noise:
  noise_scale_train: 0.0003
  noise_scale_test: 0.0
  noise_gamma: 1.0
  noise_seed: 42
  noise_also_prev: false
  noise_prev_scale_train: 1.0e-05
  noise_feature_update_mode: exact
loss:
  loss_type: mse
  exp_mises: false
  stress_weight: 1.0
validation:
  enable_val: true
  validation_interval: 1
  profile_loss_stress_vs_acceleration: false
  enable_wandb_watch: false
  enable_rollout_val: false
  log_test_media: true
  plot_diagnostics_advanced: false
  selected_validation_trajectories: null
  selected_test_trajectories: null
data:
  dataset_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large
  train_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/train
  val_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/val
  test_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/test
  stats_path: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/train/stats/stats.json
  preprocess_config_path: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/config.yaml
  extra_info_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/extra_info
lr_scheduler:
  type: plateau
  factor: 0.7
  patience: 7
  min_lr: 1.0e-07
  early_stopping_patience: 15
  threshold: 0.0001
residuals:
  residual_list: []
  curvature_loss_weight: 1.0
devices:
  devices: 1
seed: 42
use_wandb: true
initial_lr: 0.001
optimizer: adam
```
Preprocess config
```
dataset:
  dataset_size: large
  limit_timesteps: 50
paths:
  dataset_type: rotated_beam
  base_dir: dataset
  raw_data_dir: /scratch/izar/senoner/dataset/raw/raw/rotated_beam/large
  save_dir: /scratch/izar/senoner/dataset/processed/rotated_beam
  extra_info_dir: /scratch/izar/senoner/dataset/raw/raw/rotated_beam/large/extra_info
normalizer:
  isotropic_normalizer: true
  normalizer_method: z
features:
  mises_log: true
  enable_absolute_pos: true
  enable_past_pos_displ: true
  enable_node_norms: true
  enable_edge_velocity: true
  enable_force_edge: true
  enable_stretch_edge: true
  enable_rotation_angle: true
device: cpu
dt: 0.08
seed: 42
time_discretization_method: second_order_central
node_channels: 29
edge_channels: 28
train_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/train
test_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/test
val_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/val
extra_info_dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/extra_info
```
## `fem_generation/`
### `fem_generation/beam_fea_solver.py`
```
beam_fea_solver.py
Implementation of a 3D beam finite element solver using FEniCS.
One dataset has forces only in the y direction, while the "rotated" in various directions.
No sample/trajectory should have force in the x-direction though, i.e. the one orthogonal to the wall. 
```
### `fem_generation/fem2torch_helpers.py`
```
fem2torch_helpers.py
This file contains helper functions for converting FEniCS meshes to PyTorch Geometric graphs.
```
### `fem_generation/main_run_simulation.py`
```
main_run_simulation.py
Generates train / val / test parameter sets from a YAML config file,
then runs beam FEA simulations by calling fea_simulation()
The idea is that
1) validation has values in-distribution (i.e. in the range train), but different from train
2) test has out-of-distribution values (outside the range of train), and I generate it in a way such that there
are test samples that have only one feature (between force_x, force_y, force_z, lengths, width) out of distribution,
and then some that have 2, some 3, and so on. Every test sample has at least one feature out of distribution.
A test sample with beam always still for example (0,0,0) is explicitly added in test
```
### `fem_generation/visualization_fea.py`
```
visualization_fea.py
This file contains helper functions for visualizing FEA results.
```
## `gnn/`
### `gnn/eda.py`
```
eda.py
Main file to run in order to do Exploratory Data Analysis.
- The del command are there to avoid out of memory errors.
This script does the following:
1. Preprocess the data and save it in a format that is easy to use for EDA.
2. Plot the distributions of the features.
3. Plot uniformly sampled trajectories (10%, i.e. for 1000 samples 100)
- For train the normalized distributions are computed via train stats, for val and test also with train stats.
- To avoid code repetition, the preprocessing reuses the same feature builders used in preprocess.py.
- All the possible normalization are done.
```
### `gnn/eda/eda_features_helpers.py`
```
eda_features_helpers.py
Feature names for EDA plots must stay aligned with build_node_features / build_edge_features in
src/gnn/preprocessing/build_features.py. Use verify_eda_feature_names_match_builders() (runs
probe_feature_layout) to fail fast on drift.
```
### `gnn/eda/eda_preprocessor.py`
```
Builds EDA-ready structures from preprocessed graphs (feature extraction, normalization runs, trajectory sampling for plots).
```
### `gnn/mesh_numerical_helpers/compute_angle.py`
```
compute_angle.py
Contains helper functions for computing angles.
The main access point should be two:
 - in preprocess to compute theta_max
 - in train to compute the angle physics informed residual.
```
### `gnn/mesh_numerical_helpers/compute_volume.py`
```
compute_volume.py
Functions to compute volume of tetrahedra.
```
### `gnn/mesh_numerical_helpers/misc_numerical_helpers.py`
```
misc_numerical_helpers.py
Various helper functions for numerical mesh operations.
```
### `gnn/mesh_numerical_helpers/time_discretizer.py`
```
time_discretizer.py
Contains time discretization (differentiation and integration) helper functions to obtain velocity, acceleration from
position and viceversa. Implements a strategy pattern.
```
### `gnn/misc_helpers/compute_r2.py`
```
compute_r2.py
R^2 helpers using scikit-learn for plot annotations.
```
### `gnn/misc_helpers/custom_pbar.py`
```
custom_pbar.py
Custom progress bar for Lightning to shorten displayed names.
```
### `gnn/misc_helpers/misc.py`
```
misc.py
Contains miscellaneous helper functions.
```
### `gnn/misc_helpers/yamlobject.py`
```
yamlobject.py
Maps yaml keys to instance attributes, via flattening the cfg.
```
### `gnn/models/clofnet/clofnet.py`
```
clofnet.py
```
### `gnn/models/clofnet/clofnet_gaussian_vel.py`
```
velocity-aware ClofNet using Gaussian radial basis edge encoding (see class docstring for paper mapping).
```
### `gnn/models/clofnet/gaussian_layer.py`
```
layers.py
Auxiliary layers used by ClofNet variants.
This file contains two kinds of components:
1. Scalar/edge embedding helpers, especially `GaussianLayer`.
2. An attention-heavy equivariant block used in alternative experiments for
   molecular conformation generation.
```
### `gnn/models/clofnet/gcl/aggregators.py`
```
helpers_gcl.py
```
### `gnn/models/clofnet/gcl/gcl.py`
```
gcl.py
Message-passing blocks used by ClofNet and related equivariant GNN baselines.
Paper mapping for the parts that implement ClofNet most directly:
- EGNN baseline update: Eq. (1), Section 2.
- Complete local frame construction: Eq. (2), Section 4.
- Scalarization of vectors into frame coefficients: Eqs. (3)-(4), Section 4.
- Invertibility / vector reconstruction intuition: Eq. (7), Section 4.
- Vectorization back to an equivariant vector: Eq. (8), Section 4.
- Naive ClofNet message-passing block: Eqs. (16)-(19), Appendix A.3.1.
- GTB / attention refinement: Eqs. (20)-(24), Appendix A.3.1.
Important implementation notes:
- `Clof_GCL` is the closest code realization of the ClofNet update rule in this
  file. It learns scalar message features, predicts three scalar coefficients,
  and reconstructs an equivariant vector in the learned local frame.
- The additive velocity term used by `E_GCL_vel` and `Clof_GCL` is a repository design choice for dynamics modeling.
  The paper discusses vector features such as velocity in the scalarization inputs, but not this exact additive formula.
- Several classes in this file are baseline or utility layers and are only partially related to the ClofNet paper.
  Their docstrings say so explicitly.
```
### `gnn/models/clofnet/gcl/gcl_clofnet.py`
```
clofnet_gcl.py
ClofNet graph block with complete local frames.
```
### `gnn/models/clofnet/gcl/gcl_egnn.py`
```
egnn_gcl.py
Defines two GCL classes inspired from EGNN. One with velocity, one without.
- Since squared distances are invariant to rotation and translation, the produced message m_ij is invariant too.
- `h` and `edge_feat`, that means rotation/translation invariance
- For coordinates `coord`, that means SE(3) equivariance
EGNN GCL:
    m_ij = phi_m(h_i, h_j, r_ij, e_ij)
    x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
    h_i' = phi_h(h_i, sum_j m_ij)
Where
    r_ij = ||x_i - x_j||^2
EGNN_GCL_VELOCITY
    m_ij = phi_m(h_i, h_j, r_ij, e_ij)
    x_i <- x_i  + mean_j ((x_i - x_j) * phi_x(m_ij)) + psi(h_i) * vel_i
    h_i' = phi_h(h_i, sum_j m_ij)
How is velocity equivariant? Let Q rotation matrix
   psi_i * vel_i -> psi_i * (Q vel_i) = Q(psi_i * vel_i).
TODO: EgnnGCL is INHERITED AND THUS USED, EgnnGCLVelocity is not
```
### `gnn/models/clofnet/unused/gcl_attention_equivariant.py`
```
attention_gcl_equivariant.py
```
### `gnn/models/clofnet/unused/gcl_radial_force_velocity.py`
```
radial_force_velocity_gcl.py
TODO: NOT USED BY US, JUST BASELINES
```
### `gnn/models/meshgraphnet/meshgraphnet.py`
```
meshgraphnet.py
Meshgraphnet model (Encoder --> PRocessor --> Decoder)
```
### `gnn/models/models_helpers.py`
```
models_helpers.py
Contains helper functions for building models.
```
### `gnn/preprocess.py`
```
preprocess.py
Preprocess simulation data into graph format. Note one file = one trajectory.
One trajectory = same graph that changes features over time (topology aka adjacency matrix doesn't change)
Saves redundant features as pyg attributes to perform rollout more easily without having to use tensor slices.
Validation/test:
    1. load files and stats
    2. for each file:
        2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
        2.b. normalize with stats loaded in 1
    3. for each file: save
Train:
    1. load files
    2. for each file:
        2.a. compute velocity, acceleration, von mises stress, edge and other features (node and edge)
             and add noise
        2.b. compute normalization stuff (mean + stddev)
    3. for each file: save
```
### `gnn/preprocessing/build_features.py`
```
build_features.py
Builds features for the GNN and physical quantities.
```
### `gnn/preprocessing/inmemorytimestepdataset.py`
```
inmemorytimestepdataset.py
In-memory dataset for graph data.
Two versions:
- InMemoryTimeStepDataset: loads all preprocessed graphs into memory for fast training.
- InMemoryNoisedDataset: wraps InMemoryTimeStepDataset to inject per-node Gaussian noise on position x(t)
    at every __getitem__ call, with RNG seeds varying by training epoch (see set_noise_epoch).
```
### `gnn/preprocessing/noiser.py`
```
noiser.py
FIXME: DEPRECATED
Contains abstract class Noiser and GaussianNoiser to add noise to training data.
```
### `gnn/preprocessing/normalizer.py`
```
normalizer.py
Contains Normalizer class and its subclasses.
```
### `gnn/preprocessing/preprocessor.py`
```
Preprocessor implementations and shared preprocessing utilities (e.g. filename ordering, graph transforms) used by `preprocess.py` and EDA.
```
### `gnn/preprocessing/stats.py`
```
stats.py
Contains Stats dataclass and StatsModule to compute and retrieve statistics for normalization.
```
### `gnn/train.py`
```
train.py
Main training script.
```
### `gnn/training/losses.py`
```
losses.py
Contains loss functions and residuals.
```
### `gnn/training/optimizers.py`
```
optimizers.py
Contains helper functions for creating optimizers.
```
### `gnn/training/rollout.py`
```
rollout.py
Takes care of doing rollout.
Only thing from ground truth graph, i.e. is not predicted, is the node force, that acts as an actuator.
```
### `gnn/training/scheduler.py`
```
scheduler.py
Learning rate schedulers
```
### `gnn/training/trainer.py`
```
trainer.py
Implements a LightningModule subclass to train models.
```
### `gnn/training/wandb_loggers.py`
```
wandb_loggers.py
Contains helper functions for logging to W&B.
```
### `gnn/validate.py`
```
validate.py
Standalone evaluation script. Loads a Lightning checkpoint from a completed
training run and runs the same final evaluation path (test_step + on_test_epoch_end)
that train.py uses, producing rollout metrics and diagnostic plots.
Configuration is read from conf/validate/validate.yaml.
```
### `gnn/validation/plot_diagnostic.py`
```
plot_diagnostic.py
Advanced diagnostic plots for rollouts.
```
### `gnn/validation/plot_overall.py`
```
plot_overall.py
Contains functions for generating summary plots of model evaluation, with main access point overall_plot().
```
### `gnn/validation/plot_rollout.py`
```
plot_rollout.py
Contains all files that plot metrics or diagnostics for rollout, in depth for each trajectory.
Main access points: in_depth_plots()
```
### `gnn/validation/plot_train.py`
```
plot_train.py
Contains functions for plotting training history specific stuff.
```
