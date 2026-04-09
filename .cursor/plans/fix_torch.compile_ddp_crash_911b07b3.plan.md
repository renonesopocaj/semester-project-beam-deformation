---
name: Fix torch.compile DDP crash
overview: "Fix two crashes when training with `torch.compile` and multi-GPU DDP: (1) CUDA index out-of-bounds in scatter_add caused by incompatible `torch_scatter` + CUDA-graph recompilation on the smaller last batch, and (2) shared-memory exhaustion when `ddp_spawn` tries to pickle a compiled model."
todos:
  - id: fix-scatter
    content: Replace `torch_scatter.scatter_add` with native `torch.Tensor.scatter_add_` in `meshgraphnet.py`
    status: pending
  - id: fix-compile-mode
    content: Change compile mode from `reduce-overhead` to `default` and remove `fullgraph=True` in `maybe_compile_model()`
    status: pending
  - id: fix-drop-last
    content: Add `drop_last=True` to the train DataLoader so all batches have identical shapes
    status: pending
  - id: fix-ddp-strategy
    content: Change `ddp_spawn` fallback to `ddp` in `slurm_trainer_devices_and_strategy()`
    status: pending
isProject: false
---

# Fix torch.compile + Multi-GPU Training Crashes

## Observations

- Single GPU + torch.compile: **works**
- 2 GPU + torch.compile + batch_size=256 + fp32 + 8 messages: **works** (or at least passes sanity + first batches)
- 2 GPU + torch.compile + batch_size=384 + bf16 + 10 messages (torchrun/DDP): **OOB crash**
- 2 GPU + torch.compile + batch_size=384 + fp32 + 16 messages (ddp_spawn): **mmap crash**
- All graphs have identical topology / node count, so the only variable batch-size happens on the **last partial batch** per rank.

## Root Cause Analysis

### Error 1 -- CUDA "index out of bounds" (torchrun / DDP)

Two factors combine:

1. `**torch_scatter.scatter_add` is not compatible with `torch.compile`**. It is a third-party C++/CUDA extension; when TorchDynamo traces into it, the Inductor backend generates incorrect Triton kernels. This is a [known issue](https://github.com/rusty1s/pytorch_scatter/issues/440).
2. `**reduce-overhead` mode captures CUDA graphs**, which bake in fixed tensor shapes. With DDP, each rank processes `total_samples / 2` samples. When `batch_size=384`, `26460 / 384 = 68` full batches + 1 partial batch of 348 graphs. The Inductor compiles a kernel expecting 86400 nodes (384 * 225), but the last batch has only 78300 nodes (348 * 225). The compiled scatter kernel accesses index >= 78300 which is out of bounds for the smaller output buffer. This explains why `batch_size=256` seems to work (it may simply not have reached the last batch yet, or its partial size triggers a clean recompilation).

### Error 2 -- "Cannot allocate memory" (ddp_spawn fallback)

When `WORLD_SIZE` is not set (no `torchrun`), the code falls back to `ddp_spawn`. This strategy **pickles the model** to pass it to spawned processes. With `torch.compile`, the compiled model carries extra cached state that exhausts `/dev/shm` shared memory.

## Planned Changes

### 1. Replace `torch_scatter.scatter_add` with native PyTorch scatter

In [src/gnn/models/meshgraphnet/meshgraphnet.py](src/gnn/models/meshgraphnet/meshgraphnet.py), replace:

```python
from torch_scatter import scatter_add
# ...
agg_received_edges = scatter_add(edge_attr_, receivers.long(), dim=0, dim_size=num_nodes)
```

with PyTorch's built-in `Tensor.scatter_add_`:

```python
agg_received_edges = torch.zeros(num_nodes, edge_attr_.size(1),
                                  device=edge_attr_.device, dtype=edge_attr_.dtype)
agg_received_edges.scatter_add_(0, receivers.long().unsqueeze(-1).expand_as(edge_attr_), edge_attr_)
```

This uses native PyTorch operations that TorchDynamo/Inductor can trace and compile correctly. It is semantically identical to the `torch_scatter` version.

### 2. Change `torch.compile` mode from `reduce-overhead` to `default`

In [src/gnn/train.py](src/gnn/train.py) `maybe_compile_model()`, change:

```python
compile_mode = "reduce-overhead" if torch.cuda.is_available() else "default"
return torch.compile(model, mode=compile_mode, fullgraph=True)
```

to:

```python
return torch.compile(model, mode="default")
```

- `reduce-overhead` uses CUDA graphs which replay the exact same kernel with fixed buffer sizes. The last (smaller) batch causes the OOB. `default` mode provides Inductor optimizations (kernel fusion, memory planning) without CUDA graph constraints.
- `fullgraph=True` is removed to allow graceful graph-break fallbacks if needed.

### 3. Add `drop_last=True` to the train DataLoader

In [src/gnn/train.py](src/gnn/train.py), add `drop_last=True` to the train `DataLoader`:

```python
train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=train_cfg.num_workers, ...,
                              drop_last=True)
```

Since all graphs have identical topology, every full batch has exactly the same tensor shapes. Dropping the last partial batch guarantees `torch.compile` never encounters a shape mismatch, which is the cleanest way to avoid recompilation or CUDA graph replay issues. The loss of a few samples per epoch is negligible.

### 4. Change `ddp_spawn` to `ddp` in the multi-GPU fallback path

In [src/gnn/train.py](src/gnn/train.py) `slurm_trainer_devices_and_strategy()`, change:

```python
if requested_devices > 1:
    return requested_devices, "ddp_spawn"
```

to:

```python
if requested_devices > 1:
    return requested_devices, "ddp"
```

Lightning's `ddp` strategy re-launches the script as subprocesses rather than pickling the model. This avoids the shared memory exhaustion that `ddp_spawn` causes with compiled models.



NOTES

I have problems also with this config

```shellscript
#!/bin/bash
#SBATCH --job-name=semester-project
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=39
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "STDOUT file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "STDERR file: $(pwd)/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
echo "Dataset root expected at: $(pwd)/dataset"
echo "Artifacts expected at: $(pwd)/saved_models"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate semesterproject

mkdir -p saved_models
# Two ranks (matches devices=two_gpu): explicit torchrun avoids Lightning staying at world_size 1.
torchrun --standalone --nproc_per_node=1 -m src.gnn.train \
  computational=in_memory_slurm_1gpu \
  lr_scheduler=plateau \
  training=slurm \
  training.total_epochs=100 \
  model=mgn_large \
  devices=one_gpu \
  data=slurm \
  data.dataset_dir="/scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large"

```

which yields

[rank: 0] Seed set to 42

GPU available: True (cuda), used: True

TPU available: False, using: 0 TPU cores

wandb: WARNING The anonymous setting has no effect and will be removed in a future version.

wandb: WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id d2txokdj.

wandb: Tracking run with wandb version 0.25.1

wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

wandb: Run data is saved locally in /home/senoner/semester-project/saved_models/2026-04-09_03-03-43/wandb/offline-run-20260409_030344-d2txokdj

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name         | Type         | Params | Mode 

------------------------------------------------------

0 | model        | MeshGraphNet | 36.6 M | train

1 | node_stats   | StatsModule  | 0      | train

2 | edge_stats   | StatsModule  | 0      | train

3 | target_stats | StatsModule  | 0      | train

------------------------------------------------------

36.6 M    Trainable params

0         Non-trainable params

36.6 M    Total params

146.205   Total estimated model params size (MB)

282       Modules in train mode

0         Modules in eval mode

/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argumen `to` num_workers=3 `in the` DataLoader` to improve performance.

/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argumen `to` num_workers=3 `in the` DataLoader` to improve performance.

Error executing job with overrides: ['computational=in_memory_slurm_1gpu', 'lr_scheduler=plateau', 'training=slurm', '[training.total](http://training.total)_epochs=100', 'model=mgn_large', 'devices=one_gpu', 'data=slurm', 'data.dataset_dir=/scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large']

Traceback (most recent call last):

  File "/home/senoner/semester-project/src/gnn/[train.py](http://train.py)", line 356, in main

    train(train_cfg, preprocess_cfg, device, cfg, preprocess_dict)

  File "/home/senoner/semester-project/src/gnn/[train.py](http://train.py)", line 315, in train

    fit_[trainer.fit](http://trainer.fit)(model=lightning_module, train_dataloaders=train_dataloader,

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/[trainer.py](http://trainer.py)", line 560, in fit

    call._call_and_handle_interrupt(

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/[call.py](http://call.py)", line 49, in *call*and_handle_interrupt

    return trainer_fn(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/[trainer.py](http://trainer.py)", line 598, in *fit*impl

    self._run(model, ckpt_path=ckpt_path)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/[trainer.py](http://trainer.py)", line 1011, in _run

    results = self._run_stage()

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/[trainer.py](http://trainer.py)", line 1055, in *run*stage

    [self.fit](http://self.fit)_[loop.run](http://loop.run)()

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/fit_[loop.py](http://loop.py)", line 216, in run

    self.advance()

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/fit_[loop.py](http://loop.py)", line 458, in advance

    self.epoch_[loop.run](http://loop.run)(self._data_fetcher)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/training_epoch_[loop.py](http://loop.py)", line 152, in run

    self.advance(data_fetcher)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/training_epoch_[loop.py](http://loop.py)", line 348, in advance

    batch_output = self.automatic_[optimization.run](http://optimization.run)(trainer.optimizers[0], batch_idx, kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/optimization/[automatic.py](http://automatic.py)", line 192, in run

    self._optimizer_step(batch_idx, closure)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/optimization/[automatic.py](http://automatic.py)", line 270, in *optimizer*step

    call._call_lightning_module_hook(

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/[call.py](http://call.py)", line 177, in *call*lightning_module_hook

    output = fn(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/core/[module.py](http://module.py)", line 1366, in optimizer_step

    optimizer.step(closure=optimizer_closure)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/core/[optimizer.py](http://optimizer.py)", line 154, in step

    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/strategies/[strategy.py](http://strategy.py)", line 239, in optimizer_step

    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/plugins/precision/[precision.py](http://precision.py)", line 123, in optimizer_step

    return optimizer.step(closure=closure, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/optim/[optimizer.py](http://optimizer.py)", line 485, in wrapper

    out = func(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/optim/[optimizer.py](http://optimizer.py)", line 79, in *use*grad

    ret = func(self, *args,* *kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/optim/[adam.py](http://adam.py)", line 225, in step

    loss = closure()

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/plugins/precision/[precision.py](http://precision.py)", line 109, in *wrap*closure

    closure_result = closure()

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/optimization/[automatic.py](http://automatic.py)", line 146, in **call**

    self._result = self.closure(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/utils/_[contextlib.py](http://contextlib.py)", line 116, in decorate_context

    return func(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/optimization/[automatic.py](http://automatic.py)", line 131, in closure

    step_output = self._step_fn()

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/loops/optimization/[automatic.py](http://automatic.py)", line 319, in *training*step

    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/trainer/[call.py](http://call.py)", line 329, in *call*strategy_hook

    output = fn(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/lightning/pytorch/strategies/[strategy.py](http://strategy.py)", line 391, in training_step

    return self.lightning_[module.training](http://module.training)_step(*args, **kwargs)

  File "/home/senoner/semester-project/src/gnn/training/[trainer.py](http://trainer.py)", line 415, in training_step

    pred = self(batch)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/nn/modules/[module.py](http://module.py)", line 1751, in *wrapped*call_impl

    return self._call_impl(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/nn/modules/[module.py](http://module.py)", line 1762, in *call*impl

    return forward_call(*args, **kwargs)

  File "/home/senoner/semester-project/src/gnn/training/[trainer.py](http://trainer.py)", line 319, in forward

    return self.model(batch)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/nn/modules/[module.py](http://module.py)", line 1751, in *wrapped*call_impl

    return self._call_impl(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/nn/modules/[module.py](http://module.py)", line 1762, in *call*impl

    return forward_call(*args, **kwargs)

  File "/home/senoner/semester-project/src/gnn/models/meshgraphnet/[meshgraphnet.py](http://meshgraphnet.py)", line 172, in forward

    node_embedding, edge_embedding = processor(edge_index, node_embedding, edge_embedding)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/nn/modules/[module.py](http://module.py)", line 1751, in *wrapped*call_impl

    return self._call_impl(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/nn/modules/[module.py](http://module.py)", line 1762, in *call*impl

    return forward_call(*args, **kwargs)

  File "/home/senoner/semester-project/src/gnn/models/meshgraphnet/[meshgraphnet.py](http://meshgraphnet.py)", line 74, in forward

    edge_attr_ = self.edge_net(torch.hstack((node_embedding[senders], node_embedding[receivers], edge_embedding)))

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.38 GiB. GPU 0 has a total capacity of 31.74 GiB of which 2.48 GiB is free. Including non-PyTorch memory, this process has 29.25 GiB memory in use. Of the allocated memory 28.10 GiB is allocated by PyTorch, and 803.85 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  ([https://pytorch.org/docs/stable/notes/cuda.html#environment-variables](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables))

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

E0409 03:03:59.864000 1774491 site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 0 (pid: 1774495) of binary: /home/senoner/miniconda3/envs/semesterproject/bin/python

Traceback (most recent call last):

  File "/home/senoner/miniconda3/envs/semesterproject/bin/torchrun", line 6, in <module>

    sys.exit(main())

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper

    return f(*args, **kwargs)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/distributed/[run.py](http://run.py)", line 892, in main

    run(args)

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/distributed/[run.py](http://run.py)", line 883, in run

    elastic_launch(

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/distributed/launcher/[api.py](http://api.py)", line 139, in **call**

    return launch_agent(self._config, self._entrypoint, list(args))

  File "/home/senoner/miniconda3/envs/semesterproject/lib/python3.10/site-packages/torch/distributed/launcher/[api.py](http://api.py)", line 270, in launch_agent

    raise ChildFailedError(

torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 

============================================================

src.gnn.train FAILED

------------------------------------------------------------

Failures:

  <NO_OTHER_FAILURES>

------------------------------------------------------------

Root Cause (first observed failure):

[0]:

  time      : 2026-04-09_03:03:59

  host      : i49.izar.cluster

  rank      : 0 (local_rank: 0)

  exitcode  : 1 (pid: 1774495)

  error_file: <N/A>

  traceback : To enable traceback see: [https://pytorch.org/docs/stable/elastic/errors.html](https://pytorch.org/docs/stable/elastic/errors.html)

============================================================

Job ID: 2827512

Working directory: /home/senoner/semester-project

STDOUT file: /home/senoner/semester-project/slurm-semester-project-2827512.out

STDERR file: /home/senoner/semester-project/slurm-semester-project-2827512.err

Dataset root expected at: /home/senoner/semester-project/dataset

Artifacts expected at: /home/senoner/semester-project/saved_models

================================================================================

computational:

  num_workers: 0

  batch_size: 256

  precision: 32-true

  torch_compile: false

  profiling: false

  pin_memory: true

  rollout_batch_size: 1

  fused_adam: true

model:

  model_name: MGN

  num_messages: 15

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

================================================================================

Extra info dir: /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/extra_info

Selected validation trajectories: [1, 11, 24, 34, 46, 45, 76, 93, 111, 120]

Selected test trajectories: [3, 8, 12, 25, 32, 49, 51, 77, 84, 109, 124]

Using device: cuda

Loading stats from /scratch/izar/senoner/dataset/processed/rotated_beam/z_isotr_centr_log_abs_xtm1_normnode_edgevel_edgeforce_stretch_rot/large/train...

	Node features: mean shape = torch.Size([28]), std shape = torch.Size([28])

	Edge features: mean shape = torch.Size([28]), std shape = torch.Size([28])

	Target features: mean shape = torch.Size([4]), std shape = torch.Size([4])

Train dataset:

	Noise injection enabled: scale=0.0003, gamma=1.0

	Found 1080 sample files

	Loaded the dataset with 52920 samples (1080 trajectories)

Validation dataset:

	Found 135 sample files

	Loaded the dataset with 6974 samples (135 trajectories)

	Total parameters: 36,551,172

Lightning SLURM Trainer setup: train_cfg.devices=1 (type=int), torch.cuda.device_count()=1, CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', Trainer(devices=1, strategy=None)

Training ID: 2026-04-09_03-03-43

	 Initial learning rate: 0.001

	LR scheduler: plateau

	 Total epochs: 100

	Validation interval: 1

	 Loss type: mse

	Using wandb: True

	 Model directory: /home/senoner/semester-project/saved_models/2026-04-09_03-03-43

Sanity Checking: |          | 0/? [00:00<?, ?it/s]

Sanity Checking: |          | 0/? [00:00<?, ?it/s]

Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]

Sanity Checking DataLoader 0:  50%|█████     | 1/2 [00:04<00:04,  0.20it/s]

Sanity Checking DataLoader 0: 100%|██████████| 2/2 [00:09<00:00,  0.20it/s]

                                                                           

Training: |          | 0/? [00:00<?, ?it/s]

Training: |          | 0/? [00:00<?, ?it/s]

Epoch 0:   0%|          | 0/207 [00:00<?, ?it/s][1;34mwandb[0m: 

[1;34mwandb[0m: You can sync this run to the cloud by running:

[1;34mwandb[0m: [1mwandb sync /home/senoner/semester-project/saved_models/2026-04-09_03-03-43/wandb/offline-run-20260409_030344-d2txokdj[0m

[1;34mwandb[0m: Find logs at: [1;35msaved_models/2026-04-09_03-03-43/wandb/offline-run-20260409_030344-d2txokdj/logs[0m

Epoch 0:   0%|          | 0/207 [00:01<?, ?it/s]