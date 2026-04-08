"""
inmemorytimestepdataset.py

In-memory dataset for graph data.
Two versions:
- InMemoryTimeStepDataset: loads all preprocessed graphs into memory for fast training.
- InMemoryNoisedDataset: wraps InMemoryTimeStepDataset to inject per-node Gaussian noise on position x(t)
    at every __getitem__ call, with RNG seeds varying by training epoch (see set_noise_epoch).
"""

from torch_geometric.data import Dataset, Data
from src.gnn.preprocessing.build_features import make_scripted_feature_builders
from src.gnn.preprocessing.normalizer import (
    build_affine_normalizer_tensors,
    apply_affine_normalization,
    apply_affine_denormalization,
)
import multiprocessing
import os
import torch

# Large stride so (base_seed + epoch * stride + per_sample_offset) stays well separated across epochs.
_NOISE_EPOCH_SEED_STRIDE = 1_000_003
# torch.Generator.manual_seed expects a non-negative int in a bounded range; fold mixed seeds into this range.
_TORCH_SEED_MOD = 2**63 - 1


def _fold_torch_seed(seed):
    """
    Fold an arbitrary integer into a safe range for torch.Generator.manual_seed.

    Args:
        seed: int
            Unbounded mixed seed.
    return: int
        Non-negative seed in [0, 2**63 - 2].
    """
    return int(seed) % _TORCH_SEED_MOD

class InMemoryTimeStepDataset(Dataset):
    """
    In-memory dataset for graph data.
    Loads all preprocessed graphs into memory for fast training.
    Each graph_*.pt file corresponds to one trajectory (a sequence of timesteps).
    """
    data: list
    trajectory_boundaries: list[tuple[int, int]]

    def __init__(self, sample_dir, device):
        self.data = []
        self.trajectory_boundaries = []
        sample_files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.startswith("graph_")
                               and f.endswith(".pt")])
        print(f"\tFound {len(sample_files)} sample files")
        for sample_file in sample_files:
            sample_data = torch.load(sample_file, map_location=device, weights_only=False)
            start = len(self.data)
            self.data.extend(sample_data)  # Flatten all time steps into one list
            self.trajectory_boundaries.append((start, len(self.data)))

        print(f"\tLoaded the dataset with {len(self.data)} samples ({self.num_trajectories} trajectories)")

    @property
    def num_trajectories(self):
        return len(self.trajectory_boundaries)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class InMemoryNoisedDataset(InMemoryTimeStepDataset):
    """
    Wraps InMemoryTimeStepDataset to inject per-node Gaussian noise on position x(t)
    at every __getitem__ call. Noise RNG seeds depend on (training epoch, sample index) so each
    epoch uses a different draw at the same noise scales; call set_noise_epoch from the trainer
    at the start of each training epoch (see LitModule.on_train_epoch_start).

    Noise affects:
      - Node features: fully recomputed from noised position/velocity via build_valued_node_features
      - Edge features: recomputed from noised positions via build_edge_features
      - Target acceleration: corrected via a_tilde = a_star - (1 + gamma) * eps / dt^2
      - x_pos_t and x_vel_t: propagated as noised (for curvature loss coherence)

    Unchanged:
      - x(t-1) (ground truth), x(t+1) (target), rest position, von Mises stress, node_force, x_bc
    """
    noise_scale: float
    gamma: float
    dt: float
    node_stats_mean: torch.Tensor
    node_stats_std: torch.Tensor
    edge_stats_mean: torch.Tensor
    edge_stats_std: torch.Tensor
    target_stats_mean_acc: torch.Tensor
    target_stats_std_acc: torch.Tensor
    base_seed: int
    node_norm_scale: torch.Tensor
    node_norm_bias: torch.Tensor
    node_denorm_scale: torch.Tensor
    edge_norm_scale: torch.Tensor
    edge_norm_bias: torch.Tensor
    edge_denorm_scale: torch.Tensor
    target_acc_norm_scale: torch.Tensor
    noise_feature_update_mode: str

    def __init__(self, sample_dir, device, noise_scale, gamma, dt, node_stats, edge_stats, target_stats, base_seed,
                 preprocess_cfg, noise_also_prev, noise_prev_scale, noise_feature_update_mode):
        """
        Args:
            sample_dir: str
                Directory containing preprocessed graph_*.pt files.
            device: str or torch.device
                Device to load data onto.
            noise_scale: float
                Standard deviation of Gaussian noise added to positions.
            gamma: float
                Damping coefficient for acceleration correction.
            dt: float
                Simulation time step size.
            node_stats: Stats
                Node feature normalization statistics.
            edge_stats: Stats
                Edge feature normalization statistics.
            target_stats: Stats
                Target normalization statistics.
            base_seed: int
                Base seed mixed with training epoch and sample index for noise RNG (see set_noise_epoch).
            preprocess_cfg: PreprocessYaml
            noise_also_prev: bool
                Whether to also inject noise on the previous-step position x(t-1).
            noise_prev_scale: float
                Scale of noise on x(t-1), typically much smaller than noise_scale_train.
            noise_feature_update_mode: str
                Feature update mode for noisy samples: "exact" or "taylor".
        """
        super().__init__(sample_dir, device)
        if preprocess_cfg.time_discretization_method != "second_order_central":
            raise ValueError(
                f"Noise injection only supports second_order_central time discretization, "
                f"got '{preprocess_cfg.time_discretization_method}'. The acceleration correction "
                f"formula -(1+gamma)*eps/dt^2 is derived for central differences and is incorrect "
                f"for other schemes (e.g. Newmark).")
        self.noise_scale = noise_scale
        self.gamma = gamma
        self.dt = dt
        self.node_stats_mean = node_stats.mean.clone()
        self.node_stats_std = node_stats.std.clone()
        self.edge_stats_mean = edge_stats.mean.clone()
        self.edge_stats_std = edge_stats.std.clone()
        self.target_stats_mean_acc = target_stats.mean[:3].clone()
        self.target_stats_std_acc = target_stats.std[:3].clone()
        self.base_seed = int(base_seed)
        self.preprocess_cfg = preprocess_cfg
        self.noise_also_prev = noise_also_prev
        self.noise_prev_scale = noise_prev_scale
        self.noise_feature_update_mode = str(noise_feature_update_mode).lower()
        if self.noise_feature_update_mode not in {"exact", "taylor"}:
            raise ValueError(
                f"Unknown noise_feature_update_mode '{noise_feature_update_mode}'. Expected 'exact' or 'taylor'."
            )
        (self.node_norm_scale, self.node_norm_bias,
         self.node_denorm_scale) = build_affine_normalizer_tensors(self.node_stats_mean, self.node_stats_std,
                                                                    self.preprocess_cfg.normalizer_method)
        (self.edge_norm_scale, self.edge_norm_bias,
         self.edge_denorm_scale) = build_affine_normalizer_tensors(self.edge_stats_mean, self.edge_stats_std,
                                                                    self.preprocess_cfg.normalizer_method)
        self.target_acc_norm_scale, _, _ = build_affine_normalizer_tensors(
            self.target_stats_mean_acc, self.target_stats_std_acc, self.preprocess_cfg.normalizer_method
        )
        self._build_node_features_scripted, self._build_edge_features_scripted = make_scripted_feature_builders(
            self.preprocess_cfg)
        self._noise_epoch = multiprocessing.Value('i', 0)

    def set_noise_epoch(self, epoch):
        """
        Set training epoch used to derive noise seeds so each epoch draws different noise at the same scales.

        Args:
            epoch: int
                Current training epoch (0-based), typically Lightning's current_epoch.
        return: None
        """
        self._noise_epoch.value = int(epoch)

    def _normalize_node_features(self, node_attr_raw):
        return apply_affine_normalization(node_attr_raw, self.node_norm_scale, self.node_norm_bias)

    def _normalize_edge_features(self, edge_attr_raw):
        return apply_affine_normalization(edge_attr_raw, self.edge_norm_scale, self.edge_norm_bias)

    def _denormalize_node_features(self, node_attr_norm):
        return apply_affine_denormalization(node_attr_norm, self.node_stats_mean, self.node_denorm_scale,
                                            self.preprocess_cfg.normalizer_method)

    def _denormalize_edge_features(self, edge_attr_norm):
        return apply_affine_denormalization(edge_attr_norm, self.edge_stats_mean, self.edge_denorm_scale,
                                            self.preprocess_cfg.normalizer_method)

    def _build_features_exact(self, data, x_noised, v_noised, x_pos_tm1_prev):
        node_attr_raw, _ = self._build_node_features_scripted(x_noised, v_noised, x_pos_tm1_prev, data.rest_pos,
                                                              data.node_force)
        edge_attr_raw, _ = self._build_edge_features_scripted(x_noised, v_noised, x_pos_tm1_prev, data.rest_pos,
                                                              data.node_force, data.edge_index)
        return node_attr_raw, edge_attr_raw

    def _build_features_taylor(self, data, noise_added_t, noise_added_prev_tm1):
        x_clean = data.x_pos_t
        v_clean = data.x_vel_t
        x_tm1_clean = data.x_pos_tm1_prev
        delta_x_t = noise_added_t
        delta_x_tm1 = torch.zeros_like(x_tm1_clean) if noise_added_prev_tm1 is None else noise_added_prev_tm1
        delta_v_t = noise_added_t / self.dt
        if noise_added_prev_tm1 is not None:
            delta_v_t = delta_v_t - noise_added_prev_tm1 / self.dt

        node_attr_base_raw = self._denormalize_node_features(data.x[:, 1:])
        edge_attr_base_raw = self._denormalize_edge_features(data.edge_attr)

        def node_feature_fn(x_t, v_t, x_tm1):
            out, _ = self._build_node_features_scripted(x_t, v_t, x_tm1, data.rest_pos, data.node_force)
            return out

        def edge_feature_fn(x_t, v_t, x_tm1):
            out, _ = self._build_edge_features_scripted(x_t, v_t, x_tm1, data.rest_pos, data.node_force,
                                                        data.edge_index)
            return out

        _, node_delta_raw = torch.autograd.functional.jvp(
            node_feature_fn,
            (x_clean, v_clean, x_tm1_clean),
            (delta_x_t, delta_v_t, delta_x_tm1),
            create_graph=False,
            strict=False
        )
        _, edge_delta_raw = torch.autograd.functional.jvp(
            edge_feature_fn,
            (x_clean, v_clean, x_tm1_clean),
            (delta_x_t, delta_v_t, delta_x_tm1),
            create_graph=False,
            strict=False
        )
        return node_attr_base_raw + node_delta_raw, edge_attr_base_raw + edge_delta_raw

    def __getitem__(self, idx):
        """
        Returns a noised copy of the sample at idx.
        Noise seeds depend on (epoch set via set_noise_epoch, sample index, base_seed) so each epoch
        uses different noise at the same scales.


        Args:
            idx: int
                Sample index.
        return: Data
            PyG Data with noised input features, recomputed edge features, and corrected acceleration target.
        """
        data = self.data[idx]
        epoch_part = int(self._noise_epoch.value) * _NOISE_EPOCH_SEED_STRIDE
        seed_t = self.base_seed + epoch_part + 2 * int(idx)
        seed_tm1 = self.base_seed + epoch_part + 2 * int(idx) + 1
        noise_generator_t = torch.Generator(device=data.x_pos_t.device)
        noise_generator_t.manual_seed(_fold_torch_seed(seed_t))
        noise_added_t = self.noise_scale * torch.randn(data.x_pos_t.shape, dtype=data.x_pos_t.dtype,
                                                       device=data.x_pos_t.device, generator=noise_generator_t)  # (N, 3)
        fixed_mask = (data.x_bc > 0.5).squeeze(-1)  # (N,) True for wall-clamped nodes
        noise_added_t[fixed_mask] = 0.0
        # x_pos_t and x_vel_t are raw (not normalized): we add noise in raw space, then
        # rebuild node/edge features and normalize them with precomputed train statistics. The train statistics are
        # thus not affected by noise.
        x_noised = data.x_pos_t + noise_added_t
        # NOTE: eventually, noise also previous, but much less in scale
        x_pos_tm1_prev = data.x_pos_tm1_prev.clone()
        noise_added_prev_tm1 = None
        if self.noise_also_prev and self.preprocess_cfg.enable_past_pos_displ:
            noise_generator_tm1 = torch.Generator(device=data.x_pos_t.device)
            noise_generator_tm1.manual_seed(_fold_torch_seed(seed_tm1))
            noise_added_prev_tm1 = self.noise_prev_scale * torch.randn(data.x_pos_tm1_prev.shape,
                                                                       dtype=data.x_pos_tm1_prev.dtype,
                                                                       device=data.x_pos_t.device,
                                                                       generator=noise_generator_tm1)
            noise_added_prev_tm1[fixed_mask] = 0.0
            x_pos_tm1_prev += noise_added_prev_tm1
        # Propagate on velocity v[t] = x[t] + noise[t] - (x[t-1] - noise[t-1]) ---> v[t] + noise[t] - noise[t-1]
        v_noised = data.x_vel_t + noise_added_t / self.dt
        if noise_added_prev_tm1 is not None:
            v_noised = v_noised - noise_added_prev_tm1 / self.dt
        if self.noise_feature_update_mode == "taylor":
            node_attr_raw, edge_attr_raw = self._build_features_taylor(data, noise_added_t, noise_added_prev_tm1)
        else:
            node_attr_raw, edge_attr_raw = self._build_features_exact(data, x_noised, v_noised, x_pos_tm1_prev)
        node_attr_norm = self._normalize_node_features(node_attr_raw)
        new_x = torch.cat([data.x_bc, node_attr_norm], dim=1)

        # Edge features
        new_edge_attr = self._normalize_edge_features(edge_attr_raw)

        # Target acceleration correction (in normalized space)
        # position-correcting acceleration is a_P = a_star - 2 eps / dt^2
        # velocity-correcting acceleration is a_V = a_star - eps / dt^2
        # Thus a_tilde = a_star - (1 + gamma) * eps / dt^2
        acc_correction = -(1.0 + self.gamma) * noise_added_t / (self.dt ** 2)
        # new_y is target
        new_y = data.y.clone()
        # FIXME: hardcoded target handling
        new_y[:, :3] += acc_correction * self.target_acc_norm_scale

        return Data(x=new_x, edge_attr=new_edge_attr, # input node and edge features
                    y=new_y, # target
                    # Other features helpers for rollout or others
                    edge_index=data.edge_index,
                    y_acc_t=data.y_acc_t, # kept the same because it should be non-noised ground truth
                    x_pos_tm1_prev=x_pos_tm1_prev,
                    x_pos_t=x_noised,
                    x_vel_t=v_noised,
                    boundary_left_tri_index=data.boundary_left_tri_index,
                    boundary_right_tri_index=data.boundary_right_tri_index,
                    t=data.t)