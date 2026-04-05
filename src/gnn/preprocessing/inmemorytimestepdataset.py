"""
inmemorytimestepdataset.py

In-memory dataset for graph data.
"""

from torch_geometric.data import Dataset, Data
from src.gnn.preprocessing.build_features import build_edge_features, build_node_features
import os
import torch

EPS = 1e-8

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
    at every __getitem__ call, so each epoch sees a different perturbation.

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
    target_stats_std_acc: torch.Tensor
    base_seed: int

    def __init__(self, sample_dir, device, noise_scale, gamma, dt, node_stats, edge_stats, target_stats, base_seed,
                 enable_velocity_edge):
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
                Base seed used to build deterministic per-sample noise.
        """
        super().__init__(sample_dir, device)
        self.noise_scale = noise_scale
        self.gamma = gamma
        self.dt = dt
        self.node_stats_mean = node_stats.mean.clone()
        self.node_stats_std = node_stats.std.clone()
        self.edge_stats_mean = edge_stats.mean.clone()
        self.edge_stats_std = edge_stats.std.clone()
        self.target_stats_std_acc = target_stats.std[:3].clone()
        self.base_seed = int(base_seed)
        self.enable_velocity_edge = enable_velocity_edge

    def __getitem__(self, idx):
        """
        Returns a noised copy of the sample at idx.
        - A fresh eps ~ N(0, noise_scale^2) is drawn every call, so every epoch produces different perturbations.
        - One call = one epoch, so this is correct


        Args:
            idx: int
                Sample index.
        return: Data
            PyG Data with noised input features, recomputed edge features, and corrected acceleration target.
        """
        data = self.data[idx]

        # Derive noise from a deterministic per-sample seed so repeated runs are reproducible.
        noise_generator = torch.Generator(device=data.x_pos_t.device)
        noise_generator.manual_seed(self.base_seed + int(idx))
        eps = self.noise_scale * torch.randn(data.x_pos_t.shape, dtype=data.x_pos_t.dtype,
                                             device=data.x_pos_t.device, generator=noise_generator)  # (N, 3)
        fixed_mask = (data.x_bc > 0.5).squeeze(-1)  # (N,) True for wall-clamped nodes
        eps[fixed_mask] = 0.0
        x_noised = data.x_pos_t + eps
        v_noised = data.x_vel_t + eps / self.dt

        # Node features: full recomputation from raw attributes
        # node_attr_raw = build_valued_node_features(x_noised, v_noised, data.rest_pos, data.node_force,
        #                                            self.enable_velocity_edge)
        # node_attr_norm = (node_attr_raw - self.node_stats_mean) / (self.node_stats_std + EPS)
        node_attr_norm = build_node_features(x_noised, v_noised, data.rest_pos, data.node_force,
                                             self.enable_velocity_edge)
        new_x = torch.cat([data.x_bc, node_attr_norm], dim=1)

        # Edge features: full recomputation due to norms
        edge_attr_raw = build_edge_features(x_noised, data.rest_pos, data.edge_index, v_noised,
                                            self.enable_velocity_edge)
        new_edge_attr = (edge_attr_raw - self.edge_stats_mean) / (self.edge_stats_std + EPS)

        # Target acceleration correction (in normalized space)
        # position-correcting acceleration is a_P = a_star - 2 eps / dt^2
        # velocity-correcting acceleration is a_V = a_star - eps / dt^2
        # Thus a_tilde = a_star - (1 + gamma) * eps / dt^2
        acc_correction = -(1.0 + self.gamma) * eps / (self.dt ** 2)
        # new_y is target
        new_y = data.y.clone()
        new_y[:, :3] += acc_correction / (self.target_stats_std_acc + EPS)

        return Data(x=new_x, edge_attr=new_edge_attr, y=new_y, # target
                    edge_index=data.edge_index,
                    y_acc_t=data.y_acc_t, # kept the same because it should be non-noised ground truth
                    x_pos_t=x_noised,
                    x_vel_t=v_noised,
                    boundary_left_tri_index=data.boundary_left_tri_index,
                    boundary_right_tri_index=data.boundary_right_tri_index,
                    t=data.t)