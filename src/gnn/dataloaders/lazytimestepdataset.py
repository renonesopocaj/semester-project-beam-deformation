"""
lazytimestepdataset.py

Lazy (on-disk) counterparts to InMemoryTimeStepDataset and InMemoryNoisedDataset.
Data is loaded from disk on demand with a file-level LRU cache, keeping only
cache_size trajectory files in RAM at once instead of the full dataset.
Two versions:
- LazyTimeStepDataset: on-demand loading with LRU file cache, drop-in for InMemoryTimeStepDataset.
- LazyNoisedDataset: adds the same per-epoch Gaussian noise injection as InMemoryNoisedDataset,
    sharing all noise logic via _NoisedMixin.
"""

import bisect
import collections
import os
import torch
from torch_geometric.data import Dataset
from src.gnn.dataloaders.inmemorytimestepdataset import _NoisedMixin


class LazyTimeStepDataset(Dataset):
    """
    On-disk dataset that loads graph_*.pt trajectory files on demand.
    Only cache_size trajectory files are kept in RAM at any one time; the rest are
    loaded lazily and evicted with an LRU policy.

    API is identical to InMemoryTimeStepDataset: same trajectory_boundaries,
    num_trajectories, __len__, and __getitem__ signatures. Drop-in replacement
    with an extra cache_size constructor argument.

    The scan phase at __init__ loads each file once solely to record its timestep
    count, then immediately discards the data, so peak memory during construction
    is bounded by the largest single trajectory file.
    """
    file_paths: list[str]
    trajectory_boundaries: list[tuple[int, int]]
    _cumulative_lengths: list[int]
    _cache: collections.OrderedDict
    _cache_size: int
    _device: str

    def __init__(self, sample_dir, device, cache_size):
        """
        Args:
            sample_dir: str
                Directory containing preprocessed graph_*.pt files.
            device: str or torch.device
                Device to map tensors onto when loading files from disk.
            cache_size: int
                Maximum number of trajectory files held in RAM simultaneously.
                A value of 1 is sufficient for sequential (rollout) access;
                use 2-4 for training with mild shuffling across trajectories.
        """
        self._device = str(device)
        self._cache_size = int(cache_size)
        self._cache = collections.OrderedDict()
        self.file_paths = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                                  if f.startswith("graph_") and f.endswith(".pt")])
        if not self.file_paths:
            raise FileNotFoundError(f"No graph_*.pt files found in {sample_dir}")
        print(f"\tFound {len(self.file_paths)} sample files")
        # Load each file to record its length, then immediately discard to free RAM.
        lengths = []
        for path in self.file_paths:
            traj_data = torch.load(path, map_location="cpu", weights_only=False)
            lengths.append(len(traj_data))
            del traj_data
        # Prefix-sum array: _cumulative_lengths[i] = total samples across files 0 .. i-1.
        self._cumulative_lengths = [0]
        for length in lengths:
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)
        # list of (start, end) indices.
        self.trajectory_boundaries = \
            [(self._cumulative_lengths[i], self._cumulative_lengths[i + 1]) for i in range(len(self.file_paths))]
        total = self._cumulative_lengths[-1]
        print(f"\tLazy dataset indexed: {total} samples ({self.num_trajectories} trajectories), "
              f"cache_size={cache_size}")

    @property
    def num_trajectories(self):
        return len(self.trajectory_boundaries)

    def _load_file(self, file_idx):
        """
        Return the trajectory at file_idx, loading from disk if not cached and
        evicting the least-recently-used entry when the cache is full.

        Args:
            file_idx: int
                Index into self.file_paths.
        return: list[Data]
            All timestep Data objects for that trajectory.
        """
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]
        traj_data = torch.load(self.file_paths[file_idx], map_location=self._device, weights_only=False)
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)  # evict least-recently-used entry
        self._cache[file_idx] = traj_data
        return traj_data

    def _resolve_idx(self, idx):
        """
        Map a flat sample index to (file_idx, local_idx within that file) using binary search.

        Args:
            idx: int
                Flat sample index in [0, len(self)).
        return: (int, int)
            file_idx and local offset within that file.
        """
        file_idx = bisect.bisect_right(self._cumulative_lengths, idx) - 1
        local_idx = idx - self._cumulative_lengths[file_idx]
        return file_idx, local_idx

    def __getitem__(self, idx):
        file_idx, local_idx = self._resolve_idx(idx)
        traj_data = self._load_file(file_idx)
        return traj_data[local_idx]

    def __len__(self):
        return self._cumulative_lengths[-1]


class LazyNoisedDataset(_NoisedMixin, LazyTimeStepDataset):
    """
    Lazy on-disk counterpart to InMemoryNoisedDataset.
    Loads data from disk on demand (with an LRU file cache) and applies the same
    per-epoch Gaussian noise injection as InMemoryNoisedDataset at every __getitem__ call.

    All noise logic and RNG seeding are identical to InMemoryNoisedDataset and are
    shared through _NoisedMixin. The only difference is that raw Data objects come
    from disk via LazyTimeStepDataset instead of from an in-memory list.
    """

    def __init__(self, sample_dir, device, cache_size, noise_scale, gamma, dt, node_stats, edge_stats,
                 target_stats, base_seed, preprocess_cfg, noise_also_prev, noise_prev_scale,
                 noise_feature_update_mode, noise_per_epoch):
        """
        Args:
            sample_dir: str
                Directory containing preprocessed graph_*.pt files.
            device: str or torch.device
                Device to map tensors onto when loading files from disk.
            cache_size: int
                Maximum number of trajectory files held in RAM simultaneously.
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
                Base seed mixed with epoch and sample index for noise RNG (see set_noise_epoch).
            preprocess_cfg: PreprocessYaml
            noise_also_prev: bool
                Whether to also inject noise on the previous-step position x(t-1).
            noise_prev_scale: float
                Scale of noise on x(t-1), typically much smaller than noise_scale_train.
            noise_feature_update_mode: str
                Feature update mode for noisy samples: "exact" or "taylor".
            noise_per_epoch: bool
                If True, noise seeds vary each epoch; if False, fixed per sample index.
        """
        LazyTimeStepDataset.__init__(self, sample_dir, device, cache_size)
        self._init_noise(noise_scale, gamma, dt, node_stats, edge_stats, target_stats, base_seed,
                         preprocess_cfg, noise_also_prev, noise_prev_scale, noise_feature_update_mode, noise_per_epoch)

    def _get_raw(self, idx):
        """
        Return the raw (un-noised) Data object at flat index idx via the lazy disk cache.
        Delegates to LazyTimeStepDataset.__getitem__ explicitly to bypass the noise-injecting
        __getitem__ that _NoisedMixin installs higher in the MRO.

        Args:
            idx: int
        return: Data
        """
        return LazyTimeStepDataset.__getitem__(self, idx)
