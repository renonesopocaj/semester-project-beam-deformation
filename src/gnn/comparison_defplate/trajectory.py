"""
trajectory.py

Defines TrajectorySample, the common in-memory representation shared by all
loaders, metrics, and plotting code.

Signals are stored as numpy arrays of shape (T, N, C) where
  T = number of timesteps
  N = number of nodes (free + constrained)
  C = number of channels (3 for positional, 1 for scalar)

All loaders must produce exactly the five signal keys:
  "position", "displacement", "step_displacement", "velocity", "von_mises"
Only free (non-boundary) nodes are used for locality and frequency metrics;
the mask for this is stored as `free_mask` (bool, shape N).
"""

from dataclasses import dataclass, field
import numpy as np


SIGNAL_NAMES = ("position", "displacement", "step_displacement", "velocity", "von_mises")


@dataclass
class TrajectorySample:
    """
    Container for one trajectory from either dataset.

    Attributes:
        name: str
            Human-readable label, e.g. "biax_t04_fy-0.9" or "dp_000".
        edge_index: np.ndarray  shape (2, E), dtype int
            Static graph connectivity (undirected, COO format).
        free_mask: np.ndarray  shape (N,), dtype bool
            True for free/interior nodes; False for clamped/driven nodes.
        signals: dict
            Keys must be exactly SIGNAL_NAMES.  Each value has shape (T, N, C).
        extras: dict
            Cache for precomputed graph quantities (hop matrix, eigenvectors, …).
        params: dict
            Optional trajectory metadata (force values, geometry, etc.).
    """
    name: str
    edge_index: np.ndarray
    free_mask: np.ndarray
    signals: dict
    extras: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def num_nodes(self):
        return self.free_mask.shape[0]

    @property
    def num_timesteps(self):
        first_key = next(iter(self.signals))
        return self.signals[first_key].shape[0]

    def free_signal(self, signal_name):
        """
        Return signal restricted to free nodes.

        Args:
            signal_name: str
                One of SIGNAL_NAMES.
        return: np.ndarray
            Shape (T, N_free, C).
        """
        return self.signals[signal_name][:, self.free_mask, :]
