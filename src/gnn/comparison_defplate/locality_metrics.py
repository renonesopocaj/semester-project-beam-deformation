"""
locality_metrics.py

Six locality metrics that analyze how spatially local or global a signal is on
the mesh graph.  Each metric is a Metric subclass (strategy pattern).

All metrics operate on the free-node subset of the signal via
traj.free_signal(signal_name).  The signal has shape (T, N_free, C); metrics
average over channels C and over timesteps T (median) to produce one curve or
scalar per trajectory.
"""

import numpy as np
from sklearn.decomposition import PCA

from src.gnn.comparison_defplate.metrics_base import CurveMetric, ScalarMetric, CurveResult, ScalarResult
from src.gnn.comparison_defplate.graph_ops import (hop_distance_matrix, graph_diameter,
                                                   laplacian_eigenvectors, load_distance_vector)

EPS = 1e-12


def _median_over_channels(values_list):
    """
    Collapse a list of per-timestep-per-channel 1-D arrays by taking their
    element-wise median.

    Args:
        values_list: list of np.ndarray  shape (max_hop,)
    return: np.ndarray  shape (max_hop,)
    """
    return np.median(np.stack(values_list, axis=0), axis=0)


def _free_signal_flat(traj, signal_name):
    """
    Return signal as (T, N_free * C) for easy per-channel iteration.

    Args:
        traj: TrajectorySample
        signal_name: str
    return: (np.ndarray, int, int)
        flat: (T, N_free * C),  n_free: int,  n_channels: int
    """
    sig = traj.free_signal(signal_name)          # (T, N_free, C)
    t, n_free, c = sig.shape
    flat = sig.reshape(t, n_free, c)
    return flat, n_free, c


# ─── 1. Graph-distance autocorrelation ───────────────────────────────────────

class GraphAutocorrelation(CurveMetric):
    """
    Per-hop Pearson correlation of node values at graph distance k.

    ρ(k) = corr(y_i, y_j | d(i,j) == k)

    Averaged over timesteps (median) and channels (median).
    """

    @property
    def name(self):
        return "graph_autocorrelation"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: CurveResult  x=hop distances 1..D, y=median correlation
        """
        dist = hop_distance_matrix(traj)
        diam = graph_diameter(traj)
        sig, n_free, c = _free_signal_flat(traj, signal_name)   # (T, N_free, C)
        # Free-node sub-matrix of hop distances
        free_idx = np.where(traj.free_mask)[0]
        dist_free = dist[np.ix_(free_idx, free_idx)]             # (N_free, N_free)
        hops = np.arange(1, diam + 1)
        rho_per_tc = []
        for t in range(sig.shape[0]):
            for ch in range(c):
                y = sig[t, :, ch]
                rho_k = []
                for k in hops:
                    mask = dist_free == k
                    if mask.sum() < 2:
                        rho_k.append(np.nan)
                        continue
                    pairs_i, pairs_j = np.where(mask)
                    rho = np.corrcoef(y[pairs_i], y[pairs_j])[0, 1]
                    rho_k.append(float(rho))
                rho_per_tc.append(rho_k)
        rho_arr = np.nanmedian(np.array(rho_per_tc), axis=0)
        return CurveResult(x=hops.astype(float), y=rho_arr,
                           xlabel="Graph hop distance k",
                           ylabel="Correlation ρ(k)")


# ─── 2. Graph variogram ───────────────────────────────────────────────────────

class GraphVariogram(CurveMetric):
    """
    Empirical semi-variogram: γ(k) = ½ E[(y_i − y_j)² | d(i,j) == k].

    Normalized by γ(D) to lie in [0, 1].  Averaged over timesteps and channels.
    """

    @property
    def name(self):
        return "graph_variogram"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: CurveResult  x=hop distances, y=normalized semi-variogram
        """
        dist = hop_distance_matrix(traj)
        diam = graph_diameter(traj)
        sig, n_free, c = _free_signal_flat(traj, signal_name)
        free_idx = np.where(traj.free_mask)[0]
        dist_free = dist[np.ix_(free_idx, free_idx)]
        hops = np.arange(1, diam + 1)
        gamma_per_tc = []
        for t in range(sig.shape[0]):
            for ch in range(c):
                y = sig[t, :, ch]
                gamma_k = []
                for k in hops:
                    mask = dist_free == k
                    if mask.sum() < 1:
                        gamma_k.append(np.nan)
                        continue
                    pairs_i, pairs_j = np.where(mask)
                    gamma = 0.5 * np.mean((y[pairs_i] - y[pairs_j]) ** 2)
                    gamma_k.append(float(gamma))
                gamma_per_tc.append(gamma_k)
        gamma_arr = np.nanmedian(np.array(gamma_per_tc), axis=0)
        denom = gamma_arr[-1] if (np.isfinite(gamma_arr[-1]) and abs(gamma_arr[-1]) > EPS) else 1.0
        gamma_norm = gamma_arr / denom
        return CurveResult(x=hops.astype(float), y=gamma_norm,
                           xlabel="Graph hop distance k",
                           ylabel="Normalized variogram γ(k) / γ(D)")


# ─── 3. Laplacian spectral cumulative energy ──────────────────────────────────

class LaplacianSpectralEnergy(CurveMetric):
    """
    Cumulative fraction of signal energy in the first m graph-Fourier modes.

    E(m) = Σ_{ℓ≤m} |ŷ_ℓ|² / Σ_ℓ |ŷ_ℓ|²

    where ŷ_ℓ = u_ℓᵀ y, u_ℓ the ℓ-th Laplacian eigenvector.

    Uses ALL nodes (signal projected onto full eigenvectors; free nodes only
    would require computing sub-graph eigenvectors which changes the spectrum).
    Averaged over timesteps and channels.
    """

    @property
    def name(self):
        return "laplacian_spectral_energy"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: CurveResult  x=mode index fraction, y=cumulative energy fraction
        """
        _, eigvecs = laplacian_eigenvectors(traj)     # (N, N)
        sig_full = traj.signals[signal_name]          # (T, N, C), all nodes
        t, n, c = sig_full.shape
        cum_energy_per_tc = []
        for ti in range(t):
            for ch in range(c):
                y = sig_full[ti, :, ch]
                proj = eigvecs.T @ y                  # (N,) Fourier coefficients
                power = proj ** 2
                total = power.sum()
                if total < EPS:
                    continue
                cum_energy = np.cumsum(power) / total
                cum_energy_per_tc.append(cum_energy)
        if not cum_energy_per_tc:
            xs = np.linspace(0, 1, n)
            return CurveResult(x=xs, y=np.zeros(n), xlabel="Mode index fraction",
                               ylabel="Cumulative energy E(m)")
        cum_arr = np.nanmedian(np.stack(cum_energy_per_tc, axis=0), axis=0)
        xs = np.arange(1, n + 1) / n
        return CurveResult(x=xs, y=cum_arr,
                           xlabel="Mode index fraction m/N",
                           ylabel="Cumulative energy E(m)")


# ─── 4. Local variation ratio (LVR) ──────────────────────────────────────────

class LocalVariationRatio(ScalarMetric):
    """
    LVR(y) = Σ_{(i,j)∈E_free} (y_i - y_j)² / Σ_i (y_i - ȳ)²

    where E_free contains edges with both endpoints free.
    Averaged over timesteps and channels via median.
    """

    @property
    def name(self):
        return "local_variation_ratio"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: ScalarResult
        """
        sig, n_free, c = _free_signal_flat(traj, signal_name)   # (T, N_free, C)
        free_idx_set = set(np.where(traj.free_mask)[0])
        src, dst = traj.edge_index
        # Remap global edge indices to local free-node indices
        free_pairs = [(i, j) for i, j in zip(src, dst)
                      if i in free_idx_set and j in free_idx_set]
        if not free_pairs:
            return ScalarResult(value=np.nan, label="LVR")
        free_global_to_local = {g: l for l, g in enumerate(sorted(free_idx_set))}
        local_pairs = np.array([(free_global_to_local[i], free_global_to_local[j])
                                for i, j in free_pairs])
        lvr_vals = []
        for ti in range(sig.shape[0]):
            for ch in range(c):
                y = sig[ti, :, ch]
                numerator = np.sum((y[local_pairs[:, 0]] - y[local_pairs[:, 1]]) ** 2)
                denominator = np.sum((y - y.mean()) ** 2)
                if denominator < EPS:
                    continue
                lvr_vals.append(numerator / denominator)
        value = float(np.nanmedian(lvr_vals)) if lvr_vals else np.nan
        return ScalarResult(value=value, label="LVR")


# ─── 5. Boundary/load-distance dependence ────────────────────────────────────

class LoadDistanceDependence(CurveMetric):
    """
    Mean absolute signal magnitude as a function of distance from the load/
    clamped boundary:  E[|y_i| | d_load(i) = k] vs k.

    Computed on free nodes only, averaged over timesteps (median over t and ch).
    """

    @property
    def name(self):
        return "load_distance_dependence"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: CurveResult  x=load distance, y=mean |y|
        """
        load_dist = load_distance_vector(traj)       # (N,) all nodes
        free_idx = np.where(traj.free_mask)[0]
        dist_free = load_dist[free_idx]              # (N_free,)
        sig = traj.free_signal(signal_name)          # (T, N_free, C)
        max_dist = int(dist_free.max()) if dist_free.size > 0 else 0
        if max_dist == 0:
            xs = np.array([0.0])
            return CurveResult(x=xs, y=np.array([np.nan]),
                               xlabel="Load distance k (hops)", ylabel="E[|y| | d=k]")
        hops = np.arange(0, max_dist + 1)
        mean_per_tc = []
        for ti in range(sig.shape[0]):
            for ch in range(sig.shape[2]):
                y = np.abs(sig[ti, :, ch])
                means_k = []
                for k in hops:
                    mask = dist_free == k
                    means_k.append(y[mask].mean() if mask.any() else np.nan)
                mean_per_tc.append(means_k)
        means_arr = np.nanmedian(np.array(mean_per_tc), axis=0)
        return CurveResult(x=hops.astype(float), y=means_arr,
                           xlabel="Load distance k (hops)",
                           ylabel="E[|y| | d=k]")


# ─── 6. PCA cumulative variance ───────────────────────────────────────────────

class PerTrajectoryPCA(CurveMetric):
    """
    PCA on the full-field target across timesteps.

    z_t = flatten(y_t) ∈ ℝ^{N_free * C}
    PCA on matrix [z_0 ; z_1 ; … ; z_{T-1}] ∈ ℝ^{T × (N_free*C)}.

    Returns cumulative explained variance ratio vs PC index.
    """

    @property
    def name(self):
        return "per_trajectory_pca"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: CurveResult  x=PC index, y=cumulative explained variance
        """
        sig = traj.free_signal(signal_name)           # (T, N_free, C)
        t, n_free, c = sig.shape
        z = sig.reshape(t, n_free * c)                # (T, features)
        n_components = min(t, n_free * c)
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(z)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        pc_idx = np.arange(1, n_components + 1)
        return CurveResult(x=pc_idx.astype(float), y=cum_var,
                           xlabel="Number of PCs",
                           ylabel="Cumulative explained variance")


# ─── registry ─────────────────────────────────────────────────────────────────

LOCALITY_METRICS = [
    GraphAutocorrelation(),
    GraphVariogram(),
    LaplacianSpectralEnergy(),
    LocalVariationRatio(),
    LoadDistanceDependence(),
    PerTrajectoryPCA(),
]
