"""
frequency_metrics.py

Five temporal frequency metrics that characterize how rapidly a signal varies
over rollout steps.  All metrics operate on the free-node signal subset
(traj.free_signal) and summarize by taking the median over free nodes and
channels.

Signals with fewer than 4 timesteps are skipped (returned as NaN).
"""

import numpy as np
import scipy.signal as scipy_signal

from src.gnn.comparison_defplate.metrics_base import CurveMetric, ScalarMetric, CurveResult, ScalarResult

EPS = 1e-12
SAMPLING_RATE = 1.0          # step-index frequency (1 sample per rollout step)
HIGH_CUTOFF_FRACTION = 0.5   # fraction of Nyquist used as the high-freq threshold


def _welch_params(t):
    """
    Return nperseg and noverlap for Welch PSD on a series of length t.

    Args:
        t: int  series length
    return: (int, int) nperseg, noverlap
    """
    nperseg = min(64, max(4, t // 4))
    noverlap = nperseg // 2
    return nperseg, noverlap


def _compute_psd_all_series(sig_free):
    """
    Compute Welch PSD for every (node, channel) series in a free-node signal.

    Args:
        sig_free: np.ndarray  shape (T, N_free, C)
    return: (freqs, psd_matrix)
        freqs: np.ndarray  shape (F,)
        psd_matrix: np.ndarray  shape (F, N_free*C)
    """
    t, n_free, c = sig_free.shape
    if t < 4:
        return None, None
    flat = sig_free.reshape(t, n_free * c).T   # (N_free*C, T)
    # z-score each series over time
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    flat_z = (flat - mean) / (std + EPS)
    flat_z = np.nan_to_num(flat_z)
    nperseg, noverlap = _welch_params(t)
    freqs, psd = scipy_signal.welch(flat_z, fs=SAMPLING_RATE, window="hann",
                                    nperseg=nperseg, noverlap=noverlap,
                                    detrend="linear", scaling="density", axis=1)
    # psd shape: (N_free*C, F)  → transpose to (F, N_free*C)
    return freqs, psd.T


# ─── 1. Welch PSD ─────────────────────────────────────────────────────────────

class WelchPSD(CurveMetric):
    """
    Median Welch PSD across free nodes and channels vs normalized frequency.

    The PSD is log10-transformed before taking the median so that high-energy
    outliers don't dominate.
    """

    @property
    def name(self):
        return "welch_psd"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: CurveResult  x=frequency (0..0.5), y=median log10 PSD
        """
        sig = traj.free_signal(signal_name)
        freqs, psd = _compute_psd_all_series(sig)
        if freqs is None:
            return CurveResult(x=np.array([0.0, 0.5]), y=np.array([np.nan, np.nan]),
                               xlabel="Normalized frequency (cycles/step)",
                               ylabel="Median PSD (log10)")
        log_psd = np.log10(psd + EPS)           # (F, series)
        median_log = np.nanmedian(log_psd, axis=1)
        return CurveResult(x=freqs, y=median_log,
                           xlabel="Normalized frequency (cycles/step)",
                           ylabel="Median PSD (log10)")


# ─── 2. High-frequency energy ratio ──────────────────────────────────────────

class HighFreqEnergyRatio(ScalarMetric):
    """
    Fraction of spectral energy above HIGH_CUTOFF_FRACTION of Nyquist.

    HFR = Σ_{f ≥ f_c} PSD(f) / Σ_{f > 0} PSD(f)

    Median over all series (free nodes × channels).
    """

    @property
    def name(self):
        return "high_freq_energy_ratio"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: ScalarResult
        """
        sig = traj.free_signal(signal_name)
        freqs, psd = _compute_psd_all_series(sig)
        if freqs is None:
            return ScalarResult(value=np.nan, label="HFR")
        nyquist = 0.5 * SAMPLING_RATE
        high_cutoff = HIGH_CUTOFF_FRACTION * nyquist
        nonzero = freqs > 0
        high = freqs >= high_cutoff
        total_energy = psd[nonzero].sum(axis=0) + EPS
        high_energy = psd[high].sum(axis=0)
        ratios = high_energy / total_energy
        value = float(np.nanmedian(ratios))
        return ScalarResult(value=value, label="HFR")


# ─── 3. Spectral centroid ─────────────────────────────────────────────────────

class SpectralCentroid(ScalarMetric):
    """
    Frequency centroid of the PSD: Σ_f f·PSD(f) / Σ_f PSD(f), nonzero f only.

    Median over all series.  Normalized to [0, 1] by dividing by Nyquist.
    """

    @property
    def name(self):
        return "spectral_centroid"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: ScalarResult
        """
        sig = traj.free_signal(signal_name)
        freqs, psd = _compute_psd_all_series(sig)
        if freqs is None:
            return ScalarResult(value=np.nan, label="Spectral centroid")
        nonzero = freqs > 0
        f_nz = freqs[nonzero, None]      # (F_nz, 1)
        p_nz = psd[nonzero]              # (F_nz, series)
        centroids = (f_nz * p_nz).sum(axis=0) / (p_nz.sum(axis=0) + EPS)
        nyquist = 0.5 * SAMPLING_RATE
        centroids_norm = centroids / (nyquist + EPS)
        value = float(np.nanmedian(centroids_norm))
        return ScalarResult(value=value, label="Spectral centroid (normalized)")


# ─── 4. Normalized roughness ─────────────────────────────────────────────────

class NormalizedRoughness(ScalarMetric):
    """
    Var(Δx) / Var(x) per series (after z-scoring).

    Median over free nodes and channels.
    """

    @property
    def name(self):
        return "normalized_roughness"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: ScalarResult
        """
        sig = traj.free_signal(signal_name)
        t, n_free, c = sig.shape
        if t < 2:
            return ScalarResult(value=np.nan, label="Normalized roughness")
        roughness_vals = []
        for ch in range(c):
            y = sig[:, :, ch]           # (T, N_free)
            mean = y.mean(axis=0, keepdims=True)
            std = y.std(axis=0, keepdims=True)
            y_z = (y - mean) / (std + EPS)
            diff = np.diff(y_z, axis=0)  # (T-1, N_free)
            var_diff = np.var(diff, axis=0)
            var_y = np.var(y_z, axis=0)
            ratio = var_diff / (var_y + EPS)
            roughness_vals.append(ratio)
        all_ratios = np.concatenate(roughness_vals, axis=0)
        value = float(np.nanmedian(all_ratios))
        return ScalarResult(value=value, label="Normalized roughness")


# ─── 5. Lag-1 autocorrelation ─────────────────────────────────────────────────

class Lag1Autocorr(ScalarMetric):
    """
    Pearson lag-1 autocorrelation of each (node, channel) series, median over
    free nodes and channels.

    Near +1 → signal barely changes (low-frequency, easy prediction task).
    Near 0 or negative → signal changes rapidly (high-frequency).
    """

    @property
    def name(self):
        return "lag1_autocorrelation"

    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
        return: ScalarResult
        """
        sig = traj.free_signal(signal_name)
        t, n_free, c = sig.shape
        if t < 2:
            return ScalarResult(value=np.nan, label="Lag-1 ACF")
        acf_vals = []
        for ch in range(c):
            y = sig[:, :, ch]                 # (T, N_free)
            y0 = y[:-1]                       # (T-1, N_free)
            y1 = y[1:]                        # (T-1, N_free)
            mean0 = y0.mean(axis=0, keepdims=True)
            mean1 = y1.mean(axis=0, keepdims=True)
            num = ((y0 - mean0) * (y1 - mean1)).mean(axis=0)
            std0 = y0.std(axis=0)
            std1 = y1.std(axis=0)
            denom = std0 * std1
            valid = denom > EPS
            acf = np.full(n_free, np.nan)
            acf[valid] = num[valid] / denom[valid]
            acf_vals.append(acf)
        all_acf = np.concatenate(acf_vals, axis=0)
        value = float(np.nanmedian(all_acf))
        return ScalarResult(value=value, label="Lag-1 ACF")


# ─── registry ─────────────────────────────────────────────────────────────────

FREQUENCY_METRICS = [
    WelchPSD(),
    HighFreqEnergyRatio(),
    SpectralCentroid(),
    NormalizedRoughness(),
    Lag1Autocorr(),
]
