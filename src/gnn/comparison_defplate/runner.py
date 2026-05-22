"""
runner.py

Orchestrates the full biax_longer vs DeformingPlate comparison:

1. Load 10 trajectories from each dataset.
2. Compute all 11 metrics × 5 signals for every trajectory.
3. Save detail plots (220 total) and average plots (22 total).
4. Save summary.csv with per-trajectory scalar metric values plus
   Mann-Whitney U and Cliff's delta between datasets for each
   (signal, scalar_metric) combination.
5. Save the pooled Welch PSD headline figure.

Entry point: main(args) – called from scripts/run_dataset_comparison.py.
"""

import argparse
import csv
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy.stats as scipy_stats

from src.gnn.comparison_defplate.loaders import BiaxLongerLoader, DeformingPlateLoader, DeformingPlateNpzLoader
from src.gnn.comparison_defplate.locality_metrics import LOCALITY_METRICS
from src.gnn.comparison_defplate.frequency_metrics import FREQUENCY_METRICS
from src.gnn.comparison_defplate.metrics_base import ScalarResult
from src.gnn.comparison_defplate.plotting import detail_plot, average_plot, pooled_psd_plot
from src.gnn.comparison_defplate.trajectory import SIGNAL_NAMES

ALL_METRICS = LOCALITY_METRICS + FREQUENCY_METRICS


# ─── result collection ────────────────────────────────────────────────────────

def _compute_results_for_trajectories(trajectories):
    """
    Compute all metrics on all signals for a list of trajectories.

    Args:
        trajectories: list of TrajectorySample
    return: dict
        Structure: results[traj_idx][metric_name][signal_name] = CurveResult or ScalarResult
    """
    results = {}
    for traj_idx, traj in tqdm(enumerate(trajectories), total=len(trajectories)):
        results[traj_idx] = {}
        for metric in tqdm(ALL_METRICS, desc=f"{traj.name}: metrics", leave=False):
            results[traj_idx][metric.name] = {}
            for sig_name in tqdm(SIGNAL_NAMES, desc=f"{metric.name}: signals", leave=False):
                if sig_name not in traj.signals:
                    results[traj_idx][metric.name][sig_name] = None
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = metric.compute(traj, sig_name)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] {metric.name} on {sig_name} for {traj.name}: {exc}")
                    result = None
                results[traj_idx][metric.name][sig_name] = result
        print(f"  computed metrics for {traj.name}")
    return results


# ─── plotting helpers ─────────────────────────────────────────────────────────

def _save_detail_plots(trajectories, results, dataset_label, output_dir):
    """
    Save one figure per (trajectory, metric), all 5 signals overlaid.

    Args:
        trajectories: list of TrajectorySample
        results: dict from _compute_results_for_trajectories
        dataset_label: str
        output_dir: Path
    """
    detail_dir = output_dir / "detail" / dataset_label
    for traj_idx, traj in enumerate(trajectories):
        for metric in ALL_METRICS:
            results_by_signal = results[traj_idx][metric.name]
            out_path = detail_dir / traj.name / f"{metric.name}.png"
            detail_plot(metric, results_by_signal, traj.name, dataset_label, out_path)


def _save_average_plots(trajectories, results, dataset_label, output_dir):
    """
    Save one summary figure per (dataset, metric) over all trajectories.

    Args:
        trajectories: list of TrajectorySample
        results: dict from _compute_results_for_trajectories
        dataset_label: str
        output_dir: Path
    return: dict  metric_name → signal_name → list of results (for later use)
    """
    avg_dir = output_dir / "average"
    all_by_metric_signal = {}
    for metric in tqdm(ALL_METRICS, desc=f"metrics", total=len(ALL_METRICS)):
        all_results_by_signal = {sig: [] for sig in SIGNAL_NAMES}
        for traj_idx in tqdm(range(len(trajectories)), desc=f"Trajectoies", total=len(trajectories), leave=False):
            for sig_name in tqdm(SIGNAL_NAMES, desc="Signal", leave=False, total=len(SIGNAL_NAMES)):
                all_results_by_signal[sig_name].append(
                    results[traj_idx][metric.name].get(sig_name))
        out_path = avg_dir / f"{dataset_label}_{metric.name}.png"
        average_plot(metric, all_results_by_signal, dataset_label, out_path)
        all_by_metric_signal[metric.name] = all_results_by_signal
    return all_by_metric_signal


# ─── statistics ───────────────────────────────────────────────────────────────

def _cliffs_delta(a, b):
    """
    Cliff's delta: proportion of (b_j > a_i) minus (b_j < a_i) pairs.

    Args:
        a: np.ndarray  1-D
        b: np.ndarray  1-D
    return: float  in [-1, 1]
    """
    sign_mat = np.sign(b[:, None] - a[None, :])
    return float(np.mean(sign_mat))


def _compute_statistics(biax_all, dp_all):
    """
    Compute Mann-Whitney U and Cliff's delta for every (signal, scalar_metric).

    Args:
        biax_all: dict  metric_name → signal_name → list of results (biax)
        dp_all: dict  metric_name → signal_name → list of results (DP)
    return: list of dict  one row per (signal, metric) comparison
    """
    rows = []
    for metric in ALL_METRICS:
        if metric.kind != "scalar":
            continue
        for sig_name in SIGNAL_NAMES:
            a_vals = np.array([r.value for r in biax_all[metric.name].get(sig_name, [])
                               if isinstance(r, ScalarResult) and np.isfinite(r.value)])
            b_vals = np.array([r.value for r in dp_all[metric.name].get(sig_name, [])
                               if isinstance(r, ScalarResult) and np.isfinite(r.value)])
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            stat, p = scipy_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            delta = _cliffs_delta(a_vals, b_vals)
            rows.append({
                "signal": sig_name,
                "metric": metric.name,
                "biax_median": float(np.median(a_vals)),
                "dp_median": float(np.median(b_vals)),
                "mann_whitney_u": float(stat),
                "p_value": float(p),
                "cliffs_delta": delta,
            })
    return rows


def _save_summary_csv(stat_rows, output_dir):
    """
    Write summary.csv.

    Args:
        stat_rows: list of dict
        output_dir: Path
    """
    if not stat_rows:
        return
    csv_path = output_dir / "summary.csv"
    fieldnames = list(stat_rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stat_rows)
    print(f"Saved {csv_path}")


def _save_per_traj_csv(biax_trajectories, biax_results, dp_trajectories, dp_results, output_dir):
    """
    Write per-trajectory scalar values for all metrics.

    Args:
        biax_trajectories: list of TrajectorySample
        biax_results: dict
        dp_trajectories: list of TrajectorySample
        dp_results: dict
        output_dir: Path
    """
    rows = []
    datasets = [("biax_longer", biax_trajectories, biax_results)]
    if dp_trajectories:
        datasets.append(("deforming_plate", dp_trajectories, dp_results))
    for dataset_label, trajectories, results in datasets:
        for traj_idx, traj in enumerate(trajectories):
            for metric in ALL_METRICS:
                if metric.kind != "scalar":
                    continue
                for sig_name in SIGNAL_NAMES:
                    r = results[traj_idx][metric.name].get(sig_name)
                    value = r.value if isinstance(r, ScalarResult) else np.nan
                    rows.append({
                        "dataset": dataset_label,
                        "trajectory": traj.name,
                        "metric": metric.name,
                        "signal": sig_name,
                        "value": value,
                    })
    if not rows:
        return
    csv_path = output_dir / "per_trajectory.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")


# ─── entry point ──────────────────────────────────────────────────────────────

def build_arg_parser():
    """
    return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Compare biax_longer and DeformingPlate datasets.")
    parser.add_argument("--biax-dir", required=True,
                        help="Path to biax_longer test graphs directory (contains *Nsteps400*.pt).")
    parser.add_argument("--dp-tfrecord", required=True,
                        help="Path to deforming_plate test.tfrecord.")
    parser.add_argument("--n-timesteps", type=int, default=200,
                        help="Maximum timesteps to use per trajectory.")
    parser.add_argument("--n-dp-trajectories", type=int, default=10,
                        help="Number of DeformingPlate trajectories to load.")
    parser.add_argument("--output-dir", default="comparison_artifacts",
                        help="Directory for all output plots and CSVs.")
    return parser


def main(args):
    """
    Args:
        args: argparse.Namespace
            Parsed CLI arguments (from build_arg_parser).
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading biax_longer trajectories …")
    biax_loader = BiaxLongerLoader(args.biax_dir, args.n_timesteps)
    biax_trajectories = biax_loader.load_all()
    print(f"  loaded {len(biax_trajectories)} trajectories")

    print("Loading DeformingPlate trajectories …")
    dp_trajectories = []
    npz_dir = Path(args.dp_tfrecord).parent / "processed" / "test"
    try:
        if npz_dir.exists() and any(npz_dir.glob("trajectory_*.npz")):
            print(f"  using pre-converted .npz files in {npz_dir}")
            dp_loader = DeformingPlateNpzLoader(npz_dir, args.n_dp_trajectories, args.n_timesteps)
        else:
            print(f"  .npz dir not found ({npz_dir}), falling back to tfrecord (requires TF)")
            dp_loader = DeformingPlateLoader(args.dp_tfrecord, args.n_dp_trajectories, args.n_timesteps)
        dp_trajectories = dp_loader.load_all()
        print(f"  loaded {len(dp_trajectories)} trajectories")
    except (ImportError, Exception) as exc:
        print(f"  [WARN] DeformingPlate loading skipped: {exc}")
        print("  Pre-process with scripts/preprocess_deforming_plate.py (system python3 with TF)")
        print("  or run with tensorflow available to load directly from tfrecord.")

    print("=" * 60)
    print("Computing metrics for biax_longer …")
    biax_results = _compute_results_for_trajectories(biax_trajectories)

    dp_results = {}
    if dp_trajectories:
        print("Computing metrics for deforming_plate …")
        dp_results = _compute_results_for_trajectories(dp_trajectories)

    print("=" * 60)
    print("Saving detail plots …")
    _save_detail_plots(biax_trajectories, biax_results, "biax_longer", output_dir)
    if dp_trajectories:
        _save_detail_plots(dp_trajectories, dp_results, "deforming_plate", output_dir)

    print("Saving average plots …")
    biax_all = _save_average_plots(biax_trajectories, biax_results, "biax_longer", output_dir)
    dp_all = {}
    if dp_trajectories:
        dp_all = _save_average_plots(dp_trajectories, dp_results, "deforming_plate", output_dir)

    if dp_trajectories:
        print("Saving pooled Welch PSD headline figure …")
        psd_metric_name = "welch_psd"
        biax_psd = {sig: biax_all[psd_metric_name][sig] for sig in SIGNAL_NAMES}
        dp_psd = {sig: dp_all[psd_metric_name][sig] for sig in SIGNAL_NAMES}
        pooled_psd_plot(biax_psd, dp_psd, "step_displacement",
                        output_dir / "average" / "headline_welch_psd_step_displacement.png")

    print("Computing statistics …")
    stat_rows = _compute_statistics(biax_all, dp_all) if dp_trajectories else []
    if stat_rows:
        _save_summary_csv(stat_rows, output_dir)
    _save_per_traj_csv(biax_trajectories, biax_results,
                       dp_trajectories if dp_trajectories else [],
                       dp_results if dp_results else {}, output_dir)

    print("=" * 60)
    print(f"Done. All outputs in {output_dir}")
    return stat_rows, biax_trajectories, dp_trajectories
