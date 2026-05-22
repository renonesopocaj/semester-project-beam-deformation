"""
plotting.py

Two plot generators:

  detail_plot(metric, results_by_signal, traj_name, dataset_label, out_path)
    — one figure per (trajectory, metric), all five signals overlaid.

  average_plot(metric, all_results_by_signal, dataset_label, out_path)
    — one figure per (dataset, metric), summarising all trajectories.
      Curve metrics: median line + IQR shading per signal.
      Scalar metrics: violin plot per signal.

Both functions dispatch on metric.kind ("curve" or "scalar") without any
external if/else.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.gnn.comparison_defplate.metrics_base import CurveResult, ScalarResult

# ─── palette ──────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "position":         "#1f77b4",   # blue
    "displacement":     "#ff7f0e",   # orange
    "step_displacement": "#2ca02c",  # green
    "velocity":         "#d62728",   # red
    "von_mises":        "#9467bd",   # purple
}

SIGNAL_LABELS = {
    "position":          "position",
    "displacement":      "displacement",
    "step_displacement": "step displacement",
    "velocity":          "velocity",
    "von_mises":         "von Mises",
}

_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def _legend_handles(signal_names):
    handles = []
    for sig in signal_names:
        handles.append(Line2D([0], [0], color=SIGNAL_COLORS[sig],
                              label=SIGNAL_LABELS[sig], linewidth=2))
    return handles


# ─── detail plot ──────────────────────────────────────────────────────────────

def _detail_curve(ax, metric_name, results_by_signal):
    """
    Plot one curve per signal on ax.

    Args:
        ax: matplotlib Axes
        metric_name: str
        results_by_signal: dict  signal_name → CurveResult
    """
    xlabel = ylabel = ""
    for sig_name, result in results_by_signal.items():
        if result is None or not np.any(np.isfinite(result.y)):
            continue
        ax.plot(result.x, result.y, color=SIGNAL_COLORS[sig_name],
                linewidth=1.8, label=SIGNAL_LABELS[sig_name])
        xlabel = result.xlabel
        ylabel = result.ylabel
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(handles=_legend_handles(list(results_by_signal.keys())),
              fontsize=8, loc="best")


def _detail_scalar(ax, metric_name, results_by_signal):
    """
    Plot bar chart with one bar per signal on ax.

    Args:
        ax: matplotlib Axes
        metric_name: str
        results_by_signal: dict  signal_name → ScalarResult
    """
    signal_names = list(results_by_signal.keys())
    values = []
    for sig in signal_names:
        r = results_by_signal[sig]
        values.append(r.value if (r is not None and np.isfinite(r.value)) else np.nan)
    xs = np.arange(len(signal_names))
    colors = [SIGNAL_COLORS[s] for s in signal_names]
    ax.bar(xs, values, color=colors, width=0.6, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([SIGNAL_LABELS[s] for s in signal_names], rotation=25, ha="right", fontsize=9)
    if results_by_signal:
        first_r = next(r for r in results_by_signal.values() if r is not None)
        ax.set_ylabel(first_r.label, fontsize=10)


_DETAIL_DISPATCH = {
    "curve":  _detail_curve,
    "scalar": _detail_scalar,
}


def detail_plot(metric, results_by_signal, traj_name, dataset_label, out_path):
    """
    Save one figure for a single trajectory and metric, with all signals.

    Args:
        metric: Metric  (the metric whose results are given)
        results_by_signal: dict  signal_name → CurveResult or ScalarResult
        traj_name: str
        dataset_label: str  e.g. "biax_longer" or "deforming_plate"
        out_path: Path or str
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    title = f"{dataset_label} | {traj_name}\n{metric.name.replace('_', ' ')}"
    ax.set_title(title, fontsize=11)
    _DETAIL_DISPATCH[metric.kind](ax, metric.name, results_by_signal)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─── average plot ─────────────────────────────────────────────────────────────

def _average_curve(ax, metric_name, all_results_by_signal):
    """
    Median + IQR shading for each signal over all trajectories.

    Args:
        ax: matplotlib Axes
        metric_name: str
        all_results_by_signal: dict  signal_name → list of CurveResult
    """
    xlabel = ylabel = ""
    for sig_name, result_list in all_results_by_signal.items():
        valid = [r for r in result_list if r is not None and np.any(np.isfinite(r.y))]
        if not valid:
            continue
        # align all curves to the same x grid (use the most common)
        xs = valid[0].x
        ys = []
        for r in valid:
            if len(r.x) == len(xs):
                ys.append(r.y)
            else:
                # interpolate to common xs
                ys.append(np.interp(xs, r.x, r.y, left=np.nan, right=np.nan))
        ys_arr = np.array(ys)  # (n_trajs, F)
        med = np.nanmedian(ys_arr, axis=0)
        q25 = np.nanpercentile(ys_arr, 25, axis=0)
        q75 = np.nanpercentile(ys_arr, 75, axis=0)
        color = SIGNAL_COLORS[sig_name]
        ax.plot(xs, med, color=color, linewidth=2.0, label=SIGNAL_LABELS[sig_name])
        ax.fill_between(xs, q25, q75, color=color, alpha=0.2)
        xlabel = valid[0].xlabel
        ylabel = valid[0].ylabel
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(handles=_legend_handles(list(all_results_by_signal.keys())),
              fontsize=8, loc="best")


def _average_scalar(ax, metric_name, all_results_by_signal):
    """
    Violin plot: one violin per signal, all trajectories pooled into one
    distribution.

    Args:
        ax: matplotlib Axes
        metric_name: str
        all_results_by_signal: dict  signal_name → list of ScalarResult
    """
    signal_names = list(all_results_by_signal.keys())
    data_per_signal = []
    for sig in signal_names:
        vals = [r.value for r in all_results_by_signal[sig]
                if r is not None and np.isfinite(r.value)]
        data_per_signal.append(vals if vals else [np.nan])

    positions = np.arange(1, len(signal_names) + 1)
    colors = [SIGNAL_COLORS[s] for s in signal_names]
    parts = ax.violinplot(data_per_signal, positions=positions, showmedians=True,
                          showextrema=True)
    for i, (body, color) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels([SIGNAL_LABELS[s] for s in signal_names], rotation=25, ha="right", fontsize=9)
    if all_results_by_signal:
        first_list = next(lst for lst in all_results_by_signal.values() if lst)
        first_r = next((r for r in first_list if r is not None), None)
        if first_r is not None:
            ax.set_ylabel(first_r.label, fontsize=10)


_AVERAGE_DISPATCH = {
    "curve":  _average_curve,
    "scalar": _average_scalar,
}


def average_plot(metric, all_results_by_signal, dataset_label, out_path):
    """
    Save one summary figure for a whole dataset and metric.

    Args:
        metric: Metric
        all_results_by_signal: dict  signal_name → list of CurveResult/ScalarResult
            one result per trajectory
        dataset_label: str  e.g. "biax_longer" or "deforming_plate"
        out_path: Path or str
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    title = f"{dataset_label} — {metric.name.replace('_', ' ')} (all trajectories)"
    ax.set_title(title, fontsize=11)
    _AVERAGE_DISPATCH[metric.kind](ax, metric.name, all_results_by_signal)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─── pooled Welch PSD (headline figure) ──────────────────────────────────────

def pooled_psd_plot(biax_results_by_signal, dp_results_by_signal, signal_name, out_path):
    """
    Single headline figure: pooled Welch PSD of one signal for both datasets.

    Plots median PSD + IQR for biax (solid) and deforming_plate (dashed).

    Args:
        biax_results_by_signal: dict  signal_name → list of CurveResult (biax)
        dp_results_by_signal: dict  signal_name → list of CurveResult (DP)
        signal_name: str  which signal to plot
        out_path: Path or str
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Pooled Welch PSD — {SIGNAL_LABELS.get(signal_name, signal_name)}", fontsize=12)

    def _plot_dataset(result_list, label, color, linestyle):
        valid = [r for r in result_list if r is not None and np.any(np.isfinite(r.y))]
        if not valid:
            return
        xs = valid[0].x
        ys = np.array([r.y for r in valid if len(r.x) == len(xs)])
        med = np.nanmedian(ys, axis=0)
        q25 = np.nanpercentile(ys, 25, axis=0)
        q75 = np.nanpercentile(ys, 75, axis=0)
        ax.plot(xs, med, color=color, linewidth=2.2, linestyle=linestyle, label=label)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.18)

    _plot_dataset(biax_results_by_signal.get(signal_name, []), "biax_longer", "#1f77b4", "-")
    _plot_dataset(dp_results_by_signal.get(signal_name, []), "deforming_plate", "#d62728", "--")
    ax.set_xlabel("Normalized frequency (cycles/step)", fontsize=10)
    ax.set_ylabel("Median PSD (log10)", fontsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
