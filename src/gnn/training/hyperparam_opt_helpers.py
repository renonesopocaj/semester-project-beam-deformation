from pathlib import Path
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
plt.ioff()

def update_final_metrics_plain(final_metrics_plain, metric_source):
    """
    Adds final-evaluation scalar metrics from one Lightning result source to the accumulated plain dict.

    Args:
        final_metrics_plain: dict
            Accumulator mapping final_validation/* and final_test/* metric names to floats.
        metric_source: dict or list
            Either Trainer.callback_metrics or the list/dict returned by Trainer.test.
        return: dict
            The same accumulator, updated in place for convenient chaining.
    """
    if metric_source is None:
        return final_metrics_plain
    metric_dicts = []
    if isinstance(metric_source, dict):
        metric_dicts.append(metric_source)
    elif isinstance(metric_source, list):
        for item in metric_source:
            if isinstance(item, dict):
                metric_dicts.append(item)
    for metrics_map in metric_dicts:
        for key, value in metrics_map.items():
            key_str = str(key)
            if not (key_str.startswith("final_validation/") or key_str.startswith("final_test/")):
                continue
            try:
                if hasattr(value, "detach"):
                    detached = value.detach()
                    if detached.numel() != 1:
                        continue
                    final_metrics_plain[key_str] = float(detached.item())
                else:
                    final_metrics_plain[key_str] = float(value)
            except (TypeError, ValueError):
                continue
    return final_metrics_plain


def save_pareto_plot(records_list, pareto_x_key, pareto_y_key, output_dir, filename="pareto.png"):
    """
    Args:
        records_list: list of dict
            Trial records containing final_metrics maps.
        pareto_x_key: str
            X axis scalar key inside final_metrics.
        pareto_y_key: str
            Y axis scalar key inside final_metrics.
        output_dir: Path
            Directory where the plot is written.
        filename: str
            Output filename (default pareto.png); use different names for multiple pairs.
    return: Path
            Path to the saved plot.
    """
    xs = []
    ys = []
    missing_pair_count = 0
    non_finite_count = 0
    ri = 0
    while ri < len(records_list):
        rec_item = records_list[ri]
        fm = rec_item.get("final_metrics", {})
        xa = fm.get(pareto_x_key)
        ya = fm.get(pareto_y_key)
        if xa is None or ya is None:
            missing_pair_count = missing_pair_count + 1
            ri = ri + 1
            continue
        try:
            xf = float(xa)
            yf = float(ya)
        except (TypeError, ValueError):
            non_finite_count = non_finite_count + 1
            ri = ri + 1
            continue
        if (math.isfinite(xf) is False) or (math.isfinite(yf) is False):
            non_finite_count = non_finite_count + 1
            ri = ri + 1
            continue
        xs.append(xf)
        ys.append(yf)
        ri = ri + 1
    print(
        "Pareto plotting: %d valid points, %d missing-key records, %d non-finite records."
        % (len(xs), missing_pair_count, non_finite_count)
    )
    if len(xs) == 0:
        fig_empty, axe = plt.subplots(figsize=(7, 5))
        diag = (
            "No valid Pareto points.\n"
            "Expected keys:\n- %s\n- %s\n"
            "Missing-key records: %d\nNon-finite records: %d"
            % (pareto_x_key, pareto_y_key, missing_pair_count, non_finite_count)
        )
        axe.text(0.5, 0.5, diag, ha="center", va="center")
        axe.set_axis_off()
        outp_empty = Path(output_dir) / filename
        outp_empty.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outp_empty, dpi=120)
        plt.close(fig_empty)
        return outp_empty
    pairs_xy = []
    pi = 0
    while pi < len(xs):
        pairs_xy.append((xs[pi], ys[pi]))
        pi = pi + 1
    nd_local = _pareto_non_dominated_minimize(pairs_xy)
    fig, axi = plt.subplots(figsize=(7, 5))
    axi.scatter(xs, ys, c="#888888", s=42, label="all trials", alpha=0.85)
    nd_x = []
    nd_y = []
    ki = 0
    while ki < len(nd_local):
        pj = nd_local[ki]
        nd_x.append(xs[pj])
        nd_y.append(ys[pj])
        ki = ki + 1
    axi.scatter(nd_x, nd_y, c="#c0392b", s=76, marker="*", label="Pareto frontier", alpha=0.95, zorder=4)
    axi.set_xlabel(pareto_x_key)
    axi.set_ylabel(pareto_y_key)
    axi.set_title("Pareto set (multi-objective, lower-left better)")
    axi.legend(loc="best")
    axi.grid(alpha=0.25)
    out_path_png = Path(output_dir) / filename
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=170)
    plt.close(fig)
    return out_path_png

def save_sweep_performance_plot(sweep_name, x_labels, y_values, y_metric_name, output_dir):
    """
    Args:
        sweep_name: str
            Sweep identifier used in title and output filename.
        x_labels: list of str
            Labels representing swept value(s) for each option.
        y_values: list of float
            Performance values for each option (same order as x_labels).
        y_metric_name: str
            Name of the plotted metric for axis labeling.
        output_dir: Path
            Directory where plot image is written.
    return: Path
            Path to generated sweep plot image.
    """
    out_path = Path(output_dir) / ("sweep_%s_overall_mae.png" % sweep_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(x_labels) == 0 or len(y_values) == 0:
        fig_empty, ax_empty = plt.subplots(figsize=(8, 4.8))
        ax_empty.text(0.5, 0.5, "No sweep data to plot for %s" % sweep_name, ha="center", va="center")
        ax_empty.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig_empty)
        return out_path
    x_idx = list(range(len(x_labels)))
    fig, ax = plt.subplots(figsize=(max(8, int(len(x_labels) * 0.9)), 4.8))
    ax.plot(x_idx, y_values, marker="o", linewidth=1.6, alpha=0.95)
    ax.scatter(x_idx, y_values, s=40, zorder=3)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_xlabel("swept value(s)")
    ax.set_ylabel(y_metric_name)
    ax.set_title("Sweep performance: %s" % sweep_name)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path

def save_summary_metric_plot(records_list, metric_key, output_dir, filename):
    """
    Saves a summary bar/line plot with one point per trial across all sweeps.

    X-axis: sweep_name__option_label for each trial record.
    Y-axis: the value of metric_key inside final_metrics.

    Args:
        records_list: list of dict
            Trial records (each has sweep_name, option_label, final_metrics).
        metric_key: str
            Key to extract from final_metrics (e.g. final_test/overall_abs_mae).
        output_dir: Path
            Directory where the plot is written.
        filename: str
            Output filename.
    return: Path
            Path to the saved plot.
    """
    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_labels = []
    y_vals = []
    missing_count = 0
    non_finite_count = 0
    ri = 0
    while ri < len(records_list):
        rec = records_list[ri]
        val = rec.get("final_metrics", {}).get(metric_key)
        if val is not None:
            try:
                yf = float(val)
            except (TypeError, ValueError):
                missing_count = missing_count + 1
                ri = ri + 1
                continue
            if math.isfinite(yf):
                label = "%s__%s" % (rec.get("sweep_name", ""), rec.get("option_label", ""))
                x_labels.append(label)
                y_vals.append(yf)
            else:
                non_finite_count = non_finite_count + 1
        else:
            missing_count = missing_count + 1
        ri = ri + 1
    print(
        "Summary plot '%s': %d valid points, %d missing-key records, %d non-finite records."
        % (metric_key, len(x_labels), missing_count, non_finite_count)
    )
    if len(x_labels) == 0:
        diag = (
            "No data for %s\n"
            "Missing-key records: %d  Non-finite records: %d"
            % (metric_key, missing_count, non_finite_count)
        )
        fig_empty, ax_empty = plt.subplots(figsize=(8, 4.8))
        ax_empty.text(0.5, 0.5, diag, ha="center", va="center")
        ax_empty.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig_empty)
        return out_path
    x_idx = list(range(len(x_labels)))
    fig, ax = plt.subplots(figsize=(max(8, int(len(x_labels) * 0.9)), 4.8))
    ax.plot(x_idx, y_vals, marker="o", linewidth=1.6, alpha=0.95)
    ax.scatter(x_idx, y_vals, s=40, zorder=3)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel("trial (sweep__option)")
    ax.set_ylabel(metric_key)
    ax.set_title("All trials: %s" % metric_key)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _pareto_non_dominated_minimize(points_xy):
    """
    Args:
        points_xy: list of pair (float, float)
            Objective pairs to minimize jointly.
    return: list[int]
            Indices into points_xy that lie on the Pareto frontier (non-dominated).
    """
    n = len(points_xy)
    out_idx = []
    i = 0
    while i < n:
        dominated = False
        j = 0
        while j < n:
            if j == i:
                j = j + 1
                continue
            xi, yi = points_xy[i]
            xj, yj = points_xy[j]
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                dominated = True
                break
            j = j + 1
        if dominated is False:
            out_idx.append(i)
        i = i + 1
    return out_idx

def _safe_wandb_name(raw_name, fallback_name):
    """
    Args:
        raw_name: str
            Candidate W&B name.
        fallback_name: str
            Name used if the candidate has no safe characters.
    return: str
            Name containing only characters accepted by W&B artifact/media identifiers.
    """
    safe_chars = []
    ix = 0
    while ix < len(str(raw_name)):
        one_char = str(raw_name)[ix]
        if one_char.isalnum() or one_char in ("_", "-", "."):
            safe_chars.append(one_char)
        else:
            safe_chars.append("_")
        ix = ix + 1
    safe_name = "".join(safe_chars).strip("_")
    if safe_name == "":
        return fallback_name
    return safe_name


def wandb_summary_log_outputs(study_name_str, plot_paths, file_paths, wandb_project, wandb_offline_flag):
    """
    Logs all hyperparameter summary outputs to one trailing W&B summary run.

    Args:
        study_name_str: str
            Study basename for grouping.
        plot_paths: list of Path
            Plot image paths to log as W&B Images in a single run step.
        file_paths: list of Path
            Non-media files to attach to a W&B artifact.
        wandb_project: str
            W&B project matching training runs.
        wandb_offline_flag: bool
            Whether offline mode is preferred.
    """
    import wandb as wb_sdk

    run_name_here = "%s__summary" % study_name_str
    wandb_mode = "offline" if wandb_offline_flag else "online"
    artifact_name = "%s_hyperparam_outputs" % _safe_wandb_name(study_name_str, "hyperparam_study")
    with wb_sdk.init(project=wandb_project, name=run_name_here, group=study_name_str, reinit=True,
                     mode=wandb_mode) as run:
        image_payload = {}
        pp = 0
        while pp < len(plot_paths):
            plot_path = Path(plot_paths[pp])
            image_key = "hyperparam_outputs/%s" % _safe_wandb_name(plot_path.stem, "plot")
            image_payload[image_key] = wb_sdk.Image(str(plot_path), caption=plot_path.name)
            pp = pp + 1
        if len(image_payload) > 0:
            run.log(image_payload)

        artifact = wb_sdk.Artifact(name=artifact_name, type="hyperparam-summary")
        added_file = False
        fp = 0
        while fp < len(file_paths):
            file_path = Path(file_paths[fp])
            artifact.add_file(str(file_path), name=file_path.name)
            added_file = True
            fp = fp + 1
        if added_file:
            run.log_artifact(artifact)
