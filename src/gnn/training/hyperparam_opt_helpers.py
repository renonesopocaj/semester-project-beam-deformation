from pathlib import Path
import wandb as wb_sdk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
plt.ioff()

def save_pareto_plot(records_list, pareto_x_key, pareto_y_key, output_dir):
    """
    Args:
        records_list: list of dict
            Trial records containing final_metrics maps.
        pareto_x_key: str
            X axis scalar key inside final_metrics.
        pareto_y_key: str
            Y axis scalar key inside final_metrics.
        output_dir: Path
            Directory for pareto.png .
    return: Path
            Path to pareto.png .
    """
    xs = []
    ys = []
    ri = 0
    while ri < len(records_list):
        rec_item = records_list[ri]
        fm = rec_item["final_metrics"]
        xa = fm.get(pareto_x_key, float("nan"))
        ya = fm.get(pareto_y_key, float("nan"))
        xs.append(float(xa))
        ys.append(float(ya))
        ri = ri + 1
    if len(xs) == 0:
        fig_empty, axe = plt.subplots(figsize=(7, 5))
        axe.text(0.5, 0.5, "no trials to plot", ha="center")
        outp_empty = Path(output_dir) / "pareto.png"
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
    out_path_png = Path(output_dir) / "pareto.png"
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=170)
    plt.close(fig)
    return out_path_png

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

def wandb_summary_log_plot(study_name_str, pareto_path, wandb_project, wandb_offline_flag):
    """
    Logs pareto.png to W&B inside a trailing summary run.

    Args:
        study_name_str: str
            Study basename for grouping.
        pareto_path: Path
            pareto.png filesystem path.
        wandb_project: str
            W&B project matching training runs.
        wandb_offline_flag: bool
            Whether offline mode is preferred.
    """
    run_name_here = "%s__summary" % study_name_str
    wb_sdk.init(project=wandb_project, name=run_name_here, group=study_name_str, reinit=True,
                offline=wandb_offline_flag)
    wb_sdk.log({"pareto_plot": wb_sdk.Image(str(pareto_path))})
    wb_sdk.finish()
