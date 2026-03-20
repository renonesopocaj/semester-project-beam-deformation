"""
eda.py
Main file to run in order to do Exploratory Data Analysis.
The del command are there to avoid out of memory errors.
This script does the following:
1. Preprocess the data and save it in a format that is easy to use for EDA.
2. Plot the distributions of the features.
3. Plot uniformly sampled trajectories (10%, i.e. for 1000 samples 100)
To avoid code repetition, the preprocessing is done by implementing the TrainPreprocessor class with a special
EdaPreprocessor class that is used to collect the data.
All the possible normalization are done.
"""

import torch
import hydra
import gc
import os
from src.gnn.misc_helpers.yamlobject import PreprocessYaml
from src.gnn.preprocess import TrainPreprocessor
from src.gnn.preprocessing.noiser import NoNoiser
from src.gnn.preprocessing.build_features import compute_von_mises_3d
from src.gnn.validation.plot_rollout import make_ground_truth_gif
from torch_geometric.data import Data
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

EPS = 1e-6
DT = 0.08
LIMIT_TIMESTEPS = 50
DEVICE = "cpu"
LABELS = ["No norm", "Z isotr.", "Z anisotr.", "Robust isotr.", "Robust anisotr."]

def generate_ground_truth_gif_all(preprocessor, SAVE_DIR):
    """
    Generate one GIF per trajectory from the no-norm preprocessor's raw graphs.

    Args:
        preprocessor: EdaPreprocessor
            The non-normalized preprocessor whose out_graphs hold raw x_pos_t and y_sig_t.
    """
    gif_dir = Path(SAVE_DIR) / "ground_truth_gifs"
    os.makedirs(gif_dir, exist_ok=True)

    traj_idx_list = preprocessor.traj_idx_list
    out_graphs = preprocessor.out_graphs

    pbar = tqdm(enumerate(sorted(set(traj_idx_list))), desc="Ground truth GIFs")
    for i, traj in pbar:
        if i % 15 != 0:
            continue
        traj_graphs = [g for idx, g in zip(traj_idx_list, out_graphs) if idx == traj]

        for graph in traj_graphs:
            graph.von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)

        gif_path = gif_dir / f"trajectory_{traj:02d}.gif"
        # print(f"\tGenerating ground truth GIF for trajectory {traj} ({len(traj_graphs)} steps)...")xx
        make_ground_truth_gif(traj_graphs, L=1.0, W=0.1, D=0.04, gif_path=gif_path, fps=4, verbose=False)
        del gif_path
        del traj_graphs

    del traj_idx_list
    del out_graphs
    del gif_dir

def plot_all_distributions(eda_obj_list, labels, SAVE_DIR):
    """
    Args:
        eda_obj_list: list[EdaObj]
            One EdaObj per normalizer variant, holding lists of torch.Tensors for each feature.
        labels: list[str]
            Human-readable label for each normalizer variant.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    pbar = tqdm(eda_obj_list[0].__dict__, desc="Plotting distribution")
    for key in pbar:
        if len(getattr(eda_obj_list[0], key)) == 0:
            continue
        pbar.set_postfix({"key": key})
        plot_distribution(key, eda_obj_list, labels, SAVE_DIR)


def plot_distribution(attr_name, eda_obj_list, labels, SAVE_DIR):
    """
    Plots non-normalized distribution on the left y-axis and all
    normalized distributions on the right y-axis (shared scale).

    Args:
        attr_name: str
            Attribute name (used for titles/labels).
        eda_obj_list: list[EdaObj]
            List of eda objs; index 0 is non-normalized, 1..N are normalized.
        labels: list[str]
            Label for each normalizer variant.
    """
    chunks_list = []
    for idx in range(len(eda_obj_list)):
        _chunks = []
        _val = getattr(eda_obj_list[idx], attr_name)
        for t in _val:
            _chunks.append(t.reshape(-1, 1))
        _x = torch.cat(_chunks, dim=0).to(torch.float32)
        _x_np = _x.reshape(-1).detach().to(DEVICE).numpy()
        chunks_list.append(_x_np)
        del _chunks
        del _val
        del _x
        del _x_np

    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = ax_left.twinx()

    ax_left.hist(chunks_list[0], bins=200, density=True, alpha=0.3, color="gray", label=labels[0])
    ax_left.set_ylabel("Density (non-normalized)")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for data, color, label in zip(chunks_list[1:], colors, labels[1:]):
        ax_right.hist(data, bins=200, density=True, alpha=0.5, color=color,
                      label=label, histtype="step", linewidth=1.5)
    ax_right.set_ylabel("Density (normalized)")

    ax_left.set_xlabel(attr_name)
    fig.suptitle(f"{attr_name} distribution")
    ax_left.grid(True, alpha=0.3)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    fig.legend(handles_left + handles_right, labels_left + labels_right,
               loc="upper right", bbox_to_anchor=(0.98, 0.95))

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/eda_{attr_name}.png", dpi=150)
    plt.close(fig)
    del handles_left
    del labels_left
    del handles_right
    del labels_right
    del chunks_list

class EdaPreprocessor(TrainPreprocessor):
    eda_obj: object
    keep_graphs_for_gif: bool

    def __init__(self, cfg, keep_graphs_for_gif):
        """
        Args:
            cfg: object
                config object
            keep_graphs_for_gif: bool
                whether to keep full graphs after EDA collection
        """
        super().__init__(cfg)
        self.noiser = NoNoiser()
        self.keep_graphs_for_gif = keep_graphs_for_gif

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        pass

    def _save_preprocessed(self, traj_idx, out_graphs, save_dir):
        """
        Args:
            traj_idx: list
                trajectory indices
            out_graphs: list
                processed graphs
            save_dir: object
                unused here
            return: None
                fills self.eda_obj and frees memory when possible
        """
        eda_obj = EdaObj()
        has_displacement = out_graphs[0].edge_attr.shape[1] == 12

        pbar = tqdm(sorted(set(traj_idx)), desc="EDA collecting trajectories")
        for traj in pbar:
            for idx, graph in zip(traj_idx, out_graphs):
                if idx != traj:
                    continue

                eda_obj.acceleration_x.append(graph.y[:, 0])
                eda_obj.acceleration_y.append(graph.y[:, 1])
                eda_obj.acceleration_z.append(graph.y[:, 2])
                eda_obj.von_mises.append(graph.y[:, 3])

                eda_obj.position_x.append(graph.x[:, 1])
                eda_obj.position_y.append(graph.x[:, 2])
                eda_obj.position_z.append(graph.x[:, 3])
                eda_obj.velocity_x.append(graph.x[:, 4])
                eda_obj.velocity_y.append(graph.x[:, 5])
                eda_obj.velocity_z.append(graph.x[:, 6])
                eda_obj.rest_pos_x.append(graph.x[:, 7])
                eda_obj.rest_pos_y.append(graph.x[:, 8])
                eda_obj.rest_pos_z.append(graph.x[:, 9])
                eda_obj.force_x.append(graph.x[:, 10])
                eda_obj.force_y.append(graph.x[:, 11])
                eda_obj.force_z.append(graph.x[:, 12])

                if has_displacement:
                    eda_obj.displacement_pos_x.append(graph.x[:, 13])
                    eda_obj.displacement_pos_y.append(graph.x[:, 14])
                    eda_obj.displacement_pos_z.append(graph.x[:, 15])

                eda_obj.edge_relative_pos_x.append(graph.edge_attr[:, 0])
                eda_obj.edge_relative_pos_y.append(graph.edge_attr[:, 1])
                eda_obj.edge_relative_pos_z.append(graph.edge_attr[:, 2])
                eda_obj.edge_rest_relative_pos_x.append(graph.edge_attr[:, 3])
                eda_obj.edge_rest_relative_pos_y.append(graph.edge_attr[:, 4])
                eda_obj.edge_rest_relative_pos_z.append(graph.edge_attr[:, 5])

                if has_displacement:
                    eda_obj.edge_displacement_pos_x.append(graph.edge_attr[:, 6])
                    eda_obj.edge_displacement_pos_y.append(graph.edge_attr[:, 7])
                    eda_obj.edge_displacement_pos_z.append(graph.edge_attr[:, 8])
                    eda_obj.edge_relative_pos_norm.append(graph.edge_attr[:, 9])
                    eda_obj.edge_rest_relative_pos_norm.append(graph.edge_attr[:, 10])
                    eda_obj.edge_displacement_norm.append(graph.edge_attr[:, 11])
                else:
                    eda_obj.edge_relative_pos_norm.append(graph.edge_attr[:, 6])
                    eda_obj.edge_rest_relative_pos_norm.append(graph.edge_attr[:, 7])

        self.eda_obj = eda_obj

        self.targets.clear()
        self.categorical_node_features.clear()
        self.valued_node_features.clear()
        self.edge_features.clear()

        if not self.keep_graphs_for_gif:
            self.out_graphs.clear()
            self.traj_idx_list.clear()

        gc.collect()

    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, _von_mises_log):
        """
        Args:
            graph: Data
                input graph at timestep t
            _a: torch.Tensor
                acceleration target
            _x: torch.Tensor
                node positions at time t
            _v: torch.Tensor
                node velocities at time t
            _von_mises: torch.Tensor
                von mises target
            _edge_attr: torch.Tensor
                edge features
            _node_attr: torch.Tensor
                valued node features
            t: int
                timestep index
            x_next: torch.Tensor
                next positions for all timesteps
            v_next: torch.Tensor
                next velocities for all timesteps
            all_node_force: torch.Tensor
                node force over time
            rest_pos_x: torch.Tensor
                rest positions
            _categorical_node_attr: torch.Tensor
                categorical node features
            _von_mises_log: torch.Tensor
                log von mises target
            return: None
                appends rich graph objects needed for EDA and GIF generation
        """
        _target = torch.concat([_a, _von_mises], dim=1)
        self.out_graphs.append(
            Data(
                y=_target.clone(),
                y_acc_t=_a.clone(),
                y_von_mises=_von_mises.clone(),
                von_mises_log=_von_mises_log.clone(),
                y_sig_t=graph.y_sig_t.clone(),
                t=graph.time.clone(),
                x=_node_attr.clone(),
                x_next=x_next[t].clone(),
                v_next=v_next[t].clone(),
                node_force_next=all_node_force[t + 1].clone(),
                x_pos_t=_x.clone(),
                x_vel_t=_v.clone(),
                node_force=all_node_force[t].clone(),
                rest_pos=rest_pos_x.clone(),
                edge_attr=_edge_attr.clone(),
                edge_index=graph.edge_index.clone(),
                x_element_connectivity=graph.x_element_connectivity.copy()
            )
        )
        self.targets.append(_target.clone())


class EdaObj:
    acceleration_x: list
    acceleration_y: list
    acceleration_z: list
    von_mises: list
    velocity_x: list
    velocity_y: list
    velocity_z: list
    position_x: list
    position_y: list
    position_z: list
    edge_relative_pos_x: list
    edge_relative_pos_y: list
    edge_relative_pos_z: list
    edge_relative_pos_norm: list
    edge_rest_relative_pos_x: list
    edge_rest_relative_pos_y: list
    edge_rest_relative_pos_z: list
    edge_rest_relative_pos_norm: list
    force_x: list
    force_y: list
    force_z: list
    edge_displacement_pos_x: list
    edge_displacement_pos_y: list
    edge_displacement_pos_z: list
    edge_displacement_norm: list
    rest_pos_x: list
    rest_pos_y: list
    rest_pos_z: list
    displacement_pos_x: list
    displacement_pos_y: list
    displacement_pos_z: list

    def __init__(self):
        (self.acceleration_x, self.acceleration_y, self.acceleration_z, self.von_mises,
         self.velocity_x, self.velocity_y, self.velocity_z,
         self.position_x, self.position_y, self.position_z,
         self.displacement_pos_x, self.displacement_pos_y, self.displacement_pos_z,
         self.force_x, self.force_y, self.force_z,
         self.rest_pos_x, self.rest_pos_y, self.rest_pos_z,
         # edge
         self.edge_relative_pos_x, self.edge_relative_pos_y, self.edge_relative_pos_z, self.edge_relative_pos_norm,
         self.edge_rest_relative_pos_x, self.edge_rest_relative_pos_y, self.edge_rest_relative_pos_z,
         self.edge_rest_relative_pos_norm,
         self.edge_displacement_pos_x, self.edge_displacement_pos_y, self.edge_displacement_pos_z,
         self.edge_displacement_norm) = \
            ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [])

@hydra.main(version_base=None, config_path="../../conf/preprocess", config_name="eda")
def main(cfg):
    """
    Args:
        cfg: object
            Hydra-composed configuration
    """
    SPLIT = "validation"
    GIF_GENERATION_NUMBER = "uniformly_sampled" # possible values: uniformly_sampled | "all"


    cfg_obj_nonorm = PreprocessYaml(cfg)
    cfg_obj_z_isotr_norm = PreprocessYaml(cfg)
    cfg_obj_z_anisotr_norm = PreprocessYaml(cfg)
    cfg_obj_rob_isotr_norm = PreprocessYaml(cfg)
    cfg_obj_rob_anisotr_norm = PreprocessYaml(cfg)

    cfg_obj_nonorm.normalizer_method = "None"
    cfg_obj_z_isotr_norm.normalizer_method = "z"
    cfg_obj_z_anisotr_norm.normalizer_method = "z"
    cfg_obj_rob_isotr_norm.normalizer_method = "robust"
    cfg_obj_rob_anisotr_norm.normalizer_method = "robust"

    cfg_obj_nonorm.isotropic_normalizer = True
    cfg_obj_z_isotr_norm.isotropic_normalizer = True
    cfg_obj_z_anisotr_norm.isotropic_normalizer = False
    cfg_obj_rob_isotr_norm.isotropic_normalizer = True
    cfg_obj_rob_anisotr_norm.isotropic_normalizer = False

    SAVE_DIR = f"eda_artifacts/{str(cfg_obj_nonorm.dataset_type)}"

    print(f"Preprocessing size={cfg_obj_nonorm.dataset_size} dataset")
    print(f"\tSave dir of eda = {SAVE_DIR}\n")
    graphs_dir_train = Path(cfg_obj_nonorm.raw_data_dir) / "train" / "graphs"

    all_eda_obj_arg = []

    print("\n" + "=" * 60 + "\ntrain, no norm\n" + "=" * 60)
    preprocessor = EdaPreprocessor(cfg_obj_nonorm, True)
    preprocessor.preprocess_data(cfg_obj_nonorm, graphs_dir_train, None)
    all_eda_obj_arg.append(preprocessor.eda_obj)

    print("=" * 60 + "\nStarting to generate ground truth GIFs..\n" + "=" * 60 + "\n")
    generate_ground_truth_gif_all(preprocessor, SAVE_DIR)

    preprocessor.out_graphs.clear()
    preprocessor.traj_idx_list.clear()
    del preprocessor
    gc.collect()

    print("\n" + "=" * 60 + "\ntrain, z isotropic norm\n" + "=" * 60)
    preprocessor = EdaPreprocessor(cfg_obj_z_isotr_norm, False)
    preprocessor.preprocess_data(cfg_obj_z_isotr_norm, graphs_dir_train, None)
    all_eda_obj_arg.append(preprocessor.eda_obj)
    del preprocessor
    gc.collect()

    print("\n" + "=" * 60 + "\ntrain, z anisotropic norm\n" + "=" * 60)
    preprocessor = EdaPreprocessor(cfg_obj_z_anisotr_norm, False)
    preprocessor.preprocess_data(cfg_obj_z_anisotr_norm, graphs_dir_train, None)
    all_eda_obj_arg.append(preprocessor.eda_obj)
    del preprocessor
    gc.collect()

    print("\n" + "=" * 60 + "\ntrain, robust isotropic norm\n" + "=" * 60)
    preprocessor = EdaPreprocessor(cfg_obj_rob_isotr_norm, False)
    preprocessor.preprocess_data(cfg_obj_rob_isotr_norm, graphs_dir_train, None)
    all_eda_obj_arg.append(preprocessor.eda_obj)
    del preprocessor
    gc.collect()

    print("\n" + "=" * 60 + "\ntrain, robust anisotropic norm\n" + "=" * 60)
    preprocessor = EdaPreprocessor(cfg_obj_rob_anisotr_norm, False)
    preprocessor.preprocess_data(cfg_obj_rob_anisotr_norm, graphs_dir_train, None)
    all_eda_obj_arg.append(preprocessor.eda_obj)
    del preprocessor
    gc.collect()

    print("=" * 60 + "Starting to plot all distributions..." + "=" * 60 + "\n")
    plot_all_distributions(all_eda_obj_arg, LABELS, SAVE_DIR)

    del all_eda_obj_arg
    gc.collect()


if __name__ == "__main__":
    main()