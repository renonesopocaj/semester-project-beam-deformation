import torch
import hydra
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
SAVE_DIR = "eda_artifacts"
LABELS = ["No norm", "Z isotr.", "Z anisotr.", "Robust isotr.", "Robust anisotr."]


def plot_all_distributions(eda_obj_list, labels):
    """
    Args:
        eda_obj_list: list[EdaObj]
            One EdaObj per normalizer variant, holding lists of torch.Tensors for each feature.
        labels: list[str]
            Human-readable label for each normalizer variant.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    for key in eda_obj_list[0].__dict__:
        if len(getattr(eda_obj_list[0], key)) == 0:
            continue
        print(f"\tPlotting distribution of key = {key}")
        plot_distribution(key, eda_obj_list, labels)


def plot_distribution(attr_name, eda_obj_list, labels):
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

    def __init__(self, cfg):
        super().__init__(cfg)
        self.noiser = NoNoiser()

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        pass

    def _save_preprocessed(self, traj_idx, out_graphs, save_dir):
        """
        Collects features from the already-normalized graph attributes
        (graph.y, graph.x, graph.edge_attr) set by _normalize_assign_features_train.

        Column layout for graph.x  = [categorical(1), position(3), velocity(3),
                                       rest_pos(3), force(3), (displacement(3))].
        Column layout for graph.y  = [acceleration(3), von_mises(1)].
        Column layout for graph.edge_attr with displacement (12):
            [rel_pos(3), rest_rel_pos(3), disp_pos(3), rel_norm(1), rest_norm(1), disp_norm(1)].
        Column layout for graph.edge_attr without displacement (8):
            [rel_pos(3), rest_rel_pos(3), rel_norm(1), rest_norm(1)].
        """
        eda_obj = EdaObj()
        has_displacement = out_graphs[0].edge_attr.shape[1] == 12

        pbar = tqdm(sorted(set(traj_idx)), desc="EDA collecting trajectories")
        for traj in pbar:
            for idx, graph in zip(traj_idx, out_graphs):
                if idx != traj:
                    continue
                # --- Targets from graph.y ---
                eda_obj.acceleration_x.append(graph.y[:, 0])
                eda_obj.acceleration_y.append(graph.y[:, 1])
                eda_obj.acceleration_z.append(graph.y[:, 2])
                eda_obj.von_mises.append(graph.y[:, 3])

                # --- Node features from graph.x (skip col 0 = categorical BC) ---
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

                # --- Edge features from graph.edge_attr ---
                eda_obj.relative_pos_x.append(graph.edge_attr[:, 0])
                eda_obj.relative_pos_y.append(graph.edge_attr[:, 1])
                eda_obj.relative_pos_z.append(graph.edge_attr[:, 2])
                eda_obj.rest_relative_pos_x.append(graph.edge_attr[:, 3])
                eda_obj.rest_relative_pos_y.append(graph.edge_attr[:, 4])
                eda_obj.rest_relative_pos_z.append(graph.edge_attr[:, 5])

                if has_displacement:
                    eda_obj.displacement_pos_x.append(graph.edge_attr[:, 6])
                    eda_obj.displacement_pos_y.append(graph.edge_attr[:, 7])
                    eda_obj.displacement_pos_z.append(graph.edge_attr[:, 8])
                    eda_obj.relative_pos_norm.append(graph.edge_attr[:, 9])
                    eda_obj.rest_relative_pos_norm.append(graph.edge_attr[:, 10])
                    eda_obj.displacement_norm.append(graph.edge_attr[:, 11])
                else:
                    eda_obj.relative_pos_norm.append(graph.edge_attr[:, 6])
                    eda_obj.rest_relative_pos_norm.append(graph.edge_attr[:, 7])

        self.eda_obj = eda_obj
        del eda_obj
        del pbar
        del has_displacement
        del self.targets
        del self.categorical_node_features
        del self.valued_node_features
        del self.edge_features

    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, von_mises_log):
        _target = torch.concat([_a, _von_mises], dim=1)
        self.out_graphs.append(
            Data(
                y=_target.clone(),
                y_acc_t=_a.clone(),
                y_von_mises=_von_mises.clone(),
                von_mises_log=von_mises_log.clone(),
                y_sig_t=graph.y_sig_t.clone(),
                t=graph.time.clone(),
                # ==================================================================
                #                       NODE features
                # ==================================================================
                # x = [is_bc, position, velocity, rest_position, node_force]
                x=_node_attr.clone(),  # this will be the NN input feature for training
                # note: the position x_next, velocity v_next of t + 1 can be given since they can be computed
                # from x, v at time t and acceleration of t - 1 (previous step prediction)
                # note: not used for NN input, used only for building next time step graph / ground truth
                x_next=x_next[t].clone(),
                v_next=v_next[t].clone(),
                node_force_next=all_node_force[t + 1].clone(),
                # all these are already in node_attr = [x, v, u, node_force], so saving them twice
                # is only due to rollout/other
                x_pos_t=_x.clone(),
                x_vel_t=_v.clone(),
                node_force=all_node_force[t].clone(),
                rest_pos=rest_pos_x.clone(),
                # ==================================================================
                #                           EDGE features
                # ==================================================================
                edge_attr=_edge_attr.clone(),
                # features to encode mesh connectivity in two different ways
                edge_index=graph.edge_index.clone(),
                x_element_connectivity=graph.x_element_connectivity.copy()
            )
        )
        self.targets.append(torch.concat([_a, _von_mises], dim=1))
        del _target


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
    relative_pos_x: list
    relative_pos_y: list
    relative_pos_z: list
    relative_pos_norm: list
    rest_relative_pos_x: list
    rest_relative_pos_y: list
    rest_relative_pos_z: list
    rest_relative_pos_norm: list
    force_x: list
    force_y: list
    force_z: list
    displacement_pos_x: list
    displacement_pos_y: list
    displacement_pos_z: list
    displacement_norm: list
    rest_pos_x: list
    rest_pos_y: list
    rest_pos_z: list

    def __init__(self):
        (self.acceleration_x, self.acceleration_y, self.acceleration_z, self.von_mises,
         self.velocity_x, self.velocity_y, self.velocity_z,
         self.position_x, self.position_y, self.position_z,
         self.relative_pos_x, self.relative_pos_y, self.relative_pos_z, self.relative_pos_norm,
         self.rest_relative_pos_x, self.rest_relative_pos_y, self.rest_relative_pos_z,
         self.rest_relative_pos_norm,
         self.force_x, self.force_y, self.force_z,
         self.displacement_pos_x, self.displacement_pos_y, self.displacement_pos_z,
         self.displacement_norm,
         self.rest_pos_x, self.rest_pos_y, self.rest_pos_z) = (
            [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [])


@hydra.main(version_base=None, config_path="../../conf/preprocess", config_name="eda")
def main(cfg):
    """
    Args:
        cfg: omegaconf.DictConfig
            Hydra-composed configuration.
    """
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

    print(f"Preprocessing {cfg_obj_nonorm.dataset_size} dataset\n" + "=" * 60)
    print(f"\tSave dir of config.yaml = {cfg_obj_nonorm.save_dir}")
    graphs_dir_train = Path(cfg_obj_nonorm.raw_data_dir) / "train" / "graphs"

    preprocessor_train_nonorm = EdaPreprocessor(cfg_obj_nonorm)
    preprocessor_train_z_isotr_norm = EdaPreprocessor(cfg_obj_z_isotr_norm)
    preprocessor_train_z_anisotr_norm = EdaPreprocessor(cfg_obj_z_anisotr_norm)
    preprocessor_train_rob_isotr_norm = EdaPreprocessor(cfg_obj_rob_isotr_norm)
    preprocessor_train_rob_anisotr_norm = EdaPreprocessor(cfg_obj_rob_anisotr_norm)

    print("\n" + "=" * 60 + "\ntrain, no norm\n" + "=" * 60)
    preprocessor_train_nonorm.preprocess_data(cfg_obj_nonorm, graphs_dir_train, None)
    print("\n" + "=" * 60 + "\ntrain, z isotropic norm\n" + "=" * 60)
    preprocessor_train_z_isotr_norm.preprocess_data(cfg_obj_z_isotr_norm, graphs_dir_train, None)
    print("\n" + "=" * 60 + "\ntrain, z anisotropic norm\n" + "=" * 60)
    preprocessor_train_z_anisotr_norm.preprocess_data(cfg_obj_z_anisotr_norm, graphs_dir_train, None)
    print("\n" + "=" * 60 + "\ntrain, robust isotropic norm\n" + "=" * 60)
    preprocessor_train_rob_isotr_norm.preprocess_data(cfg_obj_rob_isotr_norm, graphs_dir_train, None)
    print("\n" + "=" * 60 + "\ntrain, robust anisotropic norm\n" + "=" * 60)
    preprocessor_train_rob_anisotr_norm.preprocess_data(cfg_obj_rob_anisotr_norm, graphs_dir_train, None)
    print("\n" + "=" * 60 + "\nPreprocessing complete!")

    # Plots
    all_eda_obj_arg = [preprocessor_train_nonorm.eda_obj, preprocessor_train_z_isotr_norm.eda_obj,
                       preprocessor_train_z_anisotr_norm.eda_obj, preprocessor_train_rob_isotr_norm.eda_obj,
                       preprocessor_train_rob_anisotr_norm.eda_obj]

    del preprocessor_train_z_isotr_norm
    del preprocessor_train_z_anisotr_norm
    del preprocessor_train_rob_isotr_norm
    del preprocessor_train_rob_anisotr_norm
    del cfg_obj_z_isotr_norm
    del cfg_obj_z_anisotr_norm
    del cfg_obj_rob_isotr_norm
    del cfg_obj_rob_anisotr_norm

    print("Starting to plot all distributions")
    plot_all_distributions(all_eda_obj_arg, LABELS)
    print("Starting to generate ground truth GIFs...")
    generate_ground_truth_gifs(preprocessor_train_nonorm)

    del all_eda_obj_arg
    del preprocessor_train_nonorm
    del cfg_obj_nonorm
    del graphs_dir_train


def generate_ground_truth_gifs(preprocessor):
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

    for traj in sorted(set(traj_idx_list)):
        traj_graphs = [g for idx, g in zip(traj_idx_list, out_graphs) if idx == traj]

        for graph in traj_graphs:
            graph.von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)

        gif_path = gif_dir / f"trajectory_{traj:02d}.gif"
        print(f"\tGenerating ground truth GIF for trajectory {traj} ({len(traj_graphs)} steps)...")
        make_ground_truth_gif(traj_graphs, L=1.0, W=0.1, D=0.04,
                              gif_path=gif_path, fps=4, verbose=False)
        del gif_path
        del traj_graphs

    del traj_idx_list
    del out_graphs
    del gif_dir


if __name__ == "__main__":
    main()