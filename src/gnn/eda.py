import torch
import hydra
from src.gnn.misc_helpers.yamlobject import PreprocessYaml
from src.gnn.preprocess import TrainPreprocessor, create_preprocessor
from torch_geometric.data import Data
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any

EPS = 1e-6
DT = 0.08
LIMIT_TIMESTEPS = 50
DEVICE = "cpu"

def plot_all_distributions(eda_obj_list):
    """
    Args:
        eda_obj_list: list[EdaObj]
            Object holding lists of torch.Tensors. One list for each attribute of the graph.
    """
    # For each attribute in the first eda_obj (the rest will be all the same)
    for key, _ in eda_obj_list[0].__dict__.items():
        plot_distribution(key, eda_obj_list)

def plot_distribution(attr_name, eda_obj_list):
    """
    Plots non-normalized distribution of given value, and normalized in all four ways.

    Args:
        attr_name: str
            Attribute name (used for titles/labels)
        val: list[EdaObj]
            List of eda objs, from which I will take attr_name
    """
    chunks_list = []
    for idx in range(len(eda_obj_list)):
        _chunks = []
        _val = eda_obj_list[idx].getattr(attr_name)
        for t in _val:
            _chunks.append(t.reshape(-1, 1))
        _x = torch.cat(_chunks, dim=0).to(torch.float32)
        _x_np = _x.reshape(-1).detach().to(DEVICE).numpy()
        chunks_list[idx] = _x_np

    plt.figure()
    for _x_np in chunks_list:
        plt.hist(_x_np, bins=200, density=True)
    plt.title(f"{attr_name} distribution")
    plt.xlabel(attr_name)
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"eda_artifacts/eda_{attr_name}.png")

class EdaPreprocessor(TrainPreprocessor):
    eda_obj: Any # of eda obj

    def __init__(self, cfg):
        super().__init__(cfg)

    def _save_stats_train(self, node_stats, edge_stats, target_stats, save_dir):
        pass

    def _save_preprocessed(self, traj_idx, out_graphs, save_dir):
        """
        You previously have normalized, so here you can collect the normalized version.
        Remember: disable log to have both log and nolog von mises stress
        """
        eda_obj = EdaObj()
        # For all graphs
        for traj in set(traj_idx):
            print(f"Processing trajectory: {traj}")
            for idx, graph in zip(traj_idx, out_graphs):
                if idx == traj:
                    # Acceleration
                    eda_obj.acceleration_x.append(graph.y_acc_t[:, 0])
                    eda_obj.acceleration_y.append(graph.y_acc_t[:, 1])
                    eda_obj.acceleration_z.append(graph.y_acc_t[:, 2])
                    # Von mises (not log)
                    eda_obj.von_mises.append(graph.y_von_mises)
                    # node_force
                    eda_obj.force_x.append(graph.node_force[:, 0])
                    eda_obj.force_y.append(graph.node_force[:, 1])
                    eda_obj.force_z.append(graph.node_force[:, 2])
                    # Velocity x_vel_t
                    eda_obj.velocity_x.append(graph.x_vel_t[:, 0])
                    eda_obj.velocity_y.append(graph.x_vel_t[:, 1])
                    eda_obj.velocity_z.append(graph.x_vel_t[:, 2])
                    # Position x_pos_t
                    eda_obj.position_x.append(graph.x_pos_t[:, 0])
                    eda_obj.position_y.append(graph.x_pos_t[:, 1])
                    eda_obj.position_z.append(graph.x_pos_t[:, 2])
                    # Rest position
                    # TODO:
                    # Position edges
                    eda_obj.relative_pos_x.append(graph.edge_attr[:, 0])
                    eda_obj.relative_pos_y.append(graph.edge_attr[:, 1])
                    eda_obj.relative_pos_z.append(graph.edge_attr[:, 2])
                    # Norm
                    eda_obj.relative_pos_norm.append(graph.edge_attr[:, 3])
                    # Rest position edges
                    eda_obj.rest_relative_pos_x.append(graph.edge_attr[:, 4])
                    eda_obj.rest_relative_pos_y.append(graph.edge_attr[:, 5])
                    eda_obj.rest_relative_pos_z.append(graph.edge_attr[:, 6])
                    # Norm
                    eda_obj.rest_relative_pos_norm.append(graph.edge_attr[:, 7])
                    # Stress tensor
                    eda_obj.stress_tensor_sum.append(graph.y_sig_t.sum(dim=1))
                    eda_obj.stress_tensor.append(graph.y_sig_t)
                    # Take
                    eda_obj.von_mises_log.append(graph.von_mises_log)
                    self.eda_obj = eda_obj

    def _append_outgraphs(self, graph, _a, _x, _v, _von_mises, _edge_attr, _node_attr, t, x_next, v_next,
                          all_node_force, rest_pos_x, _categorical_node_attr, von_mises_log):
        """
        Must use the same one as Validation, without normalization.
        I will change the plotting function to not normalize.
        """
        # Note: later for each data I will create the standard pyg - compatible attributes x, y, edge_attr actually
        # used for NNs
        _target = torch.concat([_a, _von_mises], dim=1)
        self.out_graphs.append(
            Data(
                # targets: current time acceleration and von mises stress
                y=_target.clone(),
                y_acc_t=_a.clone(),
                y_von_mises=_von_mises.clone(),
                von_mises_log = von_mises_log.clone(),
                # (global attribute on the graph) time step of current graph
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
                x_element_connectivity=graph.x_element_connectivity.clone()
            )
        )
        self.targets.append(torch.concat([_a, _von_mises], dim=1))

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
    stress_tensor: list
    force_x: list
    force_y: list
    force_z: list
    von_mises_log: list
    stress_tensor_sum: list

    def __init__(self):
        (self.acceleration_x, self.acceleration_y, self.acceleration_z, self.von_mises, self.velocity_x,
         self.velocity_y, self.velocity_z, self.position_x, self.position_y, self.position_z,
         self.relative_pos_x, self.relative_pos_y, self.relative_pos_z, self.stress_tensor,
         self.relative_pos_norm, self.rest_relative_pos_x, self.rest_relative_pos_y,
         self.rest_relative_pos_z, self.rest_relative_pos_norm, self.force_x, self.force_y, self.force_z,
         self.von_mises_log, self.stress_tensor_sum) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                                                        [], [], [], [], [], [], [], [])

@hydra.main(version_base=None, config_path="../../conf/preprocess", config_name="eda")
def main(cfg):
    """
    Args:
        cfg: omegaconf.DictConfig
            Hydra-composed configuration. Run with --multirun to sweep over multiple values,
            e.g.: python -m src.gnn.preprocess --multirun split=train,val,test
    """
    cfg_obj = PreprocessYaml(cfg)
    # Copy
    cfg_obj_nonorm = cfg.copy()
    cfg_obj_z_isotr_norm = cfg.copy()
    cfg_obj_z_anisotr_norm = cfg.copy()
    cfg_obj_rob_isotr_norm = cfg.copy()
    cfg_obj_rob_anisotr_norm = cfg.copy()
    # Assign normalizer
    cfg_obj_nonorm.normalizer_method = "None"
    cfg_obj_z_isotr_norm.normalizer_method = "z"
    cfg_obj_z_anisotr_norm.normalizer_method = "z"
    cfg_obj_rob_isotr_norm.normalizer_method = "robust"
    cfg_obj_rob_anisotr_norm.normalizer_method = "robust"
    # Assign everything else
    cfg_obj_nonorm.isotropic_normalizer = True
    cfg_obj_z_isotr_norm.isotropic_normalizer = True
    cfg_obj_z_anisotr_norm.isotropic_normalizer = False
    cfg_obj_rob_isotr_norm.isotropic_normalizer = True
    cfg_obj_rob_anisotr_norm.isotropic_normalizer = False

    print(f"Preprocessing {cfg_obj.dataset_size} dataset\n" + "=" * 60)
    print(f"\tSave dir of config.yaml = {cfg_obj.save_dir}")
    # Where to save results
    graphs_dir_train = Path(cfg_obj.raw_data_dir) / "train" / "graphs"

    # Create preprocessors object
    preprocessor_train_nonorm = create_preprocessor("eda", cfg_obj_nonorm, None)
    preprocessor_train_z_isotr_norm = create_preprocessor("eda", cfg_obj_z_isotr_norm, None)
    preprocessor_train_z_anisotr_norm = create_preprocessor("eda", cfg_obj_z_anisotr_norm, None)
    preprocessor_train_rob_isotr_norm = create_preprocessor("eda", cfg_obj_rob_isotr_norm, None)
    preprocessor_train_rob_anisotr_norm = create_preprocessor("eda", cfg_obj_rob_anisotr_norm, None)

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
    plot_all_distributions([preprocessor_train_nonorm.eda_obj,
                           preprocessor_train_z_isotr_norm.eda_obj,
                           preprocessor_train_z_anisotr_norm.eda_obj,
                           preprocessor_train_rob_isotr_norm.eda_obj,
                           preprocessor_train_rob_anisotr_norm.eda_obj])

if __name__ == "__main__":
    main()