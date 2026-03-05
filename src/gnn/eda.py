import torch
import numpy as np
import yaml
from src.gnn.helpers.misc.yamlobject import EdaYaml
from src.gnn.preprocessing.build_features import (calculate_kinematics, compute_von_mises_3d, build_edge_features,
                                                  build_valued_node_features)
from torch_geometric.data import Data, Batch
from pathlib import Path
import matplotlib.pyplot as plt

EPS = 1e-6
DT = 0.08
LIMIT_TIMESTEPS = 50
DEVICE = "cpu"

def _eda(cfg):
    torch.manual_seed(42)
    filenames = list(Path(cfg.raw_data_dir).glob("*.pt"))
    eda_obj = EdaObj()
    out_graphs, categorical_node_features, valued_node_features, edge_features, traj_idx, targets = (
        [], [], [], [], [], [])

    for idx_file, filename in enumerate(filenames):
        # For each file
        print(f"Processing file: {filename.name} ({idx_file + 1}/{len(filenames)})")
        dataloader = torch.load(filename, weights_only=False)

        # Create list graphs_list_batch with all graphs in the file
        graphs_list_batch = []
        for idx_data, data_idk in enumerate(dataloader):
            graph = data_idk.to(DEVICE)
            graphs_list_batch.append(graph)
            # if idx_data == 0: print_all_pyg_fields(dataloader, graph)

        # (For each file) From graphs_list (all graphs in batch), create a Batch object, and compute features
        # If you do batch.node_feature, you are creating a tensor which contains a tensor
        # (# graphs in batch, dim_feat_per_graph) where dim_feat_per_graph are the dimension of that node_feature for
        # one graph
        whole_trajectory_batch = Batch.from_data_list(graphs_list_batch)
        # Extract positions over time, NOTE: #graphs = len(graphs) = #timesteps .x attribute encodes positions
        x = whole_trajectory_batch.x.reshape(len(graphs_list_batch), -1, 3) # (#graphs, #points, 3) where 3 is (x,y,z)
        v, a = calculate_kinematics(x, DT) # v = (T, N, 3), a = (T-1, N, 3)
        v_next = v[1:] # (T-1, N, 3)
        x_next = x[1:] # (T-1, N, 3)
        rest_x0 = x[0] # Rest (first) mesh position (N, 3)
        # x_node_force is the force (scalar) in every point (x,y,z) of the graph at time t
        all_node_force = whole_trajectory_batch.x_node_force.reshape(len(graphs_list_batch), -1, 3) # (T, N, 3)
        # .x_bc contains 0,1 value with 1 = fixed, 0 = free
        is_bc = graphs_list_batch[0].x_bc

        # Inside loop: for each graph (time step) build its features
        for t, graph in enumerate(graphs_list_batch):
            # limit trajectory length to compute acceleration
            if t > LIMIT_TIMESTEPS - 2:
                break
            traj_idx.append(idx_file) # idx_file of the traj (of the file process)
            # create local (inside for) variables
            _node_force = all_node_force[t] # (N, 3)
            _x, _v, _a = x[t], v[t], a[t] # (N, 3)
            _categorical_node_attr = is_bc # (N, 1)
            # concatenate [position, velocity, rest_position, node_force] i.e. (4, N, 3) since each is (N, 3)
            _numerical_node_attr = build_valued_node_features(_x, _v, rest_x0, _node_force)
            # Obtain edge features [rel_pos, rel_pos_norm, rest_rel_pos, rest_rel_pos_norm]
            # .edge_index contains mesh connectivity
            _edge_attr = build_edge_features(_x, rest_x0, graph.edge_index, True)
            # .y_sig_t contains the full stress tensor
            # took the log because it helps making the distribution normal-like (symmetrical)
            # Output _von_mises is (N, 1) [graph.y_sig_t is (N, 9)]
            _von_mises_log = torch.log(compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + EPS)
            _von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
            node_attr = torch.concat([_categorical_node_attr, _numerical_node_attr], dim=1)

            eda_obj.acceleration_x.append(_a[:,0])
            eda_obj.acceleration_y.append(_a[:,1])
            eda_obj.acceleration_z.append(_a[:,2])
            eda_obj.von_mises.append(_von_mises)
            eda_obj.velocity_x.append(_v[:,0])
            eda_obj.velocity_y.append(_v[:,1])
            eda_obj.velocity_z.append(_v[:,2])
            eda_obj.position_x.append(_x[:, 0])
            eda_obj.position_y.append(_x[:, 1])
            eda_obj.position_z.append(_x[:, 2])
            eda_obj.relative_pos_x.append(_edge_attr[:, 0])
            eda_obj.relative_pos_y.append(_edge_attr[:, 1])
            eda_obj.relative_pos_z.append(_edge_attr[:, 2])
            eda_obj.relative_pos_norm.append(_edge_attr[:, 3])
            eda_obj.rest_relative_pos_x.append(_edge_attr[:, 4])
            eda_obj.rest_relative_pos_y.append(_edge_attr[:, 5])
            eda_obj.rest_relative_pos_z.append(_edge_attr[:, 6])
            eda_obj.rest_relative_pos_norm.append(_edge_attr[:, 7])
            eda_obj.stress_tensor_sum.append(graph.y_sig_t.sum(dim=1))
            eda_obj.stress_tensor.append(graph.y_sig_t)
            eda_obj.force_x.append(_node_force[:, 0])
            eda_obj.force_y.append(_node_force[:, 1])
            eda_obj.force_z.append(_node_force[:, 2])
            eda_obj.von_mises_log.append(_von_mises_log)

            out_graphs.append(
                Data(
                    # targets: current time acceleration and von mises stress
                    y_acc_t=a[t],
                    y_von_mises=_von_mises,
                    # time step of current graph (it was a global attribute on the graph)
                    t=graph.time,
                    # ==================================================================
                    #                       NODE features
                    # ==================================================================
                    # x = [is_bc, position, velocity, rest_position, node_force]
                    x=node_attr,
                    x_pos_t=x[t],
                    x_vel_t=v[t],
                    node_force=all_node_force[t],
                    rest_x0=rest_x0,
                    # note: the force is like the actuator: you need the next-step
                    node_force_next=all_node_force[t + 1],
                    # note: the position x_next, velocity v_next of t + 1 can be given since they can be computed from
                    # x, v at time t and acceleration of t - 1 (previous step prediction)
                    x_next=x_next[t],
                    v_next=v_next[t],
                    # ==================================================================
                    #                           EDGE features
                    # ==================================================================
                    edge_attr=_edge_attr,
                    # features to encode mesh connectivity in two different ways
                    edge_index=graph.edge_index,
                    x_element_connectivity=graph.x_element_connectivity,
                    )
            )
            # Accumulate to out_graphs for stats computation
            categorical_node_features.append(is_bc)
            # (4, N, 3)
            valued_node_features.append(build_valued_node_features(_x, _v, rest_x0, _node_force))
            edge_features.append(build_edge_features(_x, rest_x0, graph.edge_index, True))
            out_graphs.append(Data(edge_index=graph.edge_index, t=graph.time, y_acc_t=_a, x_pos_t=_x, x_vel_t=_v))
            targets.append(torch.concat([_a, _von_mises], dim=1))

    for idx in range(0, 9):
        eda_obj.__setattr__(f"stress_coord_{idx}", [])
        for stress_instance in eda_obj.stress_tensor:
            eda_obj.__getattribute__(f"stress_coord_{idx}").append(stress_instance[:, idx])

    plot_all_distributions(eda_obj)

def plot_all_distributions(eda_obj):
    """
    Args:
        eda_obj: object
            Object holding EDA attributes as (lists of) torch.Tensors
    """
    for key, val in eda_obj.__dict__.items():
        pass
        # plot_distribution(key, val)

    mean_vel, stddev_vel = compute_isotropic_stats((eda_obj.velocity_x, eda_obj.velocity_y, eda_obj.velocity_z))
    print(f"Mean and stddev of velocity are {mean_vel}, {stddev_vel}")
    plot_distribution_compare("velocity_x", eda_obj.velocity_x, mean_vel, stddev_vel, "")
    plot_distribution_compare("velocity_y", eda_obj.velocity_y, mean_vel, stddev_vel, "")
    plot_distribution_compare("velocity_z", eda_obj.velocity_z, mean_vel, stddev_vel, "")

    mean_acc, stddev_acc = compute_isotropic_stats((eda_obj.acceleration_x, eda_obj.acceleration_y,
                                                    eda_obj.acceleration_z))
    print(f"Mean and stddev of acceleration are {mean_acc}, {stddev_acc}")
    plot_distribution_compare("acceleration_x", eda_obj.acceleration_x, mean_acc, stddev_acc, "")
    plot_distribution_compare("acceleration_y", eda_obj.acceleration_y, mean_acc, stddev_acc, "")
    plot_distribution_compare("acceleration_z", eda_obj.acceleration_z, mean_acc, stddev_acc, "")

    mean_stress_tensor, stddev_stress_tensor = compute_isotropic_stats((eda_obj.stress_tensor,))
    print(f"Mean and stddev of stress tensor are {mean_stress_tensor}, {stddev_stress_tensor}")
    for idx in range(0,9):
        plot_distribution_compare("stress_tensor", eda_obj.__getattribute__(f"stress_coord_{idx}"),
                                  mean_stress_tensor, stddev_stress_tensor,f"Coordinate {idx}")

def plot_distribution(name, val):
    """
    Args:
        name: str
            Attribute name (used for titles/labels)
        val: list[torch.Tensor]
            List of tensors to concatenate; each will be reshaped to (N, 1)
    """
    chunks = []
    for t in val:
        chunks.append(t.reshape(-1, 1))
    x = torch.cat(chunks, dim=0).to(torch.float32)

    x_np = x.reshape(-1).detach().to(DEVICE).numpy()

    plt.figure()
    plt.hist(x_np, bins=200, density=True)
    plt.title(f"{name} distribution")
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"eda_artifacts/eda_{name}.png")

def plot_distribution_compare(name, val, mean, std, title):
    """
    Args:
        name: str
            Attribute name (used for titles/labels)
        val: list[torch.Tensor]
            List of tensors to concatenate; each will be reshaped to (N, 1)
    """
    chunks = []
    for t in val:
        chunks.append(t.reshape(-1, 1))
    x = torch.cat(chunks, dim=0).to(torch.float32)

    mean_anisotropic = x.mean(dim=0, keepdim=True)
    std_anisotropic = x.std(dim=0, unbiased=False, keepdim=True)

    x_norm_isotropic = (x - mean) / (std + EPS)
    x_norm_isotropic = x_norm_isotropic.reshape(-1).detach().to("cpu").numpy()
    x_norm_anis = (x - mean_anisotropic) / (std_anisotropic + EPS)
    x_norm_anis = x_norm_anis.reshape(-1).detach().to("cpu").numpy()

    bins = 200
    hist_iso, edges = np.histogram(x_norm_isotropic, bins=bins, density=True)
    hist_anis, _ = np.histogram(x_norm_anis, bins=edges, density=True)
    hist_raw, _ = np.histogram(x, bins=edges, density=True)
    ymax = max(hist_iso.max(), hist_anis.max(), hist_raw.max())
    plt.figure()
    plt.hist(x_norm_isotropic, label="Isotropic", color="r", bins=edges, density=True)
    plt.hist(x_norm_anis, label="Anisotropic", color="b", bins=edges, density=True)
    plt.hist(x, label="Not normalized", color="g", bins=edges, density=True)
    plt.ylim(0, ymax * 1.05)
    plt.title(f"{title} {name} distribution (normalized)")
    plt.xlabel(f"{title} {name} (z-score)")
    plt.ylabel(f"{name} Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_isotropic_stats(val_list):
    """
    Args:
        name: str
            Attribute name (used for titles/labels)
        val: list[torch.Tensor]
            List of tensors to concatenate; each will be reshaped to (N, 1)
    """
    chunks = []
    for val in val_list:
        for t in val:
            chunks.append(t.reshape(-1, 1))
    x = torch.cat(chunks, dim=0).to(torch.float32)
    print(x)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    return mean, std

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


if __name__ == "__main__":
    with open("conf/eda.yaml", "r") as f:
        data = yaml.safe_load(f)
    cfg_obj = EdaYaml(data)
    _eda(cfg=cfg_obj)
    print("\n" + "=" * 60 + "\nPreprocessing complete!")