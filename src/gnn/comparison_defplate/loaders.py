"""
loaders.py

Loaders for biax_longer (raw PyG .pt DataLoaders) and DeformingPlate
(.tfrecord), both returning TrajectorySample objects with the five standard
signal arrays (position, displacement, step_displacement, velocity, von_mises).

Empirical probe assertions at load time confirm expected shapes and fail loudly
on the first trajectory if anything differs.
"""

import json
import re
import numpy as np
import torch
from pathlib import Path

from src.gnn.comparison_defplate.trajectory import TrajectorySample, SIGNAL_NAMES
from src.gnn.preprocessing.build_features import compute_von_mises_3d


# ─── shared helpers ──────────────────────────────────────────────────────────

def _assert_probe(condition, msg):
    if not condition:
        raise AssertionError(f"[loader probe] {msg}")


def _to_np(tensor):
    return tensor.detach().cpu().numpy()


def _parse_filename_params(fname):
    """
    Extract physical parameters from a biax_longer filename, e.g.:
    graphsL1.0_W0.25_D0.2_..._Piy-0.9_Piz1.0_...pt

    Args:
        fname: str  filename (basename only)
    return: dict  with keys L, W, D, fy, fz (floats); missing keys absent
    """
    patterns = {
        "L": r"graphsL([+-]?[0-9]*\.?[0-9]+)",
        "W": r"_W([+-]?[0-9]*\.?[0-9]+)",
        "D": r"_D([+-]?[0-9]*\.?[0-9]+)",
        "fy": r"_Piy([+-]?[0-9]*\.?[0-9]+)",
        "fz": r"_Piz([+-]?[0-9]*\.?[0-9]+)",
    }
    params = {}
    for key, pat in patterns.items():
        m = re.search(pat, fname)
        if m:
            params[key] = float(m.group(1))
    return params


# ─── biax_longer loader ───────────────────────────────────────────────────────

class BiaxLongerLoader:
    """
    Loads all Nsteps400 trajectories from a biax_longer raw split directory.

    Expected file structure:
        <graphs_dir>/*.pt  — each file is a torch-saved DataLoader (one per trajectory)

    The Nsteps400 files are the extended-rollout variants of the selected
    trajectories.  They live alongside the shorter Nsteps50 files in the same
    graphs directory but are NOT listed in trajectory_params.json; they are
    discovered by globbing for '*Nsteps400*.pt'.

    Probes the first loaded trajectory for expected shapes:
        x: (N, 3)   x_vel_t: (N, 3)   y_sig_t: (N, 9)   x_bc: (N, 1)
        edge_index: (2, E)
    """

    def __init__(self, graphs_dir, n_timesteps):
        """
        Args:
            graphs_dir: Path or str
                Directory containing raw .pt trajectory files.
            n_timesteps: int
                Maximum number of timesteps to keep per trajectory.
        """
        self.graphs_dir = Path(graphs_dir)
        self.n_timesteps = n_timesteps
        self._probed = False

    def _select_files(self):
        """
        Return sorted list of Nsteps400 .pt paths.

        return: list of Path
        """
        files = sorted(self.graphs_dir.glob("*Nsteps400*.pt"))
        return files

    def _iter_graphs(self, pt_path):
        """
        Iterate the DataLoader stored in a raw .pt file and yield individual
        DataBatch objects (one per timestep), up to n_timesteps.

        Args:
            pt_path: Path
        return: list of DataBatch
        """
        loader = torch.load(pt_path, weights_only=False, map_location="cpu")
        graphs = []
        for i, batch in enumerate(loader):
            if i >= self.n_timesteps:
                break
            graphs.append(batch)
        return graphs

    def _probe_first(self, graphs):
        g = graphs[0]
        _assert_probe(hasattr(g, "x"), "missing field 'x'")
        _assert_probe(g.x.ndim == 2 and g.x.shape[1] == 3, f"x shape mismatch: {g.x.shape}")
        _assert_probe(hasattr(g, "x_vel_t"), "missing field 'x_vel_t'")
        _assert_probe(g.x_vel_t.shape == g.x.shape, f"x_vel_t shape {g.x_vel_t.shape} != x shape {g.x.shape}")
        _assert_probe(hasattr(g, "y_sig_t"), "missing field 'y_sig_t'")
        _assert_probe(g.y_sig_t.ndim == 2 and g.y_sig_t.shape[1] == 9, f"y_sig_t shape: {g.y_sig_t.shape}")
        _assert_probe(hasattr(g, "x_bc"), "missing field 'x_bc'")
        _assert_probe(g.x_bc.ndim == 2 and g.x_bc.shape[1] == 1, f"x_bc shape: {g.x_bc.shape}")
        _assert_probe(hasattr(g, "edge_index"), "missing field 'edge_index'")
        _assert_probe(g.edge_index.ndim == 2 and g.edge_index.shape[0] == 2,
                      f"edge_index shape: {g.edge_index.shape}")

    def _build_sample(self, graphs, path):
        """
        Convert a list of DataBatch objects into a TrajectorySample.

        Args:
            graphs: list of DataBatch
            path: Path  source file path (used to parse parameters from filename)
        return: TrajectorySample
        """
        n = graphs[0].x.shape[0]
        t = len(graphs)

        pos = np.stack([_to_np(g.x) for g in graphs], axis=0)          # (T, N, 3)
        vel = np.stack([_to_np(g.x_vel_t) for g in graphs], axis=0)    # (T, N, 3)
        sig9 = np.stack([_to_np(g.y_sig_t) for g in graphs], axis=0)   # (T, N, 9)

        # Compute von Mises per timestep
        vm_list = []
        for t_i in range(t):
            sig9_t = torch.from_numpy(sig9[t_i])
            vm_t = compute_von_mises_3d(sig9_t)  # (N,)
            vm_list.append(_to_np(vm_t))
        vm = np.stack(vm_list, axis=0)[:, :, None]  # (T, N, 1)

        disp = pos - pos[0:1]                        # (T, N, 3)
        step_disp = pos[1:] - pos[:-1]              # (T-1, N, 3)

        edge_index = _to_np(graphs[0].edge_index)   # (2, E)
        bc = _to_np(graphs[0].x_bc).squeeze(-1)     # (N,)
        free_mask = bc < 0.5                         # True = free

        params = _parse_filename_params(path.name)
        label_parts = [f"fy{params.get('fy', 0):.2f}", f"fz{params.get('fz', 0):.2f}",
                       f"L{params.get('L', 0):.2f}"]
        name = "biax_" + "_".join(label_parts)

        signals = {
            "position": pos,
            "displacement": disp,
            "step_displacement": step_disp,
            "velocity": vel,
            "von_mises": vm,
        }
        return TrajectorySample(name=name, edge_index=edge_index, free_mask=free_mask,
                                signals=signals, params=params)

    def load_all(self):
        """
        Load all Nsteps400 trajectories found in the graphs directory.

        return: list of TrajectorySample
        """
        files = self._select_files()
        if not files:
            raise FileNotFoundError(f"No *Nsteps400*.pt files found in {self.graphs_dir}")

        samples = []
        for path in files:
            graphs = self._iter_graphs(path)
            if not self._probed:
                self._probe_first(graphs)
                self._probed = True
            sample = self._build_sample(graphs, path)
            samples.append(sample)
        return samples


# ─── DeformingPlate loader ───────────────────────────────────────────────────

class DeformingPlateLoader:
    """
    In-memory loader for DeformingPlate .tfrecord trajectories.

    Reads the raw tfrecord and returns TrajectorySample objects without writing
    any .pt files.  This is a refactored version of DeformingPlatePreprocessor
    from tests/gnn/deformingplate_comparison_test.py.

    Probes the first trajectory for expected shapes:
        world_pos: (T, N, 3)   stress: (T, N, 1)   node_type: (N,)
        edge_index (derived): (2, E)
    """

    def __init__(self, tfrecord_path, n_trajectories, n_timesteps):
        """
        Args:
            tfrecord_path: Path or str
                Path to the split .tfrecord file (e.g. deforming_plate/test.tfrecord).
            n_trajectories: int
                Number of trajectories to load (first n in iteration order).
            n_timesteps: int
                Maximum number of timesteps to keep.
        """
        import os
        meta_path = Path(tfrecord_path).parent / "meta.json"
        with open(meta_path) as fh:
            self.meta = json.load(fh)
        self.tfrecord_path = str(tfrecord_path)
        self.n_trajectories = n_trajectories
        self.n_timesteps = n_timesteps
        self._probed = False

    # ── TFRecord decoding ────────────────────────────────────────────────────

    def _get_tf(self):
        try:
            import tensorflow as tf
            return tf
        except ImportError as exc:
            raise ImportError("tensorflow is required to load DeformingPlate .tfrecord files") from exc

    def _decode_record(self, serialized, tf):
        specs = {name: tf.io.VarLenFeature(tf.string) for name in self.meta["field_names"]}
        parsed = tf.io.parse_single_example(serialized, specs)
        dtype_map = {"float32": tf.float32, "int32": tf.int32}
        output = {}
        for name, field in self.meta["features"].items():
            dtype = dtype_map[field["dtype"]]
            decoded = tf.io.decode_raw(parsed[name].values, dtype)
            decoded = tf.reshape(decoded, field["shape"])
            if field["type"] == "static":
                decoded = tf.tile(decoded, [self.meta["trajectory_length"], 1, 1])
            output[name] = decoded.numpy()
        return output

    # ── edge building ────────────────────────────────────────────────────────

    @staticmethod
    def _tetrahedra_to_edges(cells_np):
        """
        Build undirected COO edge_index from tetrahedral connectivity.

        Args:
            cells_np: np.ndarray  shape (num_cells, 4), integer node indices
        return: np.ndarray  shape (2, E), dtype int64
        """
        cells = torch.as_tensor(cells_np, dtype=torch.long)
        local_pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long)
        edges = cells[:, local_pairs].reshape(-1, 2).t().contiguous()
        import torch_geometric.utils as pyg_utils
        edge_index = pyg_utils.to_undirected(edges)
        edge_index = pyg_utils.coalesce(edge_index)
        return edge_index.numpy().astype(np.int64)

    # ── probing ──────────────────────────────────────────────────────────────

    def _probe_first(self, record):
        t_clip = min(self.n_timesteps, self.meta["trajectory_length"])
        wp = record["world_pos"]
        st = record["stress"]
        nt = record["node_type"]
        _assert_probe(wp.ndim == 3 and wp.shape[2] == 3, f"world_pos shape: {wp.shape}")
        _assert_probe(st.ndim == 3 and st.shape[2] == 1, f"stress shape: {st.shape}")
        _assert_probe(nt.ndim == 3 and nt.shape[2] == 1, f"node_type shape: {nt.shape}")
        _assert_probe(wp.shape[0] >= t_clip, f"trajectory shorter than n_timesteps: {wp.shape[0]} < {t_clip}")

    # ── sample building ──────────────────────────────────────────────────────

    def _build_sample(self, record, traj_idx):
        t_clip = min(self.n_timesteps, self.meta["trajectory_length"])

        world_pos = record["world_pos"][:t_clip].astype(np.float32)   # (T, N, 3)
        stress = record["stress"][:t_clip].astype(np.float32)          # (T, N, 1)
        node_type = record["node_type"][0, :, 0].astype(np.int32)      # (N,)
        cells = record["cells"][0].astype(np.int64)                    # (num_cells, 4)

        edge_index = self._tetrahedra_to_edges(cells)                  # (2, E)

        # free = normal interior nodes (type 0); exclude clamped (type 1) and handle (type 3)
        free_mask = (node_type == 0)

        disp = world_pos - world_pos[0:1]                              # (T, N, 3)
        step_disp = world_pos[1:] - world_pos[:-1]                    # (T-1, N, 3)
        # Velocity for DP: same as step_displacement (MGN target convention)
        vel = step_disp.copy()                                         # (T-1, N, 3)

        signals = {
            "position": world_pos,
            "displacement": disp,
            "step_displacement": step_disp,
            "velocity": vel,
            "von_mises": stress,
        }
        return TrajectorySample(name=f"dp_{traj_idx:03d}", edge_index=edge_index,
                                free_mask=free_mask, signals=signals,
                                params={"traj_idx": traj_idx})

    def load_all(self):
        """
        Load the first n_trajectories trajectories from the tfrecord.

        return: list of TrajectorySample
        """
        tf = self._get_tf()
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        samples = []
        for traj_idx, serialized in enumerate(dataset):
            if traj_idx >= self.n_trajectories:
                break
            record = self._decode_record(serialized, tf)
            if not self._probed:
                self._probe_first(record)
                self._probed = True
            sample = self._build_sample(record, traj_idx)
            samples.append(sample)
        return samples


# ─── DeformingPlate .npz loader ──────────────────────────────────────────────

class DeformingPlateNpzLoader:
    """
    Loads DeformingPlate trajectories from pre-converted .npz files.

    Expected directory layout (produced by scripts/preprocess_deforming_plate.py):
        <npz_dir>/trajectory_00000.npz
        <npz_dir>/trajectory_00001.npz
        ...

    Each file must contain:
        world_pos  float32  (T, N, 3)
        stress     float32  (T, N, 1)
        node_type  int32    (N,)
        cells      int32    (num_cells, 4)

    No TensorFlow dependency — only numpy and pathlib.

    Probes the first loaded trajectory for expected shapes:
        world_pos: (T, N, 3)   stress: (T, N, 1)   node_type: (N,)
    """

    def __init__(self, npz_dir, n_trajectories, n_timesteps):
        """
        Args:
            npz_dir: Path or str
                Directory containing trajectory_NNNNN.npz files.
            n_trajectories: int
                Number of trajectories to load.
            n_timesteps: int
                Maximum number of timesteps to keep per trajectory.
        """
        self.npz_dir = Path(npz_dir)
        self.n_trajectories = n_trajectories
        self.n_timesteps = n_timesteps
        self._probed = False

    def _select_files(self):
        """
        Return sorted list of .npz paths, limited to n_trajectories.

        return: list of Path
        """
        files = sorted(self.npz_dir.glob("trajectory_*.npz"))
        return files[:self.n_trajectories]

    def _probe_first(self, data):
        wp = data["world_pos"]
        st = data["stress"]
        nt = data["node_type"]
        _assert_probe(wp.ndim == 3 and wp.shape[2] == 3, f"world_pos shape: {wp.shape}")
        _assert_probe(st.ndim == 3 and st.shape[2] == 1, f"stress shape: {st.shape}")
        _assert_probe(nt.ndim == 1, f"node_type shape: {nt.shape}")

    def _build_sample(self, data, traj_idx):
        """
        Convert arrays from one .npz file into a TrajectorySample.

        Args:
            data: dict-like  (NpzFile)
            traj_idx: int
        return: TrajectorySample
        """
        t_clip = min(self.n_timesteps, data["world_pos"].shape[0])

        world_pos = data["world_pos"][:t_clip].astype(np.float32)   # (T, N, 3)
        stress = data["stress"][:t_clip].astype(np.float32)          # (T, N, 1)
        node_type = data["node_type"].astype(np.int32)               # (N,)
        cells = data["cells"].astype(np.int64)                       # (num_cells, 4)

        edge_index = DeformingPlateLoader._tetrahedra_to_edges(cells)  # (2, E)

        # free = normal interior nodes (type 0)
        free_mask = (node_type == 0)

        disp = world_pos - world_pos[0:1]                            # (T, N, 3)
        step_disp = world_pos[1:] - world_pos[:-1]                  # (T-1, N, 3)
        vel = step_disp.copy()                                       # (T-1, N, 3)

        signals = {
            "position": world_pos,
            "displacement": disp,
            "step_displacement": step_disp,
            "velocity": vel,
            "von_mises": stress,
        }
        return TrajectorySample(name=f"dp_{traj_idx:03d}", edge_index=edge_index,
                                free_mask=free_mask, signals=signals,
                                params={"traj_idx": traj_idx})

    def load_all(self):
        """
        Load all matching .npz trajectories from npz_dir.

        return: list of TrajectorySample
        """
        files = self._select_files()
        if not files:
            raise FileNotFoundError(
                f"No trajectory_*.npz files found in {self.npz_dir}. "
                "Run scripts/preprocess_deforming_plate.py first."
            )

        samples = []
        for traj_idx, path in enumerate(files):
            data = np.load(path)
            if not self._probed:
                self._probe_first(data)
                self._probed = True
            sample = self._build_sample(data, traj_idx)
            samples.append(sample)
        return samples
