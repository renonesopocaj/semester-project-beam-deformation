"""
preprocess_deforming_plate.py

One-time conversion of a DeformingPlate .tfrecord split into per-trajectory
.npz files.  This script has NO project imports — it only uses tensorflow,
numpy, json, argparse, and pathlib — so it can be run with the system Python
3.9 installation where TensorFlow is available, independently of the
fenicsproject conda environment.

Usage
-----
python3 scripts/preprocess_deforming_plate.py \
    --tfrecord deforming_plate/test.tfrecord \
    --out-dir  deforming_plate/processed/test

Each output file is named ``trajectory_NNNNN.npz`` and contains:
    world_pos  : float32, shape (T, N, 3)
    stress     : float32, shape (T, N, 1)
    node_type  : int32,   shape (N,)
    cells      : int32,   shape (num_cells, 4)
"""

import argparse
import json
import numpy as np
from pathlib import Path


def _get_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise ImportError(
            "tensorflow is required; run this script with the system Python 3.9 "
            "where TF is installed: python3 scripts/preprocess_deforming_plate.py ..."
        ) from exc


def _decode_record(serialized, meta, tf):
    """
    Decode one serialised TFRecord example using the dataset meta.json schema.

    Args:
        serialized: tf.Tensor  raw bytes of one record
        meta: dict  parsed meta.json
        tf: module  tensorflow module
    return: dict  field_name → numpy array
    """
    specs = {name: tf.io.VarLenFeature(tf.string) for name in meta["field_names"]}
    parsed = tf.io.parse_single_example(serialized, specs)
    dtype_map = {"float32": tf.float32, "int32": tf.int32}
    traj_len = meta["trajectory_length"]
    output = {}
    for name, field in meta["features"].items():
        dtype = dtype_map[field["dtype"]]
        decoded = tf.io.decode_raw(parsed[name].values, dtype)
        decoded = tf.reshape(decoded, field["shape"])
        if field["type"] == "static":
            decoded = tf.tile(decoded, [traj_len, 1, 1])
        output[name] = decoded.numpy()
    return output


def _extract_arrays(record, n_timesteps, traj_len):
    """
    Pull the four arrays we need from a decoded record, clipping to n_timesteps.

    Args:
        record: dict  decoded field arrays
        n_timesteps: int  max timesteps to keep
        traj_len: int  full trajectory length from meta
    return: (world_pos, stress, node_type, cells)
        world_pos : float32 (T, N, 3)
        stress    : float32 (T, N, 1)
        node_type : int32   (N,)
        cells     : int32   (num_cells, 4)
    """
    t_clip = min(n_timesteps, traj_len)
    world_pos = record["world_pos"][:t_clip].astype(np.float32)
    stress = record["stress"][:t_clip].astype(np.float32)
    node_type = record["node_type"][0, :, 0].astype(np.int32)
    cells = record["cells"][0].astype(np.int32)
    return world_pos, stress, node_type, cells


def preprocess(tfrecord_path, out_dir, n_trajectories, n_timesteps):
    """
    Convert tfrecord to per-trajectory .npz files.

    Args:
        tfrecord_path: Path
        out_dir: Path
        n_trajectories: int  number of trajectories to convert (0 = all)
        n_timesteps: int  max timesteps per trajectory
    """
    tf = _get_tf()
    tfrecord_path = Path(tfrecord_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = tfrecord_path.parent / "meta.json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    traj_len = meta["trajectory_length"]
    print(f"meta loaded: trajectory_length={traj_len}, "
          f"fields={meta['field_names']}")

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    saved = 0
    for idx, serialized in enumerate(dataset):
        if n_trajectories > 0 and idx >= n_trajectories:
            break
        record = _decode_record(serialized, meta, tf)
        world_pos, stress, node_type, cells = _extract_arrays(record, n_timesteps, traj_len)

        out_path = out_dir / f"trajectory_{idx:05d}.npz"
        np.savez_compressed(
            out_path,
            world_pos=world_pos,
            stress=stress,
            node_type=node_type,
            cells=cells,
        )
        saved += 1
        if saved % 10 == 0 or idx < 5:
            print(f"  saved {out_path}  world_pos={world_pos.shape}  "
                  f"stress={stress.shape}  nodes={node_type.shape[0]}  "
                  f"cells={cells.shape[0]}")

    print(f"\nDone. {saved} trajectories written to {out_dir}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert DeformingPlate .tfrecord to per-trajectory .npz files."
    )
    parser.add_argument("--tfrecord", required=True,
                        help="Path to the split .tfrecord file (e.g. deforming_plate/test.tfrecord).")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for .npz files (e.g. deforming_plate/processed/test).")
    parser.add_argument("--n-trajectories", type=int, default=0,
                        help="Number of trajectories to convert (0 = all).")
    parser.add_argument("--n-timesteps", type=int, default=200,
                        help="Maximum timesteps to keep per trajectory.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    preprocess(
        tfrecord_path=Path(args.tfrecord),
        out_dir=Path(args.out_dir),
        n_trajectories=args.n_trajectories,
        n_timesteps=args.n_timesteps,
    )
