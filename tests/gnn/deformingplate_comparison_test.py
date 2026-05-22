"""
deformingplate_comparison_test.py

Tests for the DeformingPlateLoader and BiaxLongerLoader in
src/gnn/data_comparison/loaders.py.

What the tests check (informal specification):
- DeformingPlateLoader.load_all() produces TrajectorySample objects whose
  signal arrays have the correct number of dimensions, whose T axis is capped
  at n_timesteps, whose N axis is consistent across all five signals
  (within the same trajectory), and whose edge_index is a valid 2-row integer
  array with node indices in [0, N).  The free_mask is a boolean array of
  length N with at least one True entry.
- BiaxLongerLoader.load_all() produces TrajectorySample objects satisfying the
  same shape invariants: (T, N, C) for each signal with T <= n_timesteps,
  edge_index in bounds, and free_mask boolean.
- Both loaders enforce the five expected signal keys from SIGNAL_NAMES.
"""

import unittest
from pathlib import Path

import numpy as np

BIAX_TEST_DIR = Path("dataset/raw/raw/biax_longer/small/test/graphs")
DP_TFRECORD = Path("deforming_plate/test.tfrecord")
SELECTED_TRAJS = [1, 2, 4, 6, 8, 9, 11, 14, 16, 17]
N_TIMESTEPS = 10   # small truncation for fast test
N_DP_TRAJS = 2


def _check_sample_invariants(test_case, sample, n_timesteps_max, dataset_label):
    """
    Shared assertion helper: verifies shape and type invariants of a TrajectorySample.

    Args:
        test_case: unittest.TestCase
        sample: TrajectorySample
        n_timesteps_max: int
        dataset_label: str
    """
    from src.gnn.comparison_defplate.trajectory import SIGNAL_NAMES
    ctx = f"[{dataset_label} / {sample.name}]"

    test_case.assertIsInstance(sample.name, str, f"{ctx} name must be str")
    test_case.assertEqual(sample.edge_index.ndim, 2, f"{ctx} edge_index must be 2-D")
    test_case.assertEqual(sample.edge_index.shape[0], 2, f"{ctx} edge_index row 0 = sources, row 1 = dests")
    n = sample.num_nodes
    test_case.assertGreater(n, 0, f"{ctx} must have nodes")
    test_case.assertEqual(sample.free_mask.shape, (n,), f"{ctx} free_mask shape mismatch")
    test_case.assertEqual(sample.free_mask.dtype, bool, f"{ctx} free_mask must be bool")
    test_case.assertTrue(sample.free_mask.any(), f"{ctx} at least one free node expected")

    # edge_index node indices must be in [0, N)
    e_src = sample.edge_index[0]
    e_dst = sample.edge_index[1]
    test_case.assertTrue((e_src >= 0).all() and (e_src < n).all(),
                          f"{ctx} edge_index sources out of range")
    test_case.assertTrue((e_dst >= 0).all() and (e_dst < n).all(),
                          f"{ctx} edge_index dests out of range")

    for sig_name in SIGNAL_NAMES:
        test_case.assertIn(sig_name, sample.signals, f"{ctx} missing signal '{sig_name}'")
        arr = sample.signals[sig_name]
        test_case.assertEqual(arr.ndim, 3, f"{ctx} signal '{sig_name}' must be 3-D (T, N, C)")
        t_actual, n_actual, c = arr.shape
        test_case.assertLessEqual(t_actual, n_timesteps_max,
                                  f"{ctx} signal '{sig_name}' T={t_actual} exceeds limit")
        test_case.assertEqual(n_actual, n,
                               f"{ctx} signal '{sig_name}' N={n_actual} != num_nodes={n}")
        test_case.assertGreater(c, 0, f"{ctx} signal '{sig_name}' must have ≥1 channel")
        test_case.assertTrue(np.isfinite(arr).any(), f"{ctx} signal '{sig_name}' is all NaN/inf")


class TestDeformingPlateLoader(unittest.TestCase):

    def test_loads_correct_shapes_and_edge_validity(self):
        """
        Verifies that DeformingPlateLoader.load_all() returns n_trajectories
        TrajectorySample objects, each with:
        - five signal arrays of shape (T, N, C) with T <= n_timesteps,
        - a 2-row edge_index with all indices in [0, N),
        - a boolean free_mask of shape (N,) with at least one True entry.
        """
        if not DP_TFRECORD.exists():
            self.skipTest(f"DeformingPlate tfrecord not found at {DP_TFRECORD}")

        from src.gnn.comparison_defplate.loaders import DeformingPlateLoader
        loader = DeformingPlateLoader(DP_TFRECORD, N_DP_TRAJS, N_TIMESTEPS)
        samples = loader.load_all()

        self.assertEqual(len(samples), N_DP_TRAJS)
        for sample in samples:
            _check_sample_invariants(self, sample, N_TIMESTEPS, "deforming_plate")

    def test_velocity_equals_step_displacement(self):
        """
        Verifies that for DeformingPlate, the 'velocity' signal equals
        'step_displacement' (both derived as world_pos[t+1] - world_pos[t]).
        Both must have the same shape and near-identical values.
        """
        if not DP_TFRECORD.exists():
            self.skipTest(f"DeformingPlate tfrecord not found at {DP_TFRECORD}")

        from src.gnn.comparison_defplate.loaders import DeformingPlateLoader
        loader = DeformingPlateLoader(DP_TFRECORD, 1, N_TIMESTEPS)
        samples = loader.load_all()
        sample = samples[0]
        vel = sample.signals["velocity"]
        step = sample.signals["step_displacement"]
        self.assertEqual(vel.shape, step.shape)
        np.testing.assert_allclose(vel, step, rtol=0, atol=1e-6,
                                   err_msg="velocity must equal step_displacement for DP")


class TestBiaxLongerLoader(unittest.TestCase):

    def test_loads_correct_shapes_and_edge_validity(self):
        """
        Verifies that BiaxLongerLoader.load_all() returns TrajectorySample
        objects (one per selected trajectory that has Nsteps400) with:
        - five signal arrays of shape (T, N, C) with T <= n_timesteps,
        - a 2-row edge_index with all indices in [0, N),
        - a boolean free_mask of shape (N,) with at least one True entry.
        """
        if not BIAX_TEST_DIR.exists():
            self.skipTest(f"biax_longer test graphs not found at {BIAX_TEST_DIR}")

        from src.gnn.comparison_defplate.loaders import BiaxLongerLoader
        loader = BiaxLongerLoader(BIAX_TEST_DIR, N_TIMESTEPS)
        samples = loader.load_all()

        self.assertGreater(len(samples), 0, "No samples loaded")
        for sample in samples:
            _check_sample_invariants(self, sample, N_TIMESTEPS, "biax_longer")

    def test_velocity_is_not_all_zero(self):
        """
        Verifies that the velocity signal loaded from biax_longer raw files is
        not identically zero (the raw .pt files carry pre-computed x_vel_t).
        """
        if not BIAX_TEST_DIR.exists():
            self.skipTest(f"biax_longer test graphs not found at {BIAX_TEST_DIR}")

        from src.gnn.comparison_defplate.loaders import BiaxLongerLoader
        loader = BiaxLongerLoader(BIAX_TEST_DIR, N_TIMESTEPS)
        samples = loader.load_all()
        sample = samples[0]
        vel = sample.signals["velocity"]
        # At least some velocity should be non-zero (beam is vibrating)
        self.assertTrue(np.any(np.abs(vel) > 1e-12),
                        "velocity signal is all-zero; expected pre-computed x_vel_t")


if __name__ == "__main__":
    unittest.main()
