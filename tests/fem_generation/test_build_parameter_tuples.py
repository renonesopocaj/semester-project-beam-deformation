"""
test_build_parameter_tuples.py

Unittests for split semantics in build_parameter_tuples (module pydocs) via build_all_splits.
Covers forces and geometry (lengths, widths, depths): val in-range off-grid, test OOD buckets,
per-axis OOD presence when configured, and constant train axes (excluding the injected (0,0,0) test row for forces).
"""

import unittest
from itertools import product

from src.fem_generation.build_parameter_tuples import (
    DECIMALS,
    TOL,
    _build_zero_force_test_combo,
    _count_ood_features,
    _in_range,
    _is_constant_range,
    _list_ood_feature_names,
    _split_target_counts,
    _train_axis_values,
    _train_count,
    _train_ranges,
    build_all_splits,
)


def _feature_from_combo(combo, train_axis_key):
    """
    Return the scalar value for one train config axis key on a combo tuple.

    Args:
        combo: tuple
            ((fx, fy, fz), L, W, D)
        train_axis_key: str
            one of force_x, force_y, force_z, lengths, widths, depths
    return: float
        feature value
    """
    (fx, fy, fz), L, W, D = combo
    if train_axis_key == "force_x":
        return float(fx)
    if train_axis_key == "force_y":
        return float(fy)
    if train_axis_key == "force_z":
        return float(fz)
    if train_axis_key == "lengths":
        return float(L)
    if train_axis_key == "widths":
        return float(W)
    if train_axis_key == "depths":
        return float(D)
    raise ValueError("unknown train_axis_key: %r" % (train_axis_key,))


def _combo_rounded_key(combo):
    """
    Map a parameter combo to a tuple of rounded floats for stable equality checks.

    Args:
        combo: tuple
            ((fx, fy, fz), L, W, D)
    return: tuple
        rounded components
    """
    (fx, fy, fz), L, W, D = combo
    return (
        round(float(fx), DECIMALS),
        round(float(fy), DECIMALS),
        round(float(fz), DECIMALS),
        round(float(L), DECIMALS),
        round(float(W), DECIMALS),
        round(float(D), DECIMALS),
    )


def _cfg_fy_only():
    """
    In-memory config shaped like conf/fem_generation YAMLs: only force_y varies; only force_y has OOD ranges.

    return: dict
        cfg dict for build_all_splits
    """
    return {
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "train": {
            "force_x": {"start": 0.0, "stop": 0.0, "n": 1},
            "force_y": {"start": -1.25, "stop": 1.25, "n": 16},
            "force_z": {"start": 0.0, "stop": 0.0, "n": 1},
            "lengths": {"start": 0.7, "stop": 0.7, "n": 1},
            "widths": {"start": 0.05, "stop": 0.05, "n": 1},
            "depths": {"start": 0.05, "stop": 0.05, "n": 1},
        },
        "test_ood_ranges": {
            "force_x": {},
            "force_y": {
                "below": {"start": -1.7, "stop": -1.3},
                "above": {"start": 1.3, "stop": 1.7},
            },
            "force_z": {},
            "lengths": {},
            "widths": {},
            "depths": {},
        },
    }


def _cfg_fy_fz():
    """
    In-memory config shaped like conf/fem_generation YAMLs: force_y and force_z vary; OOD on both.

    return: dict
        cfg dict for build_all_splits
    """
    return {
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "train": {
            "force_x": {"start": 0.0, "stop": 0.0, "n": 1},
            "force_y": {"start": -0.5, "stop": 0.5, "n": 4},
            "force_z": {"start": -0.5, "stop": 0.5, "n": 4},
            "lengths": {"start": 0.7, "stop": 0.7, "n": 1},
            "widths": {"start": 0.05, "stop": 0.05, "n": 1},
            "depths": {"start": 0.05, "stop": 0.05, "n": 1},
        },
        "test_ood_ranges": {
            "force_x": {},
            "force_y": {
                "below": {"start": -1.0, "stop": -0.6},
                "above": {"start": 0.6, "stop": 1.0},
            },
            "force_z": {
                "below": {"start": -1.0, "stop": -0.6},
                "above": {"start": 0.6, "stop": 1.0},
            },
            "lengths": {},
            "widths": {},
            "depths": {},
        },
    }


def _cfg_geom_only_ood():
    """
    Forces constant (non-zero); lengths, widths, depths vary; OOD only on geometry axes.

    return: dict
        cfg dict for build_all_splits
    """
    return {
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "train": {
            "force_x": {"start": 0.0, "stop": 0.0, "n": 1},
            "force_y": {"start": 1.0, "stop": 1.0, "n": 1},
            "force_z": {"start": 0.0, "stop": 0.0, "n": 1},
            "lengths": {"start": 0.6, "stop": 0.8, "n": 3},
            "widths": {"start": 0.04, "stop": 0.06, "n": 3},
            "depths": {"start": 0.08, "stop": 0.12, "n": 3},
        },
        "test_ood_ranges": {
            "force_x": {},
            "force_y": {},
            "force_z": {},
            "lengths": {
                "below": {"start": 0.35, "stop": 0.55},
                "above": {"start": 0.85, "stop": 1.0},
            },
            "widths": {
                "below": {"start": 0.02, "stop": 0.035},
                "above": {"start": 0.065, "stop": 0.08},
            },
            "depths": {
                "below": {"start": 0.04, "stop": 0.065},
                "above": {"start": 0.13, "stop": 0.16},
            },
        },
    }


def _cfg_all_axes():
    """
    Synthetic in-memory config: all six split-defining axes vary with disjoint OOD below/above ranges.

    return: dict
        cfg dict for build_all_splits
    """
    return {
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "train": {
            "force_x": {"start": -0.1, "stop": 0.1, "n": 2},
            "force_y": {"start": -0.5, "stop": 0.5, "n": 2},
            "force_z": {"start": -0.5, "stop": 0.5, "n": 2},
            "lengths": {"start": 0.7, "stop": 0.9, "n": 2},
            "widths": {"start": 0.04, "stop": 0.06, "n": 2},
            "depths": {"start": 0.08, "stop": 0.12, "n": 2},
        },
        "test_ood_ranges": {
            "force_x": {
                "below": {"start": -0.45, "stop": -0.15},
                "above": {"start": 0.15, "stop": 0.45},
            },
            "force_y": {
                "below": {"start": -0.9, "stop": -0.55},
                "above": {"start": 0.55, "stop": 0.9},
            },
            "force_z": {
                "below": {"start": -0.9, "stop": -0.55},
                "above": {"start": 0.55, "stop": 0.9},
            },
            "lengths": {
                "below": {"start": 0.45, "stop": 0.65},
                "above": {"start": 0.95, "stop": 1.1},
            },
            "widths": {
                "below": {"start": 0.02, "stop": 0.035},
                "above": {"start": 0.065, "stop": 0.08},
            },
            "depths": {
                "below": {"start": 0.04, "stop": 0.065},
                "above": {"start": 0.13, "stop": 0.16},
            },
        },
    }


def _max_ood_feature_count(cfg):
    """
    Upper bound on per-sample OOD feature count given which axes declare nonempty test_ood_ranges.

    Args:
        cfg: dict
            config with test_ood_ranges
    return: int
        min(6, number of axes with nonempty OOD config)
    """
    ood = cfg["test_ood_ranges"]
    axis_names = ["force_x", "force_y", "force_z", "lengths", "widths", "depths"]
    n_nonempty = 0
    for axis_name in axis_names:
        if ood.get(axis_name):
            n_nonempty += 1
    return min(6, n_nonempty)


class TestBuildParameterTuplesPydocs(unittest.TestCase):
    """
    Assert train / val / test semantics from build_parameter_tuples module pydocs.
    """

    def _assert_splits_respect_pydocs(self, cfg, splits):
        """
        Run all pydoc checks for one cfg.

        Args:
            cfg: dict
                in-memory fem_generation-style config
            splits: dict
                output of build_all_splits(cfg)
        """
        n_train, n_val, n_test = _split_target_counts(cfg)
        train_ranges = _train_ranges(cfg)
        self.assertEqual(len(splits["train"]), n_train)
        self.assertEqual(len(splits["val"]), n_val)
        self.assertEqual(len(splits["test"]), n_test + 1)

        fx, fy, fz, lengths, widths, depths = _train_axis_values(cfg)
        forces = list(product(fx, fy, fz))
        expected_train = list(product(forces, lengths, widths, depths))
        self.assertEqual(len(expected_train), n_train)
        got_train_keys = {_combo_rounded_key(c) for c in splits["train"]}
        expected_train_keys = {_combo_rounded_key(c) for c in expected_train}
        self.assertEqual(got_train_keys, expected_train_keys)

        train_key_set = {_combo_rounded_key(c) for c in splits["train"]}
        for combo in splits["val"]:
            (fxv, fyv, fzv), Lv, Wv, Dv = combo
            self.assertTrue(_in_range(fxv, train_ranges["fx"][0], train_ranges["fx"][1], TOL))
            self.assertTrue(_in_range(fyv, train_ranges["fy"][0], train_ranges["fy"][1], TOL))
            self.assertTrue(_in_range(fzv, train_ranges["fz"][0], train_ranges["fz"][1], TOL))
            self.assertTrue(_in_range(Lv, train_ranges["L"][0], train_ranges["L"][1], TOL))
            self.assertTrue(_in_range(Wv, train_ranges["W"][0], train_ranges["W"][1], TOL))
            self.assertTrue(_in_range(Dv, train_ranges["D"][0], train_ranges["D"][1], TOL))
            self.assertNotIn(_combo_rounded_key(combo), train_key_set)

        algo_test = splits["test"][:-1]
        for combo in algo_test:
            self.assertGreaterEqual(_count_ood_features(combo, train_ranges), 1)

        ood_counts = {_count_ood_features(c, train_ranges) for c in algo_test}
        max_k = _max_ood_feature_count(cfg)
        self.assertEqual(ood_counts, set(range(1, max_k + 1)))

        zero_expected = _build_zero_force_test_combo(cfg)
        self.assertEqual(splits["test"][-1], zero_expected)

        axis_names = ["force_x", "force_y", "force_z", "lengths", "widths", "depths"]
        for axis_name in axis_names:
            if not cfg["test_ood_ranges"].get(axis_name):
                continue
            found = False
            for combo in algo_test:
                if axis_name in _list_ood_feature_names(combo, train_ranges):
                    found = True
                    break
            self.assertTrue(
                found,
                "expected at least one algorithmic test sample with OOD on %r" % (axis_name,),
            )

    def test_pydocs_fy_only(self):
        """
        Single varying force axis with OOD on that axis only (typical easiest-single style).
        """
        cfg = _cfg_fy_only()
        splits = build_all_splits(cfg)
        self._assert_splits_respect_pydocs(cfg, splits)

    def test_pydocs_fy_fz(self):
        """
        Two varying force axes with OOD on both (typical rotated / fixedgeom style).
        """
        cfg = _cfg_fy_fz()
        splits = build_all_splits(cfg)
        self._assert_splits_respect_pydocs(cfg, splits)

    def test_pydocs_all_axes(self):
        """
        All six axes vary with OOD on every axis (covers exact-k buckets up to k=6).
        """
        cfg = _cfg_all_axes()
        splits = build_all_splits(cfg)
        self._assert_splits_respect_pydocs(cfg, splits)

    def test_pydocs_geom_only_ood(self):
        """
        OOD only on lengths, widths, depths (no force OOD); validates geometry part of the spec.
        """
        cfg = _cfg_geom_only_ood()
        splits = build_all_splits(cfg)
        self._assert_splits_respect_pydocs(cfg, splits)

    def test_train_count_helper_matches_product(self):
        """
        _train_count matches Cartesian product size for the all-axes fixture.
        """
        cfg = _cfg_all_axes()
        self.assertEqual(_train_count(cfg), 64)


class TestConstantTrainAxes(unittest.TestCase):
    """
    Train axes with start=stop (constant) must keep that value in every train/val/test tuple.
    """

    def _assert_constant_axes_match_config(self, cfg, splits):
        """
        For each train axis with a constant range, every combo in every split uses that value.

        Args:
            cfg: dict
                config with train section
            splits: dict
                train, val, test lists from build_all_splits
        """
        train = cfg["train"]
        axis_names = ["force_x", "force_y", "force_z", "lengths", "widths", "depths"]
        for axis_name in axis_names:
            axis_cfg = train[axis_name]
            if not _is_constant_range(axis_cfg["start"], axis_cfg["stop"]):
                continue
            expected = float(axis_cfg["start"])
            for split_name in ("train", "val", "test"):
                combos = splits[split_name]
                if split_name == "test" and axis_name in (
                    "force_x",
                    "force_y",
                    "force_z",
                ):
                    combos = combos[:-1]
                for combo in combos:
                    actual = _feature_from_combo(combo, axis_name)
                    self.assertAlmostEqual(
                        actual,
                        expected,
                        places=DECIMALS - 2,
                        msg="%s split combo has wrong %s: %r vs %r"
                        % (split_name, axis_name, actual, expected),
                    )

    def test_constant_axes_fixed_fy_only_fixture(self):
        """
        Typical YAML-style cfg: fixed L, W, D and fx, fz while fy varies; those constants never drift.
        """
        cfg = _cfg_fy_only()
        splits = build_all_splits(cfg)
        self._assert_constant_axes_match_config(cfg, splits)

    def test_constant_forces_fixed_geom_ood_fixture(self):
        """
        Geometry-only OOD fixture: fixed (0,1,0) force triple must appear on every tuple.
        """
        cfg = _cfg_geom_only_ood()
        splits = build_all_splits(cfg)
        self._assert_constant_axes_match_config(cfg, splits)

    def test_constant_width_depth_while_length_varies(self):
        """
        Lengths vary; width and depth start=stop with n=1 stay fixed across train, val, and test.
        """
        cfg = {
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "train": {
                "force_x": {"start": 0.0, "stop": 0.0, "n": 1},
                "force_y": {"start": 0.5, "stop": 0.5, "n": 1},
                "force_z": {"start": 0.0, "stop": 0.0, "n": 1},
                "lengths": {"start": 0.65, "stop": 0.85, "n": 4},
                "widths": {"start": 0.03, "stop": 0.03, "n": 1},
                "depths": {"start": 0.09, "stop": 0.09, "n": 1},
            },
            "test_ood_ranges": {
                "force_x": {},
                "force_y": {},
                "force_z": {},
                "lengths": {
                    "below": {"start": 0.4, "stop": 0.55},
                    "above": {"start": 0.9, "stop": 1.0},
                },
                "widths": {},
                "depths": {},
            },
        }
        splits = build_all_splits(cfg)
        self._assert_constant_axes_match_config(cfg, splits)
        for combo in splits["train"] + splits["val"] + splits["test"]:
            self.assertAlmostEqual(_feature_from_combo(combo, "widths"), 0.03, places=8)
            self.assertAlmostEqual(_feature_from_combo(combo, "depths"), 0.09, places=8)


if __name__ == "__main__":
    unittest.main()
