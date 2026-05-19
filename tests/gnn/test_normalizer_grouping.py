"""
Tests for isotropic and anisotropic normalization in normalizer.py.

Informal specification:
- make_isotropic_groups_from_split produces a contiguous list of (start, length) groups where the
  first `split_idx` channels are packed as length-3 triplets and the remaining channels are
  length-1 scalars; both the boundary indices and the total coverage of every channel are tested
  exactly.
- The target group layout for von_mises mode is [(0, 3), (3, 1)] and for full_stress is
  [(0, 3), (3, 6)]; both the preprocessor helper and the TargetStrategy subclasses produce
  identical groups, so there is one source of truth.
- For isotropic normalizers (ZIsotropicNormalizer, RobustIsotropicNormalizer), all channels inside
  a length >= 2 group share exactly the same mean/std (derived from values pooled across the whole
  group), while length-1 groups keep independent per-column statistics.
- For anisotropic normalizers (ZAnisotropicZNormalizer, RobustAnisotropicNormalizer), every column
  gets its own independent mean/std regardless of the group layout, and these differ from the
  isotropic pooled values when raw column distributions are non-uniform.
- After normalization, isotropic-normalized values reflect the pooled stats; anisotropic-normalized
  values reflect per-column stats; on the test tensor these two differ.
"""

import pytest
import torch
import numpy as np
from types import SimpleNamespace

from src.gnn.preprocessing.normalizer import (
    make_isotropic_groups_from_split,
    ZIsotropicNormalizer,
    ZAnisotropicZNormalizer,
    RobustIsotropicNormalizer,
    RobustAnisotropicNormalizer,
)
from src.gnn.training.target_strategy import create_target_strategy


# Local reference that mirrors preprocessor.make_target_isotropic_groups and
# TargetStrategy.target_isotropic_groups; kept here to avoid importing preprocessor.py
# (which pulls in torch_geometric).
def _expected_target_groups(mode):
    if mode == "full_stress":
        return [(0, 3), (3, 6)]
    if mode == "von_mises":
        return [(0, 3), (3, 1)]
    raise ValueError(f"Unknown target mode: {mode}")

# Must match the EPS used in Normalizer.normalize / denormalize.
EPS = 1e-8


# ---- deterministic test tensors ----

def _feature_tensor():
    """
    4-column float64 tensor.

    Columns 0-2 form a vector group with intentionally different per-column distributions so that
    isotropic pooling produces stats that diverge from anisotropic per-column stats.  Column 3 is
    a scalar with a much larger range.

    Pooled (isotropic) stats for cols 0-2 over the 9 values [1..9]:
        mean = 45/9 = 5.0
        var  = 285/9 - 25 = 60/9  →  std = sqrt(60/9)

    Per-column (anisotropic) stats:
        col 0 [1,4,7]:  mean = 4, std = sqrt(6)
        col 1 [2,5,8]:  mean = 5, std = sqrt(6)
        col 2 [3,6,9]:  mean = 6, std = sqrt(6)

    Column 3 [100, 200, 300]:
        mean = 200,  var = 20000/3  →  std = sqrt(20000/3)
        median = 200, IQR = 100
    """
    return torch.tensor([
        [1.0, 2.0, 3.0, 100.0],
        [4.0, 5.0, 6.0, 200.0],
        [7.0, 8.0, 9.0, 300.0],
    ], dtype=torch.float64)


def _vm_target_tensor():
    """
    4-column float64 target for von_mises mode: [acc_x, acc_y, acc_z, vm].
    Identical distribution to _feature_tensor so expected stats are the same.
    """
    return torch.tensor([
        [1.0, 2.0, 3.0, 100.0],
        [4.0, 5.0, 6.0, 200.0],
        [7.0, 8.0, 9.0, 300.0],
    ], dtype=torch.float64)


def _full_stress_target_tensor():
    """
    9-column float64 target for full_stress mode: [acc_x, acc_y, acc_z, s11, s12, s13, s22, s23, s33].

    The 6 stress columns have distinct per-column distributions (multiples of 10 from 10 to 180).
    Pooled stress stats over all 18 values 10, 20, ..., 180:
        mean = 1710/18 = 95.0
    Per-column anisotropic stress means: 70, 80, 90, 100, 110, 120.
    """
    return torch.tensor([
        [1.0,  2.0,  3.0,  10.0,  20.0,  30.0,  40.0,  50.0,  60.0],
        [4.0,  5.0,  6.0,  70.0,  80.0,  90.0, 100.0, 110.0, 120.0],
        [7.0,  8.0,  9.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0],
    ], dtype=torch.float64)


# ---- group-index tests ----

def test_make_isotropic_groups_two_vector_groups_plus_scalars():
    """
    Informal spec: split_idx=6, total_dim=8 creates two length-3 groups at starts 0 and 3, then
    two length-1 groups at starts 6 and 7; the output covers all 8 channels without gaps.
    """
    assert make_isotropic_groups_from_split(6, 8) == [(0, 3), (3, 3), (6, 1), (7, 1)]


def test_make_isotropic_groups_all_scalars():
    """
    Informal spec: split_idx=0 turns every channel into its own length-1 group; the result is a
    list of individual scalars covering all total_dim channels.
    """
    assert make_isotropic_groups_from_split(0, 3) == [(0, 1), (1, 1), (2, 1)]


def test_make_isotropic_groups_all_vector():
    """
    Informal spec: split_idx == total_dim (both divisible by 3) produces only length-3 groups with
    no scalar tail; the list length equals total_dim / 3.
    """
    assert make_isotropic_groups_from_split(6, 6) == [(0, 3), (3, 3)]


def test_make_isotropic_groups_one_triplet_plus_scalars():
    """
    Informal spec: split_idx=3, total_dim=5 creates exactly one length-3 group followed by two
    length-1 scalars covering the remaining channels.
    """
    assert make_isotropic_groups_from_split(3, 5) == [(0, 3), (3, 1), (4, 1)]


# ---- target-group index tests ----

def test_target_groups_von_mises():
    """
    Informal spec: the target normalization layout for von_mises mode is [(0, 3), (3, 1)], pooling
    the three acc channels and leaving vm as an independent length-1 scalar group; both the
    expected layout and the TargetStrategy implementation agree.
    """
    cfg = SimpleNamespace(
        exp_mises=False, stress_weight=1.0,
        full_stress_loss_mode="components", full_stress_vm_weight=0.0,
    )
    assert create_target_strategy("von_mises", cfg).target_isotropic_groups() == [(0, 3), (3, 1)]


def test_target_groups_full_stress():
    """
    Informal spec: the target normalization layout for full_stress mode is [(0, 3), (3, 6)], pooling
    acc channels together and pooling all 6 stress components into one group; both the expected
    layout and the TargetStrategy implementation agree.
    """
    cfg = SimpleNamespace(
        exp_mises=False, stress_weight=1.0,
        full_stress_loss_mode="components", full_stress_vm_weight=0.0,
    )
    assert create_target_strategy("full_stress", cfg).target_isotropic_groups() == [(0, 3), (3, 6)]


def test_target_groups_strategy_matches_expected_contract():
    """
    Informal spec: target_isotropic_groups() on each TargetStrategy subclass returns the same
    layout as the local reference _expected_target_groups(); both the strategy and the canonical
    contract agree, so there is exactly one source of truth.
    """
    cfg = SimpleNamespace(
        exp_mises=False, stress_weight=1.0,
        full_stress_loss_mode="components", full_stress_vm_weight=0.0,
    )
    for mode in ("von_mises", "full_stress"):
        strategy = create_target_strategy(mode, cfg)
        assert strategy.target_isotropic_groups() == _expected_target_groups(mode)


# ---- cross-normalizer structural invariants (parametrized) ----

@pytest.mark.parametrize("normalizer_cls", [ZIsotropicNormalizer, RobustIsotropicNormalizer])
def test_isotropic_group_channels_share_identical_stats(normalizer_cls):
    """
    Informal spec: for both Z-isotropic and Robust-isotropic normalizers, all channels belonging to
    a length-3 group always carry bit-identical mean and std; a length-1 scalar channel is unaffected
    by pooling and carries different stats from the vector group.
    """
    t = _feature_tensor()
    groups = [(0, 3), (3, 1)]
    stats = normalizer_cls().compute_stats([t], groups)

    assert stats.mean[0] == stats.mean[1]
    assert stats.mean[1] == stats.mean[2]
    assert stats.std[0] == stats.std[1]
    assert stats.std[1] == stats.std[2]
    # Scalar channel has a much larger mean than the vector group.
    assert stats.mean[3] != stats.mean[0]


@pytest.mark.parametrize("normalizer_cls", [ZAnisotropicZNormalizer, RobustAnisotropicNormalizer])
def test_anisotropic_vector_channels_have_distinct_means(normalizer_cls):
    """
    Informal spec: for both Z-anisotropic and Robust-anisotropic normalizers, each column is
    summarised independently, so columns 0, 1, 2 — which have different per-column distributions —
    produce different mean values even though they belong to the same declared group.
    """
    t = _feature_tensor()
    groups = [(0, 3), (3, 1)]
    stats = normalizer_cls().compute_stats([t], groups)

    means = [float(stats.mean[i]) for i in range(3)]
    assert len(set(means)) == 3, "Expected all three vector channel means to be distinct"


# ---- Z-isotropic stats: exact value tests ----

class TestZIsotropicStats:

    def test_vector_group_pooled_mean(self):
        """
        Informal spec: the mean for a vector group equals the arithmetic mean of all 9 pooled values
        (3 channels × 3 rows), and all three channels carry the same value.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        stats = ZIsotropicNormalizer().compute_stats([t], groups)

        # Pooled 1..9: sum=45, count=9, mean=5.0
        expected = torch.tensor(5.0, dtype=torch.float64)
        for i in range(3):
            assert torch.isclose(stats.mean[i], expected)

    def test_vector_group_pooled_std(self):
        """
        Informal spec: the std for a vector group equals sqrt(pooled_var), where pooled_var is
        (pooled_sum_sq / pooled_count) - mean^2; all three channels carry the identical value.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        stats = ZIsotropicNormalizer().compute_stats([t], groups)

        # sum_sq over [1..9] = 285, count=9, mean=5; var = 285/9 - 25 = 60/9
        expected = torch.tensor(np.sqrt(60.0 / 9.0), dtype=torch.float64)
        for i in range(3):
            assert torch.isclose(stats.std[i], expected, atol=1e-10)

    def test_scalar_channel_mean(self):
        """
        Informal spec: the scalar (length-1) channel mean equals the column mean of that channel
        only; it is not influenced by the vector group pool.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        stats = ZIsotropicNormalizer().compute_stats([t], groups)

        assert torch.isclose(stats.mean[3], torch.tensor(200.0, dtype=torch.float64))

    def test_scalar_channel_std(self):
        """
        Informal spec: the scalar channel std comes from its own column values [100, 200, 300];
        var = (100^2 + 200^2 + 300^2)/3 - 200^2 = 140000/3 - 40000 = 20000/3.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        stats = ZIsotropicNormalizer().compute_stats([t], groups)

        expected = torch.tensor(np.sqrt(20000.0 / 3.0), dtype=torch.float64)
        assert torch.isclose(stats.std[3], expected, atol=1e-8)

    def test_isotropic_mean_differs_from_anisotropic(self):
        """
        Informal spec: isotropic pooling (all vector channels get mean=5.0) diverges from anisotropic
        per-column means (4.0, 5.0, 6.0); the two normalizers produce different stats on non-uniform data.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        iso_stats = ZIsotropicNormalizer().compute_stats([t], groups)
        aniso_stats = ZAnisotropicZNormalizer().compute_stats([t], groups)

        assert not torch.allclose(iso_stats.mean[:3], aniso_stats.mean[:3])
        assert torch.isclose(aniso_stats.mean[0], torch.tensor(4.0, dtype=torch.float64))
        assert torch.isclose(aniso_stats.mean[1], torch.tensor(5.0, dtype=torch.float64))
        assert torch.isclose(aniso_stats.mean[2], torch.tensor(6.0, dtype=torch.float64))


# ---- Z-isotropic target stats ----

class TestZIsotropicTargetStats:

    def test_von_mises_acc_channels_pooled(self):
        """
        Informal spec: in von_mises mode the acc group pools three channels; all three share the
        pooled mean 5.0 and the same std, distinct from the vm scalar channel.
        """
        t = _vm_target_tensor()
        groups = _expected_target_groups("von_mises")
        stats = ZIsotropicNormalizer().compute_stats([t], groups)

        expected_acc_mean = torch.tensor(5.0, dtype=torch.float64)
        for i in range(3):
            assert torch.isclose(stats.mean[i], expected_acc_mean)
        assert stats.std[0] == stats.std[1] == stats.std[2]
        # vm channel mean is independent
        assert torch.isclose(stats.mean[3], torch.tensor(200.0, dtype=torch.float64))
        assert not torch.isclose(stats.mean[0], stats.mean[3])

    def test_full_stress_stress_channels_pooled(self):
        """
        Informal spec: in full_stress mode all 6 stress channels share bit-identical mean and std
        equal to the pooled mean over all 18 stress values (10, 20, ..., 180), which is 95.0.
        """
        t = _full_stress_target_tensor()
        groups = _expected_target_groups("full_stress")
        stats = ZIsotropicNormalizer().compute_stats([t], groups)

        stress_means = [float(stats.mean[i]) for i in range(3, 9)]
        stress_stds = [float(stats.std[i]) for i in range(3, 9)]
        assert len(set(stress_means)) == 1, "All stress channel means must be identical under isotropic"
        assert len(set(stress_stds)) == 1, "All stress channel stds must be identical under isotropic"

        # Pooled mean: sum(10, 20, ..., 180) / 18 = 1710 / 18 = 95.0
        assert torch.isclose(stats.mean[3], torch.tensor(95.0, dtype=torch.float64))

    def test_full_stress_stress_channels_differ_under_anisotropic(self):
        """
        Informal spec: Z-anisotropic stats give distinct per-column means to the stress channels
        (70, 80, ..., 120), confirming that isotropic pooling genuinely merges their statistics.
        """
        t = _full_stress_target_tensor()
        groups = _expected_target_groups("full_stress")
        iso_stats = ZIsotropicNormalizer().compute_stats([t], groups)
        aniso_stats = ZAnisotropicZNormalizer().compute_stats([t], groups)

        iso_stress_means = [float(iso_stats.mean[i]) for i in range(3, 9)]
        assert len(set(iso_stress_means)) == 1

        aniso_stress_means = [float(aniso_stats.mean[i]) for i in range(3, 9)]
        assert len(set(aniso_stress_means)) == 6, "Every anisotropic stress channel must have a distinct mean"


# ---- Robust isotropic stats: exact value tests ----

class TestRobustIsotropicStats:

    def test_vector_group_pooled_median(self):
        """
        Informal spec: the median for a vector group is computed on all 9 pooled values; for [1..9]
        the median is 5.0 and all three channels receive this identical value.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        stats = RobustIsotropicNormalizer().compute_stats([t], groups)

        assert stats.mean[0] == stats.mean[1] == stats.mean[2]
        assert torch.isclose(stats.mean[0], torch.tensor(5.0, dtype=torch.float64))

    def test_vector_group_pooled_iqr(self):
        """
        Informal spec: the IQR for a vector group is derived from the pooled distribution; for [1..9]
        numpy gives q1=3.0, q3=7.0, IQR=4.0; all three channels carry this identical value.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        stats = RobustIsotropicNormalizer().compute_stats([t], groups)

        pooled = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        expected_iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
        assert stats.std[0] == stats.std[1] == stats.std[2]
        assert torch.isclose(stats.std[0], torch.tensor(expected_iqr, dtype=torch.float64))

    def test_scalar_channel_own_median_and_iqr(self):
        """
        Informal spec: a scalar (length-1) channel is not pooled with the vector group; its median
        and IQR come from column 3 only — median=200.0, IQR=100.0.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        stats = RobustIsotropicNormalizer().compute_stats([t], groups)

        assert torch.isclose(stats.mean[3], torch.tensor(200.0, dtype=torch.float64))
        assert torch.isclose(stats.std[3], torch.tensor(100.0, dtype=torch.float64))

    def test_isotropic_median_differs_from_anisotropic(self):
        """
        Informal spec: isotropic pooled medians (5.0 for all vector channels) differ from the
        anisotropic per-column medians (4.0, 5.0, 6.0), confirming the two variants diverge.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        iso_stats = RobustIsotropicNormalizer().compute_stats([t], groups)
        aniso_stats = RobustAnisotropicNormalizer().compute_stats([t], groups)

        assert not torch.allclose(iso_stats.mean[:3], aniso_stats.mean[:3])
        assert torch.isclose(aniso_stats.mean[0], torch.tensor(4.0, dtype=torch.float64))
        assert torch.isclose(aniso_stats.mean[1], torch.tensor(5.0, dtype=torch.float64))
        assert torch.isclose(aniso_stats.mean[2], torch.tensor(6.0, dtype=torch.float64))


# ---- Robust isotropic target stats ----

class TestRobustIsotropicTargetStats:

    def test_von_mises_acc_channels_share_robust_stats(self):
        """
        Informal spec: RobustIsotropicNormalizer in von_mises mode pools acc channels; all three
        carry the same median and IQR, and the vm scalar channel has its own distinct stats.
        """
        t = _vm_target_tensor()
        groups = _expected_target_groups("von_mises")
        stats = RobustIsotropicNormalizer().compute_stats([t], groups)

        assert stats.mean[0] == stats.mean[1] == stats.mean[2]
        assert stats.std[0] == stats.std[1] == stats.std[2]
        assert not torch.isclose(stats.mean[0], stats.mean[3])

    def test_full_stress_stress_channels_share_robust_stats(self):
        """
        Informal spec: in full_stress mode RobustIsotropicNormalizer pools all 6 stress channels;
        every stress channel carries the same pooled median and IQR.
        """
        t = _full_stress_target_tensor()
        groups = _expected_target_groups("full_stress")
        stats = RobustIsotropicNormalizer().compute_stats([t], groups)

        stress_medians = [float(stats.mean[i]) for i in range(3, 9)]
        stress_iqrs = [float(stats.std[i]) for i in range(3, 9)]
        assert len(set(stress_medians)) == 1, "All stress channel medians must be identical under isotropic"
        assert len(set(stress_iqrs)) == 1, "All stress channel IQRs must be identical under isotropic"

    def test_full_stress_stress_channels_differ_under_anisotropic_robust(self):
        """
        Informal spec: RobustAnisotropicNormalizer assigns distinct per-column medians to stress
        channels; the isotropic pooled median genuinely differs from the anisotropic per-column medians.
        """
        t = _full_stress_target_tensor()
        groups = _expected_target_groups("full_stress")
        iso_stats = RobustIsotropicNormalizer().compute_stats([t], groups)
        aniso_stats = RobustAnisotropicNormalizer().compute_stats([t], groups)

        iso_stress_medians = [float(iso_stats.mean[i]) for i in range(3, 9)]
        assert len(set(iso_stress_medians)) == 1

        aniso_stress_medians = [float(aniso_stats.mean[i]) for i in range(3, 9)]
        assert len(set(aniso_stress_medians)) == 6


# ---- normalized output value tests ----

class TestNormalizationOutputValues:

    def test_z_isotropic_row_values_use_pooled_stats(self):
        """
        Informal spec: after normalizing with isotropic stats, row 0 of the vector group is shifted
        by the pooled mean 5.0 and scaled by the pooled std; the expected formula matches
        (raw - 5.0) / (pooled_std + EPS) for each channel in the group.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        normalizer = ZIsotropicNormalizer()
        stats = normalizer.compute_stats([t], groups)
        normed = normalizer.normalize(t, stats)

        pooled_std = float(np.sqrt(60.0 / 9.0))
        for col, raw in enumerate([1.0, 2.0, 3.0]):
            expected = (raw - 5.0) / (pooled_std + EPS)
            assert torch.isclose(normed[0, col], torch.tensor(expected, dtype=torch.float64), atol=1e-8)

    def test_z_isotropic_scalar_channel_uses_own_stats(self):
        """
        Informal spec: the scalar channel (col 3) is normalized by its own mean/std, not the vector
        group pooled stats; its first-row normalized value matches (100 - 200) / (scalar_std + EPS).
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        normalizer = ZIsotropicNormalizer()
        stats = normalizer.compute_stats([t], groups)
        normed = normalizer.normalize(t, stats)

        scalar_std = float(np.sqrt(20000.0 / 3.0))
        expected = (100.0 - 200.0) / (scalar_std + EPS)
        assert torch.isclose(normed[0, 3], torch.tensor(expected, dtype=torch.float64), atol=1e-8)

    def test_z_anisotropic_each_column_uses_own_stats(self):
        """
        Informal spec: Z-anisotropic normalization centers each channel on its own per-column mean;
        for row 0, channel 0 is centered on 4.0, channel 1 on 5.0, channel 2 on 6.0, each divided
        by sqrt(6) (the common per-column std) plus EPS.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        normalizer = ZAnisotropicZNormalizer()
        stats = normalizer.compute_stats([t], groups)
        normed = normalizer.normalize(t, stats)

        col_std = float(np.sqrt(6.0))
        expected_means = [4.0, 5.0, 6.0]
        for col, (raw, mean) in enumerate(zip([1.0, 2.0, 3.0], expected_means)):
            expected = (raw - mean) / (col_std + EPS)
            assert torch.isclose(normed[0, col], torch.tensor(expected, dtype=torch.float64), atol=1e-8)

    def test_z_isotropic_and_anisotropic_produce_different_outputs(self):
        """
        Informal spec: on the feature tensor where vector columns have different per-column means,
        Z-isotropic and Z-anisotropic normalization produce different output tensors for the vector
        group; the scalar channel output may coincide only by coincidence but the vector group must differ.
        """
        t = _feature_tensor()
        groups = [(0, 3), (3, 1)]
        iso_normalizer = ZIsotropicNormalizer()
        aniso_normalizer = ZAnisotropicZNormalizer()
        iso_normed = iso_normalizer.normalize(t, iso_normalizer.compute_stats([t], groups))
        aniso_normed = aniso_normalizer.normalize(t, aniso_normalizer.compute_stats([t], groups))

        assert not torch.allclose(iso_normed[:, :3], aniso_normed[:, :3])
