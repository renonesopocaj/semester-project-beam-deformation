"""
Tests that online edge augmentation seed folding is deterministic for fixed integer inputs.

This checks the informal contract used for reproducible validation/test/rollout augmentation,
not the full PyG batch pipeline.
"""

from src.gnn.training.online_edge_augmentation import fold_online_seed


def test_fold_online_seed_is_deterministic_for_fixed_parts():
    """
    Informal specification: the same sequence of integer parts must always yield the same folded seed,
    so eval-time augmentation can be reproduced from (base_seed, split tags, indices).
    """
    a = fold_online_seed(42, 1, 2, 3, 4)
    b = fold_online_seed(42, 1, 2, 3, 4)
    assert a == b
    assert isinstance(a, int)
    assert 0 <= a <= 2**31 - 1


def test_fold_online_seed_changes_when_parts_change():
    """
    Informal specification: changing any folded part should change the output seed with high probability,
    so different batches/trajectories do not share the same RNG stream by accident.
    """
    base = fold_online_seed(42, 1, 2, 3, 4)
    changed = fold_online_seed(42, 1, 2, 3, 5)
    assert changed != base
