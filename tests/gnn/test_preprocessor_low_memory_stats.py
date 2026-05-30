"""
Tests for the low-memory preprocessing stats accumulator.

Informal specification:
- Streaming z-stat accumulation must match the existing in-memory Z normalizers exactly up to
  floating-point tolerance.
- Isotropic mode pools configured vector groups, while anisotropic mode keeps columns independent.
"""

import torch

from src.gnn.preprocessing.normalizer import ZAnisotropicZNormalizer, ZIsotropicNormalizer
from src.gnn.preprocessing.preprocessor import RunningZStatsAccumulator, update_accumulator


def _tensor_chunks():
    """
    Args:
        return: list[torch.Tensor]
            Deterministic chunks with non-uniform vector-channel distributions.
    """
    return [
        torch.tensor([[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]], dtype=torch.float64),
        torch.tensor([[7.0, 8.0, 9.0, 30.0]], dtype=torch.float64),
    ]


def test_streamed_z_isotropic_stats_match_in_memory_normalizer():
    """
    Informal spec: the low-memory accumulator with group pooling produces the same mean/std as
    ZIsotropicNormalizer.compute_stats on the same tensor chunks.
    """
    chunks = _tensor_chunks()
    groups = [(0, 3), (3, 1)]
    accumulator = update_accumulator(None, chunks)
    streamed = accumulator.to_stats(groups, True)
    expected = ZIsotropicNormalizer().compute_stats(chunks, groups)

    assert torch.allclose(streamed.mean, expected.mean)
    assert torch.allclose(streamed.std, expected.std)


def test_streamed_z_anisotropic_stats_match_in_memory_normalizer():
    """
    Informal spec: the low-memory accumulator without group pooling produces the same mean/std as
    ZAnisotropicZNormalizer.compute_stats on the same tensor chunks.
    """
    chunks = _tensor_chunks()
    groups = [(0, 3), (3, 1)]
    accumulator = RunningZStatsAccumulator(chunks[0])
    for tensor in chunks:
        accumulator.update(tensor)
    streamed = accumulator.to_stats(groups, False)
    expected = ZAnisotropicZNormalizer().compute_stats(chunks, groups)

    assert torch.allclose(streamed.mean, expected.mean)
    assert torch.allclose(streamed.std, expected.std)
