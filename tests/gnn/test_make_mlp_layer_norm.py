"""
test_make_mlp_layer_norm.py

Checks that make_mlp inserts LayerNorm at the correct positions when mlp_layer_norm=True,
and produces no LayerNorm at all when mlp_layer_norm=False. Specifically:
- For a 2-layer MLP with mlp_layer_norm=True: the sequence is Linear, LayerNorm, Act, Linear.
- For a 3-layer MLP with mlp_layer_norm=True: Linear, LayerNorm, Act, Linear, LayerNorm, Act, Linear.
- For any n_layers with mlp_layer_norm=False: no LayerNorm appears, only Linear and Act layers.
- final_activation=True appends an extra Act after the last Linear (regardless of layer norm flag).
"""

import torch.nn as nn
import pytest
from src.gnn.models.clofnet.clof_helpers import make_mlp


def relu_factory():
    return nn.ReLU()


def _module_types(mlp):
    return [type(m) for m in mlp]


def test_two_layer_norm_on():
    mlp = make_mlp(2, 8, 16, 4, relu_factory, False, True)
    types = _module_types(mlp)
    assert types == [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear]


def test_three_layer_norm_on():
    mlp = make_mlp(3, 8, 16, 4, relu_factory, False, True)
    types = _module_types(mlp)
    assert types == [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear]


def test_two_layer_norm_off():
    mlp = make_mlp(2, 8, 16, 4, relu_factory, False, False)
    types = _module_types(mlp)
    assert nn.LayerNorm not in types
    assert types == [nn.Linear, nn.ReLU, nn.Linear]


def test_three_layer_norm_off():
    mlp = make_mlp(3, 8, 16, 4, relu_factory, False, False)
    types = _module_types(mlp)
    assert nn.LayerNorm not in types
    assert types == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]


def test_final_activation_norm_on():
    mlp = make_mlp(2, 8, 16, 4, relu_factory, True, True)
    types = _module_types(mlp)
    assert types == [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear, nn.ReLU]


def test_final_activation_norm_off():
    mlp = make_mlp(2, 8, 16, 4, relu_factory, True, False)
    types = _module_types(mlp)
    assert types == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU]


def test_output_layer_never_has_layer_norm():
    """LayerNorm must never appear right before or at the last Linear layer."""
    for norm in [True, False]:
        for n in [2, 3]:
            mlp = make_mlp(n, 8, 16, 4, relu_factory, False, norm)
            types = _module_types(mlp)
            assert types[-1] == nn.Linear
            if len(types) >= 2:
                assert types[-2] != nn.LayerNorm
