import torch.nn as nn

def build_mlp_d(in_size, hidden_size, out_size, num_layers=1, lay_norm=True, use_sigmoid=False, use_softmax=False):
    """
    Build a multi-layer perceptron with optional layer normalization.

    Args:
        in_size: int
        hidden_size: int
        out_size: int
        num_layers: int
        lay_norm: bool
        use_sigmoid: bool
        use_softmax: bool
    """
    if use_sigmoid and use_softmax:
        raise ValueError("Only one of use_sigmoid or use_softmax can be true.")

    layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]

    # Add intermediate layers
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

    # Add the output layer
    layers.append(nn.Linear(hidden_size, out_size))

    module = nn.Sequential(*layers)

    # Optionally add layer normalization
    if lay_norm:
        module = nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))

    if use_sigmoid:
        module = nn.Sequential(module, nn.Sigmoid())

    if use_softmax:
        module = nn.Sequential(module, nn.Softmax(dim=-1))

    return module