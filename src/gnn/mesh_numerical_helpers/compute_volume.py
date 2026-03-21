import torch

def volume(V, T):
    """
    Calculate volumes of each tetrahedra. Used in plot_volume_change. Does it by
    1. Computes four vertex as vectors of a tetrahedron  a,b,c,d
    2. Computes volume by doing vol = -1/6 (a-d) dot ((b-c) x (c-d))

    Args:
        V: torch.Tensor
            Vertex positions (N, 3)
        T: torch.Tensor
            Tetrahedra connectivity (M, 4)

    Returns: torch.Tensor
        Volumes for each tetrahedron (M,)
    """
    a = V[T[:, 0], :]
    b = V[T[:, 1], :]
    c = V[T[:, 2], :]
    d = V[T[:, 3], :]
    # vol = -1/6 (a-d) dot ((b-c) x (c-d))
    vols = -torch.sum(torch.mul(a - d, torch.cross(b - c, c - d, dim=1)), dim=1) / 6.0
    return vols