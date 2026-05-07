"""
edge_augmentation.py

Offline edge-index augmentation for preprocessing (topology changes only).
Strategy-style API: EdgeAugmenter subclasses compose a pipeline built by the factory.
"""

import math
from abc import ABC, abstractmethod

import torch


def edge_augmentation_traj_seed(base_seed, traj_file_idx):
    """
    Args:
        base_seed: int
            preprocess seed
        traj_file_idx: int
            trajectory file index
    return: int
        deterministic combined seed for RNG
    """
    return int(base_seed) + int(traj_file_idx) * 100003


class AugmentationContext:
    """
    Mutable state passed through an augmentation pipeline.

    Attributes:
        edge_index: torch.Tensor
            directed COO (2, E), updated in place by strategies
        num_nodes: int
        rest_pos: torch.Tensor
            shape (N, 3)
        boundary_condition: torch.Tensor
            x_bc wall mask
        traj_seed: int
            per-trajectory RNG seed
        device: torch.device
    """

    edge_index: torch.Tensor
    num_nodes: int
    rest_pos: torch.Tensor
    boundary_condition: torch.Tensor
    traj_seed: int
    device: torch.device

    def __init__(self, edge_index, num_nodes, rest_pos, boundary_condition, traj_seed, device):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.rest_pos = rest_pos
        self.boundary_condition = boundary_condition
        self.traj_seed = traj_seed
        self.device = device


class EdgeAugmenter(ABC):
    """
    Abstract edge augmentation step: mutates AugmentationContext.edge_index.
    """

    @abstractmethod
    def augment(self, ctx):
        """
        Args:
            ctx: AugmentationContext
        return: None
            updates ctx.edge_index in place
        """
        raise NotImplementedError("augment")


class CompositeEdgeAugmenter(EdgeAugmenter):
    """
    Runs a sequence of EdgeAugmenter steps in order.
    """

    _steps: list

    def __init__(self, steps):
        """
        Args:
            steps: list
                ordered list of EdgeAugmenter instances
        """
        self._steps = steps

    def augment(self, ctx):
        for step in self._steps:
            step.augment(ctx)


class FullyConnectedAugmenter(EdgeAugmenter):
    """
    Replace topology with a directed clique (all ordered pairs i != j).
    """

    def augment(self, ctx):
        ctx.edge_index = _fully_connected_directed(ctx.num_nodes, ctx.device)


class KHopAugmenter(EdgeAugmenter):
    """
    Merge in directed edges for pairs within k hops on the mesh (undirected geodesic distance).
    """

    _k_hops: int

    def __init__(self, k_hops):
        """
        Args:
            k_hops: int
                positive hop count
        """
        self._k_hops = k_hops

    def augment(self, ctx):
        ctx.edge_index = _k_hop_edges(ctx.edge_index, ctx.num_nodes, self._k_hops, ctx.device)


class RandomOfflineAugmenter(EdgeAugmenter):
    """
    Per-node random extra neighbors: M = round(ratio * N) per node.
    """

    _ratio: float

    def __init__(self, ratio):
        """
        Args:
            ratio: float
                in [0, 1]
        """
        self._ratio = ratio

    def augment(self, ctx):
        extra = _random_offline_edges(ctx.num_nodes, self._ratio, ctx.traj_seed, ctx.device)
        ctx.edge_index = _merge_directed_edges(ctx.edge_index, extra[0].tolist(), extra[1].tolist(), ctx.device)


class WallAllAugmenter(EdgeAugmenter):
    """
    Add directed clique on all wall (clamped) nodes.
    """

    def augment(self, ctx):
        w = _wall_edges_all(ctx.rest_pos, ctx.boundary_condition, ctx.device)
        ctx.edge_index = _merge_directed_edges(ctx.edge_index, w[0].tolist(), w[1].tolist(), ctx.device)


class WallPseudoRandomAugmenter(EdgeAugmenter):
    """
    Add directed clique on corners plus sampled border/internal wall nodes.
    """

    _wall_border_percentage: float
    _wall_internal_percentage: float

    def __init__(self, wall_border_percentage, wall_internal_percentage):
        """
        Args:
            wall_border_percentage: float
            wall_internal_percentage: float
        """
        self._wall_border_percentage = wall_border_percentage
        self._wall_internal_percentage = wall_internal_percentage

    def augment(self, ctx):
        w = _wall_edges_pseudo_random(ctx.rest_pos, ctx.boundary_condition, self._wall_border_percentage,
                                      self._wall_internal_percentage, ctx.traj_seed, ctx.device)
        ctx.edge_index = _merge_directed_edges(ctx.edge_index, w[0].tolist(), w[1].tolist(), ctx.device)


def create_edge_augmenter_from_config(k_hop, random_offline, add_wall_edges, wall_border_percentage,
                                        wall_internal_percentage):
    """
    Factory: builds the CompositeEdgeAugmenter pipeline from preprocess flags.

    Args:
        k_hop: int or str
        random_offline: float
        add_wall_edges: str or None
        wall_border_percentage: float
        wall_internal_percentage: float
    return: EdgeAugmenter
        composite or single-step augmenter (still EdgeAugmenter API)
    """
    validate_edge_augmentation_config(k_hop, random_offline, add_wall_edges, wall_border_percentage,
                                      wall_internal_percentage)
    is_fc = k_hop == "fully connected" or k_hop == "fully_connected"
    if is_fc:
        return CompositeEdgeAugmenter([FullyConnectedAugmenter()])
    steps = []
    if isinstance(k_hop, int) and k_hop > 0:
        steps.append(KHopAugmenter(k_hop))
    if random_offline > 0.0:
        steps.append(RandomOfflineAugmenter(random_offline))
    if add_wall_edges == "all":
        steps.append(WallAllAugmenter())
    elif add_wall_edges == "pseudo_random":
        steps.append(WallPseudoRandomAugmenter(wall_border_percentage, wall_internal_percentage))
    return CompositeEdgeAugmenter(steps)


def _num_nodes_from_inputs(edge_index, rest_pos):
    """
    Args:
        edge_index: torch.Tensor
            shape (2, E), long
        rest_pos: torch.Tensor
            shape (N, 3)
    return: int
        number of nodes N
    """
    n_from_pos = int(rest_pos.shape[0])
    n_from_edge = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    return max(n_from_pos, n_from_edge)


def _undirected_neighbor_sets(edge_index, num_nodes, device):
    """
    Build undirected adjacency as list of sets.

    Args:
        edge_index: torch.Tensor
            shape (2, E)
        num_nodes: int
        device: torch.device
    return: list of set
        neighbors per node
    """
    nbr = [set() for _ in range(num_nodes)]
    ei = edge_index.to(device=device, dtype=torch.long)
    src = ei[0].tolist()
    dst = ei[1].tolist()
    for s, d in zip(src, dst):
        if s == d:
            continue
        nbr[s].add(d)
        nbr[d].add(s)
    return nbr


def _merge_directed_edges(existing, src_list, dst_list, device):
    """
    Append directed edges and return coalesced unique (2, E) long tensor.

    Args:
        existing: torch.Tensor or None
            shape (2, E0) or None
        src_list: list of int
        dst_list: list of int
        device: torch.device
    return: torch.Tensor
        shape (2, E)
    """
    if existing is None or existing.numel() == 0:
        src_all = list(src_list)
        dst_all = list(dst_list)
    else:
        e = existing.to(device=device, dtype=torch.long)
        src_all = e[0].tolist() + list(src_list)
        dst_all = e[1].tolist() + list(dst_list)
    pairs = set()
    for s, d in zip(src_all, dst_all):
        if s == d:
            continue
        pairs.add((int(s), int(d)))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    s_out = [p[0] for p in pairs]
    d_out = [p[1] for p in pairs]
    return torch.tensor([s_out, d_out], dtype=torch.long, device=device)


def _fully_connected_directed(num_nodes, device):
    """
    All ordered pairs i != j.

    Args:
        num_nodes: int
        device: torch.device
    return: torch.Tensor
        edge_index (2, E)
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _k_hop_edges(edge_index, num_nodes, k_hops, device):
    """
    Add directed edges for every pair with undirected shortest-path distance in [1, k_hops].

    Args:
        edge_index: torch.Tensor
            mesh connectivity (2, E)
        num_nodes: int
        k_hops: int
            positive hop count
        device: torch.device
    return: torch.Tensor
        augmented edge_index including original mesh edges
    """
    if k_hops <= 0:
        return edge_index.to(device=device, dtype=torch.long)
    nbr = _undirected_neighbor_sets(edge_index, num_nodes, device)
    src_all = []
    dst_all = []
    for start in range(num_nodes):
        dist = [-1] * num_nodes
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= k_hops:
                continue
            for v in nbr[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for v in range(num_nodes):
            d = dist[v]
            if v != start and d >= 1 and d <= k_hops:
                src_all.append(start)
                dst_all.append(v)
    return _merge_directed_edges(edge_index, src_all, dst_all, device)


def _random_offline_edges(num_nodes, ratio, traj_seed, device):
    """
    For each node, add directed edges to M distinct random nodes with M = round(ratio * num_nodes).

    Args:
        num_nodes: int
        ratio: float
            in [0, 1]
        traj_seed: int
            combined seed for this trajectory
        device: torch.device
    return: torch.Tensor
        edge_index (2, E) of extra edges only (empty if ratio == 0)
    """
    if num_nodes <= 1 or ratio <= 0.0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    m = int(round(ratio * float(num_nodes)))
    if m <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(int(traj_seed))
    src_all = []
    dst_all = []
    cap = num_nodes - 1
    take = m if m < cap else cap
    k_take = take if take < cap else cap
    if k_take <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    for u in range(num_nodes):
        rand = torch.rand(num_nodes, generator=g, device=device)
        rand[u] = -1.0
        _, idx = torch.topk(rand, k=k_take, largest=True)
        for v in idx.tolist():
            if v != u:
                src_all.append(u)
                dst_all.append(v)
    return torch.tensor([src_all, dst_all], dtype=torch.long, device=device)


def _wall_node_indices(boundary_condition):
    """
    Args:
        boundary_condition: torch.Tensor
            x_bc, shape (N, 1) or (N,)
    return: torch.Tensor
            long indices of wall (clamped) nodes, shape (W,)
    """
    bc = boundary_condition
    if bc.dim() == 2:
        mask = bc.squeeze(1) > 0.5
    else:
        mask = bc > 0.5
    return torch.nonzero(mask, as_tuple=False).squeeze(1).long()


def _wall_plane_projection(rest_pos, wall_idx):
    """
    Project wall rest positions onto a 2D basis in the wall plane via SVD.

    Args:
        rest_pos: torch.Tensor
            (N, 3)
        wall_idx: torch.Tensor
            (W,) long
    return: tuple
        (s, t, s_min, s_max, t_min, t_max, extent) each tensor or float for bounds
    """
    device = rest_pos.device
    pts = rest_pos[wall_idx].float()
    center = pts.mean(dim=0, keepdim=True)
    centered = pts - center
    if centered.shape[0] < 2:
        z = torch.zeros(centered.shape[0], 2, device=device)
        return z[:, 0], z[:, 1], 0.0, 0.0, 0.0, 0.0, 1.0
    try:
        _u, svals, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        z = torch.zeros(centered.shape[0], 2, device=device)
        return z[:, 0], z[:, 1], 0.0, 0.0, 0.0, 0.0, 1.0
    basis = vh[:2, :].T
    proj = centered @ basis
    s = proj[:, 0]
    t = proj[:, 1]
    s_min = float(s.min().item())
    s_max = float(s.max().item())
    t_min = float(t.min().item())
    t_max = float(t.max().item())
    extent = max(s_max - s_min, t_max - t_min, 1e-8)
    return s, t, s_min, s_max, t_min, t_max, extent


def _four_corner_global_indices(wall_idx, s, t, s_min, s_max, t_min, t_max, device):
    """
    Greedy assignment of four rectangle corners to nearest wall nodes (unique).

    Args:
        wall_idx: torch.Tensor
            (W,) global mesh indices
        s: torch.Tensor
            (W,) first projected coordinate
        t: torch.Tensor
            (W,) second projected coordinate
        s_min: float
        s_max: float
        t_min: float
        t_max: float
        device: torch.device
    return: list of int
        up to four global node indices (corners)
    """
    targets_s = [s_min, s_min, s_max, s_max]
    targets_t = [t_min, t_max, t_min, t_max]
    picked = set()
    corners = []
    for ts, tt in zip(targets_s, targets_t):
        best_j = None
        best_d = None
        for j in range(wall_idx.shape[0]):
            gj = int(wall_idx[j].item())
            if gj in picked:
                continue
            ds = float(s[j].item()) - ts
            dt = float(t[j].item()) - tt
            d = ds * ds + dt * dt
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            gj = int(wall_idx[best_j].item())
            picked.add(gj)
            corners.append(gj)
    return corners


def _classify_wall_nodes(wall_idx, s, t, s_min, s_max, t_min, t_max, extent, corner_globals):
    """
    Split wall nodes into corners (given), border (perimeter not corner), internal.

    Args:
        wall_idx: torch.Tensor
        s: torch.Tensor
        t: torch.Tensor
        bounds and extent: float
        corner_globals: list of int
    return: tuple
        (border_idx_globals, internal_idx_globals) lists of global indices
    """
    tol = max(1e-6 * extent, 1e-9)
    corner_set = set(corner_globals)
    border = []
    internal = []
    for j in range(wall_idx.shape[0]):
        g = int(wall_idx[j].item())
        if g in corner_set:
            continue
        sj = float(s[j].item())
        tj = float(t[j].item())
        on_edge = (abs(sj - s_min) < tol or abs(sj - s_max) < tol or abs(tj - t_min) < tol or abs(tj - t_max) < tol)
        if on_edge:
            border.append(g)
        else:
            internal.append(g)
    return border, internal


def _sample_subset(indices, ratio, traj_seed, salt, device):
    """
    Sample floor(ratio * len(indices)) distinct indices with deterministic RNG.

    Args:
        indices: list of int
        ratio: float
            in [0, 1]
        traj_seed: int
        salt: int
            distinguishes streams
        device: torch.device
    return: list of int
    """
    if not indices or ratio <= 0.0:
        return []
    n = len(indices)
    k = int(math.floor(ratio * float(n)))
    if k <= 0:
        return []
    if k >= n:
        return list(indices)
    g = torch.Generator(device=device)
    g.manual_seed(int(traj_seed) + int(salt))
    perm = torch.randperm(n, generator=g, device=device)
    out = [indices[int(perm[i].item())] for i in range(k)]
    return out


def _clique_directed(node_list, device):
    """
    Directed complete graph on listed nodes (both directions).

    Args:
        node_list: list of int
        device: torch.device
    return: torch.Tensor
        edge_index (2, E)
    """
    if len(node_list) <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    src = []
    dst = []
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            if i != j:
                src.append(node_list[i])
                dst.append(node_list[j])
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _wall_edges_all(rest_pos, boundary_condition, device):
    """
    Clique on all wall nodes.

    Args:
        rest_pos: torch.Tensor
        boundary_condition: torch.Tensor
        device: torch.device
    return: torch.Tensor
        edge_index
    """
    wall_idx = _wall_node_indices(boundary_condition)
    if wall_idx.numel() <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    nodes = [int(wall_idx[i].item()) for i in range(wall_idx.shape[0])]
    return _clique_directed(nodes, device)


def _wall_edges_pseudo_random(rest_pos, boundary_condition, wall_border_percentage, wall_internal_percentage,
                              traj_seed, device):
    """
    Corners always included; sample border and internal wall nodes by ratios; clique on union.

    Args:
        rest_pos: torch.Tensor
        boundary_condition: torch.Tensor
        wall_border_percentage: float
        wall_internal_percentage: float
        traj_seed: int
        device: torch.device
    return: torch.Tensor
        edge_index
    """
    wall_idx = _wall_node_indices(boundary_condition)
    if wall_idx.numel() <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    s, t, s_min, s_max, t_min, t_max, extent = _wall_plane_projection(rest_pos, wall_idx)
    corners = _four_corner_global_indices(wall_idx, s, t, s_min, s_max, t_min, t_max, device)
    border_list, internal_list = _classify_wall_nodes(wall_idx, s, t, s_min, s_max, t_min, t_max, extent, corners)
    sampled_border = _sample_subset(border_list, wall_border_percentage, traj_seed, 17, device)
    sampled_internal = _sample_subset(internal_list, wall_internal_percentage, traj_seed, 29, device)
    nodes = list(dict.fromkeys(corners + sampled_border + sampled_internal))
    return _clique_directed(nodes, device)


def validate_edge_augmentation_config(k_hop, random_offline, add_wall_edges, wall_border_percentage,
                                      wall_internal_percentage):
    """
    Raises ValueError on invalid combinations (fully connected vs other augmentations).

    Args:
        k_hop: int or str
        random_offline: float
        add_wall_edges: str or None
        wall_border_percentage: float
        wall_internal_percentage: float
    return: None
    """
    if random_offline < 0.0 or random_offline > 1.0:
        raise ValueError("edge_augmentation.random_offline must be in [0, 1].")
    if wall_border_percentage < 0.0 or wall_border_percentage > 1.0:
        raise ValueError("edge_augmentation.wall_border_percentage must be in [0, 1].")
    if wall_internal_percentage < 0.0 or wall_internal_percentage > 1.0:
        raise ValueError("edge_augmentation.wall_internal_percentage must be in [0, 1].")
    if add_wall_edges is not None and add_wall_edges not in ("all", "pseudo_random"):
        raise ValueError('edge_augmentation.add_wall_edges must be null, "all", or "pseudo_random".')
    is_fc = k_hop == "fully connected" or k_hop == "fully_connected"
    if is_fc:
        if random_offline > 0.0:
            raise ValueError('edge_augmentation: k_hop "fully connected" is incompatible with random_offline > 0.')
        if add_wall_edges is not None:
            raise ValueError('edge_augmentation: k_hop "fully connected" is incompatible with add_wall_edges.')
    if isinstance(k_hop, int):
        if k_hop < 0:
            raise ValueError("edge_augmentation.k_hop int must be >= 0.")
    elif not is_fc:
        raise ValueError('edge_augmentation.k_hop must be a non-negative int or "fully connected".')


def build_augmented_edge_index(edge_index, rest_pos, boundary_condition, k_hop, random_offline, add_wall_edges,
                               wall_border_percentage, wall_internal_percentage, traj_seed, device):
    """
    Apply preprocess edge augmentation policies and return a directed COO edge_index.

    Args:
        edge_index: torch.Tensor
            raw mesh edges (2, E)
        rest_pos: torch.Tensor
            (N, 3) rest positions
        boundary_condition: torch.Tensor
            x_bc for wall mask
        k_hop: int or str
            hop count or "fully connected"
        random_offline: float
            ratio in [0, 1]
        add_wall_edges: str or None
            "all", "pseudo_random", or None
        wall_border_percentage: float
        wall_internal_percentage: float
        traj_seed: int
            per-trajectory RNG seed
        device: torch.device
    return: torch.Tensor
        augmented edge_index on device
    """
    augmenter = create_edge_augmenter_from_config(k_hop, random_offline, add_wall_edges, wall_border_percentage,
                                                   wall_internal_percentage)
    num_nodes = _num_nodes_from_inputs(edge_index, rest_pos)
    ei = edge_index.to(device=device, dtype=torch.long)
    ctx = AugmentationContext(ei, num_nodes, rest_pos, boundary_condition, traj_seed, device)
    augmenter.augment(ctx)
    return ctx.edge_index
