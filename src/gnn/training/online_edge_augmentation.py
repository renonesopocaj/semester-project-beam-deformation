"""
Online random edge augmentation for training and evaluation.

Merges extra directed edges into each graph in a PyG batch: per source node u,
adds M = round(ratio * N) edges u -> v where v is not u and v is not already
adjacent to u on the undirected view of the current edge_index. If fewer than
M candidates exist, uses all candidates. Recomputes edge_attr on the merged
topology so edge_index and edge_attr stay consistent.
"""

import torch

from src.gnn.preprocessing.edge_augmentation import _merge_directed_edges

# Split codes for seed folding (caller may use these or pass distinct ints).
SPLIT_TRAIN_ONLINE = 0
SPLIT_VAL_DURING_FIT = 1
SPLIT_VAL_FINAL = 2
SPLIT_TEST_FINAL = 3
SPLIT_ROLLOUT_VAL = 4
SPLIT_ROLLOUT_TEST = 5
SPLIT_NOROLL_PER_TRAJ = 6
SPLIT_PREDICT = 7


def fold_online_seed(*parts):
    """
    Mix integer parts into a single non-negative seed for torch.Generator.

    Args:
        parts: int
            Arbitrary integer tags (epoch, batch index, graph index, etc.).
    return: int
        Folded seed in [0, 2**31 - 1].
    """
    h = 5381
    for p in parts:
        h = ((h << 5) + h) + int(p)
        h = h & 0x7FFFFFFF
    return int(h)


def num_graphs_in_pyg_batch(batch):
    """
    Args:
        batch: Data
            PyG Batch or single graph.
    return: int
        Number of graphs in the batch.
    """
    device = batch.x_pos_t.device
    ptr = _ptr_for_batch(batch, device)
    return int(ptr.numel()) - 1


def _ptr_for_batch(batch, device):
    """
    Args:
        batch: Data
            PyG Batch or single graph Data.
        device: torch.device
            Device for ptr tensor.
    return: torch.Tensor
        ptr of shape (num_graphs + 1,) long, cumulative node starts.
    """
    ptr_attr = getattr(batch, "ptr", None)
    if ptr_attr is not None and ptr_attr.numel() > 0:
        return ptr_attr.to(device=device, dtype=torch.long)
    num_graphs = int(getattr(batch, "num_graphs", 1))
    if num_graphs <= 1:
        n = int(batch.x_pos_t.shape[0])
        return torch.tensor([0, n], dtype=torch.long, device=device)
    b = batch.batch.to(device=device, dtype=torch.long)
    counts = torch.bincount(b, minlength=num_graphs)
    return torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(dim=0)], dim=0)


def _neighbor_sets_undirected(edge_index, num_nodes, device):
    """
    Args:
        edge_index: torch.Tensor
            Shape (2, E), long.
        num_nodes: int
            Node count N.
        device: torch.device
            Device for tensors.
    return: list
        list of set, length N, undirected adjacency (both directions).
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


def _random_online_extra_src_dst(edge_index, num_nodes, ratio, device, generator):
    """
    Directed extra edges only (caller merges with existing edge_index).

    Args:
        edge_index: torch.Tensor
            Current graph edges (2, E).
        num_nodes: int
            N.
        ratio: float
            In [0, 1]; M = round(ratio * N) per source node.
        device: torch.device
            Computation device.
        generator: torch.Generator
            RNG stream positioned by caller.
    return: (list, list)
        Parallel src and dst index lists (global indices within this graph).
    """
    if num_nodes <= 1 or ratio <= 0.0:
        return [], []
    m = int(round(ratio * float(num_nodes)))
    if m <= 0:
        return [], []
    nbr = _neighbor_sets_undirected(edge_index, num_nodes, device)
    src_all = []
    dst_all = []
    for u in range(num_nodes):
        rand = torch.rand(num_nodes, device=device, dtype=torch.float32, generator=generator)
        mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        mask[u] = False
        for v in nbr[u]:
            mask[int(v)] = False
        num_cand = int(mask.sum().item())
        if num_cand == 0:
            continue
        take = m if m < num_cand else num_cand
        rand[~mask] = -1.0
        _, idx = torch.topk(rand, k=take, largest=True)
        for v in idx.tolist():
            src_all.append(u)
            dst_all.append(v)
    return src_all, dst_all


class OnlineRandomEdgeAugmenter:
    """
    Strategy object: applies online random edge augmentation and rebuilds edge_attr.
    """

    edge_feature_builder: object
    normalizer: object
    edge_stats: object

    def __init__(self, edge_feature_builder, normalizer, edge_stats):
        """
        Args:
            edge_feature_builder: callable
                (x_t, v_t, x_tm1_prev, rest_pos, node_force_t, edge_index) -> (edge_feat_raw, _).
            normalizer: object
                Normalizer with normalize(tensor, stats).
            edge_stats: object
                Stats-like object with mean and std buffers (e.g. StatsModule).
        """
        self.edge_feature_builder = edge_feature_builder
        self.normalizer = normalizer
        self.edge_stats = edge_stats

    def apply(self, batch, ratio, per_graph_seeds):
        """
        If ratio <= 0, returns the same batch reference unchanged. Otherwise clones the batch and sets
        edge_index / edge_attr on the clone so underlying dataset tensors are not mutated.

        Args:
            batch: Data
                PyG Batch or single-graph Data with x_pos_t, x_vel_t, x_pos_tm1_prev, rest_pos, node_force, edge_index.
            ratio: float
                random_online in [0, 1].
            per_graph_seeds: list
                One int seed per graph in the batch, length num_graphs.
        return: Data
            Original batch if ratio <= 0, else a clone with updated edges.
        """
        if ratio <= 0.0:
            return batch
        out = batch.clone()
        device = out.x_pos_t.device
        ptr = _ptr_for_batch(out, device)
        num_graphs = int(ptr.numel()) - 1
        if len(per_graph_seeds) != num_graphs:
            raise ValueError("per_graph_seeds length must match number of graphs in batch.")
        global_src = []
        global_dst = []
        base_ei = out.edge_index.to(device=device, dtype=torch.long)
        for g in range(num_graphs):
            start = int(ptr[g].item())
            end = int(ptr[g + 1].item())
            n_local = end - start
            if n_local <= 0:
                continue
            mask = (base_ei[0] >= start) & (base_ei[0] < end) & (base_ei[1] >= start) & (base_ei[1] < end)
            local_ei = base_ei[:, mask].clone()
            local_ei[0] -= start
            local_ei[1] -= start
            gen = torch.Generator(device=device)
            gen.manual_seed(int(per_graph_seeds[g]))
            extra_s, extra_d = _random_online_extra_src_dst(local_ei, n_local, ratio, device, gen)
            extra_s_g = [start + int(s) for s in extra_s]
            extra_d_g = [start + int(d) for d in extra_d]
            global_src.extend(extra_s_g)
            global_dst.extend(extra_d_g)
        merged = _merge_directed_edges(base_ei, global_src, global_dst, device)
        edge_raw, _ = self.edge_feature_builder(out.x_pos_t, out.x_vel_t, out.x_pos_tm1_prev, out.rest_pos,
                                                out.node_force, merged)
        out.edge_index = merged
        out.edge_attr = self.normalizer.normalize(edge_raw, self.edge_stats)
        return out
