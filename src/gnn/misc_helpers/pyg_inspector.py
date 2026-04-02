import torch

class PyGInspector:
    def __init__(self, max_repr_chars, float_precision):
        """
        Args:
            max_repr_chars: int
                Maximum length of repr() printed for non-tensor objects
            float_precision: int
                Number of decimals for float summaries
        return: None
            Initializes the inspector
        """
        self.max_repr_chars = int(max_repr_chars)
        self.float_precision = int(float_precision)

    def describe_data(self, data, max_keys, max_values, show_tensor_stats):
        """
        Args:
            data: object
                torch_geometric.data.Data (or similar) instance
            max_keys: int
                Maximum number of keys/attributes to print
            max_values: int
                Maximum number of values to print for tensors/containers
            show_tensor_stats: bool
                If True, prints min/max/mean for floating tensors (can be costly)
        return: None
            Prints a readable summary of graph attributes and their types
        """
        keys = self._keys_list(data)
        print(f"type(data) = {type(data)}")
        print(f"num_keys = {len(keys)}")

        self._print_basic_graph_stats(data)

        for i, key in enumerate(keys[:max_keys]):
            value = self._safe_getitem(data, key)
            line = self._format_value(key, value, max_values, show_tensor_stats)
            print(line)

        if len(keys) > max_keys:
            print(f"... ({len(keys) - max_keys} more keys not shown)")

    def describe_batch(self, batch, max_keys, max_values, show_tensor_stats, graphs_to_sample, per_graph_max_keys, per_graph_max_values):
        """
        Args:
            batch: object
                torch_geometric.data.Batch instance
            max_keys: int
                Maximum number of batch keys/attributes to print
            max_values: int
                Maximum number of values to print for batch tensors/containers
            show_tensor_stats: bool
                If True, prints min/max/mean for floating tensors (can be costly)
            graphs_to_sample: int
                Number of graphs to sample from the batch (via to_data_list)
            per_graph_max_keys: int
                Max keys to print per sampled graph
            per_graph_max_values: int
                Max values to print per sampled graph tensor/containers
        return: None
            Prints summary of a Batch and optionally a few example graphs inside it
        """
        print("=== Batch summary ===")
        self.describe_data(batch, max_keys, max_values, show_tensor_stats)

        num_graphs = self._infer_num_graphs_from_batch(batch)
        if num_graphs is not None:
            print(f"num_graphs (inferred) = {num_graphs}")

        data_list = self._safe_to_data_list(batch)
        if data_list is None:
            print("Could not call batch.to_data_list(); skipping per-graph samples.")
            return

        sample_n = min(int(graphs_to_sample), len(data_list))
        print(f"=== Sampling {sample_n} graphs from batch ===")
        for i in range(sample_n):
            print(f"\n--- graph[{i}] ---")
            self.describe_data(data_list[i], per_graph_max_keys, per_graph_max_values, show_tensor_stats)

    def scan_graphs(self, graph_list, max_graphs):
        """
        Args:
            graph_list: object
                Iterable of Data objects (e.g., list, dataset, dataloader output)
            max_graphs: int
                Maximum number of graphs to scan
        return: dict
            Mapping key -> set of observed python types (as strings)
        """
        schema = {}
        n = 0
        for g in graph_list:
            if n >= int(max_graphs):
                break
            keys = self._keys_list(g)
            for key in keys:
                value = self._safe_getitem(g, key)
                tname = type(value).__name__
                if key not in schema:
                    schema[key] = set()
                schema[key].add(tname)
            n += 1

        print(f"Scanned {n} graphs. Keys observed: {len(schema)}")
        for key in sorted(schema.keys()):
            types_str = ", ".join(sorted(schema[key]))
            print(f"{key}: {types_str}")

        return schema

    def _keys_list(self, data):
        keys_attr = None
        try:
            keys_attr = data.keys
        except Exception:
            return []
        try:
            if callable(keys_attr):
                return list(keys_attr())
            return list(keys_attr)
        except Exception:
            return []

    def _safe_getitem(self, data, key):
        try:
            return data[key]
        except Exception:
            return "<unreadable>"

    def _format_value(self, key, value, max_values, show_tensor_stats):
        if isinstance(value, torch.Tensor):
            return self._format_tensor_line(key, value, max_values, show_tensor_stats)

        if isinstance(value, (list, tuple)):
            preview = self._preview_container(value, max_values)
            return f"{key}: {type(value).__name__} len={len(value)} preview={preview}"

        if isinstance(value, dict):
            preview = self._preview_dict(value, max_values)
            return f"{key}: dict len={len(value)} preview={preview}"

        rep = self._short_repr(value)
        return f"{key}: {type(value).__name__} {rep}"

    def _format_tensor_line(self, key, t, max_values, show_tensor_stats):
        shape = tuple(t.shape)
        dtype = str(t.dtype)
        device = str(t.device)

        flat_preview = self._tensor_preview(t, max_values)
        base = f"{key}: Tensor shape={shape} dtype={dtype} device={device} preview={flat_preview}"

        if not show_tensor_stats:
            return base

        stats = self._tensor_stats(t)
        if stats is None:
            return base
        return f"{base} stats={stats}"

    def _tensor_preview(self, t, max_values):
        try:
            if t.numel() == 0:
                return "[]"
            v = t.detach()
            if v.is_cuda:
                v = v.to("cpu")
            v = v.reshape(-1)[:int(max_values)]
            return v.tolist()
        except Exception:
            return "<preview_failed>"

    def _tensor_stats(self, t):
        try:
            if t.numel() == 0:
                return None
            if not t.dtype.is_floating_point:
                return None
            v = t.detach()
            if v.is_cuda:
                v = v.to("cpu")
            v = v.reshape(-1)
            mn = float(v.min().item())
            mx = float(v.max().item())
            mean = float(v.mean().item())
            fmt = f"{{:.{self.float_precision}f}}"
            return {"min": float(fmt.format(mn)), "max": float(fmt.format(mx)), "mean": float(fmt.format(mean))}
        except Exception:
            return None

    def _preview_container(self, seq, max_values):
        try:
            out = []
            for i, item in enumerate(seq):
                if i >= int(max_values):
                    break
                out.append(self._short_repr(item))
            return out
        except Exception:
            return "<preview_failed>"

    def _preview_dict(self, d, max_values):
        try:
            out = []
            for i, k in enumerate(d.keys()):
                if i >= int(max_values):
                    break
                out.append((self._short_repr(k), self._short_repr(d[k])))
            return out
        except Exception:
            return "<preview_failed>"

    def _short_repr(self, value):
        try:
            r = repr(value)
            if len(r) > self.max_repr_chars:
                return r[:self.max_repr_chars] + "..."
            return r
        except Exception:
            return "<repr_failed>"

    def _print_basic_graph_stats(self, data):
        try:
            n = data.num_nodes
            print(f"num_nodes = {n}")
        except Exception:
            pass
        try:
            eidx = self._safe_getitem(data, "edge_index")
            if isinstance(eidx, torch.Tensor) and eidx.dim() == 2 and eidx.size(0) == 2:
                print(f"num_edges = {int(eidx.size(1))}")
        except Exception:
            pass

    def _infer_num_graphs_from_batch(self, batch):
        try:
            bvec = self._safe_getitem(batch, "batch")
            if isinstance(bvec, torch.Tensor) and bvec.numel() > 0:
                return int(bvec.max().item()) + 1
        except Exception:
            pass
        try:
            return int(batch.num_graphs)
        except Exception:
            return None

    def _safe_to_data_list(self, batch):
        try:
            return batch.to_data_list()
        except Exception:
            return None

def print_all_pyg_fields(dataloader, graph):
    inspector = PyGInspector(140, 4)
    print("\n\n\n\t\t\t Inspector\n\n\n")
    inspector.scan_graphs(dataloader, 50)
    print("\n\n\n\t\t\tPrint pyg fields\n\n\n")
    print_pyg_fields(graph, 50, 6)
    print("\n\n\n\t\t\tOther\n\n\n")
    # and show the core features if present:
    keys_attr = graph.keys
    keys = list(keys_attr()) if callable(keys_attr) else list(keys_attr)

    if "x" in keys:
        print("x:", graph["x"].shape)
    if "edge_index" in keys:
        print("edge_index:", graph["edge_index"].shape)
    if "edge_attr" in keys:
        print("edge_attr:", graph["edge_attr"].shape)

    # show first actual graph:
    dl = graph.to_data_list()
    print("num_graphs_in_batch:", len(dl))
    print("first graph keys:", list(dl[0].keys()) if not callable(dl[0].keys) else list(dl[0].keys()))

def print_pyg_fields(data, max_keys, max_preview):
    """
    Args:
        data: object
            PyG Data or Batch-like object
        max_keys: int
            Maximum number of keys to print
        max_preview: int
            How many values to preview for tensors
    return: None
        Prints available keys and tensor metadata
    """
    keys_attr = data.keys
    keys = list(keys_attr()) if callable(keys_attr) else list(keys_attr)

    print(f"type = {type(data)}")
    print(f"keys ({len(keys)}): {keys[:max_keys]}{' ...' if len(keys) > max_keys else ''}\n")

    for k in keys[:max_keys]:
        v = data[k]
        if torch.is_tensor(v):
            preview = v.reshape(-1)[:max_preview].detach()
            if preview.is_cuda:
                preview = preview.to("cpu")
            print(f"{k}:\n\t Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device} preview={preview.tolist()}")
        else:
            print(f"{k}:\n\t {type(v)}")