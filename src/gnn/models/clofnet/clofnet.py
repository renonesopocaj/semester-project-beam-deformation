"""
clofnet.py

ClofNet variants for predicting acceleration (equivariant) and stress targets from PyG batches.
Base class handles scalarization of stacked R^3 node/edge features using split_idx_node / split_idx_edge,
a fixed EquiFrame from input positions, fused edge embeddings, and a stress head.
Common notation
                        Δᵢ⁽q,θ⁾(m) = w_c * (1 / |𝒩(i)|) * ∑ⱼ∈𝒩(i) Vectorize_θ(mᵢⱼ, Fᵢⱼ⁰)
                        q ∈ {a, v, x}
                        Fᵢⱼ⁰ = EquiFrame(xᶜᵉⁿᵗᵢ⁰, xᶜᵉⁿᵗⱼ⁰)
                        xᶜᵉⁿᵗ = field - centroid(field)
                        field = x_pos_t  (frame_source="abs")
                              | u(t) = x_pos_t - rest_pos  (frame_source="displacement")
                        ẽᵢⱼ⁰ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(a⁰, Fᵢⱼ⁰)])
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from torch import nn

from src.gnn.models.clofnet.scalarizer_localframe import (coord2localframe, frame_basis_from_edges, build_s_ij)
from src.gnn.models.clofnet.clof_helpers import (split_edge_features, split_node_features, make_mlp)
from src.gnn.models.clofnet.clof_aggregators import unsorted_segment_sum
from src.gnn.models.clofnet.clof_mp_blocks import (DirectAccelerationMPBlock, MultiHeadMPBlock,
                                                   RecurrentMessageInitializer, RecurrentMPBlock,
                                                   RecurrentMultiHeadMPBlock, SingleVectorizeMPBlock)


class ClofNetBase(nn.Module, ABC):
    """
    Shared ClofNet acceleration pipeline: slice features, embed invariants, build frame from positions,
    scalarize equivariant chunks + running acceleration into s_ij, fuse with edge scalars, stress head.
    """

    hidden_dim: int
    n_layers: int
    split_idx_node: int
    split_idx_edge: int
    enable_norm_diff: bool
    use_neighbor_diff_norms: bool
    act_fn_factory: object
    frame_source: str
    include_u_tm1: bool
    rescalarize_per_layer: bool
    include_v_branch: bool
    include_force: bool
    stress_out_dim: int
    m_ij_init: str
    resnet_style: bool
    eps_mode: str
    n_layers_gate_mlp: int
    n_layers_node_mlp: int
    n_layers_message_mlp: int
    normalize_delta_a: bool
    mlp_layer_norm: bool

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels, use_neighbor_diff_norms,
                 frame_source, include_u_tm1, rescalarize_per_layer, include_v_branch, include_force, stress_out_dim,
                 m_ij_init, resnet_style, eps_mode, n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp,
                 normalize_delta_a, mlp_layer_norm):
        super().__init__()
        self.act_fn_factory = act_fn_factory
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.split_idx_node = split_idx_node
        self.split_idx_edge = split_idx_edge
        self.enable_norm_diff = enable_norm_diff
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.tanh = tanh
        self.attention_enable = attention_enable
        self.clamp_enable = clamp_enable
        self.vel_mlp_dim_out = vel_mlp_dim_out
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.use_neighbor_diff_norms = use_neighbor_diff_norms
        self.frame_source = frame_source
        self.include_u_tm1 = include_u_tm1
        self.rescalarize_per_layer = rescalarize_per_layer
        self.include_v_branch = include_v_branch
        self.include_force = include_force
        self.stress_out_dim = int(stress_out_dim)
        if self.stress_out_dim not in {1, 6}:
            raise ValueError("stress_out_dim must be 1 (von_mises) or 6 (full_stress), got %s" % self.stress_out_dim)
        self.m_ij_init = m_ij_init
        self.resnet_style = resnet_style
        self.eps_mode = eps_mode
        self.n_layers_gate_mlp = n_layers_gate_mlp
        self.n_layers_node_mlp = n_layers_node_mlp
        self.n_layers_message_mlp = n_layers_message_mlp
        self.normalize_delta_a = normalize_delta_a
        self.mlp_layer_norm = mlp_layer_norm
        self.in_dim_edge_mlp = fuse_hid_dim
        num_node_triplets = split_idx_node // 3
        num_edge_triplets = split_idx_edge // 3
        pseudo_dim = 2
        num_force_triplets = 1 if include_force else 0
        self.s_ij_dim = pseudo_dim + (num_node_triplets + 1 + num_force_triplets) * 6 + num_edge_triplets * 3
        self.fuse_in_dim = in_dim_edge_scalars + self.s_ij_dim
        self.embedding_node_mlp = nn.Linear(in_dim_node_embed, self.hidden_dim)
        self.fuse_mlp_embed = make_mlp(2, self.fuse_in_dim, fuse_hid_dim, fuse_hid_dim, act_fn_factory, True, mlp_layer_norm)
        stress_in_dim = 2 * self.hidden_dim + (3 if use_neighbor_diff_norms else 0)
        self.stress_mlp = make_mlp(2, stress_in_dim, self.hidden_dim, self.stress_out_dim, act_fn_factory, False,
                                   mlp_layer_norm)
        self.mp_blocks = self._build_mp_blocks()
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)

    @abstractmethod
    def _build_mp_blocks(self):
        """
        return: nn.ModuleList
            Stacked message-passing blocks for this variant.
        """
        raise NotImplementedError

    @abstractmethod
    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered, u_tm1_0, f0):
        """
        Run the variant-specific message-passing stack.

        Args:
            h0: torch.Tensor
                Initial node embeddings [N, H].
            a0: torch.Tensor
                Initial acceleration field [N, 3] (zeros).
            v0: torch.Tensor
                Initial velocity [N, 3].
            x0: torch.Tensor
                Initial centered field copy [N, 3] (field - centroid, where field is x_pos_t or u(t)).
            node_vec_chunks: torch.Tensor
                Static valued node vectors [N, K, 3] from batch.x.
            edge_vec_chunks: torch.Tensor
                Static edge vectors [E, K, 3] from edge_attr.
            edge_scalars: torch.Tensor
                Invariant edge scalars [E, S_e].
            frame_e33: torch.Tensor
                Local frame [E, 3, 3].
            radial: torch.Tensor
                Squared distances [E, 1].
            edge_index_tuple: tuple
                (row, col) directed edges.
            x_pos_centered: torch.Tensor
                Centered field used only where subclasses need to refresh frame.
            u_tm1_0: torch.Tensor or None
                Centered u(t-1) field [N, 3], or None when include_u_tm1 is False.
            f0: torch.Tensor or None
                Per-node force [N, 3], or None when include_force is False.
        return: (h_L, a_L, v_L, x_L, m_agg_L)
            Final states and per-node aggregated messages for the stress head [N, H].
        """
        raise NotImplementedError

    def _safe_a_for_fuse(self, a):
        """
        Isotropic per-node rescaling of acceleration before it enters build_s_ij.
        Only applied when normalize_delta_a is True; identity otherwise.
        Equivariant: divides all 3 components by the same per-node scalar so rotation symmetry is preserved.

        Args:
            a: torch.Tensor [N, 3]
        return: torch.Tensor [N, 3]
        """
        if not self.normalize_delta_a:
            return a
        scale = torch.norm(a, dim=-1, keepdim=True).clamp(min=1.0)
        return a / scale

    def _fuse_embed(self, acceleration_n3, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                    edge_index_tuple, node_force):
        """
        Build s_ij from the current acceleration estimate and fuse with invariant edge scalars.
        Optionally projects node_force into s_ij when include_force is True.

        Args:
            acceleration_n3: torch.Tensor [N, 3]
            node_vec_chunks: torch.Tensor [N, K, 3]
            edge_vec_chunks: torch.Tensor [E, K, 3]
            edge_scalars: torch.Tensor [E, S_e]
            frame_e33: torch.Tensor [E, 3, 3]
            edge_index_tuple: tuple
            node_force: torch.Tensor or None
                [N, 3] per-node force; passed as None when include_force is False.
        return: torch.Tensor
            fuse_embed [E, fuse_hid_dim]
        """
        s_ij = build_s_ij(acceleration_n3, node_vec_chunks, edge_vec_chunks, frame_e33, edge_index_tuple, node_force)
        fused = torch.cat([edge_scalars, s_ij], dim=-1)
        return self.fuse_mlp_embed(fused)

    def _stress_out(self, h_final, m_agg, a_l, v_l, x_l, edge_index_tuple):
        """
        Predict scalar stress from h_i^L and µ_i^L; optionally appends summed pairwise-difference norms.

        Args:
            h_final: torch.Tensor [N, H]
            m_agg: torch.Tensor [N, H]
            a_l: torch.Tensor [N, 3]
            v_l: torch.Tensor [N, 3]
            x_l: torch.Tensor [N, 3]
            edge_index_tuple: tuple
                (row, col) directed edges; row is the destination node.
        return: torch.Tensor [N, stress_out_dim]
        """
        if self.use_neighbor_diff_norms:
            row, col = edge_index_tuple
            n = h_final.size(0)
            a_mex = unsorted_segment_sum(torch.norm(a_l[row] - a_l[col], dim=1, keepdim=True), row, n)
            v_mex = unsorted_segment_sum(torch.norm(v_l[row] - v_l[col], dim=1, keepdim=True), row, n)
            x_mex = unsorted_segment_sum(torch.norm(x_l[row] - x_l[col], dim=1, keepdim=True), row, n)
            inp = torch.cat([h_final, m_agg, a_mex, v_mex, x_mex], dim=-1)
        else:
            inp = torch.cat([h_final, m_agg], dim=-1)
        return self.stress_mlp(inp)

    def _center_field(self, field, num_graphs, n_per):
        """
        Subtract per-graph centroid from a per-node R^3 field.

        Args:
            field: torch.Tensor [N, 3]
            num_graphs: int
            n_per: int
                nodes per graph (assumed uniform)
        return: torch.Tensor [N, 3]
        """
        field_resh = field.reshape(num_graphs, n_per, 3)
        centroid = torch.mean(field_resh, dim=1, keepdim=True)
        return (field_resh - centroid).reshape(-1, 3)

    def forward(self, batch):
        """
        Predict normalized acceleration (first 3) and stress channels for each node.

        Args:
            batch: torch_geometric.data.Data or Batch
                Must provide x, edge_attr, edge_index, x_pos_t, x_vel_t, rest_pos,
                x_pos_tm1_prev, num_graphs, and optionally ptr.
                When include_force is True, must also provide node_force [N, 3].
        return: torch.Tensor
            Predictions [N, 3 + stress_out_dim] = concat(acceleration, stress channels).
        """
        x_full = batch.x
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        x_pos = batch.x_pos_t
        v_phy = batch.x_vel_t
        rest_pos = batch.rest_pos
        row, col = edge_index[0], edge_index[1]
        edge_index_tuple = (row, col)
        assert int(x_full.shape[1]) == int(self.node_channels)
        assert int(edge_attr.shape[1]) == int(self.edge_channels)
        _, node_vec_chunks, node_scalars = split_node_features(x_full, self.split_idx_node)
        edge_vec_chunks, edge_scalars = split_edge_features(edge_attr, self.split_idx_edge)
        bc = x_full[:, 0:1]
        h0 = self.embedding_node_mlp(torch.cat([bc, node_scalars], dim=-1))
        num_nodes = int(x_pos.shape[0])
        a0 = torch.zeros(num_nodes, 3, device=x_pos.device, dtype=x_pos.dtype)
        v0 = v_phy
        num_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
        n_per = num_nodes // num_graphs
        # Select the field to use for the local frame and cascade initial state.
        if self.frame_source == "displacement":
            field_t = x_pos - rest_pos
        else:
            field_t = x_pos
        # Centering per graph (subtracting per-graph centroid)
        x_centered = self._center_field(field_t, num_graphs, n_per)
        x0 = x_centered
        # Centered u(t-1) field, only computed when the flag requires it.
        if self.include_u_tm1:
            x_tm1 = batch.x_pos_tm1_prev
            if self.frame_source == "displacement":
                field_tm1 = x_tm1 - rest_pos
            else:
                field_tm1 = x_tm1
            u_tm1_0 = self._center_field(field_tm1, num_graphs, n_per)
        else:
            u_tm1_0 = None
        # Node force (not centered — force is a vector field, not a position field).
        f0 = batch.node_force if self.include_force else None
        # First projection coefficients
        radial, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_centered, self.enable_norm_diff, False, self.eps_mode)
        # Construct frame
        frame_e33 = frame_basis_from_edges(a_ij, b_ij, c_ij)
        # Run message passing loop (forward through all L layers)
        h_l, a_l, v_l, x_l, m_agg_l = self._run_mp_loop(h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks,
                                                        edge_scalars, frame_e33, radial, edge_index_tuple,
                                                        x_centered, u_tm1_0, f0)
        # Final stress prediction
        stress = self._stress_out(h_l, m_agg_l, a_l, v_l, x_l, edge_index_tuple)
        return torch.cat([a_l, stress], dim=-1)


_VALID_M_IJ_INIT = frozenset({"zero", "gaussian", "edge_encoder", "context", "context_stochastic"})


class ClofNetRecurrent(ClofNetBase):
    """
    Recurrent messages version, i.e. the following.
    RecurrentMPBlock message passing:
                            ẽᵢⱼˡ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(aˡ, Fᵢⱼ⁰)])
                            mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼˡ)
                                or mᵢⱼˡ + φₘ(...) when resnet_style is True
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾))
                            aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾)
    s_ij recomputed each layer from current acceleration; only a is updated (no explicit v/x evolution).
    rescalarize_per_layer, include_v_branch, include_force have no effect on this variant.
    When include_force is True, node_force is projected into s_ij as an additional static term.
    """

    message_initializer: RecurrentMessageInitializer

    def _build_mp_blocks(self):
        in_mij = 2 * self.hidden_dim + 1 + self.in_dim_edge_mlp + self.hidden_dim
        blocks = []
        for _ in range(self.n_layers):
            blocks.append(RecurrentMPBlock(in_node_mlp_dim=2 * self.hidden_dim, output_node_dim=self.hidden_dim,
                                           hidden_dim=self.hidden_dim, in_mij_mlp_dim=in_mij,
                                           act_fn_factory=self.act_fn_factory, recurrent=self.recurrent,
                                           resnet_style=self.resnet_style,
                                           coords_weight=self.coords_weight, attention=self.attention_enable,
                                           tanh=self.tanh, clamp=self.clamp_enable,
                                           n_layers_node_mlp=self.n_layers_node_mlp,
                                           n_layers_gate_mlp=self.n_layers_gate_mlp,
                                           n_layers_message_mlp=self.n_layers_message_mlp,
                                           mlp_layer_norm=self.mlp_layer_norm))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels, use_neighbor_diff_norms,
                 frame_source, include_u_tm1, rescalarize_per_layer, include_v_branch, include_force, stress_out_dim,
                 m_ij_init, resnet_style, eps_mode, n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp,
                 normalize_delta_a, mlp_layer_norm):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels,
                         use_neighbor_diff_norms, frame_source, include_u_tm1, rescalarize_per_layer,
                         include_v_branch, include_force, stress_out_dim, m_ij_init, resnet_style, eps_mode,
                         n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp, normalize_delta_a,
                         mlp_layer_norm)
        if self.m_ij_init not in _VALID_M_IJ_INIT:
            raise ValueError("m_ij_init must be one of %s, got '%s'" % (sorted(_VALID_M_IJ_INIT), self.m_ij_init))
        self.message_initializer = RecurrentMessageInitializer(
            hidden_dim=self.hidden_dim,
            fuse_hid_dim=self.in_dim_edge_mlp,
            m_ij_init=self.m_ij_init,
            n_layers_message_mlp=self.n_layers_message_mlp,
            act_fn_factory=self.act_fn_factory,
            mlp_layer_norm=self.mlp_layer_norm,
        )

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered, u_tm1_0, f0):
        h = h0
        a = a0
        fuse_embed_0 = self._fuse_embed(self._safe_a_for_fuse(a0), node_vec_chunks, edge_vec_chunks,
                                        edge_scalars, frame_e33, edge_index_tuple, f0)
        m_ij = self.message_initializer(h0, edge_index_tuple, radial, fuse_embed_0)
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for mp_block_l in self.mp_blocks:
            fuse_embed = self._fuse_embed(self._safe_a_for_fuse(a), node_vec_chunks, edge_vec_chunks,
                                         edge_scalars, frame_e33, edge_index_tuple, f0)
            _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff,
                                                   False, self.eps_mode)
            h, a, m_ij, m_agg_last = mp_block_l(h, edge_index_tuple, a_ij, b_ij, c_ij, radial, fuse_embed, m_ij, a)
        return h, a, v0, x0, m_agg_last


class ClofNetSeparateMessages(ClofNetBase):
    """
    Separate message MLPs for a, v, x channels; fused s_ij fixed from zero acceleration; phi_vel on a.
                            mᵢⱼᵃ,⁽ˡ⁺¹⁾ = φₘᵃ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            mᵢⱼᵛ,⁽ˡ⁺¹⁾ = φₘᵛ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)   [only when include_v_branch]
                            mᵢⱼˣ,⁽ˡ⁺¹⁾ = φₘˣ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            mᵢⱼᶠ,⁽ˡ⁺¹⁾ = φₘᶠ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)   [only when include_force]
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i)(mᵢⱼᵃ + [mᵢⱼᵛ] + mᵢⱼˣ + [mᵢⱼᶠ])))
                            vᵢ⁽ˡ⁺¹⁾ = vᵢˡ + Δᵢ⁽ᵛ,θᵥ⁾(mᵛ,⁽ˡ⁺¹⁾)      [only when include_v_branch]
                            xᵢ⁽ˡ⁺¹⁾ = xᵢˡ + Δᵢ⁽ˣ,θₓ⁾(mˣ,⁽ˡ⁺¹⁾)
                            fᵢ⁽ˡ⁺¹⁾ = fᵢˡ + Δᵢ⁽ᶠ,θᶠ⁾(mᶠ,⁽ˡ⁺¹⁾)      [only when include_force]
                            ãᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(mᵃ,⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + [φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾] + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
                                       + [φᶠ(hᵢ⁽ˡ⁺¹⁾) ⊙ fᵢ⁽ˡ⁺¹⁾]

    When rescalarize_per_layer is True, frame F_ij^l, radial r_ij^l and s_ij^l are all recomputed
    from the current x^l and a^l before each block.
    """

    def _build_mp_blocks(self):
        in_mij = 2 * self.hidden_dim + 1 + self.in_dim_edge_mlp
        blocks = []
        for _ in range(self.n_layers):
            blocks.append(MultiHeadMPBlock(in_node_mlp_dim=2 * self.hidden_dim, output_node_dim=self.hidden_dim,
                                           hidden_dim=self.hidden_dim, in_mij_mlp_dim=in_mij,
                                           act_fn_factory=self.act_fn_factory, recurrent=self.recurrent,
                                           coords_weight=self.coords_weight, attention=self.attention_enable,
                                           tanh=self.tanh, clamp=self.clamp_enable,
                                           vel_mlp_dim_out=self.vel_mlp_dim_out,
                                           include_u_tm1=self.include_u_tm1,
                                           include_v_branch=self.include_v_branch,
                                           include_force=self.include_force,
                                           n_layers_node_mlp=self.n_layers_node_mlp,
                                           n_layers_gate_mlp=self.n_layers_gate_mlp,
                                           n_layers_message_mlp=self.n_layers_message_mlp,
                                           mlp_layer_norm=self.mlp_layer_norm))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels, use_neighbor_diff_norms,
                 frame_source, include_u_tm1, rescalarize_per_layer, include_v_branch, include_force, stress_out_dim,
                 m_ij_init, resnet_style, eps_mode, n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp,
                 normalize_delta_a, mlp_layer_norm):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels,
                         use_neighbor_diff_norms, frame_source, include_u_tm1, rescalarize_per_layer,
                         include_v_branch, include_force, stress_out_dim, m_ij_init, resnet_style, eps_mode,
                         n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp, normalize_delta_a,
                         mlp_layer_norm)

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered, u_tm1_0, f0):
        h = h0; a = a0; v = v0; x = x0
        u_tm1 = u_tm1_0 if u_tm1_0 is not None else torch.zeros_like(x0)
        f = f0 if f0 is not None else torch.zeros_like(x0)
        if not self.rescalarize_per_layer:
            fuse_embed = self._fuse_embed(a0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                                          edge_index_tuple, f0)
            _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff, False, self.eps_mode)
            radial_l = radial
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for block in self.mp_blocks:
            if self.rescalarize_per_layer:
                radial_l, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x, self.enable_norm_diff, False, self.eps_mode)
                frame_l = frame_basis_from_edges(a_ij, b_ij, c_ij)
                fuse_embed = self._fuse_embed(self._safe_a_for_fuse(a), node_vec_chunks, edge_vec_chunks,
                                              edge_scalars, frame_l, edge_index_tuple, f0)
            h, a, v, x, u_tm1, f, m_agg_last = block(h, edge_index_tuple, a_ij, b_ij, c_ij, radial_l,
                                                      fuse_embed, a, v, x, u_tm1, f)
        return h, a, v, x, m_agg_last


class ClofNetRecurrentSeparateMessages(ClofNetBase):
    """
    Separate multi-head messages with recurrent branch state.

                            ẽᵢⱼˡ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(aˡ, Fᵢⱼ⁰)])
                            mᵢⱼᵃ,⁽ˡ⁺¹⁾ = φₘᵃ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼᵃ,ˡ)
                                or mᵢⱼᵃ,ˡ + φₘᵃ(...) when resnet_style is True
                            mᵢⱼᵛ,⁽ˡ⁺¹⁾ = φₘᵛ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼᵛ,ˡ)   [only when include_v_branch]
                            mᵢⱼˣ,⁽ˡ⁺¹⁾ = φₘˣ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼˣ,ˡ)
                            mᵢⱼᵘᵗᵐ¹,⁽ˡ⁺¹⁾ = φₘᵘᵗᵐ¹(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼᵘᵗᵐ¹,ˡ)
                                                        [only when include_u_tm1]
                            mᵢⱼᶠ,⁽ˡ⁺¹⁾ = φₘᶠ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼᶠ,ˡ)   [only when include_force]

    The remaining h, v, x, u(t-1), force, and acceleration equations match ClofNetSeparateMessages.
    When rescalarize_per_layer is True, frame F_ij^l, radial r_ij^l, and s_ij^l are recomputed from x^l
    and a^l before each block; otherwise the initial frame F_ij^0 and radial r_ij^0 are reused.
    """

    message_initializers: nn.ModuleDict

    def _build_mp_blocks(self):
        in_mij = 2 * self.hidden_dim + 1 + self.in_dim_edge_mlp + self.hidden_dim
        blocks = []
        for _ in range(self.n_layers):
            blocks.append(RecurrentMultiHeadMPBlock(in_node_mlp_dim=2 * self.hidden_dim,
                                                    output_node_dim=self.hidden_dim,
                                                    hidden_dim=self.hidden_dim, in_mij_mlp_dim=in_mij,
                                                    act_fn_factory=self.act_fn_factory,
                                                    recurrent=self.recurrent,
                                                    resnet_style=self.resnet_style,
                                                    coords_weight=self.coords_weight,
                                                    attention=self.attention_enable, tanh=self.tanh,
                                                    clamp=self.clamp_enable,
                                                    vel_mlp_dim_out=self.vel_mlp_dim_out,
                                                    include_u_tm1=self.include_u_tm1,
                                                    include_v_branch=self.include_v_branch,
                                                    include_force=self.include_force,
                                                    n_layers_node_mlp=self.n_layers_node_mlp,
                                                    n_layers_gate_mlp=self.n_layers_gate_mlp,
                                                    n_layers_message_mlp=self.n_layers_message_mlp,
                                                    mlp_layer_norm=self.mlp_layer_norm))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels, use_neighbor_diff_norms,
                 frame_source, include_u_tm1, rescalarize_per_layer, include_v_branch, include_force, stress_out_dim,
                 m_ij_init, resnet_style, eps_mode, n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp,
                 normalize_delta_a, mlp_layer_norm):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels,
                         use_neighbor_diff_norms, frame_source, include_u_tm1, rescalarize_per_layer,
                         include_v_branch, include_force, stress_out_dim, m_ij_init, resnet_style, eps_mode,
                         n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp, normalize_delta_a,
                         mlp_layer_norm)
        if self.m_ij_init not in _VALID_M_IJ_INIT:
            raise ValueError("m_ij_init must be one of %s, got '%s'" % (sorted(_VALID_M_IJ_INIT), self.m_ij_init))
        initializers = {}
        for branch_name in self._message_branch_names():
            initializers[branch_name] = RecurrentMessageInitializer(
                hidden_dim=self.hidden_dim,
                fuse_hid_dim=self.in_dim_edge_mlp,
                m_ij_init=self.m_ij_init,
                n_layers_message_mlp=self.n_layers_message_mlp,
                act_fn_factory=self.act_fn_factory,
                mlp_layer_norm=self.mlp_layer_norm,
            )
        self.message_initializers = nn.ModuleDict(initializers)

    def _message_branch_names(self):
        """
        Enumerate active recurrent multi-head message branches.

        return: list[str]
            Branch keys used in message_initializers and block message_state dictionaries.
        """
        branch_names = ["a", "x"]
        if self.include_v_branch:
            branch_names.append("v")
        if self.include_u_tm1:
            branch_names.append("u_tm1")
        if self.include_force:
            branch_names.append("f")
        return branch_names

    def _initialize_message_state(self, h0, edge_index_tuple, radial, fuse_embed_0):
        """
        Initialize every active recurrent branch message before the first MP layer.

        Args:
            h0: torch.Tensor [N, H]
            edge_index_tuple: tuple
                (row, col) directed edges.
            radial: torch.Tensor [E, 1]
            fuse_embed_0: torch.Tensor [E, fuse_hid_dim]
        return: dict[str, torch.Tensor]
            Initial branch messages keyed by active branch name.
        """
        message_state = {}
        for branch_name in self._message_branch_names():
            message_state[branch_name] = self.message_initializers[branch_name](h0, edge_index_tuple, radial,
                                                                                fuse_embed_0)
        return message_state

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered, u_tm1_0, f0):
        h = h0; a = a0; v = v0; x = x0
        u_tm1 = u_tm1_0 if u_tm1_0 is not None else torch.zeros_like(x0)
        f = f0 if f0 is not None else torch.zeros_like(x0)
        fuse_embed_0 = self._fuse_embed(self._safe_a_for_fuse(a0), node_vec_chunks, edge_vec_chunks,
                                        edge_scalars, frame_e33, edge_index_tuple, f0)
        message_state = self._initialize_message_state(h0, edge_index_tuple, radial, fuse_embed_0)
        if not self.rescalarize_per_layer:
            _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff,
                                                   False, self.eps_mode)
            radial_l = radial
            frame_l = frame_e33
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for block in self.mp_blocks:
            if self.rescalarize_per_layer:
                radial_l, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x, self.enable_norm_diff,
                                                              False, self.eps_mode)
                frame_l = frame_basis_from_edges(a_ij, b_ij, c_ij)
            fuse_embed = self._fuse_embed(self._safe_a_for_fuse(a), node_vec_chunks, edge_vec_chunks,
                                          edge_scalars, frame_l, edge_index_tuple, f0)
            h, a, v, x, u_tm1, f, message_state, m_agg_last = block(
                h, edge_index_tuple, a_ij, b_ij, c_ij, radial_l, fuse_embed, message_state, a, v, x, u_tm1, f
            )
        return h, a, v, x, m_agg_last


class ClofNetExtraLayer(ClofNetBase):
    """
    Single message m_ij; three vectorization heads; phi_vel gates on a; fixed s_ij from zero acceleration.
    SingleVectorizeMPBlock message passing:
                            mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            μᵢ⁽ˡ⁺¹⁾ = ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, μᵢ⁽ˡ⁺¹⁾))
                            vᵢ⁽ˡ⁺¹⁾ = vᵢˡ + Δᵢ⁽ᵛ,θᵥ⁾(m⁽ˡ⁺¹⁾)          [only when include_v_branch]
                            xᵢ⁽ˡ⁺¹⁾ = xᵢˡ + Δᵢ⁽ˣ,θₓ⁾(m⁽ˡ⁺¹⁾)
                            fᵢ⁽ˡ⁺¹⁾ = fᵢˡ + Δᵢ⁽ᶠ,θᶠ⁾(m⁽ˡ⁺¹⁾)          [only when include_force]
                            ãᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + [φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾] + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
                                       + [φᶠ(hᵢ⁽ˡ⁺¹⁾) ⊙ fᵢ⁽ˡ⁺¹⁾]

    When rescalarize_per_layer is True, frame F_ij^l, radial r_ij^l and s_ij^l are all recomputed
    from the current x^l and a^l before each block.
    """

    def _build_mp_blocks(self):
        in_mij = 2 * self.hidden_dim + 1 + self.in_dim_edge_mlp
        blocks = []
        for _ in range(self.n_layers):
            blocks.append(SingleVectorizeMPBlock(in_node_mlp_dim=2 * self.hidden_dim, output_node_dim=self.hidden_dim,
                                                 hidden_dim=self.hidden_dim, in_mij_mlp_dim=in_mij,
                                                 act_fn_factory=self.act_fn_factory, recurrent=self.recurrent,
                                                 coords_weight=self.coords_weight, attention=self.attention_enable,
                                                 tanh=self.tanh, clamp=self.clamp_enable,
                                                 vel_mlp_dim_out=self.vel_mlp_dim_out,
                                                 include_u_tm1=self.include_u_tm1,
                                                 include_v_branch=self.include_v_branch,
                                                 include_force=self.include_force,
                                                 n_layers_node_mlp=self.n_layers_node_mlp,
                                                 n_layers_gate_mlp=self.n_layers_gate_mlp,
                                                 n_layers_message_mlp=self.n_layers_message_mlp,
                                                 mlp_layer_norm=self.mlp_layer_norm))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels, use_neighbor_diff_norms,
                 frame_source, include_u_tm1, rescalarize_per_layer, include_v_branch, include_force, stress_out_dim,
                 m_ij_init, resnet_style, eps_mode, n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp,
                 normalize_delta_a, mlp_layer_norm):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels,
                         use_neighbor_diff_norms, frame_source, include_u_tm1, rescalarize_per_layer,
                         include_v_branch, include_force, stress_out_dim, m_ij_init, resnet_style, eps_mode,
                         n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp, normalize_delta_a,
                         mlp_layer_norm)

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered, u_tm1_0, f0):
        h = h0; a = a0; v = v0; x = x0
        u_tm1 = u_tm1_0 if u_tm1_0 is not None else torch.zeros_like(x0)
        f = f0 if f0 is not None else torch.zeros_like(x0)
        if not self.rescalarize_per_layer:
            fuse_embed = self._fuse_embed(a0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                                          edge_index_tuple, f0)
            _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff, False, self.eps_mode)
            radial_l = radial
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for block in self.mp_blocks:
            if self.rescalarize_per_layer:
                radial_l, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x, self.enable_norm_diff, False, self.eps_mode)
                frame_l = frame_basis_from_edges(a_ij, b_ij, c_ij)
                fuse_embed = self._fuse_embed(self._safe_a_for_fuse(a), node_vec_chunks, edge_vec_chunks,
                                              edge_scalars, frame_l, edge_index_tuple, f0)
            h, a, v, x, u_tm1, f, m_agg_last = block(h, edge_index_tuple, a_ij, b_ij, c_ij, radial_l,
                                                      fuse_embed, a, v, x, u_tm1, f)
        return h, a, v, x, m_agg_last


class ClofNetDirectAcceleration(ClofNetBase):
    """
    Fixed s_ij, no extra velocity layer, no velocity/position update.
    Acceleration receives vectorized messages plus direct gates from h^l times frozen velocity and position.
                        mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                        aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾) + φᵥ(hᵢˡ) ⊙ vᵢ⁰ + φₓ(hᵢˡ) ⊙ xᵢ⁰
                        hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾))
    rescalarize_per_layer, include_v_branch, include_force have no effect on this variant.
    When include_force is True, node_force is projected into s_ij as an additional static term.
    """

    def _build_mp_blocks(self):
        in_mij = 2 * self.hidden_dim + 1 + self.in_dim_edge_mlp
        blocks = []
        for _ in range(self.n_layers):
            blocks.append(DirectAccelerationMPBlock(in_node_mlp_dim=2 * self.hidden_dim,
                                                    output_node_dim=self.hidden_dim,
                                                    hidden_dim=self.hidden_dim, in_mij_mlp_dim=in_mij,
                                                    act_fn_factory=self.act_fn_factory, recurrent=self.recurrent,
                                                    coords_weight=self.coords_weight,
                                                    attention=self.attention_enable, tanh=self.tanh,
                                                    clamp=self.clamp_enable,
                                                    vel_mlp_dim_out=self.vel_mlp_dim_out,
                                                    include_u_tm1=self.include_u_tm1,
                                                    n_layers_node_mlp=self.n_layers_node_mlp,
                                                    n_layers_gate_mlp=self.n_layers_gate_mlp,
                                                    n_layers_message_mlp=self.n_layers_message_mlp,
                                                    mlp_layer_norm=self.mlp_layer_norm))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels, use_neighbor_diff_norms,
                 frame_source, include_u_tm1, rescalarize_per_layer, include_v_branch, include_force, stress_out_dim,
                 m_ij_init, resnet_style, eps_mode, n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp,
                 normalize_delta_a, mlp_layer_norm):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels,
                         use_neighbor_diff_norms, frame_source, include_u_tm1, rescalarize_per_layer,
                         include_v_branch, include_force, stress_out_dim, m_ij_init, resnet_style, eps_mode,
                         n_layers_gate_mlp, n_layers_node_mlp, n_layers_message_mlp, normalize_delta_a,
                         mlp_layer_norm)

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered, u_tm1_0, f0):
        h = h0
        a = a0
        fuse_embed = self._fuse_embed(a0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                                      edge_index_tuple, f0)
        _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff, False, self.eps_mode)
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        u_tm1_pass = u_tm1_0 if u_tm1_0 is not None else torch.zeros_like(x0)
        for block in self.mp_blocks:
            h, a, m_agg_last = block(h, edge_index_tuple, a_ij, b_ij, c_ij, radial, fuse_embed, a, v0, x0,
                                     u_tm1_pass)
        return h, a, v0, x0, m_agg_last
