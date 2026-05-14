"""
clofnet.py

ClofNet variants for predicting acceleration (equivariant) and von Mises stress (scalar) from PyG batches.
Base class handles scalarization of stacked R^3 node/edge features using split_idx_node / split_idx_edge,
a fixed EquiFrame from input positions, fused edge embeddings, and a stress head.
Common notation
                        Δᵢ⁽q,θ⁾(m) = w_c * (1 / |𝒩(i)|) * ∑ⱼ∈𝒩(i) Vectorize_θ(mᵢⱼ, Fᵢⱼ⁰)
                        q ∈ {a, v, x}
                        Fᵢⱼ⁰ = EquiFrame(xᵢ⁰, xⱼ⁰)
                        ẽᵢⱼ⁰ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(a⁰, Fᵢⱼ⁰)])
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from torch import nn

from src.gnn.models.clofnet.scalarizer_localframe import (coord2localframe, frame_basis_from_edges, build_s_ij)
from src.gnn.models.clofnet.clof_helpers import (split_edge_features, split_node_features)
from src.gnn.models.clofnet.clof_mp_blocks import (DirectAccelerationMPBlock, MultiHeadMPBlock, RecurrentMPBlock,
                                                   SingleVectorizeMPBlock)


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
    act_fn_factory: object

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels):
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
        self.in_dim_edge_mlp = fuse_hid_dim
        num_node_triplets = split_idx_node // 3
        num_edge_triplets = split_idx_edge // 3
        pseudo_dim = 2
        self.s_ij_dim = pseudo_dim + (num_node_triplets + 1) * 6 + num_edge_triplets * 3
        self.fuse_in_dim = in_dim_edge_scalars + self.s_ij_dim
        self.embedding_node_mlp = nn.Linear(in_dim_node_embed, self.hidden_dim)
        self.fuse_mlp_embed = nn.Sequential(nn.Linear(self.fuse_in_dim, fuse_hid_dim), act_fn_factory(),
                                            nn.Linear(fuse_hid_dim, fuse_hid_dim), act_fn_factory())
        stress_in_dim = 2 * self.hidden_dim + 3
        self.stress_mlp = nn.Sequential(nn.Linear(stress_in_dim, self.hidden_dim), act_fn_factory(),
                                        nn.Linear(self.hidden_dim, 1))
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
                     edge_index_tuple, x_pos_centered):
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
                Initial equivariant position copy [N, 3] (centered per graph).
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
                Centered positions used only where subclasses need to refresh frame (unused for fixed frame).
        return: (h_L, a_L, v_L, x_L, m_agg_L)
            Final states and per-node aggregated messages for the stress head [N, H].
        """
        raise NotImplementedError

    def _fuse_embed(self, acceleration_n3, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                    edge_index_tuple):
        """
        Build s_ij from the current acceleration estimate and fuse with invariant edge scalars.

        Args:
            acceleration_n3: torch.Tensor [N, 3]
            node_vec_chunks: torch.Tensor [N, K, 3]
            edge_vec_chunks: torch.Tensor [E, K, 3]
            edge_scalars: torch.Tensor [E, S_e]
            frame_e33: torch.Tensor [E, 3, 3]
            edge_index_tuple: tuple
        return: torch.Tensor
            fuse_embed [E, fuse_hid_dim]
        """
        s_ij = build_s_ij(acceleration_n3, node_vec_chunks, edge_vec_chunks, frame_e33, edge_index_tuple)
        fused = torch.cat([edge_scalars, s_ij], dim=-1)
        return self.fuse_mlp_embed(fused)

    def _stress_out(self, h_final, m_agg, a_l, v_l, x_l):
        """
        Predict scalar stress from final h, aggregated messages, and norms of equivariant fields.

        Args:
            h_final: torch.Tensor [N, H]
            m_agg: torch.Tensor [N, H]
            a_l, v_l, x_l: torch.Tensor [N, 3]
        return: torch.Tensor [N, 1]
        """
        norm_a = torch.norm(a_l, dim=1, keepdim=True)
        norm_v = torch.norm(v_l, dim=1, keepdim=True)
        norm_x = torch.norm(x_l, dim=1, keepdim=True)
        inp = torch.cat([h_final, m_agg, norm_a, norm_v, norm_x], dim=-1)
        return self.stress_mlp(inp)

    def forward(self, batch):
        """
        Predict normalized acceleration (first 3) and stress (last 1) for each node.

        Args:
            batch: torch_geometric.data.Data or Batch
                Must provide x, edge_attr, edge_index, x_pos_t, x_vel_t, num_graphs, and optionally ptr.
        return: torch.Tensor
            Predictions [N, 4] = concat(acceleration, stress_scalar).
        """
        x_full = batch.x
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        x_pos = batch.x_pos_t
        v_phy = batch.x_vel_t
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
        if hasattr(batch, "ptr") and batch.ptr is not None:
            n_per = int(batch.ptr[1] - batch.ptr[0])
            num_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
        else:
            n_per = num_nodes
            num_graphs = 1
        # Centering graph (subtracting centroid)
        x_resh = x_pos.reshape(num_graphs, n_per, 3)
        centroid = torch.mean(x_resh, dim=1, keepdim=True)
        x_centered = (x_resh - centroid).reshape(-1, 3)
        x0 = x_centered
        # First projection coefficients
        radial, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_centered, self.enable_norm_diff, False)
        # Construct frame
        frame_e33 = frame_basis_from_edges(a_ij, b_ij, c_ij)
        # Run message passing loop (forward through all L layers)
        h_l, a_l, v_l, x_l, m_agg_l = self._run_mp_loop(h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks,
                                                        edge_scalars, frame_e33, radial, edge_index_tuple,
                                                        x_centered)
        # Final stress prediction
        stress = self._stress_out(h_l, m_agg_l, a_l, v_l, x_l)
        return torch.cat([a_l, stress], dim=-1)


class ClofNetRecurrent(ClofNetBase):
    """
    Recurrent messages version, i.e. the following.
    RecurrentMPBlock message passing:
                            ẽᵢⱼˡ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(aˡ, Fᵢⱼ⁰)])
                            mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼˡ)
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾))
                            aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾)
    s_ij recomputed each layer from current acceleration; only a is updated (no explicit v/x evolution).
    """

    def _build_mp_blocks(self):
        in_mij = 2 * self.hidden_dim + 1 + self.in_dim_edge_mlp + self.hidden_dim
        blocks = []
        for _ in range(self.n_layers):
            blocks.append(RecurrentMPBlock(in_node_mlp_dim=2 * self.hidden_dim, output_node_dim=self.hidden_dim,
                                           hidden_dim=self.hidden_dim, in_mij_mlp_dim=in_mij,
                                           act_fn_factory=self.act_fn_factory, recurrent=self.recurrent,
                                           coords_weight=self.coords_weight, attention=self.attention_enable,
                                           tanh=self.tanh, clamp=self.clamp_enable))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels)

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered):
        h = h0
        a = a0
        num_edges = int(edge_index_tuple[0].shape[0])
        # m_ij initialized as zero.
        m_ij = torch.zeros(num_edges, self.hidden_dim, device=h.device, dtype=h.dtype)
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for mp_block_l in self.mp_blocks:
            fuse_embed = self._fuse_embed(a, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                                         edge_index_tuple)
            _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff,
                                                   False)
            h, a, m_ij, m_agg_last = mp_block_l(h, edge_index_tuple, a_ij, b_ij, c_ij, radial, fuse_embed, m_ij, a)
        return h, a, v0, x0, m_agg_last


class ClofNetSeparateMessages(ClofNetBase):
    """
    Separate message MLPs for a, v, x channels; fused s_ij fixed from zero acceleration; phi_vel on a.
                            mᵢⱼᵃ,⁽ˡ⁺¹⁾ = φₘᵃ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            mᵢⱼᵛ,⁽ˡ⁺¹⁾ = φₘᵛ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            mᵢⱼˣ,⁽ˡ⁺¹⁾ = φₘˣ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i)(mᵢⱼᵃ,⁽ˡ⁺¹⁾ + mᵢⱼᵛ,⁽ˡ⁺¹⁾ + mᵢⱼˣ,⁽ˡ⁺¹⁾)))
                            vᵢ⁽ˡ⁺¹⁾ = vᵢˡ + Δᵢ⁽ᵛ,θᵥ⁾(mᵛ,⁽ˡ⁺¹⁾)
                            xᵢ⁽ˡ⁺¹⁾ = xᵢˡ + Δᵢ⁽ˣ,θₓ⁾(mˣ,⁽ˡ⁺¹⁾)
                            ãᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(mᵃ,⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾ + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
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
                                           vel_mlp_dim_out=self.vel_mlp_dim_out))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels)

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered):
        h = h0; a = a0; v = v0; x = x0
        fuse_embed = self._fuse_embed(a0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                                      edge_index_tuple)
        _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff, False)
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for block in self.mp_blocks:
            h, a, v, x, m_agg_last = block(h, edge_index_tuple, a_ij, b_ij, c_ij, radial, fuse_embed, a, v, x)
        return h, a, v, x, m_agg_last


class ClofNetExtraLayer(ClofNetBase):
    """
    Single message m_ij; three vectorization heads; phi_vel on a; fixed s_ij from zero acceleration.
    SingleVectorizeMPBlock message passing:
                            mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            μᵢ⁽ˡ⁺¹⁾ = ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, μᵢ⁽ˡ⁺¹⁾))
                            vᵢ⁽ˡ⁺¹⁾ = vᵢˡ + Δᵢ⁽ᵛ,θᵥ⁾(m⁽ˡ⁺¹⁾)
                            xᵢ⁽ˡ⁺¹⁾ = xᵢˡ + Δᵢ⁽ˣ,θₓ⁾(m⁽ˡ⁺¹⁾)
                            ãᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾ + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
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
                                                 vel_mlp_dim_out=self.vel_mlp_dim_out))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels)

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered):
        h = h0; a = a0; v = v0; x = x0
        fuse_embed = self._fuse_embed(a0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                                      edge_index_tuple)
        _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff, False)
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for block in self.mp_blocks:
            h, a, v, x, m_agg_last = block(h, edge_index_tuple, a_ij, b_ij, c_ij, radial, fuse_embed, a, v, x)
        return h, a, v, x, m_agg_last


class ClofNetDirectAcceleration(ClofNetBase):
    """
    Fixed s_ij, no extra velocity layer, no velocity/position update.
    Acceleration receives vectorized messages plus direct gates from h^l times frozen velocity and position.
                        mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                        aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾) + φᵥ(hᵢˡ) ⊙ vᵢ⁰ + φₓ(hᵢˡ) ⊙ xᵢ⁰
                        hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾))
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
                                                    vel_mlp_dim_out=self.vel_mlp_dim_out))
        return nn.ModuleList(blocks)

    def __init__(self, in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                 coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable, vel_mlp_dim_out,
                 split_idx_node, split_idx_edge, node_channels, edge_channels):
        super().__init__(in_dim_node_embed, in_dim_edge_scalars, hidden_dim, fuse_hid_dim, act_fn_factory, n_layers,
                         coords_weight, recurrent, enable_norm_diff, tanh, attention_enable, clamp_enable,
                         vel_mlp_dim_out, split_idx_node, split_idx_edge, node_channels, edge_channels)

    def _run_mp_loop(self, h0, a0, v0, x0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33, radial,
                     edge_index_tuple, x_pos_centered):
        h = h0
        a = a0
        fuse_embed = self._fuse_embed(a0, node_vec_chunks, edge_vec_chunks, edge_scalars, frame_e33,
                                      edge_index_tuple)
        _, a_ij, b_ij, c_ij = coord2localframe(edge_index_tuple, x_pos_centered, self.enable_norm_diff, False)
        m_agg_last = torch.zeros(h.shape[0], self.hidden_dim, device=h.device, dtype=h.dtype)
        for block in self.mp_blocks:
            h, a, m_agg_last = block(h, edge_index_tuple, a_ij, b_ij, c_ij, radial, fuse_embed, a, v0, x0)
        return h, a, v0, x0, m_agg_last
