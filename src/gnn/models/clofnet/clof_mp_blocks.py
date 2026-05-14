"""
clof_mp_blocks.py

Message-passing blocks for ClofNet acceleration-target variants (recurrent, multi-head, single-message).
Common notation
                        Δᵢ⁽q,θ⁾(m) = w_c * (1 / |𝒩(i)|) * ∑ⱼ∈𝒩(i) Vectorize_θ(mᵢⱼ, Fᵢⱼ⁰)
                        q ∈ {a, v, x}
                        Fᵢⱼ⁰ = EquiFrame(xᵢ⁰, xⱼ⁰)
                        ẽᵢⱼ⁰ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(a⁰, Fᵢⱼ⁰)])
"""

import torch
from torch import nn
from src.gnn.models.clofnet.clof_aggregators import unsorted_segment_mean, unsorted_segment_sum

class RecurrentMPBlock(nn.Module):
    """
    Variant 6: recurrent edge messages m_ij^{l+1} = phi_m(h_i, h_j, r_ij, fused_edge, m_ij^l),
    node update, then acceleration update from vectorized m.

    RecurrentMPBlock message passing:
                            ẽᵢⱼˡ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(aˡ, Fᵢⱼ⁰)])
                            mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼˡ, mᵢⱼˡ)
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾))
                            aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾)
    """
    coords_weight: float
    skip_connection: bool
    attention: bool
    tanh: bool
    clamp: bool

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim, act_fn_factory, recurrent,
                 coords_weight, attention, tanh, clamp):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        self.node_h_mp_mlp = nn.Sequential(nn.Linear(in_node_mlp_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, output_node_dim))
        linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
        torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
        vec_layers = [nn.Linear(hidden_dim, hidden_dim), act_fn_factory(), linlayer_final]
        if self.tanh:
            vec_layers.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.vectorization_mlp = nn.Sequential(*vec_layers)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.message_mij_mlp = nn.Sequential(nn.Linear(in_mij_mlp_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, hidden_dim), act_fn_factory())
        self.layer_norm_h = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, m_prev, a):
        """
        One recurrent ClofNet-style block: message with m_prev, vectorize into delta-a, aggregate messages into h.

        Args:
            h: torch.Tensor
                Node embeddings [N, H].
            edge_index: tuple
                (row, col) directed edges.
            coord_diff_e3, coord_cross_e3, coord_vertical_e3: torch.Tensor
                Frame directions [E, 3].
            radial: torch.Tensor
                Squared distances [E, 1].
            fuse_embed: torch.Tensor
                Fused invariant edge embedding [E, F].
            m_prev: torch.Tensor
                Previous edge messages [E, H].
            a: torch.Tensor
                Current acceleration field [N, 3].
        return: (h_new, a_new, m_new, m_sum_per_node)
            h_new: torch.Tensor [N, H]
            a_new: torch.Tensor [N, 3]
            m_new: torch.Tensor [E, H]
            m_sum_per_node: torch.Tensor [N, H] sum of m_new over incoming edges to each node.
        """
        row, col = edge_index
        residue = h
        in_feat = torch.cat([h[row], h[col], radial, fuse_embed, m_prev], dim=1)
        m_new = self.message_mij_mlp(in_feat)
        if self.attention:
            m_new = m_new * self.att_mlp(m_new)
        agg = unsorted_segment_sum(m_new, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        h_upd = self.node_h_mp_mlp(in_node)
        if self.skip_connection:
            h_upd = h + h_upd
        h_new = self.layer_norm_h(residue + h_upd)
        coeff = self.vectorization_mlp(m_new)
        vec_e3 = (coord_diff_e3 * coeff[:, 0:1] + coord_cross_e3 * coeff[:, 1:2]
                  + coord_vertical_e3 * coeff[:, 2:3])
        if self.clamp:
            vec_e3 = torch.clamp(vec_e3, min=-100000, max=100000)
        delta_a = unsorted_segment_mean(vec_e3, row, num_segments=a.size(0)) * self.coords_weight
        a_new = a + delta_a
        m_sum_per_node = agg
        return h_new, a_new, m_new, m_sum_per_node


class MultiHeadMPBlock(nn.Module):
    """
    Three independent messages m^a, m^v, m^x; three vectorizations; phi_vel-style gates on a only.

                            mᵢⱼᵃ,⁽ˡ⁺¹⁾ = φₘᵃ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            mᵢⱼᵛ,⁽ˡ⁺¹⁾ = φₘᵛ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            mᵢⱼˣ,⁽ˡ⁺¹⁾ = φₘˣ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i)(mᵢⱼᵃ,⁽ˡ⁺¹⁾ + mᵢⱼᵛ,⁽ˡ⁺¹⁾ + mᵢⱼˣ,⁽ˡ⁺¹⁾)))
                            vᵢ⁽ˡ⁺¹⁾ = vᵢˡ + Δᵢ⁽ᵛ,θᵥ⁾(mᵛ,⁽ˡ⁺¹⁾)
                            xᵢ⁽ˡ⁺¹⁾ = xᵢˡ + Δᵢ⁽ˣ,θₓ⁾(mˣ,⁽ˡ⁺¹⁾)
                            ãᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(mᵃ,⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾ + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
    """

    coords_weight: float
    skip_connection: bool
    attention: bool
    tanh: bool
    clamp: bool

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim, act_fn_factory, recurrent,
                 coords_weight, attention, tanh, clamp, vel_mlp_dim_out):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        # Node MLP (2 layer)
        self.node_h_mp_mlp = nn.Sequential(nn.Linear(in_node_mlp_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, output_node_dim))
        # Vectorization MLPs
        def make_vec_mlp():
            linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
            torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
            layers = [nn.Linear(hidden_dim, hidden_dim), act_fn_factory(), linlayer_final]
            if self.tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)
        self.vectorization_mlp_a = make_vec_mlp()
        self.vectorization_mlp_v = make_vec_mlp()
        self.vectorization_mlp_x = make_vec_mlp()
        if self.tanh:
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        # Soft attention MLPs
        if self.attention:
            self.att_mlp_a = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.att_mlp_v = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.att_mlp_x = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # Message MLPs (3 layers)
        self.message_mij_mlp_a = nn.Sequential(nn.Linear(in_mij_mlp_dim, hidden_dim), act_fn_factory(),
                                                 nn.Linear(hidden_dim, hidden_dim), act_fn_factory(),
                                                 nn.Linear(hidden_dim, hidden_dim), act_fn_factory())
        self.message_mij_mlp_v = nn.Sequential(nn.Linear(in_mij_mlp_dim, hidden_dim), act_fn_factory(),
                                               nn.Linear(hidden_dim, hidden_dim), act_fn_factory(),
                                               nn.Linear(hidden_dim, hidden_dim), act_fn_factory())
        self.message_mij_mlp_x = nn.Sequential(nn.Linear(in_mij_mlp_dim, hidden_dim), act_fn_factory(),
                                               nn.Linear(hidden_dim, hidden_dim), act_fn_factory(),
                                               nn.Linear(hidden_dim, hidden_dim), act_fn_factory())
        # Gate mlps to add v, x in acceleration equivariant update (2 layer)
        self.phi_vel_v = nn.Sequential(nn.Linear(output_node_dim, hidden_dim), act_fn_factory(),
                                       nn.Linear(hidden_dim, vel_mlp_dim_out))
        self.phi_vel_x = nn.Sequential(nn.Linear(output_node_dim, hidden_dim), act_fn_factory(),
                                       nn.Linear(hidden_dim, vel_mlp_dim_out))
        self.layer_norm_h = nn.LayerNorm(hidden_dim)

    def _message_branch(self, mlp, h_row, h_col, radial, fuse_embed, att_mlp):
        m = mlp(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
        if self.attention:
            m = m * att_mlp(m)
        return m

    def _vectorize_delta(self, vec_mlp, m_ij, coord_diff_e3, coord_cross_e3, coord_vertical_e3, edge_index, field):
        row = edge_index[0]
        coeff = vec_mlp(m_ij)
        vec_e3 = (coord_diff_e3 * coeff[:, 0:1] + coord_cross_e3 * coeff[:, 1:2]
                  + coord_vertical_e3 * coeff[:, 2:3])
        if self.clamp:
            vec_e3 = torch.clamp(vec_e3, min=-100000, max=100000)
        delta = unsorted_segment_mean(vec_e3, row, num_segments=field.size(0)) * self.coords_weight
        return delta

    def forward(self, h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x):
        """
        One block: three messages, shared h update from sum of messages, v/x/a equivariant updates with phi_vel on a.

        Args:
            h: torch.Tensor [N, H]
            edge_index: tuple (row, col)
            coord_diff_e3, coord_cross_e3, coord_vertical_e3: torch.Tensor [E, 3]
            radial: torch.Tensor [E, 1]
            fuse_embed: torch.Tensor [E, F]
            a, v, x: torch.Tensor [N, 3] current equivariant fields
        return: (h_new, a_new, v_new, x_new, m_sum_per_node)
            m_sum_per_node: sum over edges of (m_a + m_v + m_x) aggregated to nodes [N, H].
        """
        row, col = edge_index
        residue = h
        h_row, h_col = h[row], h[col]
        if self.attention:
            m_a = self._message_branch(self.message_mij_mlp_a, h_row, h_col, radial, fuse_embed, self.att_mlp_a)
            m_v = self._message_branch(self.message_mij_mlp_v, h_row, h_col, radial, fuse_embed, self.att_mlp_v)
            m_x = self._message_branch(self.message_mij_mlp_x, h_row, h_col, radial, fuse_embed, self.att_mlp_x)
        else:
            m_a = self.message_mij_mlp_a(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
            m_v = self.message_mij_mlp_v(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
            m_x = self.message_mij_mlp_x(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
        m_sum = m_a + m_v + m_x
        agg = unsorted_segment_sum(m_sum, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        h_upd = self.node_h_mp_mlp(in_node)
        if self.skip_connection:
            h_upd = h + h_upd
        h_new = self.layer_norm_h(residue + h_upd)
        v_new = v + self._vectorize_delta(self.vectorization_mlp_v, m_v, coord_diff_e3, coord_cross_e3,
                                          coord_vertical_e3, edge_index, v)
        x_new = x + self._vectorize_delta(self.vectorization_mlp_x, m_x, coord_diff_e3, coord_cross_e3,
                                          coord_vertical_e3, edge_index, x)
        a_mid = a + self._vectorize_delta(self.vectorization_mlp_a, m_a, coord_diff_e3, coord_cross_e3,
                                          coord_vertical_e3, edge_index, a)
        gate_v = self.phi_vel_v(h_new)
        gate_x = self.phi_vel_x(h_new)
        a_new = a_mid + gate_v * v_new + gate_x * x_new
        return h_new, a_new, v_new, x_new, agg


class SingleVectorizeMPBlock(nn.Module):
    """
    Variant 3: single message m_ij; three vectorization heads from the same m; phi_vel gates on a.

    SingleVectorizeMPBlock message passing:
                            mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            μᵢ⁽ˡ⁺¹⁾ = ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾
                            hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, μᵢ⁽ˡ⁺¹⁾))
                            vᵢ⁽ˡ⁺¹⁾ = vᵢˡ + Δᵢ⁽ᵛ,θᵥ⁾(m⁽ˡ⁺¹⁾)
                            xᵢ⁽ˡ⁺¹⁾ = xᵢˡ + Δᵢ⁽ˣ,θₓ⁾(m⁽ˡ⁺¹⁾)
                            ãᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾ + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
    """

    coords_weight: float
    skip_connection: bool
    attention: bool
    tanh: bool
    clamp: bool

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim, act_fn_factory, recurrent,
                 coords_weight, attention, tanh, clamp, vel_mlp_dim_out):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        self.node_h_mp_mlp = nn.Sequential(nn.Linear(in_node_mlp_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, output_node_dim))
        # Vectorization mlps
        def make_vec_mlp():
            linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
            torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
            layers = [nn.Linear(hidden_dim, hidden_dim), act_fn_factory(), linlayer_final]
            if self.tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        self.vectorization_mlp_a = make_vec_mlp()
        self.vectorization_mlp_v = make_vec_mlp()
        self.vectorization_mlp_x = make_vec_mlp()
        # Attention mlp
        if self.tanh:
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # Message mlp (3 layer)
        self.message_mij_mlp = nn.Sequential(nn.Linear(in_mij_mlp_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, hidden_dim), act_fn_factory())
        # Gate mlps to add v, x in acceleration equivariant update (2 layer)
        self.phi_vel_v = nn.Sequential(nn.Linear(output_node_dim, hidden_dim), act_fn_factory(),
                                       nn.Linear(hidden_dim, vel_mlp_dim_out))
        self.phi_vel_x = nn.Sequential(nn.Linear(output_node_dim, hidden_dim), act_fn_factory(),
                                       nn.Linear(hidden_dim, vel_mlp_dim_out))
        self.layer_norm_h = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x):
        """
        One block with shared m_ij and three vectorizations.

        Args:
            h: torch.Tensor [N, H]
            edge_index: tuple (row, col)
            coord_diff_e3, coord_cross_e3, coord_vertical_e3: torch.Tensor [E, 3]
            radial: torch.Tensor [E, 1]
            fuse_embed: torch.Tensor [E, F]
            a, v, x: torch.Tensor [N, 3]
        return: (h_new, a_new, v_new, x_new, m_sum_per_node)
        """
        row, col = edge_index
        residue = h
        m_ij = self.message_mij_mlp(torch.cat([h[row], h[col], radial, fuse_embed], dim=1))
        if self.attention:
            m_ij = m_ij * self.att_mlp(m_ij)
        agg = unsorted_segment_sum(m_ij, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        h_upd = self.node_h_mp_mlp(in_node)
        if self.skip_connection:
            h_upd = h + h_upd
        h_new = self.layer_norm_h(residue + h_upd)

        def vec_delta(vec_mlp, field):
            coeff = vec_mlp(m_ij)
            vec_e3 = (coord_diff_e3 * coeff[:, 0:1] + coord_cross_e3 * coeff[:, 1:2]
                      + coord_vertical_e3 * coeff[:, 2:3])
            if self.clamp:
                vec_e3 = torch.clamp(vec_e3, min=-100000, max=100000)
            return unsorted_segment_mean(vec_e3, row, num_segments=field.size(0)) * self.coords_weight

        v_new = v + vec_delta(self.vectorization_mlp_v, v)
        x_new = x + vec_delta(self.vectorization_mlp_x, x)
        a_mid = a + vec_delta(self.vectorization_mlp_a, a)
        a_new = a_mid + self.phi_vel_v(h_new) * v_new + self.phi_vel_x(h_new) * x_new
        return h_new, a_new, v_new, x_new, agg


class DirectAccelerationMPBlock(nn.Module):
    """
    Single message and one acceleration vectorization; velocity and position are frozen inputs that affect
    acceleration only through s_ij and direct phi(h^l) gates.

                        mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                        aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾) + φᵥ(hᵢˡ) ⊙ vᵢ⁰ + φₓ(hᵢˡ) ⊙ xᵢ⁰
                        hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾))
    """

    coords_weight: float
    skip_connection: bool
    attention: bool
    tanh: bool
    clamp: bool

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim, act_fn_factory, recurrent,
                 coords_weight, attention, tanh, clamp, vel_mlp_dim_out):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        self.node_h_mp_mlp = nn.Sequential(nn.Linear(in_node_mlp_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, output_node_dim))
        linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
        torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
        # Vectorization MLP
        vec_layers = [nn.Linear(hidden_dim, hidden_dim), act_fn_factory(), linlayer_final]
        if self.tanh:
            vec_layers.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.vectorization_mlp = nn.Sequential(*vec_layers)
        # Attention MLP
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # Message MLP (3 layers)
        self.message_mij_mlp = nn.Sequential(nn.Linear(in_mij_mlp_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, hidden_dim), act_fn_factory(),
                                             nn.Linear(hidden_dim, hidden_dim), act_fn_factory())
        # Gate MLPs to add x, v to acceleration in equivariant update (2 layers)
        self.phi_vel = nn.Sequential(nn.Linear(output_node_dim, hidden_dim), act_fn_factory(),
                                     nn.Linear(hidden_dim, vel_mlp_dim_out))
        self.phi_pos = nn.Sequential(nn.Linear(output_node_dim, hidden_dim), act_fn_factory(),
                                     nn.Linear(hidden_dim, vel_mlp_dim_out))
        self.layer_norm_h = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x):
        """
        One block with direct frozen velocity/position injection into acceleration from h^l.

        Args:
            h: torch.Tensor [N, H]
            edge_index: tuple (row, col)
            coord_diff_e3, coord_cross_e3, coord_vertical_e3: torch.Tensor [E, 3]
            radial: torch.Tensor [E, 1]
            fuse_embed: torch.Tensor [E, F]
            a: torch.Tensor [N, 3]
            v: torch.Tensor [N, 3]
            x: torch.Tensor [N, 3]
        return: (h_new, a_new, m_sum_per_node)
            h_new: torch.Tensor [N, H]
            a_new: torch.Tensor [N, 3]
            m_sum_per_node: torch.Tensor [N, H]
        """
        row, col = edge_index
        residue = h
        m_ij = self.message_mij_mlp(torch.cat([h[row], h[col], radial, fuse_embed], dim=1))
        if self.attention:
            m_ij = m_ij * self.att_mlp(m_ij)
        coeff = self.vectorization_mlp(m_ij)
        vec_e3 = (coord_diff_e3 * coeff[:, 0:1] + coord_cross_e3 * coeff[:, 1:2]
                  + coord_vertical_e3 * coeff[:, 2:3])
        if self.clamp:
            vec_e3 = torch.clamp(vec_e3, min=-100000, max=100000)
        delta_a = unsorted_segment_mean(vec_e3, row, num_segments=a.size(0)) * self.coords_weight
        a_new = a + delta_a + self.phi_vel(h) * v + self.phi_pos(h) * x
        agg = unsorted_segment_sum(m_ij, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        h_upd = self.node_h_mp_mlp(in_node)
        if self.skip_connection:
            h_upd = h + h_upd
        h_new = self.layer_norm_h(residue + h_upd)
        return h_new, a_new, agg
