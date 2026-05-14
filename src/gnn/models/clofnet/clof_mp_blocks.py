"""
clof_mp_blocks.py

Message-passing blocks for ClofNet acceleration-target variants (recurrent, multi-head, single-message).
Common notation
                        Δᵢ⁽q,θ⁾(m) = w_c * (1 / |𝒩(i)|) * ∑ⱼ∈𝒩(i) Vectorize_θ(mᵢⱼ, Fᵢⱼ⁰)
                        q ∈ {a, v, x}
                        Fᵢⱼ⁰ = EquiFrame(xᶜᵉⁿᵗᵢ⁰, xᶜᵉⁿᵗⱼ⁰)
                        ẽᵢⱼ⁰ = ψ_fuse([eᵢⱼˢᶜᵃˡᵃʳ, sᵢⱼ(a⁰, Fᵢⱼ⁰)])
"""

import torch
from torch import nn
from src.gnn.models.clofnet.clof_aggregators import unsorted_segment_mean, unsorted_segment_sum


def make_mlp(n_layers, in_dim, hidden_dim, out_dim, act_fn_factory, final_activation):
    """
    Build a sequential MLP with n_layers linear layers.

    Args:
        n_layers: int
            Total number of Linear layers.
        in_dim: int
            Input dimension of the first layer.
        hidden_dim: int
            Width of all intermediate layers.
        out_dim: int
            Output dimension of the last layer.
        act_fn_factory: callable
            Zero-argument factory returning an activation module.
        final_activation: bool
            Whether to append an activation after the last linear layer.
    return: nn.Sequential
    """
    layers = []
    for i in range(n_layers):
        d_in = in_dim if i == 0 else hidden_dim
        d_out = out_dim if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1 or final_activation:
            layers.append(act_fn_factory())
    return nn.Sequential(*layers)

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
                 coords_weight, attention, tanh, clamp, n_layers_node_mlp, n_layers_gate_mlp, n_layers_message_mlp):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        self.node_h_mp_mlp = make_mlp(n_layers_node_mlp, in_node_mlp_dim, hidden_dim, output_node_dim, act_fn_factory, False)
        linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
        torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
        vec_layers = [nn.Linear(hidden_dim, hidden_dim), act_fn_factory(), linlayer_final]
        if self.tanh:
            vec_layers.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.vectorization_mlp = nn.Sequential(*vec_layers)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.message_mij_mlp = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
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

    When include_u_tm1 is True, an additional channel is added:
                            mᵢⱼᵘᵗᵐ¹,⁽ˡ⁺¹⁾ = φₘᵘᵗᵐ¹(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            u_tm1ᵢ⁽ˡ⁺¹⁾ = u_tm1ᵢˡ + Δᵢ⁽ᵘᵗᵐ¹,θᵤₜₘ₁⁾(mᵘᵗᵐ¹,⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + [φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾] + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
                                       + φᵤₜₘ₁(hᵢ⁽ˡ⁺¹⁾) ⊙ u_tm1ᵢ⁽ˡ⁺¹⁾ + [φᶠ(hᵢ⁽ˡ⁺¹⁾) ⊙ fᵢ⁽ˡ⁺¹⁾]
    """

    coords_weight: float
    skip_connection: bool
    attention: bool
    tanh: bool
    clamp: bool
    include_u_tm1: bool
    include_v_branch: bool
    include_force: bool

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim, act_fn_factory, recurrent,
                 coords_weight, attention, tanh, clamp, vel_mlp_dim_out, include_u_tm1, include_v_branch,
                 include_force, n_layers_node_mlp, n_layers_gate_mlp, n_layers_message_mlp):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        self.include_u_tm1 = include_u_tm1
        self.include_v_branch = include_v_branch
        self.include_force = include_force
        # Node MLP
        self.node_h_mp_mlp = make_mlp(n_layers_node_mlp, in_node_mlp_dim, hidden_dim, output_node_dim, act_fn_factory, False)
        # Vectorization MLPs
        def make_vec_mlp():
            linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
            torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
            layers = [nn.Linear(hidden_dim, hidden_dim), act_fn_factory(), linlayer_final]
            if self.tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)
        self.vectorization_mlp_a = make_vec_mlp()
        self.vectorization_mlp_x = make_vec_mlp()
        if self.tanh:
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        # Soft attention MLPs
        if self.attention:
            self.att_mlp_a = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.att_mlp_x = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # Message MLPs
        self.message_mij_mlp_a = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
        self.message_mij_mlp_x = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
        # Gate mlp to add x in acceleration equivariant update
        self.phi_vel_x = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        self.layer_norm_h = nn.LayerNorm(hidden_dim)
        # v branch (only built when flag is on)
        if self.include_v_branch:
            self.vectorization_mlp_v = make_vec_mlp()
            if self.attention:
                self.att_mlp_v = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.message_mij_mlp_v = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
            self.phi_vel_v = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        # u(t-1) channel (only built when flag is on)
        if self.include_u_tm1:
            self.message_mij_mlp_u_tm1 = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
            if self.attention:
                self.att_mlp_u_tm1 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.vectorization_mlp_u_tm1 = make_vec_mlp()
            self.phi_vel_u_tm1 = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        # force channel (only built when flag is on)
        if self.include_force:
            self.vectorization_mlp_f = make_vec_mlp()
            if self.attention:
                self.att_mlp_f = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.message_mij_mlp_f = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
            self.phi_vel_f = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)

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

    def forward(self, h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x,
                u_tm1, f):
        """
        One block: independent messages per channel, shared h update, equivariant updates for a/x and
        optionally v, u_tm1, f.

        Args:
            h: torch.Tensor [N, H]
            edge_index: tuple (row, col)
            coord_diff_e3, coord_cross_e3, coord_vertical_e3: torch.Tensor [E, 3]
            radial: torch.Tensor [E, 1]
            fuse_embed: torch.Tensor [E, F]
            a, v, x: torch.Tensor [N, 3] current equivariant fields
            u_tm1: torch.Tensor [N, 3] previous-step displacement field (ignored when include_u_tm1 is False)
            f: torch.Tensor [N, 3] per-node force field (ignored when include_force is False)
        return: (h_new, a_new, v_new, x_new, u_tm1_new, f_new, m_sum_per_node)
            m_sum_per_node: sum over edges of aggregated messages [N, H].
        """
        row, col = edge_index
        residue = h
        h_row, h_col = h[row], h[col]
        if self.attention:
            m_a = self._message_branch(self.message_mij_mlp_a, h_row, h_col, radial, fuse_embed, self.att_mlp_a)
            m_x = self._message_branch(self.message_mij_mlp_x, h_row, h_col, radial, fuse_embed, self.att_mlp_x)
        else:
            m_a = self.message_mij_mlp_a(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
            m_x = self.message_mij_mlp_x(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
        m_sum = m_a + m_x
        if self.include_v_branch:
            if self.attention:
                m_v = self._message_branch(self.message_mij_mlp_v, h_row, h_col, radial, fuse_embed, self.att_mlp_v)
            else:
                m_v = self.message_mij_mlp_v(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
            m_sum = m_sum + m_v
        if self.include_u_tm1:
            if self.attention:
                m_u_tm1 = self._message_branch(self.message_mij_mlp_u_tm1, h_row, h_col, radial, fuse_embed,
                                               self.att_mlp_u_tm1)
            else:
                m_u_tm1 = self.message_mij_mlp_u_tm1(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
            m_sum = m_sum + m_u_tm1
        if self.include_force:
            if self.attention:
                m_f = self._message_branch(self.message_mij_mlp_f, h_row, h_col, radial, fuse_embed, self.att_mlp_f)
            else:
                m_f = self.message_mij_mlp_f(torch.cat([h_row, h_col, radial, fuse_embed], dim=1))
            m_sum = m_sum + m_f
        agg = unsorted_segment_sum(m_sum, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        h_upd = self.node_h_mp_mlp(in_node)
        if self.skip_connection:
            h_upd = h + h_upd
        h_new = self.layer_norm_h(residue + h_upd)
        x_new = x + self._vectorize_delta(self.vectorization_mlp_x, m_x, coord_diff_e3, coord_cross_e3,
                                          coord_vertical_e3, edge_index, x)
        a_mid = a + self._vectorize_delta(self.vectorization_mlp_a, m_a, coord_diff_e3, coord_cross_e3,
                                          coord_vertical_e3, edge_index, a)
        a_new = a_mid + self.phi_vel_x(h_new) * x_new
        if self.include_v_branch:
            v_new = v + self._vectorize_delta(self.vectorization_mlp_v, m_v, coord_diff_e3, coord_cross_e3,
                                              coord_vertical_e3, edge_index, v)
            a_new = a_new + self.phi_vel_v(h_new) * v_new
        else:
            v_new = v
        if self.include_u_tm1:
            u_tm1_new = u_tm1 + self._vectorize_delta(self.vectorization_mlp_u_tm1, m_u_tm1, coord_diff_e3,
                                                      coord_cross_e3, coord_vertical_e3, edge_index, u_tm1)
            a_new = a_new + self.phi_vel_u_tm1(h_new) * u_tm1_new
        else:
            u_tm1_new = u_tm1
        if self.include_force:
            f_new = f + self._vectorize_delta(self.vectorization_mlp_f, m_f, coord_diff_e3, coord_cross_e3,
                                              coord_vertical_e3, edge_index, f)
            a_new = a_new + self.phi_vel_f(h_new) * f_new
        else:
            f_new = f
        return h_new, a_new, v_new, x_new, u_tm1_new, f_new, agg


class SingleVectorizeMPBlock(nn.Module):
    """
    Variant 3: single message m_ij; three vectorization heads from the same m; phi_vel gates on a.

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

    When include_u_tm1 is True, an additional dedicated message drives a u(t-1) update:
                            mᵢⱼᵘᵗᵐ¹,⁽ˡ⁺¹⁾ = φₘᵘᵗᵐ¹(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                            u_tm1ᵢ⁽ˡ⁺¹⁾ = u_tm1ᵢˡ + Δᵢ⁽ᵘᵗᵐ¹,θᵤₜₘ₁⁾(mᵘᵗᵐ¹,⁽ˡ⁺¹⁾)
                            aᵢ⁽ˡ⁺¹⁾ = ãᵢ⁽ˡ⁺¹⁾ + [φᵥ(hᵢ⁽ˡ⁺¹⁾) ⊙ vᵢ⁽ˡ⁺¹⁾] + φₓ(hᵢ⁽ˡ⁺¹⁾) ⊙ xᵢ⁽ˡ⁺¹⁾
                                       + φᵤₜₘ₁(hᵢ⁽ˡ⁺¹⁾) ⊙ u_tm1ᵢ⁽ˡ⁺¹⁾ + [φᶠ(hᵢ⁽ˡ⁺¹⁾) ⊙ fᵢ⁽ˡ⁺¹⁾]

    The force channel shares the main message m_ij (no dedicated message MLP, analogous to v/x in this variant).
    """

    coords_weight: float
    skip_connection: bool
    attention: bool
    tanh: bool
    clamp: bool
    include_u_tm1: bool
    include_v_branch: bool
    include_force: bool

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim, act_fn_factory, recurrent,
                 coords_weight, attention, tanh, clamp, vel_mlp_dim_out, include_u_tm1, include_v_branch,
                 include_force, n_layers_node_mlp, n_layers_gate_mlp, n_layers_message_mlp):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        self.include_u_tm1 = include_u_tm1
        self.include_v_branch = include_v_branch
        self.include_force = include_force
        self.node_h_mp_mlp = make_mlp(n_layers_node_mlp, in_node_mlp_dim, hidden_dim, output_node_dim, act_fn_factory, False)
        # Vectorization mlps
        def make_vec_mlp():
            linlayer_final = nn.Linear(hidden_dim, 3, bias=False)
            torch.nn.init.xavier_uniform_(linlayer_final.weight, gain=0.001)
            layers = [nn.Linear(hidden_dim, hidden_dim), act_fn_factory(), linlayer_final]
            if self.tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        self.vectorization_mlp_a = make_vec_mlp()
        self.vectorization_mlp_x = make_vec_mlp()
        # Attention mlp
        if self.tanh:
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # Message mlp
        self.message_mij_mlp = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
        # Gate mlp to add x in acceleration equivariant update
        self.phi_vel_x = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        self.layer_norm_h = nn.LayerNorm(hidden_dim)
        # v branch (only built when flag is on)
        if self.include_v_branch:
            self.vectorization_mlp_v = make_vec_mlp()
            self.phi_vel_v = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        # u(t-1) channel (only built when flag is on)
        if self.include_u_tm1:
            self.message_mij_mlp_u_tm1 = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
            if self.attention:
                self.att_mlp_u_tm1 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.vectorization_mlp_u_tm1 = make_vec_mlp()
            self.phi_vel_u_tm1 = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        # force channel (only built when flag is on; shares the main m_ij — no dedicated message MLP)
        if self.include_force:
            self.vectorization_mlp_f = make_vec_mlp()
            self.phi_vel_f = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)

    def forward(self, h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x,
                u_tm1, f):
        """
        One block with shared m_ij, independent vectorizations, and optional v / u(t-1) / force channels.

        Args:
            h: torch.Tensor [N, H]
            edge_index: tuple (row, col)
            coord_diff_e3, coord_cross_e3, coord_vertical_e3: torch.Tensor [E, 3]
            radial: torch.Tensor [E, 1]
            fuse_embed: torch.Tensor [E, F]
            a, v, x: torch.Tensor [N, 3]
            u_tm1: torch.Tensor [N, 3] previous-step displacement field (ignored when include_u_tm1 is False)
            f: torch.Tensor [N, 3] per-node force field (ignored when include_force is False)
        return: (h_new, a_new, v_new, x_new, u_tm1_new, f_new, m_sum_per_node)
        """
        row, col = edge_index
        residue = h
        m_ij = self.message_mij_mlp(torch.cat([h[row], h[col], radial, fuse_embed], dim=1))
        if self.attention:
            m_ij = m_ij * self.att_mlp(m_ij)
        agg = unsorted_segment_sum(m_ij, row, num_segments=h.size(0))
        if self.include_u_tm1:
            m_u_tm1 = self.message_mij_mlp_u_tm1(torch.cat([h[row], h[col], radial, fuse_embed], dim=1))
            if self.attention:
                m_u_tm1 = m_u_tm1 * self.att_mlp_u_tm1(m_u_tm1)
            agg = agg + unsorted_segment_sum(m_u_tm1, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        h_upd = self.node_h_mp_mlp(in_node)
        if self.skip_connection:
            h_upd = h + h_upd
        h_new = self.layer_norm_h(residue + h_upd)

        def vec_delta(vec_mlp, field, msg):
            coeff = vec_mlp(msg)
            vec_e3 = (coord_diff_e3 * coeff[:, 0:1] + coord_cross_e3 * coeff[:, 1:2]
                      + coord_vertical_e3 * coeff[:, 2:3])
            if self.clamp:
                vec_e3 = torch.clamp(vec_e3, min=-100000, max=100000)
            return unsorted_segment_mean(vec_e3, row, num_segments=field.size(0)) * self.coords_weight

        x_new = x + vec_delta(self.vectorization_mlp_x, x, m_ij)
        a_mid = a + vec_delta(self.vectorization_mlp_a, a, m_ij)
        a_new = a_mid + self.phi_vel_x(h_new) * x_new
        if self.include_v_branch:
            v_new = v + vec_delta(self.vectorization_mlp_v, v, m_ij)
            a_new = a_new + self.phi_vel_v(h_new) * v_new
        else:
            v_new = v
        if self.include_u_tm1:
            u_tm1_new = u_tm1 + vec_delta(self.vectorization_mlp_u_tm1, u_tm1, m_u_tm1)
            a_new = a_new + self.phi_vel_u_tm1(h_new) * u_tm1_new
        else:
            u_tm1_new = u_tm1
        if self.include_force:
            f_new = f + vec_delta(self.vectorization_mlp_f, f, m_ij)
            a_new = a_new + self.phi_vel_f(h_new) * f_new
        else:
            f_new = f
        return h_new, a_new, v_new, x_new, u_tm1_new, f_new, agg


class DirectAccelerationMPBlock(nn.Module):
    """
    Single message and one acceleration vectorization; velocity and position are frozen inputs that affect
    acceleration only through s_ij and direct phi(h^l) gates.

                        mᵢⱼ⁽ˡ⁺¹⁾ = φₘ(hᵢˡ, hⱼˡ, rᵢⱼ, ẽᵢⱼ⁰)
                        aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾) + φᵥ(hᵢˡ) ⊙ vᵢ⁰ + φₓ(hᵢˡ) ⊙ xᵢ⁰
                        hᵢ⁽ˡ⁺¹⁾ = LN(κ * hᵢˡ + φₕ(hᵢˡ, ∑ⱼ∈𝒩(i) mᵢⱼ⁽ˡ⁺¹⁾))

    When include_u_tm1 is True, a frozen u(t-1) gate is added:
                        aᵢ⁽ˡ⁺¹⁾ = aᵢˡ + Δᵢ⁽ᵃ,θₐ⁾(m⁽ˡ⁺¹⁾) + φᵥ(hᵢˡ) ⊙ vᵢ⁰ + φₓ(hᵢˡ) ⊙ xᵢ⁰
                                   + φᵤₜₘ₁(hᵢˡ) ⊙ u_tm1ᵢ⁰
    """

    coords_weight: float
    skip_connection: bool
    attention: bool
    tanh: bool
    clamp: bool
    include_u_tm1: bool

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim, act_fn_factory, recurrent,
                 coords_weight, attention, tanh, clamp, vel_mlp_dim_out, include_u_tm1, n_layers_node_mlp,
                 n_layers_gate_mlp, n_layers_message_mlp):
        super().__init__()
        self.coords_weight = coords_weight
        self.skip_connection = recurrent
        self.attention = attention
        self.tanh = tanh
        self.clamp = clamp
        self.include_u_tm1 = include_u_tm1
        self.node_h_mp_mlp = make_mlp(n_layers_node_mlp, in_node_mlp_dim, hidden_dim, output_node_dim, act_fn_factory, False)
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
        # Message MLP
        self.message_mij_mlp = make_mlp(n_layers_message_mlp, in_mij_mlp_dim, hidden_dim, hidden_dim, act_fn_factory, True)
        # Gate MLPs to add x, v to acceleration in equivariant update
        self.phi_vel = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        self.phi_pos = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)
        self.layer_norm_h = nn.LayerNorm(hidden_dim)
        # Gate MLP for frozen u(t-1) (only built when flag is on)
        if self.include_u_tm1:
            self.phi_u_tm1 = make_mlp(n_layers_gate_mlp, output_node_dim, hidden_dim, vel_mlp_dim_out, act_fn_factory, False)

    def forward(self, h, edge_index, coord_diff_e3, coord_cross_e3, coord_vertical_e3, radial, fuse_embed, a, v, x,
                u_tm1):
        """
        One block with direct frozen velocity/position injection into acceleration from h^l,
        plus optional frozen u(t-1) gate.

        Args:
            h: torch.Tensor [N, H]
            edge_index: tuple (row, col)
            coord_diff_e3, coord_cross_e3, coord_vertical_e3: torch.Tensor [E, 3]
            radial: torch.Tensor [E, 1]
            fuse_embed: torch.Tensor [E, F]
            a: torch.Tensor [N, 3]
            v: torch.Tensor [N, 3]
            x: torch.Tensor [N, 3]
            u_tm1: torch.Tensor [N, 3] frozen previous-step displacement (ignored when include_u_tm1 is False)
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
        if self.include_u_tm1:
            a_new = a_new + self.phi_u_tm1(h) * u_tm1
        agg = unsorted_segment_sum(m_ij, row, num_segments=h.size(0))
        in_node = torch.cat([h, agg], dim=1)
        h_upd = self.node_h_mp_mlp(in_node)
        if self.skip_connection:
            h_upd = h + h_upd
        h_new = self.layer_norm_h(residue + h_upd)
        return h_new, a_new, agg
