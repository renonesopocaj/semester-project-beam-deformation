"""
clofnet_gcl_beforeadaptation.py

ClofNet graph message passing block with complete local frames.
- `edge_model(...)` produces invariant edge embeddings `edge_feat`.
- `coord_mlp(edge_feat)` predicts three scalar coefficients: coff = [alpha_ij, beta_ij, gamma_ij]
- `coord_model(...)` reconstructs the vector update
      trans_ij = alpha_ij * a_ij + beta_ij * b_ij + gamma_ij * c_ij
    where this repository uses:
      a_ij = coord_diff
      b_ij = coord_cross
      c_ij = coord_vertical
- Those vectorized edge updates are averaged per node and added to the coordinates.
"""

from torch import nn
import torch
from src.gnn.models.clofnet.aggregators import unsorted_segment_mean, unsorted_segment_sum
from src.gnn.models.clofnet.clof_helpers import coord2localframe


class ClofMPBlock(nn.Module):

    def __init__(self, in_node_mlp_dim, output_node_dim, hidden_dim, in_mij_mlp_dim,
                 act_fn,
                 recurrent, coords_weight, attention, norm_diff, tanh, vel_mlp_dim_out, out_basis_dim, clamp,
                 enable_vel):
        """
        self, in_node_mlp_dim, output_node_dim, hidden_dim,
                 in_edge_mlp_dim,
                 act_fn=nn.ReLU(),
                 recurrent=True,
                 coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False,
                 out_basis_dim=3

        Args:
            input_dim_node: int
                Input node feature dimension.
            output_node_dim: int
                Output node feature dimension.
            hidden_dim: int
                Hidden feature dimension.
            edges_in_d: int
                Extra edge feature dimension.
            nodes_att_dim: int
                Extra node attribute dimension.
            act_fn: nn.Module
                Activation used by the MLPs.
            recurrent: bool
                Whether node updates are residual.
            coords_weight: float
                Scaling for vectorized coordinate updates.
            attention: bool
                Whether to gate edge features.
            norm_diff: bool
                Whether to normalize basis vectors using sqrt(radial) + 1 and the
                analogous norm for the cross-product direction.
            tanh: bool
                Whether to tanh-squash the three basis coefficients predicted by
                `coord_mlp`.
        """
        # in_edge_mlp_dim = input_dim_node * 2 + 1 + edges_in_d
        super().__init__()
        # input_edge = input_node_dim * 2
        # edge_coords_nf = 1
        # in_edge_mlp_dim = input_edge + edges_in_d + edge_coords_nf
        # in_node_mlp_dim = hidden_dim + input_node_dim + nodes_att_dim
        self.coords_weight, self.skip_connection, self.attention, self.norm_diff, self.tanh, self.clamp = \
            (coords_weight, recurrent, attention, norm_diff, tanh, clamp)
        self.node_h_mp_mlp = nn.Sequential(nn.Linear(in_node_mlp_dim, hidden_dim),
                                           act_fn,
                                           nn.Linear(hidden_dim, output_node_dim))

        # Initialize layer. Why no bias? Because it would be strange to add a bias to the coord_mlp
        linlayer_final_coordmlp = nn.Linear(hidden_dim, out_basis_dim, bias=False)
        torch.nn.init.xavier_uniform_(linlayer_final_coordmlp.weight, gain=0.001)
        vectorization_mlp = [nn.Linear(hidden_dim, hidden_dim),
                            act_fn,
                            linlayer_final_coordmlp] # (hidden_dim, out_basis_dim)
        # EGNN explicitly says adding tanh bounds the output, which improves stability but can hurt accuracy,
        # and they note they did not use it in their paper.
        if self.tanh:
            vectorization_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.vectorization_mlp = nn.Sequential(*vectorization_mlp)
        # Soft Attention MLP
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1),
                                         nn.Sigmoid())
        self.norm_diff = norm_diff
        self.enable_vel = enable_vel
        if self.enable_vel:
            self.vel_mlp = nn.Sequential(nn.Linear(in_node_mlp_dim, hidden_dim),
                                         act_fn,
                                         nn.Linear(hidden_dim, vel_mlp_dim_out))
        # Note: message_mij_mlp is overrided from the EgnnGCL init with 3 layers instead of 2
        self.message_mij_mlp = nn.Sequential(nn.Linear(in_mij_mlp_dim, hidden_dim),
                                             act_fn,
                                             nn.Linear(hidden_dim, hidden_dim),
                                             act_fn,
                                             nn.Linear(hidden_dim, hidden_dim),
                                             act_fn)
        self.layer_norm_h = nn.LayerNorm(hidden_dim)
        # diff: h^(l+1) used in MLP vel instead of h^(l)
        self.eq1 = False if eq_clof != 1 else True
        # diff: eq1 + m_ij explicitly passed
        self.eq2 = False if eq_clof != 2 else True
        # diff: eq2 + explicit updating of v via v_i := v_i + 1/N * \sum _{j \in N(i)} v_j
        self.eq3 = False if eq_clof != 3 else True
        # diff: eq2 + nonlinearities
        self.eq4 = False if eq_clof != 4 else True

    def coord2localframe(self, edge_index, coord_xi):
        """
        Construct the complete local frame per edge.
        This is the code counterpart of paper Eq. (2):
            a_ij = (x_i - x_j) / ||x_i - x_j||
            b_ij = (x_i x x_j) / ||x_i x x_j||
            c_ij = a_ij x b_ij

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord_xi: torch.Tensor
                Coordinates, shape [N, 3].

        return: (radial, coord_diff, coord_cross, coord_vertical)
            radial: torch.Tensor
                Squared distances, shape [E, 1].
            coord_diff: torch.Tensor
                First basis vector a_ij, shape [E, 3].
            coord_cross: torch.Tensor
                Second basis vector b_ij, shape [E, 3].
            coord_vertical: torch.Tensor
                Third basis vector c_ij, shape [E, 3].
        """
        radial, coord_diff, coord_cross, coord_vertical = coord2localframe(edge_index, coord_xi, self.norm_diff,
                                                                           crossdim1=False)
        return radial, coord_diff, coord_cross, coord_vertical

    def message_mij_model_forward(self, h_source, h_target, radial, eij_and_sij):
        """
        Compute edge features/messages from node states and radial geometry.
        Realizes
                    m_ij = phi_m(h_i, h_j, r_ij, e_ij)
        + optionallly attention i.e. \hat m_ij = m_ij * attention_weight(M) if M is all the m_ij forall i,j

        Args:
            h_source: torch.Tensor
                Source-node features, shape [E, input_node].
            h_target: torch.Tensor
                Target-node features, shape [E, input_node].
            radial: torch.Tensor
                Squared pairwise distances, shape [E, 1].
            eij_and_sij: torch.Tensor or None
                Optional extra edge attributes, shape [E, edges_in_d].

        return: torch.Tensor
            Edge features/messages, shape [E, hidden_nf].
        """
        # if edge_attr is None:
        #     in_feat = torch.cat([source, target, radial], dim=1)
        # else:
        in_feat = torch.cat([h_source, h_target, radial, eij_and_sij], dim=1)
        embed_mij = self.message_mij_mlp(in_feat)
        # out_pred: [E, hidden_nf] att_val = self.att_mlp(out_pred): [E, 1]
        # I do a term by term mul: this is basically a "soft importance" weight on each edge message, for which given
        # an edge e_ij i have a weight and if the weight is near to 0 the edge is ignored. Since the gate is
        # a scalar computed from invariant edge features, it does not break the equivariance argument.
        if self.attention:
            attention_forward = self.att_mlp(embed_mij)
            embed_mij = embed_mij * attention_forward

        return embed_mij

    def node_model_forward(self, h_node_features, edge_index, edge_attr):
        """
        Aggregate messages and update node embeddings, for non equivariant features.
        This is the code-level realization of the node update role of Eq. (17):
                            h_i^{l+1} = phi_h(h_i^l, sum_j m_ij)

        Args:
            h_node_features: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor
                Edge messages/features, shape [E, hidden_nf].

        return: (out, agg_input)
            out: torch.Tensor
                Updated node features, shape [N, output_nf].
            agg_input: torch.Tensor
                The concatenated tensor fed to the node MLP.
        """
        row = edge_index[0]
        # NOTE: this risks causing stuff blowing up (being too big in value)
        aggregation = unsorted_segment_sum(edge_attr, row, num_segments=h_node_features.size(0))
        # if node_attr is not None:
        in_feat = torch.cat([h_node_features, aggregation], dim=1)
        # else:
        #     in_feat = torch.cat([x, aggregation], dim=1)
        h_node_embedded = self.node_h_mp_mlp(in_feat)
        if self.skip_connection:
            h_node_embedded = h_node_features + h_node_embedded
        return h_node_embedded, in_feat

    def coord_mlp_forward(self, coord_xi, edge_index, coord_diff_aij, coord_cross_bij, coord_vertical_cij, mij_message):
        """
        Vectorize learned scalar coefficients back into a 3D coordinate update.
        This is the main implementation of the paper's vectorization idea in Eq. (8). The learned edge embedding
         `edge_feat` is mapped to three scalar coefficients:
                                coff_ij = [alpha_ij, beta_ij, gamma_ij]
        Then the update is reconstructed in the local frame:
                        trans_ij = alpha_ij * a_ij + beta_ij  * b_ij + gamma_ij * c_ij
        Finally, edgewise transforms are averaged per destination node and added to the coordinates.
        Overall:
                     coord_xi := coord_xi + 1/N * sum_j embedding(m_ij) * vectorize(vectorizatio_mlp(mij_message))

        Args:
            coord_xi: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord_diff_aij: torch.Tensor
                First basis vector a_ij, shape [E, 3].
            coord_cross_bij: torch.Tensor
                Second basis vector b_ij, shape [E, 3].
            coord_vertical_cij: torch.Tensor
                Third basis vector c_ij, shape [E, 3].
            mij_message: torch.Tensor
                Invariant edge embeddings/messages, shape [E, hidden_nf].

        return: torch.Tensor
            Updated coordinates, shape [N, 3].
        """
        row = edge_index[0]
        # MLP using m_ij to obtain (alpha_ij, beta_ij, gamma_ij) coefficients to reproject on (a_ij, b_ij, c_ij)
        vectorization_coeff = self.vectorization_mlp(mij_message) # [N, 3]
        # this is vectorization V_ij reconstructing from local frame to then compute coordinates update
        # vectorization block:  alpha_ij * a_ij + beta_ij * b_ij + gamma_ij * c_ij
        vectorize_mij = (coord_diff_aij * vectorization_coeff[:, :1] + coord_cross_bij * vectorization_coeff[:, 1:2]
                         + coord_vertical_cij * vectorization_coeff[:, 2:3]) # [N, 3]
        # stability choice
        vectorize_mij = torch.clamp(vectorize_mij, min=-100000, max=100000)
        # edge e is typically col[e] -> row[e], so you're saying to take all edges that come INSIDE a given i, for all
        # nodes i, and then use aggregation over neighbours only
        aggregation_vectorized = unsorted_segment_mean(vectorize_mij, row, num_segments=coord_xi.size(0)) # [N, 3]
        coord_xi = coord_xi + aggregation_vectorized * self.coords_weight # [N, 3]
        return coord_xi

    def forward(self, h, edge_index, coord_xi, vel, eij_sij_embed):
        """
        Run one ClofNet block.
        Flow relative to the paper:
            1. rebuild the local frame from current coordinates
            2. compute invariant edge messages
            3. vectorize those messages into coordinate updates
            4. add an implementation-specific velocity drift
            5. aggregate messages into node embeddings
        Notes on residuals:
        - `node_model(...)` may already add a residual when `recurrent=True`.
        - This class then adds another residual followed by `LayerNorm`.
                                        h = residue + h
        - This double-residual structure is preserved exactly from the original repository code.

        Args:
            h: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord_xi: torch.Tensor
                Coordinates, shape [N, 3].
            vel: torch.Tensor
                Velocities, shape [N, 3].
            eij_sij_embed: torch.Tensor or None
                Optional edge features, shape [E, edges_in_d].

        return: (h, coord, edge_attr)
            h: torch.Tensor
                Updated node features.
            coord: torch.Tensor
                Updated coordinates.
            edge_attr: torch.Tensor or None
                Returned unchanged for interface compatibility.
        """
        row, col = edge_index
        residue = h
        # reproject on new local frame F_ij
        r_ij, coord_diff_a_ij, coord_cross_b_ij, coord_vertical_c_ij = self.coord2localframe(edge_index, coord_xi)
        # embedding message m_ij <-- (h_i, h_j, r_ij, e_ij)
        embedded_mij = self.message_mij_model_forward(h[row], h[col], r_ij, eij_sij_embed)
        # embedding coord feature x_i <-- x_i + 1/N * sum_j m_ij * vectorize(coord_mlp(m_ij))
        coord_xi = self.coord_mlp_forward(coord_xi, edge_index, coord_diff_a_ij, coord_cross_b_ij, coord_vertical_c_ij,
                                          embedded_mij)
        # Node forward pass
        h, _ = self.node_model_forward(h, edge_index, embedded_mij)
        # Add velocity to coordinates
        if self.enable_vel:
            coord_xi = coord_xi + self.vel_mlp(torch.concat(h, eij_sij_embed)) * vel
        # node feat + add skip connection + layer norm, i.e. h <-- h + phi_h(h, sum_j m_ij)
        h = residue + h
        h = self.layer_norm_h(h)
        return h, coord_xi, eij_sij_embed