import torch
from torch import nn
from src.gnn.models.clofnet.gcl.helpers_gcl import unsorted_segment_mean, unsorted_segment_sum


class EgnnGCL(nn.Module):
    """
    Equivariant graph convolution layer with EGNN-style radial coordinate update.

    Relation to the paper:
    - This block is closest to the EGNN baseline summarized in Eq. (1), which
      the ClofNet paper explicitly references before introducing complete local
      frames.
    - It is not yet the ClofNet block because coordinate updates are restricted
      to the normalized relative direction `coord_diff` rather than the full
      local frame `(a, b, c)` of Eq. (2).

    Core equations represented here:
        m_ij = phi_m(h_i, h_j, r_ij, e_ij)
        x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        h_i' = phi_h(h_i, sum_j m_ij)

    Here:
        r_ij = ||x_i - x_j||^2
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False, out_basis_dim=1):
        """
        Args:
            input_nf: int
                Input node feature dimension.
            output_nf: int
                Output node feature dimension.
            hidden_nf: int
                Hidden dimension.
            edges_in_d: int
                Extra edge feature dimension.
            nodes_att_dim: int
                Extra node attribute dimension concatenated before node update.
            act_fn: nn.Module
                Activation function used by the MLPs.
            recurrent: bool
                Whether node updates are residual.
            coords_weight: float
                Scaling factor applied to the aggregated coordinate update.
            attention: bool
                Whether to gate edge features with a learned scalar attention.
            clamp: bool
                Legacy flag kept for compatibility. Coordinate updates are always
                clamped in this implementation.
            norm_diff: bool
                Whether to normalize coordinate differences by sqrt(radial) + 1.
            tanh: bool
                Whether to squash the coordinate coefficient with tanh.
            out_basis_dim: int
                Output dimension of `coord_mlp`. For EGNN-style updates this is 1;
                for ClofNet frame-based vectorization this becomes 3.
        """
        super().__init__()
        input_edge = input_nf * 2
        edge_coords_nf = 1

        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.clamp = clamp

        self.edge_mlp = nn.Sequential(nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
                                      act_fn,
                                      nn.Linear(hidden_nf, hidden_nf),
                                      act_fn,)

        self.node_mlp = nn.Sequential(nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                                      act_fn,
                                      nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, out_basis_dim, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]

        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3

        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1),
                                         nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        """
        Compute edge features/messages from node states and radial geometry.

        Args:
            source: torch.Tensor
                Source-node features, shape [E, input_nf].
            target: torch.Tensor
                Target-node features, shape [E, input_nf].
            radial: torch.Tensor
                Squared pairwise distances, shape [E, 1].
            edge_attr: torch.Tensor or None
                Optional extra edge attributes, shape [E, edges_in_d].

        return: torch.Tensor
            Edge features/messages, shape [E, hidden_nf].
        """
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)

        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        Aggregate messages and update node embeddings.

        This is the code-level realization of the node update role of Eq. (17):
            h_i^{l+1} = phi_h(h_i^l, sum_j m_ij)

        Args:
            x: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor
                Edge messages/features, shape [E, hidden_nf].
            node_attr: torch.Tensor or None
                Optional extra node attributes concatenated before the node MLP.

            return: (out, agg_input)
                out: torch.Tensor
                    Updated node features, shape [N, output_nf].
                agg_input: torch.Tensor
                    The concatenated tensor fed to the node MLP.
        """
        row, col = edge_index
        del col

        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)

        out = self.node_mlp(agg)

        if self.recurrent:
            out = x + out

        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        Update coordinates along the relative direction only.

        This is the EGNN-style coordinate update corresponding to Eq. (1):
            x_i <- x_i + sum_j (x_i - x_j) * phi_x(m_ij)

        In this implementation:
            trans_ij = coord_diff_ij * coord_mlp(edge_feat_ij)

        and nodewise updates are the mean over incoming edge transforms.

        Args:
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord_diff: torch.Tensor
                Relative directions, shape [E, 3].
            edge_feat: torch.Tensor
                Edge messages/features, shape [E, hidden_nf].

            return: torch.Tensor
                Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)

        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        """
        Compute squared distances and relative directions for each edge.

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].

            return: (radial, coord_diff)
                radial: torch.Tensor
                    Squared distances, shape [E, 1].
                coord_diff: torch.Tensor
                    Relative directions x_i - x_j, shape [E, 3].
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        """
        Run one equivariant message-passing layer.

        Args:
            h: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            edge_attr: torch.Tensor or None
                Optional edge features, shape [E, edges_in_d].
            node_attr: torch.Tensor or None
                Optional node features, shape [N, nodes_att_dim].

            return: (h, coord, edge_attr)
                h: torch.Tensor
                    Updated node features.
                coord: torch.Tensor
                    Updated coordinates.
                edge_attr: torch.Tensor or None
                    Returned unchanged for interface compatibility.
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        del agg

        return h, coord, edge_attr

class EgnnGCLVelocity(EgnnGCL):
    """
    EGNN-style equivariant block with an additional learned velocity term.

    Relation to the paper:
    - Closest to the EGNN baseline update and not the full ClofNet block.
    - The velocity addition is a repository-side dynamics adaptation.

    Coordinate update:
        x_i <- x_i
             + mean_j ((x_i - x_j) * phi_x(m_ij))
             + psi(h_i) * vel_i

    Args:
        input_nf: int
            Input node feature dimension.
        output_nf: int
            Output node feature dimension.
        hidden_nf: int
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
            Scaling for the aggregated coordinate update.
        attention: bool
            Whether to gate edge features.
        norm_diff: bool
            Whether to normalize relative directions.
        tanh: bool
            Whether to apply tanh on the scalar coefficient used in the radial
            coordinate update.
    """

    def __init__( self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                  coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        super().__init__(input_nf,
                         output_nf,
                         hidden_nf,
                         edges_in_d=edges_in_d,
                         nodes_att_dim=nodes_att_dim,
                         act_fn=act_fn,
                         recurrent=recurrent,
                         coords_weight=coords_weight,
                         attention=attention,
                         norm_diff=norm_diff,
                         tanh=tanh)
        self.norm_diff = norm_diff

        self.coord_mlp_vel = nn.Sequential(nn.Linear(input_nf, hidden_nf),
                                           act_fn,
                                           nn.Linear(hidden_nf, 1))

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        """
        Run one layer with both radial equivariant updates and velocity drift.

        Args:
            h: torch.Tensor
                Node features, shape [N, input_nf].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            vel: torch.Tensor
                Velocities, shape [N, 3].
            edge_attr: torch.Tensor or None
                Optional edge features.
            node_attr: torch.Tensor or None
                Optional node features.

            return: (h, coord, edge_attr)
                h: torch.Tensor
                    Updated node features.
                coord: torch.Tensor
                    Updated coordinates.
                edge_attr: torch.Tensor or None
                    Returned unchanged.
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        coord = coord + self.coord_mlp_vel(h) * vel

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        del agg

        return h, coord, edge_attr
