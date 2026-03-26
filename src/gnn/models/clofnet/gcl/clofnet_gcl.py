from src.gnn.models.clofnet.gcl.egnn_gcl import EgnnGCL
from torch import nn
import torch
from src.gnn.models.clofnet.gcl.helpers_gcl import unsorted_segment_mean

class ClofGCL(EgnnGCL):
    """
    ClofNet graph block with complete local frames.

    This is the key ClofNet layer in the file.

    Paper mapping:
    - Eq. (2): build a complete local frame
          a_ij = (x_i - x_j) / ||x_i - x_j||
          b_ij = (x_i x x_j) / ||x_i x x_j||
          c_ij = a_ij x b_ij
    - Eqs. (3)-(4): scalarization turns vectors into invariant coefficients
      relative to that frame.
    - Eq. (8): vectorization reconstructs an equivariant vector from three
      scalar coefficients and the basis vectors.
    - Eqs. (16)-(19): message passing learns invariant messages, then those
      messages are vectorized back into coordinate updates.

    How the code realizes Eq. (8):
    - `edge_model(...)` produces invariant edge embeddings `edge_feat`.
    - `coord_mlp(edge_feat)` predicts three scalar coefficients:
          coff = [alpha_ij, beta_ij, gamma_ij]
    - `coord_model(...)` reconstructs the vector update
          trans_ij = alpha_ij * a_ij + beta_ij * b_ij + gamma_ij * c_ij
      where this repository uses:
          a_ij = coord_diff
          b_ij = coord_cross
          c_ij = coord_vertical
    - Those vectorized edge updates are averaged per node and added to the
      coordinates.

    The extra term
        coord += coord_mlp_vel(h) * vel
    is not a paper equation. It is a repository-level dynamics addition.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        """
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
                        tanh=tanh,
                        out_basis_dim=3)
        self.norm_diff = norm_diff

        self.coord_mlp_vel = nn.Sequential(nn.Linear(input_nf, hidden_nf),
                                            act_fn,
                                            nn.Linear(hidden_nf, 1))

        self.edge_mlp = nn.Sequential(nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf),
                                      act_fn,
                                      nn.Linear(hidden_nf, hidden_nf),
                                      act_fn,
                                      nn.Linear(hidden_nf, hidden_nf),
                                      act_fn)

        self.layer_norm = nn.LayerNorm(hidden_nf)

    def coord2localframe(self, edge_index, coord):
        """
        Construct the complete local frame per edge.

        This is the code counterpart of paper Eq. (2):
            a_ij = (x_i - x_j) / ||x_i - x_j||
            b_ij = (x_i x x_j) / ||x_i x x_j||
            c_ij = a_ij x b_ij

        Implementation details:
        - The repository uses `+ 1` instead of a tiny epsilon in the
          normalization denominator, mirroring the broader codebase style.
        - The frame is "complete" because it spans three directions instead of
          only the radial direction, which addresses the directional
          degeneracy problem discussed in Section 3.

        Args:
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord: torch.Tensor
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
        row, col = edge_index

        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1).unsqueeze(1)

        coord_cross = torch.cross(coord[row], coord[col])

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

            cross_norm = torch.sqrt(torch.sum(coord_cross ** 2, dim=1).unsqueeze(1)) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return radial, coord_diff, coord_cross, coord_vertical

    def coord_model(self, coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat):
        """
        Vectorize learned scalar coefficients back into a 3D coordinate update.

        This is the main implementation of the paper's vectorization idea in
        Eq. (8). The learned edge embedding `edge_feat` is mapped to three
        scalar coefficients:
            coff_ij = [alpha_ij, beta_ij, gamma_ij]

        Then the update is reconstructed in the local frame:
            trans_ij =
                alpha_ij * a_ij
              + beta_ij  * b_ij
              + gamma_ij * c_ij

        Finally, edgewise transforms are averaged per destination node and
        added to the coordinates.

        Args:
            coord: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            coord_diff: torch.Tensor
                First basis vector a_ij, shape [E, 3].
            coord_cross: torch.Tensor
                Second basis vector b_ij, shape [E, 3].
            coord_vertical: torch.Tensor
                Third basis vector c_ij, shape [E, 3].
            edge_feat: torch.Tensor
                Invariant edge embeddings/messages, shape [E, hidden_nf].

            return: torch.Tensor
                Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        coff = self.coord_mlp(edge_feat)

        trans = (
            coord_diff * coff[:, :1]
            + coord_cross * coff[:, 1:2]
            + coord_vertical * coff[:, 2:3]
        )

        trans = torch.clamp(trans, min=-100, max=100)

        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg * self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
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
        - This class then adds another residual:
              h = residue + h
          followed by `LayerNorm`.
        - This double-residual structure is preserved exactly from the original
          repository code.

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
        residue = h

        radial, coord_diff, coord_cross, coord_vertical = self.coord2localframe(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat)

        coord = coord + self.coord_mlp_vel(h) * vel

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        del agg

        h = residue + h
        h = self.layer_norm(h)

        return h, coord, edge_attr
