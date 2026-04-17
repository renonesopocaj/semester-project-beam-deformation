class EgnnGCLVelocity(EgnnGCL):
    """
    EGNN-style equivariant block with an additional learned velocity term.
    Relation to the paper:
    - Closest to the EGNN baseline update and not the full ClofNet block.
    - The velocity addition is a repository-side dynamics adaptation.
    Coordinate update:
        x_i <- x_i  + mean_j ((x_i - x_j) * phi_x(m_ij)) + psi(h_i) * vel_i
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
                Scaling for the aggregated coordinate update.
            attention: bool
                Whether to gate edge features.
            norm_diff: bool
                Whether to normalize relative directions.
            tanh: bool
                Whether to apply tanh on the scalar coefficient used in the radial
                coordinate update.
        """
        super().__init__(input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim,
                         act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention,
                         norm_diff=norm_diff, tanh=tanh)
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

        edge_feat = self.message_mij_model_forward(h[row], h[col], radial, edge_attr)
        coord = self.coord_model_forward(coord, edge_index, coord_diff, edge_feat)
        coord = coord + self.coord_mlp_vel(h) * vel

        h, _ = self.node_model_forward(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr
