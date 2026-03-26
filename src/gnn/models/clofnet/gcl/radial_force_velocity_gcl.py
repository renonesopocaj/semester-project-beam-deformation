from src.gnn.models.clofnet.gcl.gcl import AbstractGCL, unsorted_segment_mean
import torch
from torch import nn

class RadialForceGCL(AbstractGCL):
    """
    Radial-force coordinate update block without learned node features.

    Relation to the paper:
    - This is a baseline-style block, not a ClofNet block.
    - It is closest to the radial-only coordinate update idea used by EGNN
      in Eq. (1), where motion is constrained to the relative direction
      x_i - x_j.
    - It does not implement the complete local frame of Eq. (2), so it cannot
      express the richer vectorization used by ClofNet Eq. (8).

    Implemented edge force:
        x_diff = x_i - x_j
        r_ij = ||x_i - x_j||
        e_out = phi([r_ij, e_ij])
        m_ij = x_diff * e_out

    Node update:
        x_i <- x_i + mean_j m_ij - reg * x_i
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        """
        Args:
        nf: int
            Hidden dimension inside the radial force MLP.
        edge_attr_nf: int
            Number of extra edge attributes appended to the radial distance.
        reg: float
            Damping coefficient applied directly to coordinates.
        act_fn: nn.Module
            Activation used in the radial force MLP.
        clamp: bool
            Whether to clamp force vectors elementwise to [-100, 100].
        """
        super().__init__()
        self.clamp = clamp
        self.reg = reg

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)

    def edge_model_forward(self, source, target, edge_attr):
        """
        Predict radial force vectors per edge.

        Args:
            source: torch.Tensor
                Source coordinates, shape [E, 3].
            target: torch.Tensor
                Target coordinates, shape [E, 3].
            edge_attr: torch.Tensor
                Extra edge attributes, shape [E, edge_attr_nf].

        return: torch.Tensor
            Edge force vectors, shape [E, 3].
        """
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out

        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)

        return m_ij

    def node_model_forward(self, x, edge_index, edge_attr):
        """
        Aggregate radial force vectors and update coordinates.

        Args:
            x: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor
                Per-edge force vectors, shape [E, 3].

        return: torch.Tensor
            Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x * self.reg
        return x_out


class GCLRadialForceVelocity(nn.Module):
    """
    Radial-force block with an additional learned velocity drift term.

    Relation to the paper:
    - This is a dynamics-oriented utility block in the repository.
    - The radial part is still closest to EGNN Eq. (1), not to ClofNet's
      complete-frame vectorization.
    - The additive velocity term is an implementation choice.

    Update rule:
        m_ij = (x_i - x_j) * phi([||x_i - x_j||, e_ij])
        x_i <- x_i + coords_weight * mean_j m_ij + vel_i * psi(||vel_i||)

    Args:
        nf: int
            Hidden dimension in the MLPs.
        edge_attr_nf: int
            Number of extra edge attributes.
        act_fn: nn.Module
            Activation used in the MLPs.
        coords_weight: float
            Scaling factor for the aggregated coordinate update.
    """

    def __init__(self, nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0,):
        super().__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(nn.Linear(1, nf),
                                           act_fn,
                                           nn.Linear(nf, 1))
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh(),)

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        """
        Update coordinates using radial messages plus a velocity drift.

        Args:
            x: torch.Tensor
                Coordinates, shape [N, 3].
            vel_norm: torch.Tensor
                Velocity norms, shape [N, 1].
            vel: torch.Tensor
                Velocities, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_attr: torch.Tensor or None
                Optional per-edge attributes.

        return: (coord, edge_attr)
            coord: torch.Tensor
                Updated coordinates, shape [N, 3].
            edge_attr: torch.Tensor or None
                Returned unchanged for interface compatibility.
        """
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x = x + vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        """
        Build radial coordinate messages.

        Args:
            source: torch.Tensor
                Source coordinates, shape [E, 3].
            target: torch.Tensor
                Target coordinates, shape [E, 3].
            edge_attr: torch.Tensor
                Extra per-edge attributes, shape [E, edge_attr_nf].

        return: torch.Tensor
            Per-edge coordinate messages, shape [E, 3]
        """
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        """
        Aggregate radial messages and update coordinates.

        Args:
            x: torch.Tensor
                Coordinates, shape [N, 3].
            edge_index: tuple[torch.Tensor, torch.Tensor]
                Directed edge list as (row, col).
            edge_m: torch.Tensor
                Per-edge coordinate messages, shape [E, 3].

            return: torch.Tensor
                Updated coordinates, shape [N, 3].
        """
        row, col = edge_index
        del col

        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out