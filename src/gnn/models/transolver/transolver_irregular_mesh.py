import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from src.gnn.models.transolver.embedding import sin_timestep_embedding
import numpy as np
from src.gnn.models.transolver.physics_attention import PhysicsAttentionIrregularMesh

# Maps string names to nn.Module activation constructors for MLP blocks.
ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    """
    Pointwise feed-forward stack applied independently to each mesh node vector.

    This is the Transolver FeedForward sublayer in spirit: a sequence of Linear
    maps with activation on the last dimension only (no message passing over the
    mesh). When n_layers > 0 and res is True, each hidden block is a residual
    MLP block at width n_hidden.

    Allocated networks:
    - linear_pre: nn.Sequential(Linear(n_input, n_hidden), act). Lifts inputs to
      the working width n_hidden with nonlinearity.
    - linears: list of nn.Sequential(Linear(n_hidden, n_hidden), act). Optional
      repeated hidden transforms; residual adds x when res is True.
    - linear_post: nn.Linear(n_hidden, n_output). Projects back to output width.
    """

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        """
        Build the MLP tower.

        Args:
            n_input: int
                Input feature width per node.
            n_hidden: int
                Hidden width (and expansion target after linear_pre).
            n_output: int
                Output feature width per node.
            n_layers: int
                Number of hidden blocks in linears (each may be residual).
            act: str
                Name of activation in ACTIVATION map.
            res: bool
                If True, each linears[i] output is added to its input when n_layers > 0.

        return: None
        """
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        # When True, inner blocks add identity (residual MLP tower).
        self.res = res
        # First projection up to n_hidden with nonlinearity (Transolver FFN-style expansion path).
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        # Final readout from hidden width to n_output.
        self.linear_post = nn.Linear(n_hidden, n_output)
        # n_layers residual (or plain) blocks at fixed width n_hidden.
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        """
        Run linear_pre, optional residual hidden blocks, then linear_post.

        Args:
            x: torch.Tensor
                Any tensor whose last dimension equals n_input; often (B, N, n_input).

        return: torch.Tensor
            Same leading shape as x with last dimension n_output.

        Steps:
            1) x = linear_pre(x).
            2) For each i in 0..n_layers-1: optionally x = linears[i](x) + x.
            3) x = linear_post(x).
        """
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                # Residual update at hidden width (paper FeedForward as local nonlinear stack).
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """
    One Transolver layer: pre-norm Physics-Attention residual + pre-norm FFN residual.

    Matches the paper stack (Eq. overall) with pre-norm ordering as in common ViT-style code:
    - x_hat = x + Physics-Attn(LayerNorm(x))
    - x_new = x_hat + FeedForward(LayerNorm(x_hat))
    The last layer of the stack can apply LayerNorm and a linear readout to out_dim
    channels per node (paper final linear on x^L for predicted physical quantities).

    Allocated networks:
    - ln_1, ln_2: nn.LayerNorm(hidden_dim). Normalize before attention and FFN.
    - Attn: PhysicsAttentionIrregularMesh. Full slice, token attention, deslice path.
    - mlp: MLP with n_layers=0 so it is a two-layer pointwise FFN: expand by mlp_ratio
      then contract back to hidden_dim (paper FeedForward width scaling).
    - ln_3, mlp2: only if last_layer; final norm and nn.Linear(hidden_dim, out_dim).
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        """
        Construct one encoder block.

        Args:
            num_heads: int
                Head count for PhysicsAttentionIrregularMesh.
            hidden_dim: int
                Model channel width C (paper hidden C).
            dropout: float
                Dropout inside attention and MLP tail.
            act: str
                Activation name for the FFN MLP.
            mlp_ratio: int
                Expansion ratio: FFN hidden width is hidden_dim * mlp_ratio.
            last_layer: bool
                If True, apply per-node linear head to out_dim after second residual.
            out_dim: int
                Output channels when last_layer is True.
            slice_num: int
                Number of physics slices M in Physics-Attention.

        return: None
        """
        super().__init__()
        self.last_layer = last_layer
        # Pre-norm before physics attention (paper LayerNorm before Physics-Attn).
        self.ln_1 = nn.LayerNorm(hidden_dim)
        # Irregular mesh: linear Project + slice + attention on M tokens + deslice (paper Physics-Attention).
        self.Attn = PhysicsAttentionIrregularMesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                  dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        # Pointwise FFN with expansion mlp_ratio (paper second sublayer after attention).
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            # Extra norm + linear to physical output channels (paper readout on x^L).
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        """
        Apply attention and FFN residuals; optionally return linear predictions.

        Args:
            fx: torch.Tensor
                Node features (B, N, hidden_dim).

        return: torch.Tensor
            Same shape as fx when not last_layer; else (B, N, out_dim).

        Steps:
            1) fx = fx + Attn(LN(fx)).
            2) fx = fx + mlp(LN(fx)).
            3) If last_layer: return mlp2(LN(fx)); else return fx.
        """
        # Residual physics-attention branch: x_hat = Attn(LN(x)) + x (paper Eq. overall first line).
        fx = self.Attn(self.ln_1(fx)) + fx
        # Second residual: FFN(LN(x_hat)) + x_hat (paper Eq. overall second line).
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    """
    Transolver stack for irregular meshes: embed concat(g, u), then L residual blocks.

    Paper mapping: initial node embedding corresponds to x^0 = Linear(Concat(g, u))
    (implemented here as an MLP preprocess on concatenated inputs). Repeated
    Transolver_block modules implement the L layers; the final block returns the
    per-node prediction in out_dim channels.

    Optional unified_pos replaces raw coordinates g with ref^2 distances to a
    fixed unit-square reference grid (extra geometry features beyond the paper's
    minimal Linear concat). Optional Time_Input adds a sinusoidal time embedding
    per batch element broadcast to all nodes, then refined by time_fc.

    Allocated networks:
    - preprocess: MLP mapping concat inputs to n_hidden (initial embedding).
    - time_fc: nn.Sequential(Linear, SiLU, Linear) when Time_Input; maps sinusoidal
      time vector to n_hidden per batch row.
    - blocks: ModuleList of Transolver_block; only the last uses last_layer head.
    - placeholder: nn.Parameter vector added to every node after preprocess (learnable bias).
    """

    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        """
        Build the full irregular-mesh Transolver.

        Args:
            space_dim: int
                Coordinate dimension per node when unified_pos is False (e.g. 2 for x,y).
            n_layers: int
                Number of Transolver blocks L.
            n_hidden: int
                Hidden channel width C.
            dropout: float
                Dropout in attention and MLP.
            n_head: int
                Attention head count.
            Time_Input: bool
                If True, allocate time_fc and add time conditioning in forward.
            act: str
                Activation for MLP blocks.
            mlp_ratio: int
                FFN expansion ratio inside each Transolver_block.
            fun_dim: int
                Channel count of optional per-node observed field u (paper u); concatenated after x.
            out_dim: int
                Output channels per node (predicted quantities).
            slice_num: int
                Slices M in Physics-Attention.
            ref: int
                Reference grid side length for unified_pos; produces ref^2 distance features.
            unified_pos: bool
                If True, replace x by pairwise distances to ref x ref reference points.

        return: None
        """
        super(Model, self).__init__()
        self.__name__ = 'Transolver_1D'
        # Resolution of reference 2D grid used for unified position features (ref x ref distances).
        self.ref = ref
        # If True, replace raw coords with pairwise distances to reference grid (geometry-general encoding).
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if self.unified_pos:
            # Input concat: observed channels + ref^2 distance scalars per node -> embed to n_hidden (paper Linear(Concat(g,u)) spirit).
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            # Concatenate function channels with raw space_dim coordinates per node.
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        if Time_Input:
            # Refine sinusoidal time embedding to hidden size before adding to every node.
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        # Stack of Transolver layers; last block outputs out_dim (e.g. PDE target per node).
        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        # Learnable global bias broadcast to all nodes (optional inductive shift on token sequence).
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        """
        Apply trunc_normal_ to Linear weights and init LayerNorm/BatchNorm1d.

        Args:
            None

        return: None
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Per-module init callback used by initialize_weights.

        Args:
            m: nn.Module
                Submodule being initialized.

        return: None
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        """
        Build unified position features: Euclidean distances from nodes to ref grid.

        Args:
            x: torch.Tensor
                Node coordinates, shape (B, N, 2) in [0,1]^2 (typical usage).
            batchsize: int
                Batch size B (should match x.shape[0]).

        return: torch.Tensor
            Tensor of shape (B, N, ref^2) with distances to each reference point.

        Steps:
            1) Build ref x ref reference points on the unit square.
            2) For each node, compute L2 distance to every reference point.
            3) Flatten ref dimensions to ref^2 features per node.
        """
        # x: B N 2
        # grid_ref
        dev = x.device
        dtype = x.dtype
        # 1D linspace along x for ref samples in [0,1].
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=dtype, device=dev)
        # Broadcast to (B, ref, ref, 1): x coordinate of ref grid corners.
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=dtype, device=dev)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        # Stack xy -> reference points (B, ref^2, 2) on the same device as node coordinates.
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref * self.ref, 2)

        # Euclidean distance from each mesh node to each ref point -> ref^2 scalars per node (unified position).
        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        """
        Embed inputs, optionally add time, run all Transolver blocks, return predictions.

        Args:
            x: torch.Tensor
                Per-node geometry g: coordinates (B, N, space_dim) or, after unified_pos,
                distance features (B, N, ref^2) produced inside this method.
            fx: torch.Tensor or None
                Optional per-node observed channels u; if None, preprocess uses x only.
            T: torch.Tensor or None
                Optional per-batch scalar timesteps; when not None and Time_Input,
                adds time embedding to every node.

        return: torch.Tensor
            Per-node outputs (B, N, out_dim) from the last block.

        Steps:
            1) If unified_pos: replace x by get_grid(x).
            2) Concatenate x with fx if fx is not None; preprocess MLP to n_hidden.
            3) Add placeholder bias to all nodes.
            4) If T is not None: add time_fc(sin_timestep_embedding(T)) broadcast to N.
            5) Sequentially apply each Transolver_block.
        """
        if self.unified_pos:
            # Overwrite x with distance features so preprocess sees geometry via ref grid (not raw g alone).
            x = self.get_grid(x, x.shape[0])
        if fx is not None:
            # Concatenate geometry/unified_pos channels with optional observed field u (paper Concat(g,u)).
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            # No separate u stream: embed positions (or unified distances) only.
            fx = self.preprocess(x)
        # Add shared learnable offset to every node embedding.
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            # Sinusoidal embedding of scalar time per batch row, tiled across N nodes.
            Time_emb = sin_timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            # Inject time-dependent modulation into node features (e.g. sequence / param conditioning).
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
