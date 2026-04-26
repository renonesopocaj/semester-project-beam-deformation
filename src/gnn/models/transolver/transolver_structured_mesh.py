import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from src.gnn.models.transolver.embedding import sin_timestep_embedding
import torch.utils.checkpoint as checkpoint
from src.gnn.models.transolver.physics_attention import PhysicsAttentionStructuredMesh3D

# String -> activation module mapping for MLP blocks.
ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    """
    Pointwise feed-forward MLP on the last feature dimension (per voxel / node).

    Same role as in the irregular Transolver: implements the FeedForward sublayer
    inside each Transolver layer as a width-expanded Linear stack without graph
    message passing. Used here to map concat(g, u) or distance-augmented inputs to
    hidden_dim, and inside each block for the FFN residual.

    Allocated networks:
    - linear_pre: nn.Sequential(Linear(n_input, n_hidden), act).
    - linears: optional list of (Linear(n_hidden, n_hidden), act) with residual add.
    - linear_post: nn.Linear(n_hidden, n_output).
    """

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        """
        Build the MLP.

        Args:
            n_input: int
                Last-dimension size of inputs.
            n_hidden: int
                Hidden width after linear_pre.
            n_output: int
                Output last-dimension size.
            n_layers: int
                Count of inner (n_hidden -> n_hidden) blocks.
            act: str
                Activation key in ACTIVATION.
            res: bool
                If True, add identity inside each inner block when n_layers > 0.

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
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)]
        )

    def forward(self, x):
        """
        Run pre-layer, optional residual hidden blocks, post-layer.

        Args:
            x: torch.Tensor
                Tensor with last dim n_input.

        return: torch.Tensor
            Same leading shape, last dim n_output.

        Steps:
            1) linear_pre.
            2) For each inner block, optionally add residual.
            3) linear_post.
        """
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """
    One Transolver layer on a flattened H x W x D structured grid with 3D Physics-Attention.

    Uses PhysicsAttentionStructuredMesh3D so that slice logits and token numerators
    are built from Conv3d-localized features (paper: convolutional Project on structured
    meshes) before the shared slice-token attention and deslice. Residuals follow the
    same pre-norm pattern as the irregular model: attention then FFN, each with
    LayerNorm and skip connection. Last block applies LayerNorm and Linear to out_dim.

    Allocated networks:
    - Attn: PhysicsAttentionStructuredMesh3D (slice, M-token attention, deslice).
    - mlp: MLP with hidden_dim * mlp_ratio expansion (FeedForward analogue).
    - ln_1, ln_2, optional ln_3: LayerNorm; mlp2: final readout Linear when last_layer.
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
            H=32,
            W=32,
            D=32
    ):
        """
        Construct one 3D structured Transolver block.

        Args:
            num_heads: int
                Attention heads (paper multi-head on channel subspaces).
            hidden_dim: int
                Channel width C per voxel after embedding.
            dropout: float
                Dropout in attention and output projection.
            act: str
                FFN activation name.
            mlp_ratio: int
                FFN hidden multiplier relative to hidden_dim.
            last_layer: bool
                If True, include final norm and linear to out_dim.
            out_dim: int
                Output channels when last_layer.
            slice_num: int
                Number of physics slices M.
            H: int
                Structured grid size x.
            W: int
                Structured grid size y.
            D: int
                Structured grid size z; flattened N = H*W*D.

        return: None
        """
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        # 3D structured lattice: Conv3d local mix then Physics-Attention on N=H*W*D nodes (paper structured mesh variant).
        self.Attn = PhysicsAttentionStructuredMesh3D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W, D=D)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        """
        Pre-norm attention residual then pre-norm FFN residual; optional linear head.

        Args:
            fx: torch.Tensor
                (B, H*W*D, hidden_dim) voxel features in node-major order.

        return: torch.Tensor
            Same shape as fx, or (B, N, out_dim) if last_layer.

        Steps:
            1) fx = fx + Attn(LN(fx)).
            2) fx = fx + mlp(LN(fx)).
            3) If last_layer: mlp2(LN(fx)); else fx.
        """
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    """
    Transolver for 3D structured meshes: voxels indexed in a fixed H x W x D lattice.

    Node dimension is N = H * W * D with features ordered consistently with
    PhysicsAttentionStructuredMesh3D (reshape inside attention expects this layout).
    The paper's initial embedding x^0 = Linear(Concat(g, u)) is implemented as preprocess
    MLP on concatenated per-voxel inputs. unified_pos optionally replaces raw voxel
    coordinates with ref^3 distances to a coarse reference grid in [0,1]^3; distances
    are precomputed once in __init__ as self.pos (see get_grid) and tiled per batch.

    Differences vs irregular Model worth documenting:
    - use_checkpoint: when True, each block runs under torch.utils.checkpoint.checkpoint
      to trade extra forward compute in backward for lower activation memory.
    - placeholder bias is added only in the branch where fx is None (geometry-only),
      matching the existing code path.

    Allocated networks:
    - preprocess: MLP from concat feature width to n_hidden.
    - time_fc: two-layer MLP with SiLU when Time_Input (sinusoidal time -> hidden).
    - blocks: ModuleList of Transolver_block with PhysicsAttentionStructuredMesh3D.
    - placeholder: learnable vector broadcast when no separate u channel tensor.
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
                 unified_pos=False,
                 H=32,
                 W=32,
                 D=32,
                 ):
        """
        Build the 3D structured Transolver.

        Args:
            space_dim: int
                Coordinate channels per voxel when unified_pos is False (typically 3).
            n_layers: int
                Depth L of Transolver blocks.
            n_hidden: int
                Hidden width C.
            dropout: float
                Dropout rate.
            n_head: int
                Number of attention heads.
            Time_Input: bool
                Whether to add time conditioning via time_fc.
            act: str
                Activation for MLPs.
            mlp_ratio: int
                FFN expansion inside each block.
            fun_dim: int
                Optional extra input channels per voxel (observed u) concatenated with x.
            out_dim: int
                Prediction channels per voxel from the last block.
            slice_num: int
                Slices M in Physics-Attention.
            ref: int
                Side length of coarse reference grid; unified_pos uses ref^3 distances.
            unified_pos: bool
                If True, cache distance features and use wider preprocess input.
            H: int
                Structured grid extent.
            W: int
                Structured grid extent.
            D: int
                Structured grid extent.

        return: None
        """
        super(Model, self).__init__()
        self.__name__ = 'Transolver_3D'
        # When True, forward uses activation checkpointing inside the block loop to save memory.
        self.use_checkpoint = False
        self.H = H
        self.W = W
        self.D = D
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            # Precompute distance tensor from each voxel to coarse ref^3 grid (cached; see forward).
            self.pos = self.get_grid()
            # fun_dim + ref^3 distances per voxel as input to first MLP (3D unified position).
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      H=H,
                                                      W=W,
                                                      D=D,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        """
        Initialize Linear and norm layers under this module tree.

        Args:
            None

        return: None
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Per-module initializer for trunc_normal_ on Linear weights.

        Args:
            m: nn.Module
                Child module visited by apply().

        return: None
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        """
        Precompute 3D unified position features: L2 distance from each voxel to ref^3 points.

        Args:
            batchsize: int
                Batch size used when building the cached tensor (typically 1 for static cache).

        return: torch.Tensor
            Shape (batchsize, H, W, D, ref^3) with nonnegative distances.

        Steps:
            1) Build voxel centers on a uniform H x W x D grid in [0,1]^3.
            2) Build ref x ref x ref reference voxel centers in [0,1]^3.
            3) Broadcast pairwise 3D Euclidean distances and flatten ref dims to ref^3.
        """
        size_x, size_y, size_z = self.H, self.W, self.D
        # Unit-cube coordinates for each voxel center (structured simulation grid).
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).cuda()  # B H W D 3

        # Coarser ref^3 reference grid in [0,1]^3 for multi-scale distance features.
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda()  # B ref ref ref 3

        # Pairwise 3D distance between every voxel and every ref cell via broadcast; flatten ref dims to ref^3.
        pos = torch.sqrt(
            torch.sum((grid[:, :, :, :, None, None, None, :] - grid_ref[:, None, None, None, :, :, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, size_x, size_y, size_z, self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        """
        Embed voxel inputs, optional time, run L Transolver blocks (optionally checkpointed).

        Args:
            x: torch.Tensor
                Per-voxel geometry or ignored when unified_pos uses cached self.pos; after
                unified_pos branch, x becomes (B, N, ref^3) distance features with N = HWD.
            fx: torch.Tensor or None
                Optional extra channels per voxel; if None, preprocess uses only x and
                placeholder is added.
            T: torch.Tensor or None
                Optional batch times for sinusoidal embedding.

        return: torch.Tensor
            (B, N, out_dim) predictions.

        Steps:
            1) If unified_pos: set x to tiled cached distances (B, HWD, ref^3).
            2) Concat x and fx if fx is not None; preprocess to hidden; else preprocess(x)
               and add placeholder.
            3) If T given: add time_fc(sin_timestep_embedding(T)) per batch across N.
            4) For each block: checkpoint.checkpoint(block, fx) if use_checkpoint else block(fx).
        """
        if self.unified_pos:
            # Expand cached pos to batch and flatten spatial dims -> (B, HWD, ref^3) distance features per voxel.
            x = self.pos.repeat(x.shape[0], 1, 1, 1, 1).reshape(x.shape[0], self.H * self.W * self.D,
                                                                self.ref * self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            # No u: only geometry/unified distances; placeholder bias only in this branch (differs from irregular Model).
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = sin_timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            if self.use_checkpoint:
                # Trade compute for memory: recompute block during backward.
                fx = checkpoint.checkpoint(block, fx)
            else:
                fx = block(fx)

        return fx
