import torch.nn as nn
import torch
from einops import rearrange, repeat
from abc import ABC, abstractmethod

class AbstractPhysicsAttention(ABC):
    """
    Shared implementation of Transolver Physics-Attention on mesh node features.

    The paper decomposes node features into M learnable slices, forms M
    physics-aware tokens, runs scaled dot-product attention among tokens only,
    then maps token updates back to nodes. This class holds the token attention
    and deslicing path shared by irregular (linear) and structured (conv) variants.

    Core equations (paper notation, single head; code uses multi-head split):
    - Slice weights: w_i = softmax(Project(x_i)) in R^{1 x M} with sum_j w_{i,j}=1.
    - Per-slice weighted features before normalization: s_{j,i} = w_{i,j} x_i.
    - Physics-aware token: z_j = (sum_i w_{i,j} x_i) / (sum_i w_{i,j}).
    - Token attention: q,k,v = Linear(z); z' = softmax(q k^T / sqrt(d)) v.
    - Deslice: x'_i = sum_j w_{i,j} z'_j.

    Submodules (on concrete subclasses): to_q, to_k, to_v are per-head linear maps
    on slice tokens; to_out is a linear (plus dropout) merging heads back to dim.
    """

    def clear_last_attention_state(self):
        """
        Reset the cached attention tensors captured during the previous forward pass.

        return: None
            No-op when the concrete module does not expose a capture cache.
        """
        if hasattr(self, "last_attention_state"):
            self.last_attention_state = None

    def _store_last_attention_state(self, slice_weights, slice_norm, attn):
        """
        Cache the tensors needed to recover effective node-to-node interactions.

        Args:
            slice_weights: torch.Tensor
                Slice routing weights with shape (B, H, N, G).
            slice_norm: torch.Tensor
                Slice masses with shape (B, H, G).
            attn: torch.Tensor
                Slice-token attention with shape (B, H, G, G).
        return: None
            Stores detached tensors only when capture_attention is enabled.
        """
        if not bool(getattr(self, "capture_attention", False)):
            return
        self.last_attention_state = {
            "slice_weights": slice_weights.detach(),
            "slice_norm": slice_norm.detach(),
            "slice_attention": attn.detach(),
        }

    def slice_to_out(self, x_mid, fx_mid, slice_weights):
        """
        Form slice tokens, attend over M slices, deslice to per-node features.

        Args:
            x_mid: torch.Tensor
                Per-head features used only upstream for slice logits; kept for API
                symmetry with forward() in subclasses (not used inside this method).
            fx_mid: torch.Tensor
                Per-head node features of shape (B, H, N, dim_head) aggregated into
                slice numerators before dividing by slice mass.
            slice_weights: torch.Tensor
                Non-negative weights w_{i,j} of shape (B, H, N, G) with softmax over G.

        return: torch.Tensor
            Tensor of shape (B, N, dim) after Physics-Attention and output projection.

        Steps:
            1) slice_norm_j = sum_i w_{i,j} (per head).
            2) slice_token_j = sum_i w_{i,j} fx_i then divide by slice_norm (paper z_j),
               with epsilon and per-dim_head broadcast matching code.
            3) q_j = W_q z_j, k_j = W_k z_j, v_j = W_v z_j; attn = softmax(q k^T * scale).
            4) out_slice_j = sum_t attn_{j,t} v_t (paper z'_j).
            5) out_node_i = sum_j w_{i,j} out_slice_j (paper x'_i), merge heads, Linear.
        """
        # Per-slice total membership: sum over nodes n of w_{i,j} -> shape B H G (paper Eq. token denominator).
        slice_norm = slice_weights.sum(2)  # B H G
        # Weighted sum of node features into each slice: s_j aggregates w_{i,j} * x_i (paper learn_slice / pre-token).
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        # Normalize token by slice mass so z_j = sum_i w_ij x_i / sum_i w_ij (paper Eq. token_encoding); repeat dim_head for broadcast.
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        # Map each of M slice tokens to query in head subspace (paper q = Linear(z)).
        q_slice_token = self.to_q(slice_token)
        # Keys from same slice-token tensor (paper k = Linear(z)).
        k_slice_token = self.to_k(slice_token)
        # Values from slice tokens (paper v = Linear(z)).
        v_slice_token = self.to_v(slice_token)
        # Scaled dot-product scores between slice tokens: O(M^2) per head, not O(N^2) (paper Eq. attn, sqrt(C)).
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        # Softmax over slice index so each slice attends over all slices.
        attn = self.softmax(dots)
        # Regularize attention weights during training.
        attn = self.dropout(attn)
        self._store_last_attention_state(slice_weights, slice_norm, attn)
        # Mix value vectors by attention weights -> updated slice tokens z' (paper z').
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        # Broadcast z'_j back to nodes: x'_i = sum_j w_{i,j} z'_j (paper Eq. deslice).
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        # Concatenate head outputs along channel -> single vector per node (multi-head merge).
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        # Final linear projection back to model hidden dim C.
        return self.to_out(out_x)


class PhysicsAttentionIrregularMesh(nn.Module, AbstractPhysicsAttention):
    """
    Physics-Attention for unstructured node sets (1D/2D/3D geometry in coordinate form).

    Matches the paper choice for irregular meshes: Project() is a pointwise
    linear map from each node embedding to slice logits and to per-head features.
    Attention complexity is O(N M C_head + M^2 C_head) per layer with M = slice_num.

    Allocated networks:
    - in_project_x: nn.Linear(dim, heads*dim_head). Encodes each node into the stream
      from which slice logits are computed (analogue of Project for routing).
    - in_project_fx: nn.Linear(dim, heads*dim_head). Separate stream whose outputs
      are weighted into slice tokens (numerator of z_j); lets routing and content differ.
    - in_project_slice: nn.Linear(dim_head, slice_num). Maps per-head per-node
      features to raw logits over M slices before softmax (M in the paper).
    - to_q, to_k, to_v: nn.Linear(dim_head, dim_head, bias=False). Standard QKV on
      slice tokens only (not on N nodes).
    - to_out: nn.Sequential(Linear(heads*dim_head, dim), Dropout). Output projection
      after head concatenation.
    - temperature: learnable positive scale on slice logits (implementation detail;
      paper uses softmax without an extra scalar per head).
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        """
        Build irregular-mesh Physics-Attention with multi-head slice routing.

        Args:
            dim: int
                Node hidden channel count C (model width).
            heads: int
                Number of attention heads H (paper splits channels across heads).
            dim_head: int
                Per-head width d; inner_dim = H * d.
            dropout: float
                Dropout on attention weights and on to_out.
            slice_num: int
                Number of slices M (physics-aware tokens per head).

        return: None
        """
        super().__init__()
        # Total head channels C = heads * dim_head (paper multi-head split along channel).
        inner_dim = dim_head * heads
        # Per-head feature width d used inside slice tokens and QKV.
        self.dim_head = dim_head
        self.heads = heads
        # 1/sqrt(d) scaling for dot-product attention stability (paper / sqrt(C) with per-head d).
        self.scale = dim_head ** -0.5
        # Softmax over slice index G so each node's weights sum to 1 across slices (paper Softmax on Project(x)).
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # Learnable temperature for slice logits; sharper/softer assignment over G slices.
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.capture_attention = False
        self.last_attention_state = None
        # Point-wise linear map from node dim -> head features for stream used in slice routing / x_mid (paper Project on mesh).
        self.in_project_x = nn.Linear(dim, inner_dim)
        # Separate linear for features pooled into slice tokens (paper weighted x in numerator of z_j).
        self.in_project_fx = nn.Linear(dim, inner_dim)
        # Per head: dim_head -> slice_num logits before softmax -> w_{i,*} (paper M slices).
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        # QKV on slice tokens only; each operates in R^{dim_head} per head (paper attention among M tokens).
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        # Project concatenated heads back to dim and dropout (output of Physics-Attn block).
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

    def forward(self, x):
        """
        Run slice assignment then shared slice_to_out on node sequence x.

        Args:
            x: torch.Tensor
                Node features of shape (B, N, dim), paper x in R^{N x C}.

        return: torch.Tensor
            Updated node features (B, N, dim) after one Physics-Attention block.

        Steps:
            1) fx_mid, x_mid: split in_project_fx / in_project_x outputs into
               (B, H, N, dim_head).
            2) slice_weights = softmax( in_project_slice(x_mid) / temperature )
               over last dim G (paper w_i).
            3) slice_to_out: build z, attention among M tokens, deslice, to_out.
        """
        # B N C
        B, N, C = x.shape

        # Slice
        # Embed each node into head subspaces; fx_mid is the field aggregated into slice tokens (B, heads, N, dim_head).
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
        # Parallel stream for slice logits (same shape); paper uses x_i features to produce w_i.
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
        # Softmax over G: each node distributes unit mass across slices (paper Eq. learn_slice).
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        return self.slice_to_out(x_mid, fx_mid, slice_weights)


class PhysicsAttentionStructuredMesh2D(nn.Module, AbstractPhysicsAttention):
    """
    Physics-Attention for nodes laid out on a fixed 2D H x W structured grid.

    Implements the paper appendix variant: before slice assignment, features are
    built with a local Conv2d over the grid instead of a pointwise-only linear map.
    Same slice token, token attention, and deslice equations as the irregular case;
    only the way x_mid and fx_mid are produced from x changes (local neighborhood mix).

    Allocated networks:
    - in_project_x, in_project_fx: nn.Conv2d(dim, inner_dim, kernel, padding same).
      Translation-equivariant local encoders over the 2D lattice; two streams like
      the irregular twin Linears.
    - in_project_slice, to_q, to_k, to_v, to_out: same roles as PhysicsAttentionIrregularMesh.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=101, W=31, kernel=3):  # kernel=3):
        """
        Build 2D structured-mesh Physics-Attention.

        Args:
            dim: int
                Input channel C per lattice cell after flattening to nodes.
            heads: int
                Head count H.
            dim_head: int
                Per-head width d.
            dropout: float
                Dropout rate.
            slice_num: int
                Slice count M (tokens per head).
            H: int
                Grid height; N must equal H * W.
            W: int
                Grid width.
            kernel: int
                Convolution kernel size for local mixing (odd, same padding).

        return: None
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        # Fixed grid height/width so flattened N = H*W matches conv layout (paper: conv on structured mesh).
        self.H = H
        self.W = W

        # Local 2D conv mixes neighbors before slicing (paper appendix: conv for structured/uniform grids).
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

    def forward(self, x):
        """
        Reshape flat nodes to H x W, apply conv streams, softmax slices, slice_to_out.

        Args:
            x: torch.Tensor
                Shape (B, N, C) with N = H * W.

        return: torch.Tensor
            Shape (B, N, C) after Physics-Attention.

        Steps:
            1) Reshape to (B, C, H, W) and apply in_project_fx and in_project_x.
            2) Permute conv outputs to (B, H, N, dim_head) per head layout.
            3) slice_weights = softmax(in_project_slice(x_mid) / clamp(temperature)).
            4) slice_to_out as in the paper.
        """
        # B N C
        B, N, C = x.shape
        # Lay out nodes on H x W lattice then NCHW for Conv2d (paper structured discretization).
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        # Conv features per cell, flatten to N nodes, split heads -> B H N dim_head (same role as fx_mid in irregular).
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        # Second conv stream for slice logits; shape B H N dim_head (comment B H N G was a typo: last dim is dim_head not G).
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N dim_head
        # Clamp temperature for numerical stability on grid; then softmax over slice_num (paper slice weights).
        slice_weights = self.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        return self.slice_to_out(x_mid, fx_mid, slice_weights)


class PhysicsAttentionStructuredMesh3D(nn.Module, AbstractPhysicsAttention):
    """
    Physics-Attention for nodes on a fixed 3D H x W x D structured voxel grid.

    Same Transolver token pipeline as 2D; local mixing uses Conv3d on the volume
    before per-node slice logits and token pooling.

    Allocated networks:
    - in_project_x, in_project_fx: nn.Conv3d(dim, inner_dim, kernel, padding same).
      Local volumetric encoders for routing and token content.
    - in_project_slice, to_q, to_k, to_v, to_out: identical roles to the 2D class.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, H=32, W=32, D=32, kernel=3):
        """
        Build 3D structured-mesh Physics-Attention.

        Args:
            dim: int
                Channel width C per voxel when flattened as nodes.
            heads: int
                Head count H.
            dim_head: int
                Per-head width d.
            dropout: float
                Dropout rate.
            slice_num: int
                Slice count M.
            H: int
                Grid extent in x; N = H * W * D.
            W: int
                Grid extent in y.
            D: int
                Grid extent in z (depth).
            kernel: int
                3D convolution kernel size.

        return: None
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        # 3D structured mesh: volumetric conv encodes local neighborhood (paper structured variant).
        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

    def forward(self, x):
        """
        Reshape nodes to H x W x D, apply Conv3d streams, then slice_to_out.

        Args:
            x: torch.Tensor
                Shape (B, N, C) with N = H * W * D.

        return: torch.Tensor
            Shape (B, N, C).

        Steps:
            1) Reshape to (B, C, H, W, D).
            2) Conv3d streams to x_mid and fx_mid in (B, H, N, dim_head) layout.
            3) slice_weights with clamped temperature.
            4) slice_to_out.
        """
        # B N C
        B, N, C = x.shape
        # Reshape flat N to H x W x D voxels, channels first for Conv3d.
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W D

        ### (1) Slice
        # Voxel conv output -> per-node head features for token aggregation.
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        # Parallel stream for slice routing logits; B H N dim_head.
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N dim_head

        # Slice
        slice_weights = self.softmax(self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        return self.slice_to_out(x_mid, fx_mid, slice_weights)
