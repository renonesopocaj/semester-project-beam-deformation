import math
import torch
import torch.nn as nn
from einops import rearrange


class RotaryEmbedding(nn.Module):
    """
    Build rotary position embedding (RoPE) frequency angles from scalar coordinates.

    RoPE encodes relative position by multiplying pairs of channels with a 2D rotation
    whose angle depends on position; here angles are precomputed as freqs so that
    apply_rotary_pos_emb can mix t with cos(freqs) and rotate_half(t) with sin(freqs).

    Allocated state:
    - inv_freq: buffer of shape (dim//2,) with geometrically spaced inverse frequencies
      1 / 10000^{2k/dim} for k = 0, 1, ... (standard RoPE spectrum).
    - min_freq, scale: scalars rescaling the effective angle theta = coordinates * scale / min_freq.
    """

    def __init__(self, dim, min_freq=1 / 2, scale=1.):
        """
        Register inverse frequency buffer for RoPE angles.

        Args:
            dim: int
                Full rotary dimension d (must be even for the usual pairing; forward duplicates half).
            min_freq: float
                Denominator scale inside effective angle (larger -> slower phase change).
            scale: float
                Multiplier on coordinates before dividing by min_freq.

        return: None
        """
        super().__init__()
        # Inverse frequencies for RoPE: pairs of dims get 1/10000^{2k/d} (standard rotary spectrum).
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        """
        Compute angle grid freqs of shape (..., n, dim) for rotary mixing.

        Args:
            coordinates: torch.Tensor
                Positions of shape (batch, n) or broader leading dims (... , n); real-valued.
            device: torch.device
                Target device for dtype/device alignment with inv_freq buffer.

        return: torch.Tensor
            Angles with last dimension dim: concat(freqs_half, freqs_half) so length matches
            rotate_half pairing in apply_rotary_pos_emb.

        Steps:
            1) Cast coordinates to device and dtype of inv_freq.
            2) Scale coordinates by scale / min_freq.
            3) Outer product with inv_freq -> (..., n, dim//2), then duplicate along last dim.
        """
        # coordinates [b, n]
        # Match dtype/device to buffer for stable trig.
        t = coordinates.to(device).type_as(self.inv_freq)
        # Rescale coordinates so effective angle = t * scale / min_freq.
        t = t * (self.scale / self.min_freq)
        # Outer product: each position with each frequency -> angles for sin/cos (half spectrum).
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        # Duplicate so last dim equals full d for pairing with rotate_half in apply_rotary_pos_emb.
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    """
    Split feature dimension into two halves and apply the RoPE orthogonal swap (-x2, x1).

    Args:
        x: torch.Tensor
            Tensor whose last dimension is even; pairs of dims are grouped as (x1, x2).

    return: torch.Tensor
        Same shape as x; last dim is concatenation of -x2 and x1 along paired structure.

    Steps:
        1) Rearrange last dim into two groups of size d/2 each.
        2) Return cat(-x2, x1, dim=-1).
    """
    # Pair consecutive dims (j=2 groups) for the RoPE swap trick.
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    # Classic rotate-half: (-x2, x1) concatenated is orthogonal mix used with sin/cos.
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply 1D rotary position embedding to the last dimension of t.

    Core mix: output = t * cos(freqs) + rotate_half(t) * sin(freqs), elementwise.

    Args:
        t: torch.Tensor
            Features to rotate; last dim must equal freqs last dim.
        freqs: torch.Tensor
            Angles from RotaryEmbedding.forward, same trailing shape as t for broadcast.

    return: torch.Tensor
        Rotary-transformed tensor, same shape as t.
    """
    # RoPE: modulate t by cos(freqs) and orthogonal part by sin(freqs) (paper-style 1D rotary on last dim).
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    """
    Apply independent 1D RoPE to the first and second halves of the last dimension.

    Args:
        t: torch.Tensor
            Shape (batch, heads, n, d); last dim split as d//2 for x-axes features and d//2 for y.
        freqs_x: torch.Tensor
            Rotary angles for the first half; shape broadcastable with t[..., :d//2].
        freqs_y: torch.Tensor
            Rotary angles for the second half; shape broadcastable with t[..., d//2:].

    return: torch.Tensor
        Concatenation of apply_rotary_pos_emb on each half along last dim.

    Steps:
        1) Split t into t_x, t_y along last dim.
        2) apply_rotary_pos_emb(t_x, freqs_x) and same for y.
        3) Concatenate along feature dim.
    """
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    # First half of channels treated as x-axis coordinates, second half as y-axis (2D RoPE split).
    t_x, t_y = t[..., :d // 2], t[..., d // 2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal absolute positional encoding (Vaswani et al. style).

    For sequence positions pos = 0, 1, ..., and even/odd dimensions 2i, 2i+1:
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    The table is registered as a non-trainable buffer; forward adds the first seq_len rows
    to the input and applies dropout. No trainable Linear layers: only nn.Dropout.
    """

    def __init__(self, d_model, dropout, max_len=421 * 421):
        """
        Precompute the PE matrix up to max_len positions.

        Args:
            d_model: int
                Model width (must match last dim of x in forward).
            dropout: float
                Dropout probability applied after adding PE.
            max_len: int
                Maximum sequence length allocated in the buffer table.

        return: None
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # Table of shape (max_len, d_model) filled with sinusoids (Vaswani-style absolute PE).
        pe = torch.zeros(max_len, d_model)
        # Integer positions 0 .. max_len-1 as column vector for broadcasting.
        position = torch.arange(0, max_len).unsqueeze(1)
        # Geometric progression of wavelengths in log space for even dimensions.
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # Even indices: sine; encodes position along one frequency set.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices: cosine; phase-shifted same frequencies.
        pe[:, 1::2] = torch.cos(position * div_term)
        # Batch dimension 1 for adding to (B, seq, d_model).
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add stored sinusoidal encoding to x along the sequence dimension.

        Args:
            x: torch.Tensor
                Shape (batch, seq_len, d_model).

        return: torch.Tensor
            Same shape as x after add and dropout.

        Steps:
            1) x = x + pe[:, :seq_len] (buffer frozen for gradients).
            2) Apply dropout.
        """
        # Add fixed sinusoidal bias to sequence positions; buffer not trained.
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def sin_timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Sinusoidal embedding of scalar timesteps (DDPM / score-model style).

    For half = floor(dim/2), frequencies omega_k decrease geometrically from 1 to
    about 1/max_period. The embedding is [cos(t * omega), sin(t * omega)] along the
    feature axis (length 2*half), padded with one zero if dim is odd. The argument
    repeat_only is accepted for API compatibility but is not used in this implementation.

    Args:
        timesteps: torch.Tensor
            One scalar time per batch row, shape (N,) or broadcastable to (N, 1); may be fractional.
        dim: int
            Output embedding dimension per row.
        max_period: float
            Controls the minimum frequency in the geometric progression.
        repeat_only: bool
            Unused in the current implementation; reserved for call-site compatibility.

    return: torch.Tensor
        Tensor of shape (N, dim) with sinusoidal features per timestep.
    """

    # Half the output dims drive cos/sin pairs (DDPM-style timestep embedding).
    half = dim // 2
    # Decreasing frequencies from 1 to ~1/max_period across half channels.
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    # Broadcast multiply: each batch element's scalar time * each frequency -> phase grid (N, half).
    args = timesteps[:, None].float() * freqs[None]
    # Concat cos then sin along feature dim (standard sinusoidal timestep vector).
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        # Pad with zero if dim odd so output length exactly dim.
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
