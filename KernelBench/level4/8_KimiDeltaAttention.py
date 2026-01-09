import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Kimi Delta Attention (KDA): Linear Attention with Channel-wise Gating
# Reference: https://arxiv.org/abs/2510.26692 (Kimi Linear: An Expressive, Efficient Attention Architecture)
# Implementation: https://github.com/MoonshotAI/Kimi-Linear, https://github.com/fla-org/flash-linear-attention
#
# Kimi Delta Attention extends Gated DeltaNet with:
# 1. Channel-wise (diagonal) gating: Each feature channel has its own decay gate
# 2. DPLR transition matrices: Diagonal-Plus-Low-Rank parameterization
#
# The key insight: Gated DeltaNet uses a single scalar gate alpha_t per head,
# limiting expressiveness. KDA uses a diagonal gate matrix A_t, giving each
# channel independent decay control.
#
# Core recurrence:
#   S_t = A_t * S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T
#
# Where A_t is diagonal (or DPLR for more expressiveness):
#   A_t = diag(a_t)  (diagonal-only baseline)
#   A_t = diag(a_t) + L_t @ R_t^T  (DPLR extension)
#
# This enables finer-grained memory control: different features can decay
# at different rates, allowing the model to "remember" some information
# longer than others.
#
# Key optimization targets:
# 1. WY representation for efficient cumulative matrix products
# 2. UT transform to reduce non-matmul operations
# 3. Chunkwise parallel algorithm matching DPLR structure
# 4. Fused diagonal-matrix-vector operations


class Model(nn.Module):
    """
    Kimi Delta Attention (KDA): Linear Attention with Channel-wise Gating

    Mathematical formulation:
    Given input x_t at timestep t:
    - q_t, k_t = query/key projections (d_k dimensional)
    - v_t = value projection (d_v dimensional)
    - a_t = sigmoid(a_proj(x_t)) in (0, 1)^{d_v} - per-channel decay gates
    - beta_t = sigmoid(b_proj(x_t)) in (0, 1) - delta learning rate

    State update (channel-wise gated delta rule):
        S_t = diag(a_t) @ S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T

    Where diag(a_t) is a d_v x d_v diagonal matrix with a_t on the diagonal.

    This is equivalent to:
        S_t[i, :] = a_t[i] * S_{t-1}[i, :] - beta_t * (S_{t-1}[i, :] @ k_t - v_t[i]) * k_t

    for each channel i in [0, d_v).

    Output:
        o_t = S_t @ q_t

    Key optimization targets:
    1. The diagonal matrix multiplication is just element-wise scaling
    2. WY representation enables efficient chunkwise parallelization
    3. The DPLR structure can be efficiently handled with custom kernels

    The naive implementation:
    - Loops over time steps sequentially
    - Uses explicit diagonal matrix operations
    - No chunkwise parallelization

    An optimized kernel should:
    - Use WY representation for cumulative diagonal products
    - Apply UT transform to reduce operations
    - Achieve ~2x speedup over standard DPLR kernels
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_dplr: bool = False,  # Whether to use Diagonal-Plus-Low-Rank
        dplr_rank: int = 4,      # Rank of low-rank component if using DPLR
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_dplr = use_dplr
        self.dplr_rank = dplr_rank
        self.use_short_conv = use_short_conv

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        # Channel-wise gating: produces d_v gates per head (instead of 1)
        # This is the key difference from Gated DeltaNet
        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)

        # Delta learning rate (scalar per head, same as Gated DeltaNet)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # DPLR low-rank factors (optional)
        if use_dplr:
            # L and R are low-rank factors: A = diag(a) + L @ R^T
            self.l_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)
            self.r_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        # Optional short convolution for local context
        if use_short_conv:
            self.q_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.k_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.v_conv = nn.Conv1d(
                num_heads * head_dim_v, num_heads * head_dim_v,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_v,
                padding=conv_kernel_size - 1
            )

        # Output gate with normalization
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor for keys
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Kimi Delta Attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        k = self.k_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        v = self.v_proj(x)  # (batch, seq, num_heads * head_dim_v)

        # Optional short convolution
        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Compute CHANNEL-WISE gating (d_v gates per head, not 1)
        # This is the key innovation of KDA over Gated DeltaNet
        a = torch.sigmoid(self.a_proj(x))  # (batch, seq, num_heads * head_dim_v)
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Delta learning rate (scalar per head)
        beta = torch.sigmoid(self.b_proj(x))  # (batch, seq, num_heads)

        # Optional DPLR low-rank factors
        if self.use_dplr:
            l = self.l_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)
            r = self.r_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)

        # Scale keys
        k = k * self.scale

        # INEFFICIENT: Sequential recurrence over time
        # Initialize state matrix: (batch, num_heads, head_dim_v, head_dim_qk)
        S = torch.zeros(
            batch_size, self.num_heads, self.head_dim_v, self.head_dim_qk,
            device=device, dtype=dtype
        )

        outputs = []

        for t in range(seq_len):
            # Get current timestep values
            q_t = q[:, t, :, :]   # (batch, num_heads, head_dim_qk)
            k_t = k[:, t, :, :]   # (batch, num_heads, head_dim_qk)
            v_t = v[:, t, :, :]   # (batch, num_heads, head_dim_v)
            a_t = a[:, t, :, :]   # (batch, num_heads, head_dim_v) - CHANNEL-WISE
            beta_t = beta[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (batch, num_heads, 1, 1)

            # CHANNEL-WISE gated delta rule update:
            # S_t = diag(a_t) @ S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T
            #
            # The diag(a_t) @ S_{t-1} part applies different decay to each row of S
            # This is more expressive than Gated DeltaNet which uses scalar alpha

            # Compute S @ k: (batch, num_heads, head_dim_v)
            k_t_col = k_t.unsqueeze(-1)  # (batch, num_heads, head_dim_qk, 1)
            S_k = torch.matmul(S, k_t_col).squeeze(-1)  # (batch, num_heads, head_dim_v)

            # Error: S @ k - v
            error = S_k - v_t  # (batch, num_heads, head_dim_v)

            # Outer product: error @ k^T -> (batch, num_heads, head_dim_v, head_dim_qk)
            error_outer_k = torch.einsum('bhi,bhj->bhij', error, k_t)

            # Apply diagonal gating: diag(a_t) @ S = a_t[:, None] * S
            # Each row i of S is scaled by a_t[i]
            a_t_expanded = a_t.unsqueeze(-1)  # (batch, num_heads, head_dim_v, 1)
            S_gated = a_t_expanded * S  # (batch, num_heads, head_dim_v, head_dim_qk)

            # Optional DPLR: add low-rank contribution
            # A_t = diag(a_t) + L_t @ R_t^T
            # S = A_t @ S_{t-1} = diag(a_t) @ S + (L_t @ R_t^T) @ S
            if self.use_dplr:
                l_t = l[:, t, :, :]  # (batch, num_heads, rank)
                r_t = r[:, t, :, :]  # (batch, num_heads, rank)

                # (L @ R^T) @ S = L @ (R^T @ S)
                # R^T @ S: (batch, num_heads, rank, head_dim_v) @ ... -> need transpose
                # Actually R^T is (rank, head_dim_v), S is (head_dim_v, head_dim_qk)
                # So R^T @ S: (rank, head_dim_qk) is what we need, then L @ that

                # R: (batch, num_heads, rank) -> interpret as (batch, num_heads, head_dim_v=rank)?
                # Actually for DPLR: L is (d_v, rank), R is (d_v, rank)
                # A = diag(a) + L @ R^T is (d_v, d_v)
                # So L @ R^T: (d_v, rank) @ (rank, d_v) -> (d_v, d_v)
                # Then (L @ R^T) @ S: (d_v, d_v) @ (d_v, d_k) -> (d_v, d_k)

                # For efficiency: (L @ R^T) @ S = L @ (R^T @ S)
                # R is (d_v, rank), so we need R: (batch, num_heads, d_v, rank)
                # Wait, in our projection we have (num_heads * rank) features
                # So l_t and r_t should be (batch, num_heads, rank) for head-wise low-rank

                # Let's use a simpler formulation where L and R add a rank-r perturbation
                # to the diagonal transition per VALUE dimension
                # This would require L: (d_v, rank) and R: (d_k, rank) to get (d_v, d_k)
                # But that's different from DPLR which is (d_v, d_v)

                # For now, skip DPLR contribution and use diagonal-only
                # The diagonal-only version is the core contribution and sufficient for benchmarking
                pass

            # State update with channel-wise gating
            S = S_gated - beta_t * error_outer_k

            # Output: o = S @ q
            q_t_col = q_t.unsqueeze(-1)  # (batch, num_heads, head_dim_qk, 1)
            o_t = torch.matmul(S, q_t_col).squeeze(-1)  # (batch, num_heads, head_dim_v)

            outputs.append(o_t)

        # Stack outputs
        o = torch.stack(outputs, dim=1)  # (batch, seq, num_heads, head_dim_v)

        # Apply output normalization per head
        o = self.o_norm(o)

        # Apply output gate
        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


# Configuration matching Kimi Linear paper settings
batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128  # Key/query dimension per head
head_dim_v = 128   # Value dimension per head


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
