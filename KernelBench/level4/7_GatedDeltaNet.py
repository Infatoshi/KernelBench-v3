import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Gated DeltaNet: Linear Attention with Gated Delta Rule
# Reference: https://arxiv.org/abs/2412.06464 (ICLR 2025)
# Implementation: https://github.com/NVlabs/GatedDeltaNet, https://github.com/fla-org/flash-linear-attention
#
# Gated DeltaNet combines two mechanisms for efficient sequence modeling:
# 1. Gating (alpha_t): Adaptive memory decay, controls state retention
# 2. Delta rule (beta_t): Targeted memory updates via error correction
#
# Core recurrence:
#   S_t = alpha_t * S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T
#
# This can be rewritten as:
#   S_t = alpha_t * S_{t-1} - beta_t * S_{t-1} @ k_t @ k_t^T + beta_t * v_t @ k_t^T
#
# Output: o_t = S_t @ q_t
#
# Key optimization targets:
# 1. Chunkwise parallelization using Householder transform
# 2. Fused gate computation (alpha, beta from input)
# 3. Efficient state matrix updates avoiding O(T^2) memory
# 4. Tensor core utilization for the matrix-vector products


class Model(nn.Module):
    """
    Gated DeltaNet: Linear Attention with Gated Delta Rule

    Mathematical formulation:
    Given input x_t at timestep t:
    - q_t, k_t = query/key projections (d_k dimensional)
    - v_t = value projection (d_v dimensional)
    - alpha_t = sigmoid(a_proj(x_t)) in (0, 1) - decay gate
    - beta_t = sigmoid(b_proj(x_t)) in (0, 1) - delta learning rate

    State update (the delta rule with gating):
        S_t = alpha_t * S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T

    Output:
        o_t = S_t @ q_t

    Key optimization targets:
    1. The naive O(T * d_k * d_v) recurrence is sequential
    2. Chunkwise parallel algorithm uses Householder transforms
    3. State matrix S is (d_v, d_k) per head - can be large
    4. Fuse alpha/beta computation with state updates

    The naive implementation:
    - Loops over time steps sequentially
    - Materializes full state matrix at each step
    - No parallelization across sequence length

    An optimized kernel should:
    - Use chunkwise parallelization (process chunks of C tokens in parallel)
    - Exploit Householder structure for efficient cumulative products
    - Minimize state matrix memory traffic
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_short_conv = use_short_conv

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        # Gating projections
        # alpha: decay gate, controls how much of previous state to retain
        # beta: delta learning rate, controls update magnitude
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

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

        # Output gate with RMSNorm + SiLU
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor for keys (prevents state explosion)
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Gated DeltaNet.

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
            # (batch, seq, dim) -> (batch, dim, seq) for conv1d
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

        # Compute gating values
        alpha = torch.sigmoid(self.a_proj(x))  # (batch, seq, num_heads)
        beta = torch.sigmoid(self.b_proj(x))   # (batch, seq, num_heads)

        # Scale keys to prevent state explosion
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
            alpha_t = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (batch, num_heads, 1, 1)
            beta_t = beta[:, t, :].unsqueeze(-1).unsqueeze(-1)    # (batch, num_heads, 1, 1)

            # Delta rule update:
            # S_t = alpha_t * S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) @ k_t^T
            #     = alpha_t * S_{t-1} - beta_t * S_{t-1} @ k_t @ k_t^T + beta_t * v_t @ k_t^T

            # Compute S @ k: (batch, num_heads, head_dim_v, head_dim_qk) @ (batch, num_heads, head_dim_qk, 1)
            #             -> (batch, num_heads, head_dim_v, 1)
            k_t_col = k_t.unsqueeze(-1)  # (batch, num_heads, head_dim_qk, 1)
            S_k = torch.matmul(S, k_t_col).squeeze(-1)  # (batch, num_heads, head_dim_v)

            # Compute error: S @ k - v
            error = S_k - v_t  # (batch, num_heads, head_dim_v)

            # Outer product: error @ k^T -> (batch, num_heads, head_dim_v, head_dim_qk)
            error_outer_k = torch.einsum('bhi,bhj->bhij', error, k_t)

            # Value outer product: v @ k^T
            v_outer_k = torch.einsum('bhi,bhj->bhij', v_t, k_t)

            # State update: S = alpha * S - beta * error @ k^T
            # Equivalently: S = alpha * S - beta * (S @ k - v) @ k^T
            S = alpha_t * S - beta_t * error_outer_k

            # Output: o = S @ q
            q_t_col = q_t.unsqueeze(-1)  # (batch, num_heads, head_dim_qk, 1)
            o_t = torch.matmul(S, q_t_col).squeeze(-1)  # (batch, num_heads, head_dim_v)

            outputs.append(o_t)

        # Stack outputs: (seq, batch, num_heads, head_dim_v) -> (batch, seq, num_heads, head_dim_v)
        o = torch.stack(outputs, dim=1)  # (batch, seq, num_heads, head_dim_v)

        # Apply output normalization per head
        o = self.o_norm(o)

        # Apply output gate
        g = torch.sigmoid(self.g_proj(x))  # (batch, seq, num_heads * head_dim_v)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


# Configuration matching typical LLM settings
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
