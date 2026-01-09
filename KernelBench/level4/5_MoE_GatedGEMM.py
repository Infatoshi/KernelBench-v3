import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# MoE Gated GEMM (Mixture of Experts with Fused Gating)
# Used in: Mixtral, DeepSeek-V3, Grok, DBRX, Arctic
# Reference: https://arxiv.org/abs/2401.04088 (Mixtral of Experts)
#
# In MoE, the gating mechanism selects which experts process each token.
# The naive approach:
# 1. Compute gate scores for all experts
# 2. Select top-k experts per token
# 3. Loop through selected experts, gathering tokens for each
# 4. Run expert MLP, scatter results back
#
# This sequential loop is highly inefficient. A fused kernel should:
# - Batch tokens across experts efficiently
# - Avoid explicit gather/scatter
# - Optionally fuse gate scoring with expert selection
#
# This problem focuses on the "gated dual GEMM" pattern:
# output = gate * (W_up * x) where gate comes from sigmoid(W_gate * x)


class Model(nn.Module):
    """
    MoE Expert with Gated GEMM (SiLU-gated FFN).

    This is a SINGLE expert's computation pattern, used in MoE FFN:
    output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    The "gated GEMM" refers to: SiLU(gate_proj(x)) * up_proj(x)
    This is two parallel GEMMs followed by element-wise multiply.

    Key optimization targets:
    1. Fuse gate_proj and up_proj into single memory read of x
    2. Fuse SiLU activation with multiplication
    3. Optimize memory layout for the dual GEMM pattern
    4. When batched across experts, enable parallel execution

    The naive implementation runs two separate matmuls.
    An optimized kernel should read x once and compute both projections.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # Expert weights: each expert has gate_proj, up_proj, down_proj
        # Shape: (num_experts, out_features, in_features) for batched matmul
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,              # (batch, seq_len, hidden_size)
        expert_indices: torch.Tensor, # (batch, seq_len, top_k) - selected expert indices
        expert_weights: torch.Tensor, # (batch, seq_len, top_k) - routing weights
    ) -> torch.Tensor:
        """
        MoE forward with gated dual GEMM.

        Each token is processed by top_k experts, weighted by expert_weights.
        """
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]

        # Reshape for processing
        x_flat = x.view(-1, self.hidden_size)  # (batch * seq_len, hidden)
        num_tokens = x_flat.shape[0]

        # INEFFICIENT: Loop through each expert
        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)

        for expert_idx in range(self.num_experts):
            # Find which (token, slot) pairs use this expert
            # expert_indices: (batch, seq_len, top_k)
            expert_mask = (expert_indices == expert_idx)  # (batch, seq_len, top_k)

            if not expert_mask.any():
                continue

            # Get token indices and their routing weights for this expert
            batch_idx, seq_idx, slot_idx = torch.where(expert_mask)
            token_indices = batch_idx * seq_len + seq_idx
            weights = expert_weights[batch_idx, seq_idx, slot_idx]  # (num_selected,)

            # Get tokens for this expert
            expert_input = x_flat[token_indices]  # (num_selected, hidden)

            # GATED DUAL GEMM: The main optimization target
            # gate = SiLU(expert_input @ gate_proj.T)
            # up = expert_input @ up_proj.T
            # intermediate = gate * up
            # expert_output = intermediate @ down_proj.T

            gate = F.silu(F.linear(expert_input, self.gate_proj[expert_idx]))
            up = F.linear(expert_input, self.up_proj[expert_idx])
            intermediate = gate * up
            expert_output = F.linear(intermediate, self.down_proj[expert_idx])

            # Accumulate weighted output
            output.index_add_(0, token_indices, expert_output * weights.unsqueeze(-1))

        return output.view(batch, seq_len, self.hidden_size)


# Mixtral-style configuration
batch_size = 4
seq_len = 2048
hidden_size = 4096
intermediate_size = 14336  # Mixtral uses large intermediate
num_experts = 8
top_k = 2  # Each token routed to 2 experts


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Random expert selection (in real MoE, this comes from gating network)
    expert_indices = torch.stack([
        torch.randperm(num_experts)[:top_k]
        for _ in range(batch_size * seq_len)
    ]).view(batch_size, seq_len, top_k)

    # Random routing weights (normalized)
    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)

    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_experts]
