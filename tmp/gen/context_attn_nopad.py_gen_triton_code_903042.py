import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,
                Out,
                stride_qbs, stride_qh, stride_qd,
                stride_kbs, stride_kh, stride_kd,
                stride_vbs, stride_vh, stride_vd,
                stride_obs, stride_oh, stride_od,
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    # get batch and head
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    # compute base block indices
    block_m_offset = start_m * BLOCK_M
    offs_m = block_m_offset + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize statistics for softmax
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Compute pointers for tile of Q (masked by seqlen)
    q_ptrs = (Q + cur_batch_start_loc * stride_qbs +
               cur_head * stride_qh +
               offs_m[:, None] * stride_qbs + offs_d[None, :] * stride_qd)
    mask_m = offs_m < cur_batch_seq_len
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Loop over K,V in blocks of BLOCK_N
    # Causal attention mask applied by limiting iteration
    block_end = tl.minimum(cur_batch_seq_len, (start_m + 1) * BLOCK_M)
    end_n = tl.cdiv(block_end, BLOCK_N) * BLOCK_N

    for blk_n_idx in range(0, end_n, BLOCK_N):
        offs_n_curr = blk_n_idx + offs_n
        mask_n_valid = offs_n_curr < cur_batch_seq_len
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]  # causal

        # Load K tile
        k_ptrs = (K + cur_batch_start_loc * stride_kbs +
                 cur_head * stride_kh +
                 offs_n_curr[None, :] * stride_kbs + offs_d[:, None] * stride_kd)
        k_tile = tl.load(k_ptrs, mask=mask_n_valid[None, :], other=0.0)

        # QK^T
        qk = tl.dot(q_tile, k_tile) * sm_scale
        qk_mask = (causal_mask & mask_m[:, None] & mask_n_valid[None, :])
        qk = tl.where(qk_mask, qk, float('-inf'))

        # Softmax partial computation
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_ij)
        beta = tl.exp(m_ij[:, None] - m_ij[:, None])  # placeholder diag fix
        p = tl.exp(qk - m_ij[:, None])

        l_ij = tl.sum(p, axis=1)

        # Running softmax stats update
        l_i_new = alpha * l_i + beta[:, 0] * l_ij
        acc_scale = (l_i * alpha) / l_i_new
        acc = acc * acc_scale[:, None]

        # Load V tile
        v_ptrs = (V + cur_batch_start_loc * stride_vbs +
                 cur_head * stride_vh +
                 offs_n_curr[:, None] * stride_vbs + offs_d[None, :] * stride_vd)
        v_tile = tl.load(v_ptrs, mask=mask_n_valid[:, None], other=0.0)

        # Weighted accumulate
        acc += tl.dot(p.to(v_tile.dtype), v_tile)
        l_i = l_i_new
        m_i = m_ij

    # Final output normalize
    out_tile = acc / l_i[:, None]

    # Store output
    out_ptrs = (Out + cur_batch_start_loc * stride_obs +
               cur_head * stride_oh +
               offs_m[:, None] * stride_obs + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, out_tile.to(Out.dtype.element_ty), mask=mask_m[:, None])


@torch.no_grad()
def context_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       o: torch.Tensor, b_start_loc: torch.Tensor,
                       b_seq_len: torch.Tensor, max_input_len: int):
    # shapes
    batch, num_heads, head_dim = q.shape
    sm_scale = 1.0 / (head_dim ** 0.5)

    # Heuristics for tile sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = head_dim

    # Grid setup
    grid = (batch, num_heads, triton.cdiv(max_input_len, BLOCK_M))
    num_warps = 4 if head_dim <= 64 else 8

    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len,
        o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=1
    )


##################################################################################################################################################





def test_context_attention_fwd():

    Z, H, N_CTX, D_HEAD = 4, 6, 1024, 128

    dtype = torch.float16

    Z = 3

    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)

    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)

    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)



    max_input_len = N_CTX

    Z = 4

    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")

    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")



    b_seq_len[0] = 512

    b_seq_len[1] = 1024

    b_seq_len[2] = 512

    b_seq_len[3] = 1024



    for i in range(1, Z):

        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]



    # case 1: Normal call with the given setup (should run without issue)

    result_case_1 = {}

    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len)

    result_case_1['normal'] = o.clone()



    # case 2: Alter max_input_len, making it smaller or larger to check boundary conditions

    max_input_len_case_2 = 512

    result_case_2 = {}

    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len_case_2)

    result_case_2['max_input_len_512'] = o.clone()



    # case 3: Modify batch size Z to test larger batch processing

    Z_case_3 = 8  # larger batch size

    b_start_loc_case_3 = torch.zeros((Z_case_3,), dtype=torch.int32, device="cuda")

    b_seq_len_case_3 = torch.ones((Z_case_3,), dtype=torch.int32, device="cuda")

    b_seq_len_case_3[0] = 512

    b_seq_len_case_3[1] = 1024

    for i in range(1, Z_case_3):

        b_start_loc_case_3[i] = b_start_loc_case_3[i - 1] + b_seq_len_case_3[i - 1]



    result_case_3 = {}

    context_attention_fwd(q, k, v, o, b_start_loc_case_3, b_seq_len_case_3, max_input_len)

    result_case_3['batch_size_8'] = o.clone()



    # case 4: Test with different sequence lengths to check handling of varying sequence lengths

    b_seq_len_case_4 = torch.tensor([512, 256, 1024, 512], dtype=torch.int32, device="cuda")

    b_start_loc_case_4 = torch.zeros((4,), dtype=torch.int32, device="cuda")

    for i in range(1, 4):

        b_start_loc_case_4[i] = b_start_loc_case_4[i - 1] + b_seq_len_case_4[i - 1]



    result_case_4 = {}

    context_attention_fwd(q, k, v, o, b_start_loc_case_4, b_seq_len_case_4, max_input_len)

    result_case_4['varying_seq_len'] = o.clone()



    # Return all results in a dictionary

    return {

        'result_case_1': result_case_1,

        'result_case_2': result_case_2,

        'result_case_3': result_case_3,

        'result_case_4': result_case_4

    }



# Execute and save results

result_gold = test_context_attention_fwd()
