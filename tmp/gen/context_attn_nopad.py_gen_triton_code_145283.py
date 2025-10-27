import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = start_m * BLOCK_M

    # --- offsets ---
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # broadcast helpers
    seq_start_m = tl.full([BLOCK_M], cur_batch_in_all_start_index, dtype=tl.int32)
    offs_m_real = seq_start_m + offs_m

    # --- load Q ---
    offs_q = offs_m_real[:, None] * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + offs_q, mask=(offs_m < cur_batch_seq_len)[:, None], other=0.0)

    # --- init running stats for softmax ---
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # block masking
    do_mask = block_start_loc < cur_batch_seq_len

    # loop over K/V blocks
    for start_n in range(0, do_mask * ((start_m + 1) * BLOCK_M), BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_real = start_n + offs_n

        # load K transposed (BLOCK_DMODEL, BLOCK_N)
        offs_k = (cur_batch_in_all_start_index + offs_n_real)[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + offs_k, mask=(offs_n_real < cur_batch_seq_len)[None, :], other=0.0)

        # QK = q @ k
        qk = tl.dot(q, k)                       # (BLOCK_M, BLOCK_N)
        qk = qk * sm_scale
        # causal mask
        causal_mask = offs_m[:, None] >= offs_n_real[None, :]
        qk = tl.where(causal_mask & (offs_m[:, None] < cur_batch_seq_len) & (offs_n_real[None, :] < cur_batch_seq_len), qk, float('-inf'))

        # online softmax
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij

        # acc rescale
        acc_scale = l_i / l_new * alpha
        acc_o = acc_o * acc_scale[:, None]

        # scale P and accumulate
        p_scale = beta / l_new
        p = p * p_scale[:, None]

        # load V (BLOCK_N, BLOCK_DMODEL)
        offs_v = (cur_batch_in_all_start_index + offs_n_real)[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd
        v_blk = tl.load(V + offs_v, mask=(offs_n_real < cur_batch_seq_len)[:, None], other=0.0)

        p = p.to(v_blk.dtype)
        acc_o += tl.dot(p, v_blk)

        m_i = m_new
        l_i = l_new

    # final output write
    offs_o = offs_m_real[:, None] * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    tl.store(Out + offs_o, acc_o.to(Out.type.element_ty), mask=(offs_m < cur_batch_seq_len)[:, None])


@torch.no_grad()
def context_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                          b_start_loc: torch.Tensor, b_seq_len: torch.Tensor, max_input_len: int):
    # q/k/v shapes: [total_tokens, num_heads, head_dim]
    total_tokens, num_heads, head_dim = q.shape
    assert k.shape == v.shape == (total_tokens, num_heads, head_dim)
    assert q.dtype in (torch.float16, torch.float32)

    sm_scale = 1.0 / (head_dim ** 0.5)
    batch = b_seq_len.shape[0]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim

    grid = (batch, num_heads, triton.cdiv(max_input_len, BLOCK_M))

    num_warps = 4 if head_dim <= 64 else 8

    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
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
