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
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    block_start_m = start_m * BLOCK_M

    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    mask_m = offs_m < cur_batch_seq_len

    q_ptrs = (
        Q
        + (cur_batch_start_loc + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    k_base = (
        K
        + cur_batch_start_loc * stride_kbs
        + cur_head * stride_kh
    )

    v_base = (
        V
        + cur_batch_start_loc * stride_vbs
        + cur_head * stride_vh
    )

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        offs_n_rel = start_n + offs_n
        mask_n = offs_n_rel < cur_batch_seq_len

        k_ptrs = (
            k_base
            + offs_n_rel[None, :] * stride_kbs
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            v_base
            + offs_n_rel[:, None] * stride_vbs
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        qk = tl.dot(q, k)
        qk *= sm_scale

        mask_causal = (offs_m[:, None] >= offs_n_rel[None, :]) & mask_n[None, :]
        qk = tl.where(mask_causal, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scaled = p / l_i_new[:, None]
        acc_scale = (l_i / l_i_new) * alpha
        acc = acc * acc_scale[:, None]

        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        p_scaled = p_scaled.to(v.dtype)
        acc += tl.dot(p_scaled, v)

        l_i = l_i_new
        m_i = m_i_new

    out_ptrs = (
        Out
        + (cur_batch_start_loc + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


@torch.no_grad()
def context_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, b_start_loc: torch.Tensor, b_seq_len: torch.Tensor, max_input_len: int):
    batch_size = b_start_loc.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    sm_scale = 1.0 / (head_dim ** 0.5)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = head_dim

    grid = (batch_size, num_heads, triton.cdiv(max_input_len, BLOCK_M))

    num_warps = 4

    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return o

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
