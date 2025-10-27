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
    pid_m = tl.program_id(0) # block along seqlen of one sample
    pid_h = tl.program_id(1) # head index
    pid_b = tl.program_id(2) # batch index

    start_m = pid_m * BLOCK_M
    # batch_start_loc and batch actual seqlen
    cur_batch_start_loc = tl.load(B_Start_Loc + pid_b)
    cur_batch_seq_len = tl.load(B_Seqlen + pid_b)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # compute global indices for Q, K, V in full tensors
    # Q: [tokens, num_heads, head_dim] (row-major layout)
    # where tokens is sum of all seq_lens; B_Start_Loc gives start of each sample
    cur_q_offset_in_full = cur_batch_start_loc + offs_m
    cur_k_offset_in_full = cur_batch_start_loc + offs_n

    # load query block
    # Q block: [BLOCK_M, BLOCK_DMODEL]
    q_ptr = (
        Q +
        cur_q_offset_in_full[:, None] * stride_qbs +
        pid_h * stride_qh +
        offs_d[None, :] * stride_qd
    )
    mask_q = offs_m < cur_batch_seq_len
    q = tl.load(q_ptr, mask=mask_q[:, None], other=0.0)

    # running statistics for softmax
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # block over keys and accumulate attention
    NKV = cur_batch_seq_len
    for start_n in range(0, NKV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        cur_k_offset_in_full_block = cur_batch_start_loc + offs_n

        # load K block
        k_ptr = (
            K +
            cur_k_offset_in_full_block[None, :] * stride_kbs +
            pid_h * stride_kh +
            offs_d[:, None] * stride_kd
        )
        mask_k = offs_n < cur_batch_seq_len
        k = tl.load(k_ptr, mask=mask_k[None, :], other=0.0)

        # compute attention score: q @ k.T
        s = tl.dot(q, k) * sm_scale

        # causal and seqlen mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        causal_mask = causal_mask & mask_q[:, None] & mask_k[None, :]
        s = tl.where(causal_mask, s, float('-inf'))

        # compute softmax on this block
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)

        # update acc and stats
        acc = acc * alpha[:, None]

        # load V block
        v_ptr = (
            V +
            cur_k_offset_in_full_block[None, :] * stride_vbs +
            pid_h * stride_vh +
            offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptr, mask=mask_k[None, :], other=0.0)

        # update acc
        acc += tl.dot(p.to(v.dtype), v)

        # update softmax stats
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # normalize acc
    acc = acc / l_i[:, None]

    # store output
    out_ptr = (
        Out +
        cur_q_offset_in_full[:, None] * stride_obs +
        pid_h * stride_oh +
        offs_d[None, :] * stride_od
    )
    tl.store(out_ptr, acc.to(tl.float16), mask=mask_q[:, None])


def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    """
    Context attention on variable-length sequences.

    q, k, v: float16 tensors of shape [tokens, num_heads, head_dim]
    o: output tensor
    b_start_loc: int64 tensor of shape [batch_size,], start offset of each sequence in tokens
    b_seq_len: int64 tensor of shape [batch_size,], actual sequence lengths
    max_input_len: int, max length among sequences, used for grid sizing
    """
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3 and o.ndim == 3
    assert q.shape[0] == k.shape[0] == v.shape[0] == o.shape[0]
    num_tokens, num_heads, head_dim = q.shape
    batch_size = b_start_loc.shape[0]

    sm_scale = 1.0 / (head_dim ** 0.5)

    # constants block sizes (tuned)
    BLOCK_M = 32
    BLOCK_N = 32

    # grid
    grid = (triton.cdiv(max_input_len, BLOCK_M), num_heads, batch_size)

    # launch kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len,
        o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
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
