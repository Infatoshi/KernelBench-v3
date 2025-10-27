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
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # retrieve sequence info
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = start_m * BLOCK_M

    # m_offset along sequence
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # pointers to Q tile: shape (BLOCK_M, BLOCK_DMODEL)
    q_ptrs = Q + (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize softmax statistics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # iterate over BLOCK_N sized blocks in keys/values
    end_n = tl.cdiv(cur_batch_seq_len, BLOCK_N)
    for start_n in range(0, end_n, 1):
        start_n = tl.multiple_of(start_n, 1)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # load K tile, shape (BLOCK_N, BLOCK_DMODEL)
        k_ptrs = K + (cur_batch_in_all_start_index + offs_n[None, :]) * stride_kbs + cur_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < cur_batch_seq_len, other=0.0)

        # compute attention scores (without masking) scale
        scores = tl.dot(q, k.trans())  # (BLOCK_M, BLOCK_N)
        scores = scores * sm_scale

        # causal mask: allow attending only to current and previous positions
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        scores = tl.where(causal_mask & (offs_n[None, :] < cur_batch_seq_len), scores, float("-inf"))

        # softmax logic
        m_ij = tl.max(scores, 1)  # max over N for each M
        p_ij = tl.exp(scores - m_ij[:, None])
        l_ij = tl.sum(p_ij, 1)

        # update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.exp(m_ij - m_i_new) * l_ij
        l_i_scale = l_i / l_i_new * tl.exp(m_i - m_i_new)
        acc_scale = l_i_scale
        p_ij_norm = p_ij / l_i_new[:, None]

        # load V tile, shape (BLOCK_N, BLOCK_DMODEL)
        v_ptrs = V + (cur_batch_in_all_start_index + offs_n[:, None]) * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < cur_batch_seq_len, other=0.0)

        # scale accumulator and accumulate
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p_ij_norm.to(v.dtype), v)

        m_i = m_i_new
        l_i = l_i_new

    # write back output tile
    o_ptrs = Out + (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def context_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                          b_start_loc: torch.Tensor, b_seq_len: torch.Tensor, max_input_len: int):
    """
    q: tensor of shape (sum_seqlen, num_heads, head_dim)
    k, v: same layout as q
    o: output tensor, same shape as q
    b_start_loc: long tensor (B,) with the starting position of each batch in the flattened total length
    b_seq_len: long tensor (B,) sequence lengths per batch
    max_input_len: the maximum sequence length across the batch
    """
    head_dim = q.shape[2]
    assert head_dim == BLOCK_DMODEL
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim
    sm_scale = 1.0 / (head_dim ** 0.5)

    batch_size = b_seq_len.shape[0]
    num_heads = q.shape[1]

    grid = (batch_size, num_heads, triton.cdiv(max_input_len, BLOCK_M))

    num_warps = 4 if head_dim <= 64 else 8

    _fwd_kernel[grid](
        Q=q,
        K=k,
        V=v,
        sm_scale=sm_scale,
        B_Start_Loc=b_start_loc,
        B_Seqlen=b_seq_len,
        Out=o,
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_vbs=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        stride_obs=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
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
