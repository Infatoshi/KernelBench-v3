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
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // (tl.num_programs(2))
    head_id = pid_bh % (tl.num_programs(2))

    seq_start = tl.load(B_Start_Loc + batch_id)
    seq_len = tl.load(B_Seqlen + batch_id)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    mask_m = offs_m < seq_len

    # Compute Q offset
    offs_q = (seq_start + offs_m) * stride_qbs + head_id * stride_qh + offs_d * stride_qd
    q = tl.load(Q + offs_q, mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_DMODEL), other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    max_score = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    sum_exp = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n_full = start_n + offs_n
        mask_n = offs_n_full < seq_len
        
        # Compute K offset
        offs_k = (seq_start + offs_n_full) * stride_kbs + head_id * stride_kh + offs_d * stride_kd
        k = tl.load(K + offs_k, mask=mask_n[None, :] & (offs_d[:, None] < BLOCK_DMODEL), other=0.0)

        # Compute attention scores
        scores = tl.dot(q, k) * sm_scale

        # Mask for causal attention
        causal_mask = offs_m[:, None] >= offs_n_full[None, :]
        scores = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], scores, float('-inf'))

        # Online softmax
        row_max = tl.max(scores, axis=1)
        row_max_old = max_score
        max_score = tl.maximum(max_score, row_max)

        # Rescale accumulated values
        acc_scale = tl.exp(row_max_old - max_score)
        acc = acc * acc_scale[:, None]
        sum_exp = sum_exp * acc_scale

        # Accumulate
        p = tl.exp(scores - max_score[:, None])
        sum_exp += tl.sum(p, axis=1)

        # Load V
        offs_v = (seq_start + offs_n_full) * stride_vbs + head_id * stride_vh + offs_d * stride_vd
        v = tl.load(V + offs_v, mask=mask_n[None, :] & (offs_d[:, None] < BLOCK_DMODEL), other=0.0)

        acc += tl.dot(p.to(v.dtype), v)

    # Final normalization
    acc = acc / sum_exp[:, None]

    # Write output
    offs_o = (seq_start + offs_m) * stride_obs + head_id * stride_oh + offs_d * stride_od
    tl.store(Out + offs_o, acc.to(Out.type.element_ty), mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_DMODEL))


def context_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                          b_start_loc: torch.Tensor, b_seq_len: torch.Tensor, max_input_len: int):
    # Assume q shape: [total_tokens, num_heads, head_dim]
    # total_tokens = sum(b_seq_len)
    # k, v shape: [total_tokens, num_heads, head_dim]
    # o shape: [total_tokens, num_heads, head_dim]

    total_tokens, num_heads, head_dim = q.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim

    sm_scale = 1.0 / (head_dim ** 0.5)

    grid = lambda meta: (
        triton.cdiv(max_input_len, meta['BLOCK_M']),
        total_tokens,
        num_heads
    )

    num_warps = 4

    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        num_warps=num_warps
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
