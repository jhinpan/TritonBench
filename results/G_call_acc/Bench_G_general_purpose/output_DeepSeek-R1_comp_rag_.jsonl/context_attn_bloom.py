import triton
import triton.language as tl

def wrap_kernel_launcher(kernel):
    return kernel
@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh, stride_obs, stride_oh, kv_group_num: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_idx = tl.load(B_Start_Loc + cur_batch)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (cur_batch_start_idx + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_end = tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    block_start = 0
    for start_n in range(block_start, block_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + (cur_batch_start_idx + start_n) * stride_kbs + offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        v_ptrs = V + (cur_batch_start_idx + start_n) * stride_vbs + offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=start_n + offs_n[None, :] < cur_batch_seq_len, other=0.0)
        v = tl.load(v_ptrs, mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        qk = tl.dot(q, k) * sm_scale
        causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
        qk = tl.where(causal_mask, qk, float('-inf'))
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        acc_scale = alpha / l_i_new[:, None]
        acc *= acc_scale
        p_scale = beta / l_i_new[:, None]
        acc += tl.dot((p * p_scale).to(v.dtype), v)
        l_i = l_i_new
        m_i = m_i_new
    out_ptrs = Out + (cur_batch_start_idx + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    global cached_kernel
    BLOCK = 128 if torch.cuda.get_device_capability()[0] >= 8 else 64
    d_model = q.size(-1)
    assert d_model in {16, 32, 64, 128, 256}, 'Unsupported feature dimension'
    assert q.shape[1] % k.shape[1] == 0, 'Query heads must be multiple of key heads'
    batch_size, num_heads = (b_seq_len.shape[0], q.shape[1])
    kv_group_num = q.shape[1] // k.shape[1]
    sm_scale = 1.0 / d_model ** 0.5
    grid = (batch_size, num_heads, triton.cdiv(max_input_len, BLOCK))
    num_warps = 8 if d_model >= 128 else 4
    if cached_kernel:
        return cached_kernel(grid=grid, num_warps=num_warps, args=[q, k, v, sm_scale, b_start_loc, b_seq_len, o, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0), o.stride(1), kv_group_num, BLOCK, d_model, BLOCK])
    _fwd_kernel[grid](q, k, v, sm_scale, b_start_loc, b_seq_len, o, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0), o.stride(1), kv_group_num=kv_group_num, BLOCK_M=BLOCK, BLOCK_DMODEL=d_model, BLOCK_N=BLOCK, num_warps=num_warps, num_stages=4 if d_model >= 128 else 3)
    cached_kernel = wrap_kernel_launcher(_fwd_kernel)
##################################################################################################################################################



import torch
import numpy as np

def test_context_attention_fwd():
    Z, H, N_CTX, D_HEAD = 10, 6, 500, 96
    dtype = torch.float16
    Z = 1
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * N_CTX + 7000, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * N_CTX + 7000, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    req_to_token_indexs = torch.zeros((10, Z * N_CTX + 7000), dtype=torch.int32, device="cuda")
    max_input_len = N_CTX
    Z = 1
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
    b_prompt_cache_len[0] = 0
    prompt_cache_len = 0

    b_seq_len[0] = 500
    b_req_idx[0] = 0
    req_to_token_indexs[0][: prompt_cache_len + N_CTX] = torch.tensor(
        np.arange(prompt_cache_len + N_CTX), dtype=torch.int32
    ).cuda()

    result_gold = context_attention_fwd(
        q,
        k,
        v,
        o,
        b_req_idx,
        b_start_loc,
        b_seq_len + prompt_cache_len,
        b_prompt_cache_len,
        max_input_len,
        req_to_token_indexs,
    )
    return result_gold
