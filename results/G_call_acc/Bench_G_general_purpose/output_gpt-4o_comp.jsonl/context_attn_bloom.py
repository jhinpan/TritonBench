import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q_ptr, K_ptr, V_ptr, Out_ptr, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kk, stride_kn, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_ok, batch_size, num_heads, seq_len, head_dim, scale_factor, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    Q = tl.load(Q_ptr + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    K = tl.load(K_ptr + pid_z * stride_kz + pid_h * stride_kh + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kn)
    V = tl.load(V_ptr + pid_z * stride_vz + pid_h * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn)
    scores = tl.dot(Q, K.T) * scale_factor
    scores = scores - tl.max(scores, axis=1, keepdim=True)
    exp_scores = tl.exp(scores)
    denom = tl.sum(exp_scores, axis=1, keepdim=True)
    softmax_scores = exp_scores / denom
    weighted_sum = tl.dot(softmax_scores, V)
    tl.store(Out_ptr + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, weighted_sum)
def context_attention_fwd(Q, K, V, output, batch_size, num_heads, seq_len, head_dim, device='cuda'):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = head_dim
    grid = (batch_size, num_heads, (seq_len + BLOCK_M - 1) // BLOCK_M)
    scale_factor = 1.0 / head_dim ** 0.5
    _fwd_kernel[grid](Q, K, V, output, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3), V.stride(0), V.stride(1), V.stride(2), V.stride(3), output.stride(0), output.stride(1), output.stride(2), output.stride(3), batch_size, num_heads, seq_len, head_dim, scale_factor, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL, num_warps=4, num_stages=3)
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
