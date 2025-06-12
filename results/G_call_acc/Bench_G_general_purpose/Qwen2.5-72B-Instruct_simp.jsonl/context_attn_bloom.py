import triton
import triton.language as tl
import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, Out, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_km, stride_vb, stride_vh, stride_vm, stride_ob, stride_oh, stride_om, n_head, n_ctx, scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(n_ctx, BLOCK_M)
    num_pid_n = tl.cdiv(n_ctx, BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid %= num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + (batch_id * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh)
    k_ptrs = K + (batch_id * stride_kb + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kh)
    v_ptrs = V + (batch_id * stride_vb + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vh)
    o_ptrs = Out + (batch_id * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_oh)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    qk = tl.dot(q, k, trans_b=True) * scale
    qk = tl.softmax(qk, axis=1)
    v = tl.load(v_ptrs)
    acc += tl.dot(qk, v)
    tl.store(o_ptrs, acc)
def context_attention_fwd(Q, K, V, Out, n_head, n_ctx, scale):
    batch_size, _, _ = Q.shape
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = 64
    grid = (triton.cdiv(n_ctx, BLOCK_M) * triton.cdiv(n_ctx, BLOCK_N) * batch_size,)
    stride_qb = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_qm = Q.stride(2)
    stride_kb = K.stride(0)
    stride_kh = K.stride(1)
    stride_km = K.stride(2)
    stride_vb = V.stride(0)
    stride_vh = V.stride(1)
    stride_vm = V.stride(2)
    stride_ob = Out.stride(0)
    stride_oh = Out.stride(1)
    stride_om = Out.stride(2)
    _fwd_kernel[grid](Q, K, V, Out, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_km, stride_vb, stride_vh, stride_vm, stride_ob, stride_oh, stride_om, n_head, n_ctx, scale, BLOCK_M, BLOCK_N, BLOCK_D)
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
