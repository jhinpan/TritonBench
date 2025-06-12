import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, output_ptr, batch, heads, seq_len, dim, stride_qb, stride_qh, stride_qs, stride_kb, stride_kh, stride_ks, stride_vb, stride_vh, stride_vs, stride_ob, stride_oh, stride_os, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(batch, BLOCK_SIZE)
    num_pid_h = heads
    num_pid_s = tl.cdiv(seq_len, BLOCK_SIZE)
    pid_b = pid // (num_pid_h * num_pid_s)
    pid_h = pid % (num_pid_h * num_pid_s) // num_pid_s
    pid_s = pid % (num_pid_h * num_pid_s) % num_pid_s
    block_b = pid_b * BLOCK_SIZE
    block_s = pid_s * BLOCK_SIZE
    q_block_ptr = q_ptr + block_b * stride_qb + pid_h * stride_qh + block_s * stride_qs
    q = tl.load(q_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qs + tl.arange(0, dim)[None, :], mask=None)
    acc = tl.zeros([BLOCK_SIZE, dim], dtype=tl.float32)
    scale = 1.0 / dim ** 0.5
    for k_idx in range(0, seq_len, BLOCK_SIZE):
        k_block_ptr = k_ptr + block_b * stride_kb + pid_h * stride_kh + k_idx * stride_ks
        k = tl.load(k_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_ks + tl.arange(0, dim)[None, :], mask=None)
        v_block_ptr = v_ptr + block_b * stride_vb + pid_h * stride_vh + k_idx * stride_vs
        v = tl.load(v_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_vs + tl.arange(0, dim)[None, :], mask=None)
        scores = tl.dot(q, k.transpose())
        scores = scores * scale
        scores = tl.softmax(scores)
        acc += tl.dot(scores, v)
    output_block_ptr = output_ptr + block_b * stride_ob + pid_h * stride_oh + block_s * stride_os
    output_mask = block_s + tl.arange(0, BLOCK_SIZE) < seq_len
    tl.store(output_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_os + tl.arange(0, dim)[None, :], acc, mask=output_mask[:, None])
def context_attention_fwd(q, k, v):
    """
    Forward pass for context attention.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, dim)
        k: Key tensor of shape (batch, heads, seq_len, dim)
        v: Value tensor of shape (batch, heads, seq_len, dim)
        
    Returns:
        Output tensor of shape (batch, heads, seq_len, dim)
    """
    batch, heads, seq_len, dim = q.shape
    output = torch.empty_like(q)
    BLOCK_SIZE = 32
    if torch.cuda.get_device_capability()[0] >= 7:
        BLOCK_SIZE = 64
    grid = (batch * heads * triton.cdiv(seq_len, BLOCK_SIZE),)
    _fwd_kernel[grid](q, k, v, output, batch, heads, seq_len, dim, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), output.stride(0), output.stride(1), output.stride(2), BLOCK_SIZE=BLOCK_SIZE)
    return output
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
