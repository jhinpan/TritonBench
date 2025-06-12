import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q_ptr, K_ptr, V_ptr, Out_ptr, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_om, stride_oh, stride_ob, B, H, M, N, D, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_id = pid // H
    head_id = pid % H
    Q_offset = batch_id * stride_qb + head_id * stride_qh
    K_offset = batch_id * stride_kb + head_id * stride_kh
    V_offset = batch_id * stride_vb + head_id * stride_vh
    Out_offset = batch_id * stride_ob + head_id * stride_oh
    Q = tl.load(Q_ptr + Q_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qm)
    K = tl.load(K_ptr + K_offset + tl.arange(0, BLOCK_SIZE)[None, :] * stride_kn)
    V = tl.load(V_ptr + V_offset + tl.arange(0, BLOCK_SIZE)[None, :] * stride_vn)
    QK = tl.dot(Q, K, trans_b=True) * scale
    attn_scores = tl.softmax(QK, axis=1)
    Out = tl.dot(attn_scores, V)
    tl.store(Out_ptr + Out_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_om, Out)
def context_attention_fwd(Q, K, V, scale):
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    Out = torch.empty((B, H, M, D), device=Q.device, dtype=Q.dtype)
    BLOCK_SIZE = 128
    grid = (B * H,)
    _fwd_kernel[grid](Q, K, V, Out, Q.stride(0), Q.stride(1), Q.stride(2), K.stride(0), K.stride(1), K.stride(2), V.stride(0), V.stride(1), V.stride(2), Out.stride(2), Out.stride(1), Out.stride(0), B, H, M, N, D, scale, BLOCK_SIZE=BLOCK_SIZE)
    return Out
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
