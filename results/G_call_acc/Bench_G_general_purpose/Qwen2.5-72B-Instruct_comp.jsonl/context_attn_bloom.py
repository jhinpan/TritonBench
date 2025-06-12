import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, Out, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_km, stride_vb, stride_vh, stride_vm, stride_ob, stride_oh, stride_om, n_heads, n_ctx, head_dim, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = pid // (n_heads * n_ctx)
    hid = pid % (n_heads * n_ctx) // n_ctx
    mid = pid % (n_heads * n_ctx) % n_ctx
    Q += bid * stride_qb + hid * stride_qh + mid * stride_qm
    K += bid * stride_kb + hid * stride_kh
    V += bid * stride_vb + hid * stride_vh
    Out += bid * stride_ob + hid * stride_oh + mid * stride_om
    q = tl.load(Q + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_D)[None, :] * head_dim)
    out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n in range(0, n_ctx, BLOCK_N):
        k = tl.load(K + (n + tl.arange(0, BLOCK_N))[:, None] * stride_km + tl.arange(0, BLOCK_D)[None, :] * head_dim)
        logits = tl.dot(q, k.T) * (1.0 / tl.sqrt(head_dim))
        logits = tl.softmax(logits, axis=1)
        v = tl.load(V + (n + tl.arange(0, BLOCK_N))[:, None] * stride_vm + tl.arange(0, BLOCK_D)[None, :] * head_dim)
        out += tl.dot(logits, v)
    tl.store(Out + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_D)[None, :] * head_dim, out)
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
