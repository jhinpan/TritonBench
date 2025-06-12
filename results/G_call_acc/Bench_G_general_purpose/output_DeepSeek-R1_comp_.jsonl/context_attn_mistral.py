import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, WINDOW_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    batch_start = tl.load(B_Start_Loc + pid_batch)
    seqlen = tl.load(B_Seqlen + pid_batch)
    offs_m = pid_m * BLOCK_M
    if offs_m >= seqlen:
        return
    q_offset = batch_start + offs_m
    Q_ptr = Q + q_offset * stride_qbs + pid_head * stride_qh
    q = tl.load(Q_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_qbs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd, mask=(offs_m + tl.arange(0, BLOCK_M))[:, None] < seqlen, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    start_n = tl.maximum(0, (offs_m - WINDOW_SIZE) // BLOCK_N)
    end_n = tl.minimum((seqlen + BLOCK_N - 1) // BLOCK_N, (offs_m + BLOCK_M + WINDOW_SIZE) // BLOCK_N)
    for pid_n in range(start_n, end_n):
        offs_n = pid_n * BLOCK_N
        k_offset = batch_start + offs_n
        K_ptr = K + k_offset * stride_kbs + pid_head * stride_kh
        k = tl.load(K_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_kbs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kd, mask=(offs_n + tl.arange(0, BLOCK_N))[:, None] < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        query_pos = offs_m + tl.arange(0, BLOCK_M)
        key_pos = offs_n + tl.arange(0, BLOCK_N)
        mask = (key_pos[None, :] >= query_pos[:, None] - WINDOW_SIZE) & (key_pos[None, :] <= query_pos[:, None] + WINDOW_SIZE)
        mask &= key_pos[None, :] < seqlen
        qk = tl.where(mask, qk, float('-inf'))
        m_i = tl.max(qk, 1)
        p = tl.exp(qk - m_i[:, None])
        p_sum = tl.sum(p, 1)
        p = p / p_sum[:, None]
        v_offset = batch_start + offs_n
        V_ptr = V + v_offset * stride_vbs + pid_head * stride_vh
        v = tl.load(V_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_vbs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd, mask=(offs_n + tl.arange(0, BLOCK_N))[:, None] < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
    out_offset = batch_start + offs_m
    Out_ptr = Out + out_offset * stride_obs + pid_head * stride_oh
    tl.store(Out_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_obs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od, acc.to(Out_ptr.dtype.element_ty), mask=(offs_m + tl.arange(0, BLOCK_M))[:, None] < seqlen)
def context_attention_fwd(q, k, v, o, B_Start_Loc, B_Seqlen, window_size=512, sm_scale=None):
    BLOCK_M = 64
    BLOCK_N = 128
    head_dim = q.size(-1)
    if sm_scale is None:
        sm_scale = 1.0 / head_dim ** 0.5
    max_seqlen = B_Seqlen.max().item()
    grid = (q.size(0), q.size(1), (max_seqlen + BLOCK_M - 1) // BLOCK_M)
    _fwd_kernel[grid](q, k, v, sm_scale, B_Start_Loc, B_Seqlen, o, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2), BLOCK_M=BLOCK_M, BLOCK_DMODEL=head_dim, BLOCK_N=BLOCK_N, WINDOW_SIZE=window_size, num_warps=8 if head_dim > 64 else 4, num_stages=1)
##################################################################################################################################################



def test_context_attention_fwd():
    Z, H, N_CTX, D_HEAD = 4, 6, 1024, 128
    dtype = torch.float16
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    max_input_len = N_CTX
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")

    b_seq_len[0] = 512
    b_seq_len[1] = 1024
    b_seq_len[2] = 512
    b_seq_len[3] = 1024

    for i in range(1, Z):
        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]

    results = {}
    
    # Test case 1
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, 10)
    results['test_case_1'] = o.clone()

    # Test case 2: Different sliding window
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, 20)
    results['test_case_2'] = o.clone()

    # Test case 3: Different max_input_len
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len // 2, 10)
    results['test_case_3'] = o.clone()

    # Test case 4: Different batch size
    Z = 2
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_seq_len[0] = 512
    b_seq_len[1] = 1024
    for i in range(1, Z):
        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, 10)
    results['test_case_4'] = o.clone()

    return results

result_gold = test_context_attention_fwd()
