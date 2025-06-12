import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr, x_stride_b, x_stride_m, x_stride_n, rms_w_stride_n, output_stride_b, output_stride_m, output_stride_n, N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    x_row_ptr = x_ptr + pid_batch * x_stride_b + pid_m * x_stride_m
    sum_squares = 0.0
    for block_start in tl.range(0, N_SIZE, BLOCK_N_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_N_SIZE)
        mask = offsets < N_SIZE
        x_ptrs = x_row_ptr + offsets * x_stride_n
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_squared = x * x
        sum_squares += tl.sum(x_squared, axis=0)
    variance = sum_squares / N_SIZE
    rstd = 1.0 / tl.sqrt(variance + eps)
    output_row_ptr = output_ptr + pid_batch * output_stride_b + pid_m * output_stride_m
    for block_start in tl.range(0, N_SIZE, BLOCK_N_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_N_SIZE)
        mask = offsets < N_SIZE
        x_ptrs = x_row_ptr + offsets * x_stride_n
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        w_ptrs = rms_w_ptr + offsets * rms_w_stride_n
        w = tl.load(w_ptrs, mask=mask, other=0.0)
        output_val = x * rstd * w
        output_ptrs = output_row_ptr + offsets * output_stride_n
        tl.store(output_ptrs, output_val, mask=mask)
def rmsnorm_triton_wrapper(x, weight, eps=1e-06):
    B, M, N = x.shape
    output = torch.empty_like(x)
    grid = (B, M)
    BLOCK_N = 128
    rmsnorm_triton[grid](x, weight, output, x.stride(0), x.stride(1), x.stride(2), weight.stride(0), output.stride(0), output.stride(1), output.stride(2), N_SIZE=N, eps=eps, BLOCK_N_SIZE=BLOCK_N)
    return output
def test_rmsnorm():
    B, M, N = (2, 3, 4)
    x = torch.randn(B, M, N, device='cuda')
    weight = torch.randn(N, device='cuda')
    output_triton = rmsnorm_triton_wrapper(x, weight)
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(rms + 1e-06)
    output_torch = x_normed * weight
    assert torch.allclose(output_triton, output_torch, atol=0.0001)
##################################################################################################################################################



def test_rmsnorm_triton():
    results = {}
    
    # Case 1
    batch, M, K = 2, 4, 1024
    x = torch.randn((batch, M, K), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((K,), dtype=torch.float16, device='cuda')
    eps = 1e-6
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_1'] = out

    # Case 2: Different eps value
    eps = 1e-5
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_2'] = out

    # Case 3: Different batch size
    batch, M, K = 3, 4, 1024
    x = torch.randn((batch, M, K), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((K,), dtype=torch.float16, device='cuda')
    eps = 1e-6
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_3'] = out

    # Case 4: Different M size
    batch, M, K = 2, 5, 1024
    x = torch.randn((batch, M, K), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((K,), dtype=torch.float16, device='cuda')
    eps = 1e-6
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_4'] = out

    return results

result_gold = test_rmsnorm_triton()
