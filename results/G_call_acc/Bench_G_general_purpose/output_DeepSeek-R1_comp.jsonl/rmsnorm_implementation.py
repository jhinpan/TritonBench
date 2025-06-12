import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr, N_SIZE, eps, x_batch_stride, x_m_stride, x_k_stride, rms_w_k_stride, out_batch_stride, out_m_stride, out_k_stride, BLOCK_N_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    x_start_ptr = x_ptr + batch_idx * x_batch_stride + m_idx * x_m_stride
    out_start_ptr = out_ptr + batch_idx * out_batch_stride + m_idx * out_m_stride
    sum_squares = 0.0
    for k_offset in range(0, N_SIZE, BLOCK_N_SIZE):
        k_indices = k_offset + tl.arange(0, BLOCK_N_SIZE)
        mask = k_indices < N_SIZE
        x_ptrs = x_start_ptr + k_indices * x_k_stride
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_square = x * x
        sum_squares += tl.sum(x_square, axis=0)
    mean = sum_squares / N_SIZE
    rms_val = 1.0 / tl.sqrt(mean + eps)
    for k_offset in range(0, N_SIZE, BLOCK_N_SIZE):
        k_indices = k_offset + tl.arange(0, BLOCK_N_SIZE)
        mask = k_indices < N_SIZE
        x_ptrs = x_start_ptr + k_indices * x_k_stride
        w_ptrs = rms_w_ptr + k_indices * rms_w_k_stride
        out_ptrs = out_start_ptr + k_indices * out_k_stride
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        w = tl.load(w_ptrs, mask=mask, other=0.0)
        out = x * rms_val * w
        tl.store(out_ptrs, out, mask=mask)
def rmsnorm_wrapper(x: torch.Tensor, rms_weight: torch.Tensor, eps: float=1e-05) -> torch.Tensor:
    assert x.is_cuda and rms_weight.is_cuda
    assert x.is_contiguous() and rms_weight.is_contiguous()
    batch_size, M_size, K_size = x.shape
    output = torch.empty_like(x)
    grid = (batch_size, M_size)
    BLOCK_N_SIZE = 128
    rmsnorm_triton[grid](x, rms_weight, output, K_size, eps, x.stride(0), x.stride(1), x.stride(2), rms_weight.stride(0), output.stride(0), output.stride(1), output.stride(2), BLOCK_N_SIZE=BLOCK_N_SIZE, num_warps=BLOCK_N_SIZE // 32)
    return output
##################################################################################################################################################



def test_rmsnorm():
    # Define the input tensor x with shape (batch, M, K)
    batch = 2
    M = 3
    K = 4096
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")

    # Define the rms_weights tensor with shape (K,)
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")

    # Dictionary to store the results of different test cases
    results = {}

    # Test case 1
    out1 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_1'] = out1.cpu()

    # Additional test cases for branch coverage

    # Test case 2: Different batch size
    batch = 4
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out2 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_2'] = out2.cpu()

    # Test case 3: Different M size
    M = 5
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out3 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_3'] = out3.cpu()

    # Test case 4: Larger K size
    K = 8192
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out4 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_4'] = out4.cpu()

    return results

# Execute the test function
result_gold = test_rmsnorm()
