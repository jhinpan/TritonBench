import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N'])
@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x_zeros = tl.where(cols < N, x, 0.0)
    var = tl.sum(x_zeros * x_zeros, axis=0)
    rstd = 1.0 / tl.sqrt(var + eps)
    y = x * rstd
    tl.store(Y + cols, y, mask=cols < N)
def _l2_norm_fwd(x, eps=1e-06):
    original_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    if x.stride(-1) != 1:
        x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    element_size = x.element_size()
    max_fused_size = 65536 // element_size
    BLOCK_N = min(max_fused_size, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise ValueError(f'Feature dimension {N} exceeds maximum supported size {BLOCK_N}.')
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](x, y, x.stride(0), N, eps, BLOCK_N)
    return y.reshape(original_shape)
##################################################################################################################################################



import torch

# Test the forward L2 normalization
def test_l2_norm_fwd():
    results = {}
    
    # Test case 1
    x1 = torch.randn(4, 8, device='cuda', dtype=torch.float32)
    y1 = _l2_norm_fwd(x1)
    results['test_case_1'] = y1

    # Test case 2: Different batch size
    x2 = torch.randn(2, 8, device='cuda', dtype=torch.float32)
    y2 = _l2_norm_fwd(x2)
    results['test_case_2'] = y2

    # Test case 3: Different feature size
    x3 = torch.randn(4, 4, device='cuda', dtype=torch.float32)
    y3 = _l2_norm_fwd(x3)
    results['test_case_3'] = y3

    # Test case 4: Larger tensor
    x4 = torch.randn(8, 8, device='cuda', dtype=torch.float32)
    y4 = _l2_norm_fwd(x4)
    results['test_case_4'] = y4

    return results

result_gold = test_l2_norm_fwd()
