import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(X, W, Y, stride_x_N, stride_x_hn, stride_x_hd, stride_y_N, stride_y_hn, stride_y_hd, stride_w_hn, stride_w_hd, N, eps, BLOCK_SIZE: tl.constexpr):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    x_ptr = X + seq_id * stride_x_N + head_id * stride_x_hn
    y_ptr = Y + seq_id * stride_y_N + head_id * stride_y_hn
    w_ptr = W + head_id * stride_w_hn
    mean = 0.0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        x_centered = tl.where(mask, x - mean, 0.0)
        _var += x_centered * x_centered
    var = tl.sum(_var) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(w_ptr + cols * stride_w_hd, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * weight
        tl.store(y_ptr + cols * stride_y_hd, y.to(X.dtype.element_ty), mask=mask)
def layernorm_forward(X: torch.Tensor, W: torch.Tensor, eps: float) -> torch.Tensor:
    assert X.ndim == 3, 'Input tensor must be 3D'
    assert W.ndim == 2, 'Weight tensor must be 2D'
    assert X.shape[-2:] == W.shape, 'Feature dimension mismatch between X and W'
    Y = torch.empty_like(X)
    stride_x_N, stride_x_hn, stride_x_hd = X.stride()
    stride_y_N, stride_y_hn, stride_y_hd = Y.stride()
    stride_w_hn, stride_w_hd = W.stride()
    N = X.size(-1)
    BLOCK_SIZE = 128
    grid = (X.shape[0], X.shape[1])
    _layer_norm_fwd_kernel[grid](X, W, Y, stride_x_N, stride_x_hn, stride_x_hd, stride_y_N, stride_y_hn, stride_y_hd, stride_w_hn, stride_w_hd, N, eps, BLOCK_SIZE)
    return Y
##################################################################################################################################################



import torch

# Test function for layernorm_forward
def test_layernorm_forward():
    results = {}
    
    # Test case 1: Basic functionality
    X = torch.randn(2, 3, 128, dtype=torch.float32, device='cuda')
    W = torch.randn(3, 128, dtype=torch.float32, device='cuda')
    eps = 1e-5
    y = layernorm_forward(X, W, eps)
    results['test_case_1'] = y

    # Test case 2: Different batch size
    X = torch.randn(4, 3, 128, dtype=torch.float32, device='cuda')
    W = torch.randn(3, 128, dtype=torch.float32, device='cuda')
    y = layernorm_forward(X, W, eps)
    results['test_case_2'] = y

    # Test case 3: Different feature size
    X = torch.randn(2, 3, 256, dtype=torch.float32, device='cuda')
    W = torch.randn(3, 256, dtype=torch.float32, device='cuda')
    y = layernorm_forward(X, W, eps)
    results['test_case_3'] = y

    # Test case 4: Different number of heads
    X = torch.randn(2, 4, 128, dtype=torch.float32, device='cuda')
    W = torch.randn(4, 128, dtype=torch.float32, device='cuda')
    y = layernorm_forward(X, W, eps)
    results['test_case_4'] = y

    return results

# Run the test function
result_gold = test_layernorm_forward()
