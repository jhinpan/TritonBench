import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(X, W, Y, stride_x_N, stride_x_hn, stride_x_hd, stride_y_N, stride_y_hn, stride_y_hd, stride_w_hn, stride_w_hd, N, eps, BLOCK_SIZE: tl.constexpr):
    Seq = tl.program_id(0)
    H = tl.program_id(1)
    X += Seq * stride_x_N + H * stride_x_hn
    Y += Seq * stride_y_N + H * stride_y_hn
    W += H * stride_w_hn
    mean = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    var = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        var += tl.sum(x * x, axis=0)
    var = var / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x_normalized = (x - mean) * rstd
        y = x_normalized * w
        tl.store(Y + cols, y.to(X.dtype.element_ty), mask=mask)
def layernorm_forward(X, W, eps=1e-05):
    """
    Forward pass for LayerNorm
    Args:
        X: Input tensor of shape (batch_size, hidden_dim, feature_dim)
        W: Weight tensor of shape (hidden_dim, feature_dim)
        eps: Small constant for numerical stability
    Returns:
        Normalized tensor of same shape as input
    """
    assert len(X.shape) == 3, 'Input tensor must be 3D'
    assert len(W.shape) == 2, 'Weight tensor must be 2D'
    assert X.shape[-1] == W.shape[-1], 'Feature dimensions must match'
    assert X.shape[1] == W.shape[0], 'Hidden dimensions must match'
    Y = torch.empty_like(X)
    stride_x_N, stride_x_hn, stride_x_hd = (X.stride(0), X.stride(1), X.stride(2))
    stride_y_N, stride_y_hn, stride_y_hd = (Y.stride(0), Y.stride(1), Y.stride(2))
    stride_w_hn, stride_w_hd = (W.stride(0), W.stride(1))
    BLOCK_SIZE = min(128, X.shape[-1])
    grid = (X.shape[0], X.shape[1])
    _layer_norm_fwd_kernel[grid](X, W, Y, stride_x_N, stride_x_hn, stride_x_hd, stride_y_N, stride_y_hn, stride_y_hd, stride_w_hn, stride_w_hd, X.shape[-1], eps, BLOCK_SIZE)
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
