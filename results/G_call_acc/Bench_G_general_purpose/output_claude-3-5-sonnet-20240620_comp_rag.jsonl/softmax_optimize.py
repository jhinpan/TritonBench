import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(output_ptr, input_ptr, n_rows, n_cols, row_stride_in, row_stride_out, TILE_N: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_in = input_ptr + row_idx * row_stride_in
    row_start_out = output_ptr + row_idx * row_stride_out
    col_offsets = tl.arange(0, TILE_N)
    max_val = -float('inf')
    sum_exp = 0.0
    for tile_start in range(0, n_cols, TILE_N):
        mask = col_offsets + tile_start < n_cols
        x = tl.load(row_start_in + tile_start + col_offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    for tile_start in range(0, n_cols, TILE_N):
        mask = col_offsets + tile_start < n_cols
        x = tl.load(row_start_in + tile_start + col_offsets, mask=mask, other=-float('inf'))
        x_stable = tl.exp(x - max_val)
        sum_exp += tl.sum(x_stable, axis=0)
    for tile_start in range(0, n_cols, TILE_N):
        mask = col_offsets + tile_start < n_cols
        x = tl.load(row_start_in + tile_start + col_offsets, mask=mask, other=-float('inf'))
        x_stable = tl.exp(x - max_val) / sum_exp
        tl.store(row_start_out + tile_start + col_offsets, x_stable, mask=mask)
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise softmax using Triton kernel.
    
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        Tensor of shape (M, N) containing row-wise softmax probabilities
    """
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    TILE_N = min(triton.next_power_of_2(n_cols), 512)
    num_warps = 4
    if TILE_N >= 256:
        num_warps = 8
    if TILE_N >= 512:
        num_warps = 16
    softmax_kernel_online_v2[n_rows,](output_ptr=output, input_ptr=x, n_rows=n_rows, n_cols=n_cols, row_stride_in=x.stride(0), row_stride_out=output.stride(0), TILE_N=TILE_N, num_warps=num_warps)
    return output
##################################################################################################################################################



# Comparison Test
def test_softmax():

    torch.manual_seed(0)
    
    result = {}
    
    # Case 1: M = 128, N = 512
    x1 = torch.randn(128, 512, device='cuda', dtype=torch.float32)
    result['test_case_1'] = softmax(x1)

    # Case 2: M = 64, N = 1024
    x2 = torch.randn(64, 1024, device='cuda', dtype=torch.float32)
    result['test_case_2'] = softmax(x2)

    # Case 3: M = 256, N = 128
    x3 = torch.randn(256, 128, device='cuda', dtype=torch.float32)
    result['test_case_3'] = softmax(x3)
    
    return result

# Execute test function
result_gold = test_softmax()
