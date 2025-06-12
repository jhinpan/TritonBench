import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel_online_v2(output_ptr, input_ptr, M, N, stride_om, stride_on, stride_im, stride_in, TILE_N: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, TILE_N)
    input_row_ptr = input_ptr + row_idx * stride_im
    output_row_ptr = output_ptr + row_idx * stride_om
    row_max = -float('inf')
    row_sum = 0.0
    for n in range(0, N, TILE_N):
        mask = col_offsets + n < N
        x = tl.load(input_row_ptr + col_offsets * stride_in + n * stride_in, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    for n in range(0, N, TILE_N):
        mask = col_offsets + n < N
        x = tl.load(input_row_ptr + col_offsets * stride_in + n * stride_in, mask=mask, other=-float('inf'))
        x = tl.exp(x - row_max)
        row_sum += tl.sum(x, axis=0)
        tl.store(output_row_ptr + col_offsets * stride_on + n * stride_on, x, mask=mask)
    for n in range(0, N, TILE_N):
        mask = col_offsets + n < N
        x = tl.load(output_row_ptr + col_offsets * stride_on + n * stride_on, mask=mask)
        x = x / row_sum
        tl.store(output_row_ptr + col_offsets * stride_on + n * stride_on, x, mask=mask)
def prev_multiple_of(a, b):
    return math.floor(a / b) * b
def softmax(x):
    """
    Compute softmax using a custom Triton kernel
    Args:
        x: input tensor of shape (M, N)
    Returns:
        output tensor of shape (M, N)
    """
    M, N = x.shape
    out = torch.empty_like(x)
    TILE_N = min(triton.next_power_of_2(N), 512)
    grid = (M,)
    softmax_kernel_online_v2[grid](out, x, M, N, out.stride(0), out.stride(1), x.stride(0), x.stride(1), TILE_N=TILE_N)
    return out
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
