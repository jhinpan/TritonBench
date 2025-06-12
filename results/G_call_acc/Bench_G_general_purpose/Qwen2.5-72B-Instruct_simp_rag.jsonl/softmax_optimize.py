import torch
import triton
import triton.language as tl

@triton.jit
def prev_multiple_of(a, b: tl.constexpr):
    return a // b * b
@triton.jit
def softmax_kernel_online_v2(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return
    row_start_ptr = input_ptr + row_idx * N
    col_offsets = tl.arange(0, TILE_N)
    input_ptrs = row_start_ptr + col_offsets
    tile = tl.load(input_ptrs, mask=col_offsets < N, other=-float('inf'))
    max_val = tl.max(tile, axis=0)
    tile_minus_max = tile - max_val
    exp_tile = tl.exp(tile_minus_max)
    sum_exp = tl.sum(exp_tile, axis=0)
    softmax_tile = exp_tile / sum_exp
    output_row_start_ptr = output_ptr + row_idx * N
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_tile, mask=col_offsets < N)
def softmax(x):
    M, N = x.shape
    TILE_N = triton.next_power_of_2(N)
    y = torch.empty_like(x)
    num_warps = 4
    if TILE_N >= 2048:
        num_warps = 8
    if TILE_N >= 4096:
        num_warps = 16
    softmax_kernel_online_v2[M,](y, x, M, N, TILE_N, num_warps=num_warps)
    return y
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
