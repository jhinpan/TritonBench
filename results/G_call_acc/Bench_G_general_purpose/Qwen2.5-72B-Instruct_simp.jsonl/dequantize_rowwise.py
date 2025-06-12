import triton
import triton.language as tl
import torch
import triton
import triton.language as tl
import torch

@triton.jit
def _dequantize_rowwise(x_ptr, state_x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, M)
    for row in range(row_start, row_end):
        max_val = tl.load(state_x_ptr + row)
        inv_127 = 1.0 / 127.0
        row_offset = row * N
        for col in range(N):
            x_val = tl.load(x_ptr + row_offset + col)
            output_val = x_val * max_val * inv_127
            tl.store(output_ptr + row_offset + col, output_val)
def dequantize_rowwise(x, state_x):
    M, N = x.shape
    output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    _dequantize_rowwise[grid](x, state_x, output, M, N, BLOCK_SIZE)
    return output
##################################################################################################################################################



# Test function for dequantize_rowwise
def test_dequantize_rowwise():
    results = {}

    # Test case 1: Simple case
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8, device='cuda')
    state_x = torch.tensor([4.0, 8.0], dtype=torch.float32, device='cuda')
    output = dequantize_rowwise(x, state_x)
    results['test_case_1'] = output

    # Test case 2: Larger input
    x = torch.randint(-128, 127, (10, 16), dtype=torch.int8, device='cuda')
    state_x = torch.rand(10, dtype=torch.float32, device='cuda') * 10
    output = dequantize_rowwise(x, state_x)
    results['test_case_2'] = output

    # Test case 3: Edge case with zeros
    x = torch.zeros((5, 8), dtype=torch.int8, device='cuda')
    state_x = torch.ones(5, dtype=torch.float32, device='cuda')
    output = dequantize_rowwise(x, state_x)
    results['test_case_3'] = output

    # Test case 4: Different block size
    x = torch.randint(-128, 127, (3, 32), dtype=torch.int8, device='cuda')
    state_x = torch.rand(3, dtype=torch.float32, device='cuda') * 10
    output = dequantize_rowwise(x, state_x)
    results['test_case_4'] = output

    return results

# Run the test function
result_gold = test_dequantize_rowwise()
