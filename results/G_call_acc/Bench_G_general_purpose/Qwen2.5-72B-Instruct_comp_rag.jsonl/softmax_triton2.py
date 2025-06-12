import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if n_cols > 200000 else 2
    y = torch.empty_like(x)
    grid = (n_rows, 1, 1)
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages)
    return y
##################################################################################################################################################



import torch

# Test cases for the softmax function
def test_softmax():
    result_dict = {}

    # Test case 1: Small matrix
    x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device='cuda')
    y1 = softmax(x1)
    result_dict["test_case_1"] = y1

    # Test case 2: Larger matrix
    x2 = torch.randn(128, 256, dtype=torch.float32, device='cuda')
    y2 = softmax(x2)
    result_dict["test_case_2"] = y2

    # Test case 3: Single row
    x3 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device='cuda')
    y3 = softmax(x3)
    result_dict["test_case_3"] = y3

    # Test case 4: Single column
    x4 = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device='cuda')
    y4 = softmax(x4)
    result_dict["test_case_4"] = y4

    # Test case 5: Large matrix with power of two columns
    x5 = torch.randn(64, 512, dtype=torch.float32, device='cuda')
    y5 = softmax(x5)
    result_dict["test_case_5"] = y5

    return result_dict

# Run the test cases
result_gold = test_softmax()
