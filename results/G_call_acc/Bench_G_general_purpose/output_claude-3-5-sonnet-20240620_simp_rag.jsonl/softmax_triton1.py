import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_f32 = row.to(tl.float32)
    row_max = tl.max(row_f32, axis=0)
    row_minus_max = row_f32 - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)
def softmax(input_tensor):
    """
    Wrapper function to compute softmax using the Triton kernel
    
    Args:
        input_tensor: Input tensor of shape (rows, cols)
    Returns:
        Output tensor with softmax applied to each row
    """
    n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(input_tensor)
    input_row_stride = input_tensor.stride(0)
    output_row_stride = output.stride(0)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    softmax_kernel[n_rows,](output, input_tensor, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return output
##################################################################################################################################################



import torch

def test_softmax():
    # Define the input tensor
    x = torch.randn(128, 512, device='cuda', dtype=torch.float32)

    # Compute softmax using Triton
    output = softmax(x)

    # Additional test cases to cover all branches
    results = {}

    # Test case 1: n_cols < 2048
    x1 = torch.randn(128, 1024, device='cuda', dtype=torch.float32)
    results['test_case_1'] = softmax(x1)

    # Test case 2: 2048 <= n_cols < 4096
    x2 = torch.randn(128, 2048, device='cuda', dtype=torch.float32)
    results['test_case_2'] = softmax(x2)

    # Test case 3: n_cols >= 4096
    x3 = torch.randn(128, 4096, device='cuda', dtype=torch.float32)
    results['test_case_3'] = softmax(x3)

    # Test case 4: n_cols < 2048 (original test case)
    results['test_case_4'] = output

    return results

result_gold = test_softmax()
