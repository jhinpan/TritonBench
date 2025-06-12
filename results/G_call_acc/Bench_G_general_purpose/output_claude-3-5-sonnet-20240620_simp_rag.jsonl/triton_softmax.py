import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)
def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    grid = (n_rows,)
    softmax_kernel[grid](output, x, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output
##################################################################################################################################################



# Test cases for the triton_softmax function
def test_triton_softmax():
    results = {}
    
    # Test case 1: Simple 2x2 matrix
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda")
    output1 = triton_softmax(x1)
    results['test_case_1'] = output1

    # Test case 2: 3x3 matrix with negative values
    x2 = torch.tensor([[-1.0, -2.0, -3.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.float32, device="cuda")
    output2 = triton_softmax(x2)
    results['test_case_2'] = output2

    # Test case 3: 4x4 matrix with larger values
    x3 = torch.tensor([[10.0, 20.0, 30.0, 40.0], [5.0, 15.0, 25.0, 35.0], [0.0, 0.0, 0.0, 0.0], [-10.0, -20.0, -30.0, -40.0]], dtype=torch.float32, device="cuda")
    output3 = triton_softmax(x3)
    results['test_case_3'] = output3

    # Test case 4: 1x5 matrix (single row)
    x4 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32, device="cuda")
    output4 = triton_softmax(x4)
    results['test_case_4'] = output4

    # Test case 5: 5x1 matrix (single column)
    x5 = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32, device="cuda")
    output5 = triton_softmax(x5)
    results['test_case_5'] = output5

    return results

result_gold = test_triton_softmax()
