import triton
import triton.language as tl
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    input_row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    max_val = tl.max(input_row, axis=0)
    input_row = input_row - max_val
    if mask_ptr is not None:
        mask_row = tl.load(mask_ptr + row_idx * row_stride + col_offsets, mask=col_offsets < n_cols, other=0)
        input_row = input_row + mask_row
    exp_row = tl.exp(input_row)
    sum_exp = tl.sum(exp_row, axis=0)
    output_row = exp_row / sum_exp
    output_ptrs = output_ptr + row_idx * row_stride + col_offsets
    tl.store(output_ptrs, output_row, mask=col_offsets < n_cols)
def softmax(input: torch.Tensor, mask: torch.Tensor=None, dim=-1) -> torch.Tensor:
    if dim != -1:
        raise ValueError('Currently, only the last dimension is supported for softmax.')
    if mask is not None and mask.shape != input.shape:
        raise ValueError('Mask tensor must have the same shape as the input tensor.')
    original_shape = input.shape
    input = input.view(-1, input.shape[-1])
    n_rows, n_cols = input.shape
    BLOCK_SIZE = 128
    grid = (n_rows,)
    output = torch.empty_like(input)
    if mask is not None:
        mask = mask.view(-1, mask.shape[-1])
        softmax_kernel[grid](output, input, n_cols, n_cols, mask, BLOCK_SIZE)
    else:
        softmax_kernel[grid](output, input, n_cols, n_cols, None, BLOCK_SIZE)
    output = output.view(original_shape)
    return output
##################################################################################################################################################



def test_softmax():
    # Test Case 1: Small matrix without mask
    input_tensor_1 = torch.randn(32, 128, dtype=torch.float16, device='cuda')
    output_tensor_1 = softmax(input_tensor_1)

    # Test Case 2: Small matrix with mask
    input_tensor_2 = torch.randn(32, 128, dtype=torch.float16, device='cuda')
    mask_tensor_2 = torch.randint(0, 2, (32, 128), dtype=torch.float16, device='cuda')
    output_tensor_2 = softmax(input_tensor_2, mask=mask_tensor_2)

    # Test Case 3: Larger matrix without mask
    input_tensor_3 = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    output_tensor_3 = softmax(input_tensor_3)

    # Test Case 4: Larger matrix with mask
    input_tensor_4 = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    mask_tensor_4 = torch.randint(0, 2, (1024, 512), dtype=torch.float16, device='cuda')
    output_tensor_4 = softmax(input_tensor_4, mask=mask_tensor_4)

    # Test Case 5: Very large matrix without mask
    input_tensor_5 = torch.randn(100000, 256, dtype=torch.float16, device='cuda')
    output_tensor_5 = softmax(input_tensor_5)

    # Test Case 6: Very large matrix with mask
    input_tensor_6 = torch.randn(100000, 256, dtype=torch.float16, device='cuda')
    mask_tensor_6 = torch.randint(0, 2, (100000, 256), dtype=torch.float16, device='cuda')
    output_tensor_6 = softmax(input_tensor_6, mask=mask_tensor_6)

    return {
        "test_case_1": output_tensor_1,
        "test_case_2": output_tensor_2,
        "test_case_3": output_tensor_3,
        "test_case_4": output_tensor_4,
        "test_case_5": output_tensor_5,
        "test_case_6": output_tensor_6
    }

# Run the test function
result_gold = test_softmax()
