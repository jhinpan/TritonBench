import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
def relu(x):
    """
    Wrapper function for the ReLU Triton kernel
    Args:
        x: Input tensor
    Returns:
        Output tensor after applying ReLU
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output
##################################################################################################################################################



import torch

def test_relu():
    results = {}
    
    # Test case 1: All negative values
    input_tensor = torch.tensor([-3.0, -1.0, -0.5, -2.0, -5.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_1'] = output_tensor

    # Test case 2: All positive values
    input_tensor = torch.tensor([3.0, 1.0, 0.5, 2.0, 5.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_2'] = output_tensor

    # Test case 3: Mixed values
    input_tensor = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_3'] = output_tensor

    # Test case 4: Zero values
    input_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_4'] = output_tensor

    return results

result_gold = test_relu()
