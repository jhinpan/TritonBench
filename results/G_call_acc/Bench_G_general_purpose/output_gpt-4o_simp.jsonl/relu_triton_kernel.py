import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    relu_x = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, relu_x, mask=offsets < n_elements)
def relu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
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
