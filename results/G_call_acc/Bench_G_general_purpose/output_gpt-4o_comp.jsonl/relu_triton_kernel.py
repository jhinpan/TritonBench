import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(input_ptr + offsets, mask=offsets < N, other=0.0)
    relu_result = tl.where(input_data > 0, input_data, 0.0)
    tl.store(output_ptr + offsets, relu_result, mask=offsets < N)
def relu(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    output_tensor = torch.empty_like(input_tensor)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input_tensor, output_tensor, N, BLOCK_SIZE=BLOCK_SIZE)
    return output_tensor
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
