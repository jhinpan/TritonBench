import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets, mask=offsets < N, other=0.0)
    y = tl.where(x >= 0, x, 0.0)
    tl.store(Y + offsets, y, mask=offsets < N)
def relu(X: torch.Tensor):
    assert X.is_cuda
    Y = torch.empty_like(X)
    N = X.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    relu_kernel[num_blocks,](X, Y, N, BLOCK_SIZE)
    return Y
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
