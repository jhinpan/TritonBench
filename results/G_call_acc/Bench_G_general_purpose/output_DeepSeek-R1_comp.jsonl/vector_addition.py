import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, 'Inputs must be CUDA tensors'
    assert x.shape == y.shape, 'Input shapes must match'
    output = torch.empty_like(x)
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
##################################################################################################################################################



def test_add():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    # Test case 1
    output_triton_1 = add(x, y)
    
    # Test case 2
    size_2 = 1024
    x_2 = torch.rand(size_2, device='cuda')
    y_2 = torch.rand(size_2, device='cuda')
    output_triton_2 = add(x_2, y_2)
    
    # Test case 3
    size_3 = 2048
    x_3 = torch.rand(size_3, device='cuda')
    y_3 = torch.rand(size_3, device='cuda')
    output_triton_3 = add(x_3, y_3)
    
    # Test case 4
    size_4 = 4096
    x_4 = torch.rand(size_4, device='cuda')
    y_4 = torch.rand(size_4, device='cuda')
    output_triton_4 = add(x_4, y_4)
    
    results = {
        "test_case_1": output_triton_1,
        "test_case_2": output_triton_2,
        "test_case_3": output_triton_3,
        "test_case_4": output_triton_4
    }
    
    return results

result_gold = test_add()
