import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    x_keep = x_keep != 0
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous(), 'Input x must be contiguous.'
    assert x_keep.is_contiguous(), 'Mask x_keep must be contiguous.'
    assert x.shape == x_keep.shape, 'x and x_keep must have the same shape.'
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output
##################################################################################################################################################



# Test for the dropout function
def test_dropout():
    # Dictionary to store test case results
    results = {}
    
    # Test case 1
    x = torch.randn(size=(10,)).cuda()
    p = 0.5
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
    output = dropout(x, x_keep=x_keep, p=p)
    results['test_case_1'] = output

    # Test case 2: p = 0 (no dropout)
    p = 0.0
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
    output = dropout(x, x_keep=x_keep, p=p)
    results['test_case_2'] = output

    # Test case 3: p = 1 (full dropout)
    p = 1.0
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
    output = dropout(x, x_keep=x_keep, p=p)
    results['test_case_3'] = output

    # Test case 4: different block size
    p = 0.5
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
    output = dropout(x, x_keep=x_keep, p=p)
    results['test_case_4'] = output
    
    return results

# Run tests and store result
result_gold = test_dropout()
