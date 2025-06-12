import triton
import triton.language as tl
import torch
import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    rng = tl.rand(seed, offsets)
    keep_mask = rng > p
    output = tl.where(keep_mask, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
def seeded_dropout(x, p, seed):
    x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE)
    return output
##################################################################################################################################################



import torch

# Test for the seeded_dropout function
def test_seeded_dropout():
    # Input tensor
    x = torch.randn(size=(10,)).cuda()
    results = {}
    # Test with the same seed
    results['test_case_1'] = seeded_dropout(x, p=0.5, seed=123)
    results['test_case_2'] = seeded_dropout(x, p=0.5, seed=123)
    # Test with a different seed
    results['test_case_3'] = seeded_dropout(x, p=0.5, seed=512)
    # Test with a different probability
    results['test_case_4'] = seeded_dropout(x, p=0.3, seed=123)
    return results

# Run tests
result_gold = test_seeded_dropout()
