import triton
import triton.language as tl
import torch

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    random_values = tl.rand(seed, offsets)
    keep_mask = random_values > p
    scale = 1.0 / (1.0 - p)
    output = tl.where(keep_mask, x * scale, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
def seeded_dropout(x, p, seed, block_size=1024):
    """
    Applies seeded dropout to the input tensor `x`.
    
    Args:
        x (torch.Tensor): Input tensor.
        p (float): Dropout probability.
        seed (int): Seed for random number generation.
        block_size (int): Block size for Triton kernel (default: 1024).
    
    Returns:
        torch.Tensor: Tensor with dropout applied.
    """
    x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x_ptr=x, output_ptr=output, n_elements=n_elements, p=p, seed=seed, BLOCK_SIZE=block_size)
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
