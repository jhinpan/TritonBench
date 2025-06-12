import torch
import triton
import triton.language as tl

@triton.jit
def _multinomial_sampling_kernel(Scores, Seeds, Offsets, Indices, Outputs, stride_sb, stride_st, stride_ib, stride_it, num_batchs, num_tokens, BLOCK: tl.constexpr, BLOCK_N: tl.constexpr):
    batch_block_id = tl.program_id(0)
    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)
    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)
    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb + s_off[None, :] * stride_st, mask=s_mask, other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, axis=1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, axis=1)
        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask.to(tl.int1), axis=1) > 0
        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), axis=1)
        indices = tl.load(Indices + off * stride_ib + valid_pos * stride_it, mask=found_mask & off_mask, other=-1)
        output = tl.where(found_mask, indices, output)
    tl.store(Outputs + off, output, mask=off_mask)
def multinomial_sampling(scores: torch.Tensor, seeds: torch.LongTensor, offsets: torch.LongTensor, indices: torch.Tensor=None) -> torch.Tensor:
    """
    Performs multinomial sampling using Triton kernel.
    
    Args:
        scores: 2D tensor of shape [batch_size, num_tokens]
        seeds: 1D tensor of random seeds per batch
        offsets: 1D tensor of offset counters for random generation
        indices: Optional 2D tensor of token indices (for sorted distributions)
    
    Returns:
        Sampled indices tensor of shape [batch_size]
    """
    assert scores.dim() == 2
    batch_size, num_tokens = scores.shape
    device = scores.device
    if num_tokens == 1:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if indices is None:
        indices = torch.arange(num_tokens, device=device).expand_as(scores)
    assert indices.size() == scores.size()
    outputs = indices[:, 0].clone()
    BLOCK = 8
    BLOCK_N = 128
    grid = (triton.cdiv(batch_size, BLOCK),)
    _multinomial_sampling_kernel[grid](Scores=scores, Seeds=seeds, Offsets=offsets, Indices=indices, Outputs=outputs, stride_sb=scores.stride(0), stride_st=scores.stride(1), stride_ib=indices.stride(0), stride_it=indices.stride(1), num_batchs=batch_size, num_tokens=num_tokens, BLOCK=BLOCK, BLOCK_N=BLOCK_N, num_warps=8)
    return outputs
##################################################################################################################################################



import torch

def test_multinomial_sampling():
    result_dict = {}

    # Test case 1: Basic functionality with default indices
    scores = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], dtype=torch.float32).cuda()
    seeds = torch.tensor([123, 456], dtype=torch.int64).cuda()
    offsets = torch.tensor([0, 0], dtype=torch.int64).cuda()

    outputs = multinomial_sampling(scores, seeds, offsets)
    result_dict['test_case_1'] = outputs

    # Test case 2: Providing custom indices
    indices = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64).cuda()
    outputs = multinomial_sampling(scores, seeds, offsets, indices)
    result_dict['test_case_2'] = outputs

    # Test case 3: Single token case
    scores_single_token = torch.tensor([[1.0], [1.0]], dtype=torch.float32).cuda()
    outputs = multinomial_sampling(scores_single_token, seeds, offsets)
    result_dict['test_case_3'] = outputs

    return result_dict

result_gold = test_multinomial_sampling()
