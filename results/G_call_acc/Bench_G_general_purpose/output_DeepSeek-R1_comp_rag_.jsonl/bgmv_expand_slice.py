import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(input_ptr, lora_ptr, out_ptr, N, K, lora_indices, xm_stride, xk_stride, l0_stride, lora_k_stride, lora_n_stride, cm_stride, cn_stride, slice_offset, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_N: tl.constexpr, EVEN_K: tl.constexpr, ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr):
    pid_sn = tl.program_id(0)
    cur_batch = tl.program_id(1)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride + offset_k * xk_stride)
    else:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride + offset_k * xk_stride, mask=offset_k < K, other=0)
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    split_n_length = tl.cdiv(N, SPLIT_N)
    b_base = lora_ptr + l0_stride * lora_index + pid_sn * split_n_length * lora_k_stride
    c_base = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length + slice_offset * cn_stride
    for n_start in range(0, split_n_length, BLOCK_N):
        n_offs = n_start + offset_n
        b_ptrs = b_base + n_offs[:, None] * lora_k_stride + offset_k[None, :] * lora_n_stride
        b_mask = (n_offs[:, None] < split_n_length) & (offset_k[None, :] < K)
        tiled_b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.sum(tiled_a * tiled_b, axis=1)
        if ADD_INPUTS:
            c_ptrs = c_base + n_offs * cn_stride
            c_mask = n_offs < split_n_length
            existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
            acc += existing
        tl.store(c_base + n_offs * cn_stride, acc, mask=n_offs < split_n_length)
@torch.inference_mode()
def _bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor, output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor, slice_offset: int, slice_size: int, add_inputs: bool=True) -> None:
    assert inputs.is_contiguous() and output_tensor.is_contiguous()
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert slice_size == lora_b_weights.size(-2)
    if lora_b_weights.ndim == 4:
        lora_b_weights = lora_b_weights.squeeze(1)
    assert lora_b_weights.ndim == 3
    N, K = lora_b_weights.shape[-2:]
    BLOCK_K = triton.next_power_of_2(K)
    EVEN_K = K % BLOCK_K == 0
    CAST_TYPE = inputs.dtype == torch.float32 and lora_b_weights.dtype in {torch.float16, torch.bfloat16}
    config = {'BLOCK_N': 64, 'SPLIT_N': 4, 'num_stages': 2, 'num_warps': 4}
    grid = (config['SPLIT_N'], lora_indices_tensor.size(0))
    _bgmv_expand_slice_kernel[grid](inputs, lora_b_weights, output_tensor, N, K, lora_indices_tensor, inputs.stride(0), inputs.stride(1), lora_b_weights.stride(0), lora_b_weights.stride(1), lora_b_weights.stride(2), output_tensor.stride(0), output_tensor.stride(1), slice_offset, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K, ADD_INPUTS=add_inputs, CAST_TYPE=CAST_TYPE, **config)
##################################################################################################################################################



import torch

def test_bgmv_expand_slice():
    # Define test inputs
    batch_size = 4
    hidden_size = 128
    rank = 64
    lora_num = 3

    inputs = torch.randn(batch_size, hidden_size, dtype=torch.float16, device='cuda').contiguous()
    lora_b_weights = torch.randn(lora_num, rank, hidden_size, dtype=torch.float16, device='cuda').contiguous()
    output_tensor = torch.zeros(batch_size, rank, dtype=torch.float16, device='cuda').contiguous()
    lora_indices_tensor = torch.tensor([0, 1, -1, 2], dtype=torch.int32, device='cuda')
    slice_offset = 0
    slice_size = 64

    results = {}

    # Test case 1: Basic functionality with add_inputs=True
    _bgmv_expand_slice(
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        output_tensor=output_tensor,
        lora_indices_tensor=lora_indices_tensor,
        slice_offset=slice_offset,
        slice_size=slice_size,
        add_inputs=True
    )
    results['test_case_1'] = output_tensor.clone()

    # Test case 2: Basic functionality with add_inputs=False
    output_tensor_zero = torch.zeros_like(output_tensor)
    _bgmv_expand_slice(
        inputs=inputs,
        lora_b_weights=lora_b_weights,
        output_tensor=output_tensor_zero,
        lora_indices_tensor=lora_indices_tensor,
        slice_offset=slice_offset,
        slice_size=slice_size,
        add_inputs=False
    )
    results['test_case_2'] = output_tensor_zero.clone()

    # Test case 3: With casting from float32 to float16
    inputs_float32 = inputs.to(torch.float32)
    _bgmv_expand_slice(
        inputs=inputs_float32,
        lora_b_weights=lora_b_weights,
        output_tensor=output_tensor,
        lora_indices_tensor=lora_indices_tensor,
        slice_offset=slice_offset,
        slice_size=slice_size,
        add_inputs=True
    )
    results['test_case_3'] = output_tensor.clone()

    return results

result_gold = test_bgmv_expand_slice()
