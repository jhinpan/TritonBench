import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 128}, num_warps=4), triton.Config({'BLOCK_SIZE': 256}, num_warps=4), triton.Config({'BLOCK_SIZE': 512}, num_warps=4)], key=['N'])
@triton.jit
def _swiglu_fwd_kernel(X_ptr, Y_ptr, OUT_ptr, M, N, stride_x_row, stride_x_col, stride_y_row, stride_y_col, stride_out_row, stride_out_col, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    if row_idx >= M:
        return
    col_start = col_block_idx * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y_ptrs = Y_ptr + row_idx * stride_y_row + col_offsets * stride_y_col
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    sig_y = tl.sigmoid(y)
    output = x * sig_y
    out_ptrs = OUT_ptr + row_idx * stride_out_row + col_offsets * stride_out_col
    tl.store(out_ptrs, output, mask=mask)
def _swiglu_fwd(xy: torch.Tensor) -> torch.Tensor:
    assert xy.dim() == 2, 'Input tensor must be 2-dimensional'
    assert xy.size(-1) % 2 == 0, 'Last dimension must be even'
    x = xy[..., :xy.size(-1) // 2].contiguous()
    y = xy[..., xy.size(-1) // 2:].contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_SIZE']))
    _swiglu_fwd_kernel[grid](x, y, out, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), out.stride(0), out.stride(1))
    return out
##################################################################################################################################################



# Test the forward function with different configurations
def test_swiglu_fwd():
    results = {}
    # Test case 1
    batch_size = 4
    ncols = 128
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_1'] = out.detach().cpu()

    # Test case 2
    batch_size = 8
    ncols = 256
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_2'] = out.detach().cpu()

    # Test case 3
    batch_size = 16
    ncols = 512
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_3'] = out.detach().cpu()

    # Test case 4
    batch_size = 32
    ncols = 1024
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_4'] = out.detach().cpu()

    return results

# Run the tests
result_gold = test_swiglu_fwd()
