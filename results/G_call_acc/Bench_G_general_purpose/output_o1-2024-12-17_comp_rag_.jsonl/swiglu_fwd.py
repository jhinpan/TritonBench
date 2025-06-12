import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2), triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8)], key=['ncols'])
@triton.jit
def _swiglu_fwd_kernel(X_ptr, Y_ptr, OUT_ptr, nrows, ncols, stride_x_row, stride_x_col, stride_y_row, stride_y_col, stride_out_row, stride_out_col, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    col_start = col_block_idx * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    col_idx = col_start + cols
    mask = col_idx < ncols
    x_offset = row_idx * stride_x_row + col_idx * stride_x_col
    y_offset = row_idx * stride_y_row + col_idx * stride_y_col
    out_offset = row_idx * stride_out_row + col_idx * stride_out_col
    x_data = tl.load(X_ptr + x_offset, mask=mask, other=0.0)
    y_data = tl.load(Y_ptr + y_offset, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x_data))
    swiglu_val = x_data * sig * y_data
    tl.store(OUT_ptr + out_offset, swiglu_val, mask=mask)
def _swiglu_fwd(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy.chunk(2, dim=-1)
    x = x.contiguous()
    y = y.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M, triton.cdiv(N, 64))
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
