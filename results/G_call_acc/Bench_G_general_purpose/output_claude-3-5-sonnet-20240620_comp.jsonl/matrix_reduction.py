import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_idx < M
    output = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        col_idx = n + tl.arange(0, BLOCK_N)
        col_mask = col_idx < N
        mask = row_mask[:, None] & col_mask[None, :]
        offs = row_idx[:, None] * stride_xm + col_idx[None, :] * stride_xn
        x = tl.load(x_ptr + offs, mask=mask, other=float('-inf'))
        output = tl.maximum(output, tl.max(x, axis=1))
    tl.store(y_ptr + row_idx * stride_y, output, mask=row_mask)
def load_reduce(x):
    """
    Compute row-wise maximum of input tensor x using Triton kernel
    
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        y: Output tensor of shape (M,) containing row-wise maxima
    """
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK_M = 32
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M),)
    load_reduce_kernel[grid](x_ptr=x, y_ptr=y, stride_xm=x.stride(0), stride_xn=x.stride(1), stride_y=1, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return y
def test_load_reduce():
    torch.manual_seed(0)
    M, N = (1024, 2048)
    x = torch.randn(M, N, device='cuda')
    y_triton = load_reduce(x)
    y_torch = torch.max(x, dim=1)[0]
    torch.testing.assert_close(y_triton, y_torch)
    print('Test passed!')
##################################################################################################################################################



import torch

def test_reduce():
    # 测试参数设置
    test_cases = [
        {"BLOCK_M": 16, "BLOCK_N": 16, "dtype_str": "float16"},
        {"BLOCK_M": 32, "BLOCK_N": 32, "dtype_str": "float16"},
        {"BLOCK_M": 64, "BLOCK_N": 64, "dtype_str": "float32"},
        {"BLOCK_M": 128, "BLOCK_N": 128, "dtype_str": "float32"},
    ]

    results = {}
    for i, case in enumerate(test_cases):
        BLOCK_M = case["BLOCK_M"]
        BLOCK_N = case["BLOCK_N"]
        dtype_str = case["dtype_str"]

        try:
            load_reduce(BLOCK_M, BLOCK_N, dtype_str)
            results[f"test_case_{i+1}"] = "passed"
        except Exception as e:
            results[f"test_case_{i+1}"] = f"failed: {e}"

    return results

result_gold = test_reduce()
