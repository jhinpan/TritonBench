import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_M
    offsets = block_start + tl.arange(0, BLOCK_M)
    mask = offsets < M
    x_ptrs = x_ptr + offsets[:, None] * stride_xm + tl.arange(0, BLOCK_N)[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask[:, None] & (tl.arange(0, BLOCK_N)[None, :] < N), other=-float('inf'))
    max_val = tl.max(x, axis=1)
    tl.store(y_ptr + offsets, max_val, mask=mask)
def load_reduce(x):
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N = (32, 128)
    grid = (triton.cdiv(M, BLOCK_M),)
    load_reduce_kernel[grid](x, y, x.stride(0), x.stride(1), y.stride(0), M, N, BLOCK_M, BLOCK_N)
    return y
def test_load_reduce():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = load_reduce(x)
    y_torch = torch.max(x, dim=1)[0]
    torch.testing.assert_close(y_triton, y_torch, atol=0.01, rtol=0.01)
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
