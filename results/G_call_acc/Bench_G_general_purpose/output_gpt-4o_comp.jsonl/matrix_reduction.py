import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    x_offset = pid_m * BLOCK_M * stride_xm
    y_offset = pid_m * BLOCK_M * stride_y
    x_ptrs = x_ptr + x_offset + tl.arange(0, BLOCK_N) * stride_xn
    y_ptrs = y_ptr + y_offset
    x = tl.load(x_ptrs, mask=True, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    tl.store(y_ptrs, row_max)
def load_reduce(x, BLOCK_M=128, BLOCK_N=128):
    assert x.ndim == 2, 'Input must be a 2D tensor'
    M, N = x.shape
    y = torch.empty((M,), dtype=x.dtype, device=x.device)
    grid = (M // BLOCK_M,)
    load_reduce_kernel[grid](x, y, x.stride(0), x.stride(1), y.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    expected = torch.max(x, dim=1).values
    torch.testing.assert_close(y, expected, rtol=1e-05, atol=1e-05)
    return y
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
