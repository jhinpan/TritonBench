import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    x_offset = row_start * stride_xm
    x_ptrs = x_ptr + x_offset + tl.arange(0, BLOCK_N) * stride_xn
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_N) < stride_xn)
    row_max = tl.max(x, axis=0)
    y_ptrs = y_ptr + row_start * stride_y
    tl.store(y_ptrs, row_max)
def load_reduce(x, BLOCK_M=128, BLOCK_N=128):
    M, N = x.shape
    y = torch.empty(M, dtype=x.dtype, device=x.device)
    stride_xm, stride_xn = x.stride()
    stride_y = y.stride(0)
    grid = (triton.cdiv(M, BLOCK_M),)
    load_reduce_kernel[grid](x, y, stride_xm, stride_xn, stride_y, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    y_ref = torch.max(x, dim=1).values
    torch.testing.assert_close(y, y_ref)
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
