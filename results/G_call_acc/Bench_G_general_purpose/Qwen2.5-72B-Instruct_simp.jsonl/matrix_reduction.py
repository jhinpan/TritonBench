import triton
import triton.language as tl
import numpy as np
import torch

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, M, N, stride_xm, stride_xn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    rm_end = tl.minimum(rm + BLOCK_M, M)
    rn_end = tl.minimum(rn + BLOCK_N, N)
    x_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(rm, rm_end):
        for j in range(rn, rn_end):
            x_block[i - rm, j - rn] = tl.load(x_ptr + i * stride_xm + j * stride_xn)
    max_values = tl.max(x_block, axis=1)
    for i in range(rm, rm_end):
        tl.store(y_ptr + i, max_values[i - rm])
def load_reduce(M, N, BLOCK_M, BLOCK_N):
    x = np.random.randn(M, N).astype(np.float32)
    x_ptr = triton.testing.to_triton(x, device='cuda')
    y = np.zeros(M, dtype=np.float32)
    y_ptr = triton.testing.to_triton(y, device='cuda')
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)
    load_reduce_kernel[grid](x_ptr, y_ptr, M, N, x.shape[1], 1, BLOCK_M, BLOCK_N)
    y = y_ptr.to_numpy()
    expected = np.max(x, axis=1)
    np.testing.assert_allclose(y, expected, rtol=1e-05, atol=1e-05)
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
