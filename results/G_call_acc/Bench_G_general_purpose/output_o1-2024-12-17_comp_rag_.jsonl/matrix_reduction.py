import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    partial_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    for col_start in range(0, N, BLOCK_N):
        col_idx = col_start + tl.arange(0, BLOCK_N)
        load_mask = (row_idx < M) & (col_idx < N)
        x = tl.load(x_ptr + row_idx[:, None] * stride_xm + col_idx[None, :] * stride_xn, mask=load_mask, other=-float('inf'))
        row_max = tl.max(x, axis=1)
        partial_max = tl.maximum(partial_max, row_max)
    store_mask = row_idx < M
    tl.store(y_ptr + row_idx * stride_y, partial_max, mask=store_mask)
def load_reduce(x: torch.Tensor, BLOCK_M: int=128, BLOCK_N: int=128):
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)
    load_reduce_kernel[grid](x, y, x.stride(0), x.stride(1), y.stride(0), M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.testing.assert_close(y, x.max(dim=1).values)
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
