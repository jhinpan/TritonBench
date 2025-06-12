import triton
import triton.language as tl
import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn), offsets=(row_start, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x_block = tl.load(x_block_ptr)
    row_max = tl.max(x_block, axis=1)
    y_block_ptr = tl.make_block_ptr(base=y_ptr, shape=(BLOCK_M,), strides=(stride_y,), offsets=(row_start,), block_shape=(BLOCK_M,), order=(0,))
    tl.store(y_block_ptr, row_max)
def load_reduce(x: torch.Tensor, y: torch.Tensor, BLOCK_M: int, BLOCK_N: int):
    x = x.cuda()
    y = y.cuda()
    M, N = x.shape
    assert y.shape == (M,)
    grid = (M // BLOCK_M, 1, 1)
    block = (BLOCK_M, BLOCK_N, 1)
    load_reduce_kernel[grid, block](x_ptr=x, y_ptr=y, stride_xm=x.stride(0), stride_xn=x.stride(1), stride_y=y.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.cuda.synchronize()
    y_ref = torch.max(x, dim=1).values
    assert torch.allclose(y, y_ref), 'The results do not match the reference implementation.'
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
