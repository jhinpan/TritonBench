import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    max_val = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    for col in range(0, stride_xn, BLOCK_N):
        x_block = tl.load(x_ptr + row * stride_xm + col, mask=row < stride_xm, other=-float('inf'))
        max_val = tl.max(max_val, x_block, axis=1)
    tl.store(y_ptr + row, max_val, mask=row < stride_xm)
def load_reduce(x: torch.Tensor, y: torch.Tensor):
    M, N = x.shape
    stride_xm, stride_xn = x.stride()
    stride_y = y.stride(0)
    BLOCK_M = 16
    BLOCK_N = 16
    grid = (M // BLOCK_M + (M % BLOCK_M != 0),)
    load_reduce_kernel[grid](x_ptr=x, y_ptr=y, stride_xm=stride_xm, stride_xn=stride_xn, stride_y=stride_y, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    y_torch, _ = torch.max(x, dim=1)
    assert torch.allclose(y, y_torch), "The results do not match PyTorch's max function"
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
