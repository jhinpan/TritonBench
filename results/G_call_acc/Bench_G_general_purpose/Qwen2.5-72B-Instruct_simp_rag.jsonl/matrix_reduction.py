import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(input_ptr, output_ptr, stride_im, stride_in, stride_on, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    block_end_m = min(block_start_m + BLOCK_M, M)
    block_end_n = min(block_start_n + BLOCK_N, N)
    max_value = -tl.inf
    for i in range(block_start_m, block_end_m):
        for j in range(block_start_n, block_end_n):
            value = tl.load(input_ptr + i * stride_im + j * stride_in)
            max_value = tl.max(max_value, value)
    tl.store(output_ptr + pid_m * stride_on, max_value)
def load_reduce(input: torch.Tensor, output: torch.Tensor, block_m: int, block_n: int):
    assert input.dim() == 2, 'Input must be a 2D tensor'
    assert output.dim() == 1, 'Output must be a 1D tensor'
    assert output.size(0) == input.size(0), 'Output size must match the number of rows in the input'
    M, N = input.size()
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    load_reduce_kernel[grid](input, output, input.stride(0), input.stride(1), output.stride(0), M, N, BLOCK_M=block_m, BLOCK_N=block_n)
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
