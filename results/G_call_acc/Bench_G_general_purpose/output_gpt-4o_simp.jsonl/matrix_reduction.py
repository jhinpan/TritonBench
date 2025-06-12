import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(input_ptr, output_ptr, stride_am, stride_an, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    block_start_m = pid_m * BLOCK_M
    mask_m = block_start_m + tl.arange(0, BLOCK_M) < M
    max_vals = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    for block_start_n in range(0, N, BLOCK_N):
        offs_n = block_start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs = block_start_m[:, None] * stride_am + offs_n[None, :] * stride_an
        data = tl.load(input_ptr + offs, mask=mask_m[:, None] & mask_n[None, :], other=-float('inf'))
        max_vals = tl.maximum(max_vals, tl.max(data, axis=1))
    tl.store(output_ptr + block_start_m, max_vals, mask=mask_m)
def load_reduce(input_matrix, BLOCK_M=128, BLOCK_N=128):
    M, N = input_matrix.shape
    output_vector = torch.empty(M, dtype=input_matrix.dtype, device=input_matrix.device)
    grid = (triton.cdiv(M, BLOCK_M),)
    load_reduce_kernel[grid](input_ptr=input_matrix, output_ptr=output_vector, stride_am=input_matrix.stride(0), stride_an=input_matrix.stride(1), M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return output_vector
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
