import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_ym, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    row_max = tl.zeros(BLOCK_M, dtype=tl.float32) - float('inf')
    for col_start in range(0, N, BLOCK_N):
        offs_n = col_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        block_ptr = tl.make_block_ptr(base=x_ptr, shape=(M, N), strides=(stride_xm, stride_xn), offsets=(row_start, col_start), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        block = tl.load(block_ptr, mask=mask, other=-float('inf'))
        current_max = tl.max(block, axis=1)
        row_max = tl.maximum(row_max, current_max)
    tl.store(y_ptr + offs_m * stride_ym, row_max, mask=mask_m)
def load_reduce():
    M, N = (1024, 512)
    BLOCK_M, BLOCK_N = (128, 32)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(M, device='cuda', dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M),)
    load_reduce_kernel[grid](x_ptr=x, y_ptr=y, stride_xm=x.stride(0), stride_xn=x.stride(1), stride_ym=y.stride(0), M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    y_ref = x.max(dim=1)[0]
    torch.testing.assert_close(y, y_ref, msg='Triton and Torch results differ')
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
