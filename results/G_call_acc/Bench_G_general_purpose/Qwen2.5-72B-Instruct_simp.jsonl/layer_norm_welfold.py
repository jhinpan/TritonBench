import triton
import triton.language as tl
import torch
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_no_welford(X, Y, Mean, Rstd, weight, bias, stride, N, eps, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    block_end = min(block_start + XBLOCK, stride)
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(block_start, block_end):
        x = tl.load(X + i * N)
        sum_x += x
        sum_x2 += x * x
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    for i in range(block_start, block_end):
        x = tl.load(X + i * N)
        y = (x - mean) * rstd
        y = y * tl.load(weight + i) + tl.load(bias + i)
        tl.store(Y + i * N, y)
def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3, eps=1e-05):
    device = primals_1.device
    assert primals_2.device == device and primals_3.device == device
    N = primals_1.shape[-1]
    stride = primals_1.numel() // N
    Y = torch.empty_like(primals_1)
    Mean = torch.empty((stride,), device=device, dtype=torch.float32)
    Rstd = torch.empty((stride,), device=device, dtype=torch.float32)
    XBLOCK = 128
    RBLOCK = 128
    grid = (stride // XBLOCK + (stride % XBLOCK > 0),)
    triton_red_fused_native_layer_norm_no_welford[grid](primals_1, Y, Mean, Rstd, primals_2, primals_3, stride, N, eps, XBLOCK, RBLOCK)
    return (Y, Mean, Rstd)
##################################################################################################################################################



import torch

def test_fused_native_layer_norm_no_welford():
    # Define the input shapes
    S = 128  # Number of sequences
    D = 4096  # Dimension of each sequence

    # Create input tensors with appropriate shapes and data types
    primals_1 = torch.randn(D, dtype=torch.bfloat16, device='cuda')  # Weight tensor
    primals_2 = torch.randn(D, dtype=torch.bfloat16, device='cuda')  # Bias tensor
    primals_3 = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')  # Input tensor

    # Test the fused_native_layer_norm_no_welford function
    test_case_1 = fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3)

    # Additional test cases to cover all branches
    # Test case 2: Different input size
    S2 = 256
    primals_3_case2 = torch.randn(S2, D, dtype=torch.bfloat16, device='cuda')
    test_case_2 = fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3_case2)

    # Test case 3: Different dimension size
    D2 = 2048
    primals_1_case3 = torch.randn(D2, dtype=torch.bfloat16, device='cuda')
    primals_2_case3 = torch.randn(D2, dtype=torch.bfloat16, device='cuda')
    primals_3_case3 = torch.randn(S, D2, dtype=torch.bfloat16, device='cuda')
    test_case_3 = fused_native_layer_norm_no_welford(primals_1_case3, primals_2_case3, primals_3_case3)

    # Test case 4: Edge case with minimal size
    S4 = 1
    D4 = 1
    primals_1_case4 = torch.randn(D4, dtype=torch.bfloat16, device='cuda')
    primals_2_case4 = torch.randn(D4, dtype=torch.bfloat16, device='cuda')
    primals_3_case4 = torch.randn(S4, D4, dtype=torch.bfloat16, device='cuda')
    test_case_4 = fused_native_layer_norm_no_welford(primals_1_case4, primals_2_case4, primals_3_case4)

    return {
        "test_case_1": test_case_1,
        "test_case_2": test_case_2,
        "test_case_3": test_case_3,
        "test_case_4": test_case_4,
    }

result_gold = test_fused_native_layer_norm_no_welford()
