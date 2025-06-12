import triton
import triton.language as tl
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * stride_am
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_bn
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :])
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = offs_am[:, None] + offs_k[None, :] * stride_ak < M * K
        b_mask = offs_k[:, None] * stride_bk + offs_bn[None, :] < K * N
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION:
        accumulator = tl.where(accumulator > 0, accumulator, 0.01 * accumulator)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=c_mask)
def matmul(A, B, activation=False):
    assert A.is_contiguous() and B.is_contiguous(), 'Input tensors must be contiguous'
    assert A.device == B.device, 'Input tensors must be on the same device'
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    ACTIVATION = 1 if activation else 0
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, ACTIVATION)
    return C
##################################################################################################################################################



def test_matmul():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define matrix dimensions
    M, K, N = 64, 128, 64

    # Create random matrices A and B
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    # Compute matrix multiplication using Triton with leaky_relu activation
    c_triton_leaky_relu = matmul(a, b, activation="leaky_relu")

    # Compute matrix multiplication using Triton without activation
    c_triton_no_activation = matmul(a, b, activation="")

    # Store results in a dictionary
    results = {
        "test_case_1": c_triton_leaky_relu,
        "test_case_2": c_triton_no_activation
    }
    
    return results

# Run the test
result_gold = test_matmul()
