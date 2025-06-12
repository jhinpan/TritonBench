import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == 'leaky_relu':
        accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
def matmul(a: torch.Tensor, b: torch.Tensor, activation: str='') -> torch.Tensor:
    """
    Compute matrix multiplication C = A @ B with optional activation
    
    Args:
        a: Input matrix A (M x K)
        b: Input matrix B (K x N) 
        activation: Activation function to apply ("leaky_relu" or "")
    
    Returns:
        Output matrix C (M x N)
    """
    assert a.shape[1] == b.shape[0], 'Incompatible matrix dimensions'
    assert a.is_contiguous(), 'Matrix A must be contiguous'
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c, M=M, N=N, K=K, stride_am=a.stride(0), stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cm=c.stride(0), stride_cn=c.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M, ACTIVATION=activation)
    return c
##################################################################################################################################################



def test_matmul():
    results = {}

    # Test case 1: Basic matrix multiplication without activation
    a = torch.randn((256, 64), device='cuda', dtype=torch.float16)
    b = torch.randn((64, 256), device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    results["test_case_1"] = c

    # Test case 2: Matrix multiplication with leaky ReLU activation
    c_with_activation = matmul(a, b, activation="leaky_relu")
    results["test_case_2"] = c_with_activation

    # Test case 3: Matrix multiplication with larger dimensions
    a_large = torch.randn((512, 128), device='cuda', dtype=torch.float16)
    b_large = torch.randn((128, 512), device='cuda', dtype=torch.float16)
    c_large = matmul(a_large, b_large)
    results["test_case_3"] = c_large

    # Test case 4: Matrix multiplication with larger dimensions and leaky ReLU activation
    c_large_with_activation = matmul(a_large, b_large, activation="leaky_relu")
    results["test_case_4"] = c_large_with_activation

    return results

result_gold = test_matmul()
