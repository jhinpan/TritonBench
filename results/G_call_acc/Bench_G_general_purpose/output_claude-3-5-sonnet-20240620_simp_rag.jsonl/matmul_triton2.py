import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    for block_m, block_n, block_k in [(16, 16, 16), (32, 32, 32), (64, 64, 32), (128, 128, 32)]:
        num_warps = 4 if block_n >= 64 else 2
        num_stages = 3
        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_SIZE_M': 8}, num_stages=num_stages, num_warps=num_warps))
    return configs
@triton.autotune(configs=get_configs(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Compute the matrix multiplication C = A @ B."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < N, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """Compute the matrix multiplication C = A @ B using a Triton kernel."""
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions'
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c, M=M, N=N, K=K, stride_am=a.stride(0), stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cm=c.stride(0), stride_cn=c.stride(1))
    return c
def test_matmul():
    M, N, K = (128, 128, 128)
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c_triton = triton_matmul(a, b)
    c_torch = torch.matmul(a, b)
    assert torch.allclose(c_triton, c_torch, rtol=0.01, atol=0.01)
    print('✓ Triton and PyTorch results match')
##################################################################################################################################################



import torch

# Function to compare results of Triton and PyTorch matmul
def test_matmul():
    results = {}
    
    # Test case 1
    M, K, N = 256, 256, 256
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c_triton_1 = triton_matmul(a, b)
    results['test_case_1'] = c_triton_1

    # Test case 2
    M, K, N = 64, 64, 64
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c_triton_2 = triton_matmul(a, b)
    results['test_case_2'] = c_triton_2

    # Test case 3
    M, K, N = 16, 16, 16
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c_triton_3 = triton_matmul(a, b)
    results['test_case_3'] = c_triton_3

    return results

# Run the comparison
result_gold = test_matmul()