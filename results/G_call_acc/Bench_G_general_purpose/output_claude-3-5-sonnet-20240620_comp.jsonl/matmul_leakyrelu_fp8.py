import triton
import triton.language as tl
import torch

@triton.jit
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return tl.where(x > 0, x, alpha * x)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, activation_type: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """
    Compute C = activation(A @ B) using block-level matrix multiplication
    """
    pid = tl.program_id(axis=0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K))
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)
    if activation_type == 1:
        acc = leaky_relu(acc)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask)
def matmul(a: torch.Tensor, b: torch.Tensor, activation: str=None):
    """
    Compute C = activation(A @ B)
    
    Parameters:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
        activation: Activation function to apply ('leaky_relu' or None)
    
    Returns:
        c: Output tensor of shape (M, N)
    """
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions for matrix multiplication'
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    activation_type = 1 if activation == 'leaky_relu' else 0
    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c, M=M, N=N, K=K, stride_am=a.stride(0), stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cm=c.stride(0), stride_cn=c.stride(1), activation_type=activation_type, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)
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
