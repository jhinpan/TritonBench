import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, ACTIVATION: tl.constexpr, alpha: tl.constexpr=0.01):
    """Kernel for matrix multiplication C = A @ B with optional activation."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs_m = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(num_blocks_k):
        k_offset = k * BLOCK_SIZE_K
        a_ptrs = A + (offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k_offset + offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_ptrs = B + ((k_offset + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (k_offset + offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
    if ACTIVATION == 'leaky_relu':
        acc = tl.where(acc >= 0, acc, acc * alpha)
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)
def matmul(a: torch.Tensor, b: torch.Tensor, activation: str=None, alpha: float=0.01):
    """
    High-performance matrix multiplication with optional activation.
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        activation: Optional activation ("leaky_relu")
        alpha: Slope for leaky ReLU (default: 0.01)
    
    Returns:
        torch.Tensor: Result matrix C of shape (M, N)
    """
    assert a.is_cuda and b.is_cuda, 'Inputs must be on GPU'
    assert a.dim() == 2 and b.dim() == 2, 'Inputs must be 2D matrices'
    assert a.size(1) == b.size(0), f'Dimension mismatch: {a.shape} vs {b.shape}'
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, ACTIVATION=activation, alpha=alpha, num_warps=4, num_stages=3)
    return c
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
