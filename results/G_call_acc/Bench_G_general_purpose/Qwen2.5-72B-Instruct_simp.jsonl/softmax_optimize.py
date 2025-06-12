import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel_online_v2(X, Y, M, N, TILE_N: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = M
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group // num_pid_m
    pid_m = first_pid_m + pid % num_pid_m
    pid_n = pid % group_size * TILE_N
    offs_m = pid_m
    offs_n = pid_n + tl.arange(0, TILE_N)
    X_block_ptr = X + (offs_m * N + offs_n)
    Y_block_ptr = Y + (offs_m * N + offs_n)
    x = tl.load(X_block_ptr, mask=offs_n < N, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    x = x - max_val
    z = tl.exp(x)
    z_sum = tl.sum(z, axis=0)
    z = z / z_sum
    tl.store(Y_block_ptr, z, mask=offs_n < N)
def prev_multiple_of(a, b):
    return a // b * b
def softmax(x: torch.Tensor):
    M, N = x.shape
    out = torch.empty_like(x)
    TILE_N = 128
    assert TILE_N & TILE_N - 1 == 0, 'TILE_N must be a power of 2'
    grid = (M * ((N + TILE_N - 1) // TILE_N),)
    softmax_kernel_online_v2[grid](x, out, M, N, TILE_N)
    return out
##################################################################################################################################################



# Comparison Test
def test_softmax():

    torch.manual_seed(0)
    
    result = {}
    
    # Case 1: M = 128, N = 512
    x1 = torch.randn(128, 512, device='cuda', dtype=torch.float32)
    result['test_case_1'] = softmax(x1)

    # Case 2: M = 64, N = 1024
    x2 = torch.randn(64, 1024, device='cuda', dtype=torch.float32)
    result['test_case_2'] = softmax(x2)

    # Case 3: M = 256, N = 128
    x3 = torch.randn(256, 128, device='cuda', dtype=torch.float32)
    result['test_case_3'] = softmax(x3)
    
    return result

# Execute test function
result_gold = test_softmax()
