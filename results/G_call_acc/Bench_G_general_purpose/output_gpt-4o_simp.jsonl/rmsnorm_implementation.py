import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(x_ptr, rms_weights_ptr, out_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_SIZE
    offset_n = pid_n * BLOCK_SIZE
    x_block = tl.load(x_ptr + offset_m * N * K + offset_n * K + tl.arange(0, BLOCK_SIZE))
    x_squared = x_block * x_block
    sum_x_squared = tl.sum(x_squared, axis=0)
    rms = tl.sqrt(sum_x_squared / K)
    normalized_block = x_block / rms
    rms_weights = tl.load(rms_weights_ptr + offset_n * K + tl.arange(0, BLOCK_SIZE))
    scaled_block = normalized_block * rms_weights
    tl.store(out_ptr + offset_m * N * K + offset_n * K + tl.arange(0, BLOCK_SIZE), scaled_block)
def rmsnorm_wrapper(x, rms_weights):
    x = x.contiguous()
    rms_weights = rms_weights.contiguous()
    M, N, K = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
    rmsnorm_triton[grid](x, rms_weights, out, M, N, K, BLOCK_SIZE)
    return out
##################################################################################################################################################



def test_rmsnorm():
    # Define the input tensor x with shape (batch, M, K)
    batch = 2
    M = 3
    K = 4096
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")

    # Define the rms_weights tensor with shape (K,)
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")

    # Dictionary to store the results of different test cases
    results = {}

    # Test case 1
    out1 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_1'] = out1.cpu()

    # Additional test cases for branch coverage

    # Test case 2: Different batch size
    batch = 4
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out2 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_2'] = out2.cpu()

    # Test case 3: Different M size
    M = 5
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out3 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_3'] = out3.cpu()

    # Test case 4: Larger K size
    K = 8192
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out4 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_4'] = out4.cpu()

    return results

# Execute the test function
result_gold = test_rmsnorm()
