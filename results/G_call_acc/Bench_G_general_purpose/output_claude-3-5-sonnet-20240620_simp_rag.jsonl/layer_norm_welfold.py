import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice

@triton.autotune(configs=[triton.Config({'XBLOCK': 1, 'RBLOCK': 1024}, num_stages=1, num_warps=8), triton.Config({'XBLOCK': 1, 'RBLOCK': 2048}, num_stages=1, num_warps=8)], key=['xnumel', 'rnumel'])
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask).to(tl.float32)
        _tmp3 += tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    mean = tl.sum(_tmp3, 1)[:, None] / rnumel
    tl.store(in_out_ptr0 + x0, mean, None)
    _var = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        val = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask).to(tl.float32)
        diff = val - mean
        _var += tl.broadcast_to(diff * diff, [XBLOCK, RBLOCK])
    var = tl.sum(_var, 1)[:, None] / rnumel
    inv_std = libdevice.rsqrt(var + 1e-05)
    tl.store(in_out_ptr1 + x0, inv_std, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        val = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask).to(tl.float32)
        weight = tl.load(in_ptr1 + r1, rmask).to(tl.float32)
        bias = tl.load(in_ptr2 + r1, rmask).to(tl.float32)
        normalized = (val - mean) * inv_std
        result = normalized * weight + bias
        tl.store(out_ptr0 + (r1 + rnumel * x0), result, rmask)
def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    """
    Wrapper function for layer normalization without Welford's algorithm
    Args:
        primals_1: weights
        primals_2: bias
        primals_3: input tensor
    Returns:
        Tuple of (normalized tensor, input tensor, mean, inv_std)
    """
    S, D = primals_3.shape
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        mean_buf = torch.empty((S, 1), dtype=torch.float32, device='cuda')
        inv_std_buf = torch.empty((S, 1), dtype=torch.float32, device='cuda')
        output_buf = torch.empty_like(primals_3)
        grid = lambda META: (triton.cdiv(S, META['XBLOCK']),)
        triton_red_fused_native_layer_norm_no_welford[grid](mean_buf, inv_std_buf, primals_3, primals_1, primals_2, output_buf, S, D)
    return (output_buf, primals_3, mean_buf, inv_std_buf)
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
