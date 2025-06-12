import torch
import triton
import triton.language as tl
import math

def can_use_int32_index(tensor):
    return tensor.numel() <= 1 << 31
def cfggen():
    block_m = [1, 2, 4, 8, 16, 32]
    configs = [triton.Config({'BLOCK_M': m, 'BLOCK_N': 1024}, num_warps=4) for m in block_m]
    return configs
@triton.jit
def max_kernel_1(input_ptr, mid_ptr, num_elements, BLOCK_SIZE: tl.constexpr, INT64_INDEX: tl.constexpr=False):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(values)
    tl.store(mid_ptr + pid, max_val)
@triton.jit
def max_kernel_2(mid_ptr, output_ptr, mid_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mid_ptrs = mid_ptr + offsets
    mask = offsets < mid_size
    values = tl.load(mid_ptrs, mask=mask, other=-float('inf'))
    max_val = tl.max(values)
    tl.store(output_ptr, max_val)
@triton.autotune(configs=cfggen(), key=['M', 'N'])
@triton.jit
def max_kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, INT64_INDEX: tl.constexpr=False):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    acc = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    input_ptr += rows[:, None] * N
    for block_n in range(0, N, BLOCK_N):
        cols = block_n + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        vals = tl.load(input_ptr + cols, mask=mask, other=-float('inf'), eviction_policy='evict_first')
        acc = tl.maximum(acc, tl.max(vals, axis=1))
    tl.store(output_ptr + rows, acc, mask=row_mask)
def max(input, keepdim=False):
    if input.dim() == 0:
        return input
    M = input.numel()
    block_size = triton.next_power_of_2(int(math.sqrt(M)))
    mid_size = (M + block_size - 1) // block_size
    dtype = input.dtype
    device = input.device
    mid = torch.empty(mid_size, dtype=dtype, device=device)
    use_int64_index = not can_use_int32_index(input)
    grid1 = (mid_size,)
    max_kernel_1[grid1](input, mid, M, block_size, INT64_INDEX=use_int64_index)
    block_mid = triton.next_power_of_2(mid_size)
    out_shape = [1] * input.dim() if keepdim else []
    out = torch.empty(out_shape, dtype=dtype, device=device)
    max_kernel_2[1,](mid, out, mid_size, block_mid)
    return out
def max_dim(input, dim, keepdim=False):
    if dim is None:
        return max(input, keepdim=keepdim)
    dim = dim if dim >= 0 else input.dim() + dim
    assert 0 <= dim < input.dim(), 'dim out of range'
    shape = list(input.shape)
    M = 1
    for s in shape[:dim]:
        M *= s
    K = shape[dim]
    N = 1
    for s in shape[dim + 1:]:
        N *= s
    input_flat = input.view(M, K, N).permute(0, 2, 1).contiguous().view(M * N, K)
    out_flat = torch.empty((M * N, 1), dtype=input.dtype, device=input.device)
    use_int64_index = not can_use_int32_index(input_flat)
    grid = lambda meta: (triton.cdiv(M * N, meta['BLOCK_M']),)
    max_kernel[grid](input_flat, out_flat, M * N, K, INT64_INDEX=use_int64_index)
    out_shape = list(shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)
    return out_flat.view(out_shape)
##################################################################################################################################################



def test_max():
    # 测试1：1维Tensor，验证max函数
    # 使用随机生成的长度为1024的一维Tensor
    inp1d = torch.randn(1024, device="cuda")
    # 使用自定义max函数
    out1d_custom = max(inp1d)

    # 测试2：2维Tensor，验证max_dim函数
    # 使用随机生成的1024x1024的二维Tensor
    inp2d = torch.randn(1024, 1024, device="cuda")
    # 使用自定义max_dim函数，沿着dim=1计算最大值
    out2d_custom = max_dim(inp2d, dim=1)

    # 测试3：3维Tensor，验证max_dim函数
    # 使用随机生成的128x64x32的三维Tensor
    inp3d = torch.randn(128, 64, 32, device="cuda")
    # 使用自定义max_dim函数，沿着dim=2计算最大值
    out3d_custom = max_dim(inp3d, dim=2)

    # 测试4：保持维度的测试
    # 使用随机生成的512x256的二维Tensor
    inp2d_keepdim = torch.randn(512, 256, device="cuda")
    # 使用自定义max_dim函数，保持维度的情况下计算最大值
    out2d_custom_keepdim = max_dim(inp2d_keepdim, dim=1, keepdim=True)

    # 测试5：负维度测试
    # 使用随机生成的64x128x256的三维Tensor
    inp3d_neg_dim = torch.randn(64, 128, 256, device="cuda")
    # 使用自定义max_dim函数，沿着负的维度计算最大值（等价于dim=1）
    out3d_custom_neg_dim = max_dim(inp3d_neg_dim, dim=-2)

    # 记录每个测试用例的结果
    results = {
        "test_case_1": out1d_custom,
        "test_case_2": out2d_custom,
        "test_case_3": out3d_custom,
        "test_case_4": out2d_custom_keepdim,
        "test_case_5": out3d_custom_neg_dim,
    }

    return results

result_gold = test_max()
