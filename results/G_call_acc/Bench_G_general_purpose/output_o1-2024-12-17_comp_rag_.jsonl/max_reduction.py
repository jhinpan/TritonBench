import math
import torch
import triton
import triton.language as tl

def can_use_int32_index(tensor: torch.Tensor) -> bool:
    """
    Checks if a tensor can use 32-bit indexing (i.e., the number
    of elements is within the range of 32-bit integers).
    """
    return tensor.numel() < 2 ** 31
def dim_compress(tensor: torch.Tensor, dims: list) -> torch.Tensor:
    """
    A simple helper that permutes the specified dimensions (dims) 
    to the end of the tensor shape, effectively compressing them 
    into contiguous axes for easier calculations.
    """
    ndims = tensor.ndim
    dims = sorted((d % ndims for d in dims))
    all_dims = list(range(ndims))
    for d in reversed(dims):
        all_dims.append(all_dims.pop(d))
    permuted = tensor.permute(all_dims)
    product = 1
    for d in dims:
        product *= tensor.shape[d]
    shape = list(permuted.shape)
    new_shape = shape[:ndims - len(dims)] + [product]
    return permuted.reshape(new_shape)
def cfggen():
    """
    Helper function to generate configurations for Triton autotuning.
    """
    block_m = [1, 2, 4, 8]
    configs = [triton.Config({'BLOCK_M': m, 'BLOCK_N': 1024}, num_warps=4) for m in block_m]
    return configs
@triton.jit
def max_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr, INT64_INDEX: tl.constexpr=False):
    """
    This kernel computes block-level maximum values in a 1D tensor.
    'pid' is the program id. Each block processes BLOCK_SIZE elements
    starting at offset = pid * BLOCK_SIZE. The max value for each block 
    is stored in 'mid'.
    """
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    val = tl.load(inp_ptrs, mask=mask, other=-float('inf'))
    block_max = tl.max(val)
    tl.store(mid + pid, block_max)
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """
    This kernel takes the intermediate results from 'mid',
    computes their maximum, and stores the result in 'out'.
    """
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float('inf'))
    block_max = tl.max(mid_val)
    tl.store(out, block_max)
@triton.autotune(configs=cfggen(), key=['M', 'N'])
@triton.jit
def max_kernel(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, INT64_INDEX: tl.constexpr=False):
    """
    Generic maximum reduction kernel. Each program id (pid) handles
    one block of rows. For each row in a block, it scans along N
    columns and computes the maximum. Results are stored in 'out'.
    """
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M
    inp += rows * N
    out += rows
    max_vals = tl.full([BLOCK_M, BLOCK_N], -float('inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        val = tl.load(inp + cols, mask=mask, other=-float('inf'))
        val_float32 = val.to(tl.float32)
        max_vals = tl.maximum(max_vals, val_float32)
    row_max = tl.max(max_vals, axis=1)[:, None]
    tl.store(out, row_max, mask=row_mask)
def max(inp: torch.Tensor, keepdim: bool=False) -> torch.Tensor:
    """
    Computes the maximum value across the entire input tensor.
    Uses max_kernel_1 and max_kernel_2. The final result is written 
    to 'out'. The shape of 'out' depends on 'keepdim'.
    """
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = (M + block_size - 1) // block_size
    block_mid = triton.next_power_of_2(mid_size)
    use_int64_index = not can_use_int32_index(inp)
    dtype = inp.dtype
    device = inp.device
    mid = torch.empty(mid_size, dtype=dtype, device=device)
    if keepdim:
        out_shape = [1] * inp.dim()
        out = torch.empty(out_shape, dtype=dtype, device=device)
    else:
        out = torch.empty([], dtype=dtype, device=device)
    max_kernel_1[mid_size,](inp, mid, M, block_size, INT64_INDEX=use_int64_index)
    max_kernel_2[1,](mid, out, mid_size, block_mid)
    return out
def max_dim(inp: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
    """
    Computes the maximum value of 'inp' along a specified 'dim'.
    This function internally performs a reduction along 'dim' via
    the 'max_kernel'. The output is then reshaped based on 'keepdim'.
    """
    ndims = inp.ndim
    if dim < 0:
        dim += ndims
    assert 0 <= dim < ndims, f'Invalid dim={dim} for tensor of rank {ndims}'
    shape = list(inp.shape)
    dims = [dim]
    compressed = dim_compress(inp, dims)
    N = shape[dim]
    M = compressed.numel() // N
    out_shape = shape.copy()
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    use_int64_index = not can_use_int32_index(compressed)

    def grid(meta):
        return ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
    max_kernel[grid](compressed, out, M, N, INT64_INDEX=use_int64_index)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
def grid(meta):
    return ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
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
