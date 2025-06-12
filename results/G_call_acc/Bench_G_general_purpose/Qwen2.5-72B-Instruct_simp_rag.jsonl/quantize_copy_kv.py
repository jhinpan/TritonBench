import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(K, Dest_loc, Out, Out_scale, stride_k_bs, stride_k_h, stride_k_g, stride_k_d, stride_o_bs, stride_o_h, stride_o_g, stride_o_d, stride_os_bs, stride_os_h, stride_os_g, group_size, BLOCK_GROUP_NUM: tl.constexpr, BLOCK_GROUP_DIM: tl.constexpr):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)
    dest_index = tl.load(Dest_loc + cur_index)
    src_data = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :], mask=offs_g[:, None] < group_size, other=0.0)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(tl.float16)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale)
    return
@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    quant_group_dim = 8
    assert head_dim % quant_group_dim == 0, 'error head dim, can not been supported to copy quant kv'
    grid = (seq_len, head_num)
    num_warps = 1
    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim
    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(Out.shape[0], Out.shape[1], group_size, group_dim)
    _fwd_kernel_destindex_copy_quantize_kv[grid](K, DestLoc, Out, Out_scale, K.stride(0), K.stride(1), K.stride(2), K.stride(3), Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3), Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2), group_size, BLOCK_GROUP_NUM=triton.next_power_of_2(group_size), BLOCK_GROUP_DIM=group_dim, num_warps=num_warps, num_stages=1)
    return
##################################################################################################################################################



def test_destindex_copy_quantize_kv():
    B, N_CTX, H, D = 16, 512, 8, 64
    
    # Test case 1
    src1 = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc1 = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest1 = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest1 = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()
    destindex_copy_quantize_kv(src1, dest_loc1, value_dest1, scale_dest1)

    # Test case 2 - Randomized location indices
    dest_loc2 = torch.randint(0, B * N_CTX, (B * N_CTX,), dtype=torch.int32).cuda()
    src2 = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    value_dest2 = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest2 = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()
    destindex_copy_quantize_kv(src2, dest_loc2, value_dest2, scale_dest2)

    # Test case 4 - Larger head dimension
    D4 = 256
    src3 = torch.randn((B * N_CTX, H, D4), dtype=torch.float16).cuda()
    dest_loc3 = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest3 = torch.randn((B * N_CTX, H, D4), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest3 = torch.randn((B * N_CTX, H, 1), dtype=torch.float16).cuda()
    destindex_copy_quantize_kv(src3, dest_loc3, value_dest3, scale_dest3)

    results = {
        "test_case_1": value_dest1,
        "test_case_2": value_dest2,
        "test_case_3": value_dest3,
    }

    return results

result_gold = test_destindex_copy_quantize_kv()
