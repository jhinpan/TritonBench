import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    length = tl.load(old_a_len + pid)
    old_start = tl.load(old_a_start + pid)
    new_start = tl.load(new_a_start + pid)
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, length, BLOCK_SIZE):
        old_offsets = block_start + offsets
        new_offsets = block_start + offsets
        mask = old_offsets < length
        src_ptrs = old_a_location + old_start + old_offsets
        block = tl.load(src_ptrs, mask=mask)
        dst_ptrs = new_a_location + new_start + new_offsets
        tl.store(dst_ptrs, block, mask=mask)
def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_location):
    """
    Launch the Triton kernel for variable length copying
    
    Args:
        old_a_start: Starting indices for source segments
        old_a_len: Lengths of segments
        old_location: Source data array
        new_a_start: Starting indices for destination segments
        new_location: Destination data array
    """
    BLOCK_SIZE = 128
    num_segments = len(old_a_start)
    grid = (num_segments,)
    var_len_copy_kernel_triton[grid](old_a_start, old_a_len, old_location, new_a_start, new_location, BLOCK_SIZE)
##################################################################################################################################################



import torch

def test_launch_var_len_copy_kernel_triton():
    # Define test input data
    num_arrays = 3
    BLOCK_SIZE = 256

    # Old array start indices
    old_a_start = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')

    # Lengths of each array
    old_a_len = torch.tensor([50, 150, 200], dtype=torch.int32, device='cuda')

    # Flattened old array data
    old_a_location = torch.arange(500, dtype=torch.float32, device='cuda')

    # New array start indices
    new_a_start = torch.tensor([0, 60, 260], dtype=torch.int32, device='cuda')

    # Target flattened array for copying
    new_a_location = torch.zeros(500, dtype=torch.float32, device='cuda')

    # Launch the Triton kernel
    launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location)

    # Store results in a dictionary
    results = {}
    for i in range(num_arrays):
        old_start = old_a_start[i].item()
        new_start = new_a_start[i].item()
        length = old_a_len[i].item()
        # Store the result of each test case
        results[f"test_case_{i+1}"] = torch.equal(
            old_a_location[old_start:old_start + length],
            new_a_location[new_start:new_start + length]
        )
    
    return results

result_gold = test_launch_var_len_copy_kernel_triton()
