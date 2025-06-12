import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(Logits, B_Start_Loc, B_Seqlen, Prob_Out, BLOCK_SIZE: tl.constexpr):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    start_loc = tl.load(B_Start_Loc + batch_id)
    seq_len = tl.load(B_Seqlen + batch_id)
    start_idx = start_loc + head_id * seq_len
    logits_ptr = Logits + start_idx
    logits = tl.load(logits_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < seq_len, other=-float('inf'))
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp_logits
    prob_out_ptr = Prob_Out + start_idx
    tl.store(prob_out_ptr + tl.arange(0, BLOCK_SIZE), probs, mask=tl.arange(0, BLOCK_SIZE) < seq_len)
@torch.no_grad()
def token_softmax_fwd(logits, b_start_loc, b_seqlen, prob_out, max_input_len):
    BLOCK_SIZE = min(128, max_input_len)
    num_warps = 4 if BLOCK_SIZE > 64 else 2
    num_batches = b_start_loc.shape[0]
    num_heads = logits.shape[1] // max_input_len
    grid = (num_batches, num_heads)
    _fwd_kernel_token_softmax[grid](logits, b_start_loc, b_seqlen, prob_out, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
##################################################################################################################################################



import torch

# Define the test function
def test_token_softmax_fwd():
    results = {}

    # Test case 1: Small input size
    batch_size = 2
    head_num = 2
    max_input_len = 8

    # Create random input tensors
    Logics = torch.randn((head_num, batch_size * max_input_len), dtype=torch.float32, device='cuda')
    B_Start_Loc = torch.tensor([0, max_input_len], dtype=torch.int32, device='cuda')
    B_Seqlen = torch.tensor([max_input_len, max_input_len], dtype=torch.int32, device='cuda')
    Prob_Out = torch.empty_like(Logics)

    # Call the Triton softmax function
    token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len)

    # Store the output
    results['test_case_1'] = Prob_Out.clone()

    # Test case 2: Larger input size
    batch_size = 1
    head_num = 1
    max_input_len = 16

    # Create random input tensors
    Logics = torch.randn((head_num, batch_size * max_input_len), dtype=torch.float32, device='cuda')
    B_Start_Loc = torch.tensor([0], dtype=torch.int32, device='cuda')
    B_Seqlen = torch.tensor([max_input_len], dtype=torch.int32, device='cuda')
    Prob_Out = torch.empty_like(Logics)

    # Call the Triton softmax function
    token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len)

    # Store the output
    results['test_case_2'] = Prob_Out.clone()

    return results

# Run the test function
result_gold = test_token_softmax_fwd()
