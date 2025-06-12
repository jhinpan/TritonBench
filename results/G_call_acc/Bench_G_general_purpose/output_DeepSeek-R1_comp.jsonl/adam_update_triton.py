import torch
import triton
import triton.language as tl

@triton.jit
def update_fn_kernel(p_ptr, grad_ptr, exp_avg_ptr, n_elements, lr, wd, beta1, beta2, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    p = p * (1 - lr * wd)
    diff = exp_avg - grad
    step = beta1 * diff
    p = p - lr * step
    step_non_zero = step != 0.0
    sign_step = tl.where(step > 0, 1.0, -1.0)
    adjustment = lr * sign_step
    p = tl.where(step_non_zero, p - adjustment, p)
    exp_avg = beta2 * exp_avg + (1 - beta2) * grad
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
def update_fn(p: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, lr: float, wd: float, beta1: float, beta2: float, BLOCK_SIZE: int=1024):
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda, 'Inputs must be CUDA tensors'
    assert p.is_contiguous() and grad.is_contiguous() and exp_avg.is_contiguous(), 'Inputs must be contiguous'
    n_elements = p.numel()
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    update_fn_kernel[grid](p, grad, exp_avg, n_elements, lr, wd, beta1, beta2, BLOCK_SIZE=BLOCK_SIZE)
##################################################################################################################################################



import torch

def test_update_fn():
    # Initialize input tensors
    n_elements = 128
    p1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    grad1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    exp_avg1 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)

    n_elements = 1024
    p2 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    grad2 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    exp_avg2 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)

    # Hyperparameters
    lr = 0.01
    wd = 0.01
    beta1 = 0.9
    beta2 = 0.999

    # Call the update function for different configurations
    update_fn(p1, grad1, exp_avg1, lr, wd, beta1, beta2)
    update_fn(p2, grad2, exp_avg2, lr, wd, beta1, beta2)

    # Store results in a dictionary
    results = {
        "test_case_1": (p1.clone(), exp_avg1.clone()),
        "test_case_2": (p2.clone(), exp_avg2.clone())
    }

    return results

result_gold = test_update_fn()
