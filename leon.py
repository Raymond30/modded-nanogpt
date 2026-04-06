import torch


    
class leon(torch.optim.Optimizer):
    #  function initializes the optimizer with the model's parameters and some hyperparameters from args.
    def __init__(self, 
                params,
                learning_rate = 1e-2,
                momentum = 0.9, 
                weight_decay = 0.1,
                beta2 = 0.8,
                eps = 1e-10, 
                ns_steps = 5,
                leon_scale = 0.2,
                whiten_method = 'newton_schulz',
                precond_scale = 1.,
                ):
        defaults = {'beta2': beta2, 'lr': learning_rate, 'eps': eps, 
                    'momentum': momentum, 'weight_decay': weight_decay,
                    'leon_scale': leon_scale, 'precond_scale': precond_scale, 'ns_steps': ns_steps, 'whiten_method': whiten_method
                    }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            eps = group['eps']
            momentum = group['momentum']
            beta2 = group['beta2']
            wd = group['weight_decay']
            precond_scale = group['precond_scale']
            whiten_method = group['whiten_method']
            if whiten_method == 'newton_schulz':
                zeropower_function = augmented_zeropower_via_newtonschulz5
            elif whiten_method == 'polar_express':
                zeropower_function = augmented_zeropower_via_PolarExpress
            else:
                raise ValueError(f'Unsupported whitening method: {whiten_method}')
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                grad = p.grad
                assert grad.ndim >= 2, f'1D parameter and embedding should be handled by AdamW'
                if grad.ndim == 2:
                    # Initialize state if needed
                    if len(state) == 0:
                        dim = grad.size(0) if grad.size(0) < grad.size(1) else grad.size(1)
                        state['momentum_buffer'] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                        state['second_momentum_buffer'] = torch.zeros((dim, dim), device=grad.device, dtype=grad.dtype)
                        state['step'] = 0
                    
                    state['step'] += 1

                    # Update momentum buffer
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, 1 - momentum)
                    gram_tmp = torch.mm(grad, grad.t()) if grad.size(0) < grad.size(1) else torch.mm(grad.t(), grad) 
                    second_momentum_buf = state['second_momentum_buffer']
                    second_momentum_buf.lerp_(gram_tmp, 1 - beta2)
                    del gram_tmp

                    # Bias correction 
                    # Does not seem to make much difference
                    # bc1 = 1 - momentum ** state['step']
                    # bc2 = 1 - beta2 ** state['step']
                    # buf_corrected = buf / bc1
                    # second_momentum_corrected = second_momentum_buf / bc2
                    
                    G_input = buf if grad.size(0) < grad.size(1) else buf.t()
                    A_input = precond_scale * second_momentum_buf
                    
                    try:
                        update_tmp = zeropower_function(G_input, A_input, steps=group['ns_steps'], eps=eps)
                        if torch.isnan(update_tmp).any() or torch.isinf(update_tmp).any():
                            torch.save({'G_input': G_input, 'A_input': A_input, 'steps': group['ns_steps'], 'eps': eps}, 'nan_inputs.pt')
                            print("NaN or Inf detected in zeropower_function output! Inputs saved to nan_inputs.pt")
                            raise ValueError("NaN or Inf detected in zeropower_function output!")
                    except Exception as e:
                        torch.save({'G_input': G_input, 'A_input': A_input, 'steps': group['ns_steps'], 'eps': eps}, 'nan_inputs.pt')
                        print("Exception in zeropower_function! Inputs saved to nan_inputs.pt")
                        raise e
                    
                    update = update_tmp if grad.size(0) < grad.size(1) else update_tmp.t()
                    
                    # Update inverse square root of preconditioner
                    # momentum_gram_tmp = torch.mm(state['momentum_buffer'], state['momentum_buffer'].t()) if grad.size(0) < grad.size(1) else torch.mm(state['momentum_buffer'].t(), state['momentum_buffer'])
                    # inverse_precond = self._matrix_power(precond_scale * state['precond'] + momentum_gram_tmp, eps = eps, N = self.matalg_steps)
                    # if torch.isnan(inverse_precond).any() or torch.isinf(inverse_precond).any():
                    #     print(f'[WARNING] Step {state["step"]}: NaN/Inf detected in parameter. Skipping inverse_precond update.')
                    #     inverse_precond = torch.eye(state['precond'].size(0), device=state['precond'].device, dtype=state['precond'].dtype)

                    # assert not torch.allclose(inverse_precond, torch.zeros_like(inverse_precond)), f'Inverse precond is all zeros'
                    # # Apply preconditioning
                    # update = inverse_precond @ state['momentum_buffer'] if grad.size(0) < grad.size(1) else state['momentum_buffer'] @ inverse_precond
                    
                    

                    # Apply Muon-type RMS norm scaling
                    update = (group['leon_scale'] * math.sqrt(max(p.size(-2), p.size(-1))) * update) 
                    # update.mul_((group['leon_scale'] * math.sqrt(grad.size(0) * grad.size(1))) / (torch.linalg.matrix_norm(update, ord='fro').item()+eps))
                    # state['step'] += 1
                else:
                    raise ValueError('Missing Training Param')
                
                if wd > 0: 
                    p.mul_(1 - group['lr'] * wd)
                p.add_(update, alpha=-group['lr'])
        return



        def augmented_zeropower_via_newtonschulz5(G: torch.Tensor, L: torch.Tensor, steps: int, eps: float = 1e-10) -> torch.Tensor:
    assert G.ndim >= 2
    m = G.shape[-2]
    assert L.shape[-2:] == (m, m)
    assert L.shape[:-2] == G.shape[:-2]
    return _augmented_zeropower_via_newtonschulz5_compiled(G, L, steps, eps)

@torch.compile
def _augmented_zeropower_via_newtonschulz5_compiled(G: torch.Tensor, L: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of the augmented matrix [G, \sqrt{L}] and then takes the leading block corresponding to G. Euivalently, this computes (G * G^top + L)^{-0.5} G. G and L must be aligned along the first dimension.
    """
    a, b, c = (3.4445, -4.7750,  2.0315)
    # (2,-1.5,0.5)
    # X = G.bfloat16()
    X = G
    A = L
    A = 0.5 * (A + A.transpose(-2, -1)) # ensure symmetry to prevent numerical issues
    A.diagonal(dim1=-2, dim2=-1).add_(eps)

    # Scale so that tr(GG^T + L) == 1  => lambda_max <= 1 for PSD A
    tr = G.float().pow(2).sum(dim=(-2, -1), keepdim=True) \
         + A.float().diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    tr = tr.clamp_min(1e-12)

    inv_sqrt_tr = tr.rsqrt()                      # fp32
    inv_tr = inv_sqrt_tr.square()                 # fp32

    X = X * inv_sqrt_tr.to(X.dtype)
    A = (A.float() * inv_tr).contiguous()
    
    # Build initial Gram matrix once: A0 = (GG^T + L) / tr
    A = X.float() @ X.float().mT + A

    # Perform the NS iterations
    for _ in range(steps):
        A = 0.5 * (A + A.mT)  # keep it symmetric
        B = b * A + c * (A @ A) # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + (B.to(X.dtype) @ X)
        # A <- (aI + B) A (aI + B), computed sequentially for stability
        # BA = B @ A
        # A = (a * a) * A + a * (BA + BA.mT) + BA @ B
        A = a * A + B @ A
        A = a* A + A @ B
    
    return X


def augmented_zeropower_via_PolarExpress(G: torch.Tensor, L: torch.Tensor, steps: int, eps: float = 1e-10) -> torch.Tensor:
    assert G.ndim >= 2
    m = G.shape[-2]
    assert L.shape[-2:] == (m, m)
    assert L.shape[:-2] == G.shape[:-2]
    return _augmented_zeropower_via_PolarExpress_compiled(G, L, steps, eps)

@torch.compile
def _augmented_zeropower_via_PolarExpress_compiled(G: torch.Tensor, L: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of the augmented matrix [G, \sqrt{L}] and then takes the leading block corresponding to G. Euivalently, this computes (G * G^top + L)^{-0.5} G. G and L must be aligned along the first dimension.
    """
    orig_dtype = G.dtype
    coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933), 
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601), 
    (3.9486908534822946 , -2.908902115962949, 0.5518191394370137), 
    (3.3184196573706015 , -2.488488024314874, 0.51004894012372), 
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673), 
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835), 
    (1.8750014808534479 , -1.2500016453999487, 0.3750001645474248), 
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically 
    ]
    # safety factor for numerical stability (but exclude last polynomial) 
    coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list [:-1]] + [coeffs_list [-1]]
    
    # Use float32 for stability
    X = G # Keep G in original dtype (likely bfloat16)
    A = L.float()
    A = 0.5 * (A + A.transpose(-2, -1)) # ensure symmetry to prevent numerical issues


    # Scale so that tr(GG^T + L) == 1  => lambda_max <= 1 for PSD A
    tr = X.float().pow(2).sum(dim=(-2, -1), keepdim=True) \
         + A.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    tr = tr.clamp_min(1e-12) * (1.1**2)

    inv_sqrt_tr = tr.rsqrt()                      # fp32
    inv_tr = inv_sqrt_tr.square()                 # fp32

    X = X * inv_sqrt_tr.to(X.dtype)
    A = (A * inv_tr).contiguous()
    
    # Build initial Gram matrix once: A0 = (GG^T + L) / tr
    A = (X.float() @ X.float().mT) + A
    A.diagonal(dim1=-2, dim2=-1).add_(eps) # add epsilon for stability

    # Adaptive Cholesky verification for PSD
    # with torch.no_grad():
    #     # Prevent initial eps being exactly zero, which otherwise prevents iterative growth
    #     eps = max(float(eps), 1e-12)
    #     while True:
    #         _, info = torch.linalg.cholesky_ex(A)
    #         # Use .all() if info is a batched tensor, or just info == 0 for scalars
    #         if (info == 0).all():
    #             break
    #         if eps > 1e-1 or torch.isnan(A).any() or torch.isinf(A).any():
    #             # Break to prevent infinite loop on NaN, Inf or terminally broken matrices.
    #             # Allow subsequent code or the optimizer's try/except blocks to catch the NaN!
    #             break
    #         eps *= 10
    #         A.diagonal(dim1=-2, dim2=-1).add_(eps)

    # Perform the NS iterations
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list))) 
    for a, b, c in hs:
        A = 0.5 * (A + A.mT)  # keep it symmetric
        B = b * A + c * (A @ A) # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + (B.to(X.dtype) @ X)
        # A <- (aI + B) A (aI + B), computed sequentially for stability
        # BA = B @ A
        # A = (a * a) * A + a * (BA + BA.mT) + BA @ B
        A = a * A + B @ A
        A = a* A + A @ B
    
    return X.to(orig_dtype)