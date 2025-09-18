import torch
from tqdm.auto import trange
import math

# Try to import ComfyUI samplers, but handle gracefully if not available
COMFY_NOT_AVAILABLE_MSG = "ComfyUI k-diffusion not available"
try:
    from comfy.k_diffusion.sampling import sample_euler as comfy_sample_euler  # type: ignore
    from comfy.k_diffusion.sampling import sample_euler_ancestral as comfy_sample_euler_ancestral  # type: ignore
    from comfy.k_diffusion.sampling import sample_dpm_2 as comfy_sample_dpm_2  # type: ignore
    from comfy.k_diffusion.sampling import sample_dpm_2_ancestral as comfy_sample_dpm_2_ancestral  # type: ignore
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    # Define dummy functions if ComfyUI is not available
    def comfy_sample_euler(*args, **kwargs):
        raise NotImplementedError(COMFY_NOT_AVAILABLE_MSG)
    def comfy_sample_euler_ancestral(*args, **kwargs):
        raise NotImplementedError(COMFY_NOT_AVAILABLE_MSG)
    def comfy_sample_dpm_2(*args, **kwargs):
        raise NotImplementedError(COMFY_NOT_AVAILABLE_MSG)
    def comfy_sample_dpm_2_ancestral(*args, **kwargs):
        raise NotImplementedError(COMFY_NOT_AVAILABLE_MSG)

__all__ = ['sample_ddim', 'sample_euler', 'sample_euler_ancestral', 'sample_dpm_2', 'sample_dpm_2_ancestral',
           'sample_res_2s', 'sample_res_3s', 'sample_res_2m', 'sample_res_3m']


# ------------------------ RK Sampler Infrastructure ------------------------#

def phi(h, k):
    """Phi function for exponential integration (scalar version)."""
    if k == 0:
        return torch.exp(h)
    elif k == 1:
        return (torch.exp(h) - 1) / h if h != 0 else 1.0
    else:
        # Recursive calculation for higher orders
        result = torch.exp(h)
        for i in range(1, k + 1):
            result = result - (h ** i) / math.factorial(i)
        return result / (h ** k) if h != 0 else 1.0 / math.factorial(k)


def get_rk_coefficients(rk_type, h, c2=0.5, c3=1.0):
    """Get Runge-Kutta coefficients for different RK methods."""
    if rk_type == "res_2s":
        # RES2 from RES4LYF
        a21 = c2 * phi(h * c2, 1)
        b1 = phi(h, 1) - phi(h, 2) / c2
        b2 = phi(h, 2) / c2
        
        return {
            'A': [[0, 0], [a21, 0]],
            'b': [b1, b2],
            'c': [0, c2]
        }
    
    elif rk_type == "res_3s":
        # RES3 from RES4LYF
        gamma = (3 * c3**3 - 2 * c3) / (c2 * (2 - 3 * c2))
        
        a21 = c2 * phi(h * c2, 1)
        a31 = c3 * phi(h * c3, 1) - gamma * c2 * phi(h * c3, 2)
        a32 = gamma * c2 * phi(h * c3, 2)
        
        b1 = phi(h, 1) - gamma * phi(h, 2) - (1 / (gamma * c2 + c3)) * phi(h, 2)
        b2 = gamma * (1 / (gamma * c2 + c3)) * phi(h, 2)
        b3 = (1 / (gamma * c2 + c3)) * phi(h, 2)
        
        return {
            'A': [[0, 0, 0], [a21, 0, 0], [a31, a32, 0]],
            'b': [b1, b2, b3],
            'c': [0, c2, c3]
        }
    
    else:
        raise ValueError(f"Unknown RK type: {rk_type}")


@torch.no_grad()
def sample_rk_base(noise, model, sigmas, rk_type='res_2s', show_progress=True):
    """Base RK sampler implementation."""
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Calculate step size h
        h = -torch.log(sigma_next / sigma) if sigma_next > 0 else torch.tensor(0.0)
        
        # Get RK coefficients
        coeffs = get_rk_coefficients(rk_type, h)
        A, b, c = coeffs['A'], coeffs['b'], coeffs['c']
        
        # Number of stages
        s = len(c)
        
        # Store intermediate values
        k = []
        
        for stage in range(s):
            # Calculate intermediate x
            x_stage = x
            for j in range(stage):
                x_stage = x_stage + A[stage][j] * k[j]
            
            # Scale by sigma for model input
            sigma_stage = sigma * torch.exp(h * c[stage])
            
            # Model prediction
            denoised = model(x_stage, sigma_stage)
            
            # Calculate k_stage (noise prediction)
            k_stage = (x_stage - sigma_stage * denoised) / sigma_stage
            k.append(k_stage)
        
        # Combine stages
        dx = torch.zeros_like(x)
        for stage in range(s):
            dx = dx + b[stage] * k[stage]
        
        # Update x
        x = x + h * dx
        
        # Add noise if not the last step
        if sigma_next > 0:
            x = x + sigma_next * torch.randn_like(x)
    
    return x


# ------------------------ RK Sampler Functions ------------------------#

@torch.no_grad()
def sample_res_2s(noise, model, sigmas, show_progress=True):
    """RES2 sampler from RES4LYF - 2-stage exponential RK method."""
    return sample_rk_base(noise, model, sigmas, rk_type='res_2s', show_progress=show_progress)


@torch.no_grad()
def sample_res_3s(noise, model, sigmas, show_progress=True):
    """RES3 sampler from RES4LYF - 3-stage exponential RK method."""
    return sample_rk_base(noise, model, sigmas, rk_type='res_3s', show_progress=show_progress)


@torch.no_grad()
def sample_res_2m(noise, model, sigmas, show_progress=True):
    """RES2M sampler from RES4LYF - 2-stage multistep exponential RK method."""
    # For now, fall back to single-step version
    return sample_rk_base(noise, model, sigmas, rk_type='res_2s', show_progress=show_progress)


@torch.no_grad()
def sample_res_3m(noise, model, sigmas, show_progress=True):
    """RES3M sampler from RES4LYF - 3-stage multistep exponential RK method."""
    # For now, fall back to single-step version
    return sample_rk_base(noise, model, sigmas, rk_type='res_3s', show_progress=show_progress)


@torch.no_grad()
def sample_ddim(noise, model, sigmas, eta=0., show_progress=True):
    """DDIM solver steps."""
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])
        noise_factor = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2 *
                              (1 - (1 - sigmas[i] ** 2) /
                               (1 - sigmas[i + 1] ** 2)))
        d = (x - (1 - sigmas[i] ** 2) ** 0.5 * denoised) / sigmas[i]
        x = (1 - sigmas[i + 1] ** 2) ** 0.5 * denoised + \
            (sigmas[i + 1] ** 2 - noise_factor ** 2) ** 0.5 * d
        if sigmas[i + 1] > 0:
            x += noise_factor * torch.randn_like(x)
    return x


@torch.no_grad()
def sample_euler(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_euler(model, noise, sigmas, s_churn=0.75)

    return noise


@torch.no_grad()
def sample_euler_ancestral(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_euler_ancestral(model, noise, sigmas, s_noise=.97, eta=1.8)

    return noise


@torch.no_grad()
def sample_dpm_2(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_dpm_2(model, noise, sigmas, s_churn=0.75)

    return noise


@torch.no_grad()
def sample_dpm_2_ancestral(noise, model, sigmas, show_progress=True):
    noise = comfy_sample_dpm_2_ancestral(model, noise, sigmas, s_noise=.97, eta=1.8)

    return noise
