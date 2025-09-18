"""GaussianDiffusion wraps denoising, diffusion, sampling, and loss computation
operators for diffusion models. We consider a variance preserving (VP) process
where the diffusion process is:

q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I),

where alpha_t^2 = 1 - sigma_t^2.
"""
import random
from typing import Dict, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass

import torch

from .schedules import karras_schedule, ve_to_vp, vp_to_ve
from .solvers import sample_ddim, sample_euler, sample_euler_ancestral, sample_dpm_2, sample_dpm_2_ancestral, sample_res_2s, sample_res_3s, sample_res_2m, sample_res_3m
from .compile_utils import apply_torch_compile
from .autocast_utils import autocast_context

__all__ = ['GaussianDiffusion', 'SampleConfig']


@dataclass
class SampleConfig:
    """Configuration for sampling parameters."""
    model_kwargs: Optional[Dict[str, Any]] = None
    guide_scale: Optional[float] = None
    guide_rescale: Optional[float] = None
    clamp: Optional[float] = None
    percentile: Optional[float] = None
    solver: str = 'ddim'
    steps: Union[int, torch.LongTensor] = 20
    t_max: Optional[int] = None
    t_min: Optional[int] = None
    discretization: Optional[str] = None
    discard_penultimate_step: Optional[bool] = None
    return_intermediate: Optional[str] = None
    show_progress: bool = False
    seed: int = -1
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.kwargs is None:
            self.kwargs = {}


def _i(tensor, t, x):
    """Index tensor using t and format the output according to x."""
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).float().to(x.device)


class GaussianDiffusion(object):
    """GaussianDiffusion wraps denoising, diffusion, sampling, and loss computation
    operators for diffusion models. We consider a variance preserving (VP) process
    where the diffusion process is:

    q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I),

    where alpha_t^2 = 1 - sigma_t^2.
    """

    def __init__(self, sigmas: torch.Tensor, prediction_type: str = 'eps', use_mixed_precision: bool = True):
        """Initialize the diffusion model.

        Args:
            sigmas: Noise coefficients for each timestep.
            prediction_type: Type of prediction ('x0', 'eps', 'v').
            use_mixed_precision: Whether to use mixed precision.
        """
        assert prediction_type in {'x0', 'eps', 'v'}
        self.sigmas: torch.Tensor = sigmas  # noise coefficients
        self.alphas: torch.Tensor = torch.sqrt(1 - sigmas**2)  # signal coefficients
        self.num_timesteps: int = len(sigmas)
        self.prediction_type: str = prediction_type
        self.use_mixed_precision: bool = use_mixed_precision
        
        # Apply torch.compile to key computational methods if available
        self.denoise = apply_torch_compile(self._denoise)

    def diffuse(self, x0, t, noise=None):
        """Add Gaussian noise to signal x0 according to:

        q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I).
        """
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt

    def _denoise(self,
                xt,
                t,
                s,
                model,
                model_kwargs={},
                guide_scale=None,
                guide_rescale=None,
                clamp=None,
                percentile=None):
        """Apply one step of denoising from the posterior distribution q(x_s |
        x_t, x0).

        Since x0 is not available, estimate the denoising results using the
        learned distribution p(x_s | x_t, x_hat_0 == f(x_t)).
        """
        s = t - 1 if s is None else s

        # Use mixed precision where appropriate
        with autocast_context(enabled=self.use_mixed_precision, device_type=str(xt.device).split(':')[0]):
            # hyperparams
            sigmas = _i(self.sigmas, t, xt)
            alphas = _i(self.alphas, t, xt)
            alphas_s = _i(self.alphas, s.clamp(0), xt)
            alphas_s[s < 0] = 1.
            sigmas_s = torch.sqrt(1 - alphas_s**2)

            # precompute variables
            betas = 1 - (alphas / alphas_s)**2
            coef1 = betas * alphas_s / sigmas**2
            coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
            var = betas * (sigmas_s / sigmas)**2
            log_var = torch.log(var).clamp_(-20, 20)

        # prediction
        out = self._get_model_output(xt, t, model, model_kwargs, guide_scale, guide_rescale)

        # compute x0
        x0 = self._compute_x0(out, xt, sigmas, alphas)

        # restrict the range of x0
        x0 = self._restrict_x0_range(x0, percentile, clamp)

        eps = (xt - alphas * x0) / sigmas

        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps

    def _validate_sample_inputs(self, steps, t_max, t_min, discretization, discard_penultimate_step, return_intermediate, solver):
        """Validate inputs for the sample method."""
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')
        assert solver in ('ddim', 'euler', 'euler_ancestral', 'dpm_2', 'dpm_2_ancestral')

    def _setup_sample_options(self, steps, solver, discretization, discard_penultimate_step, seed):
        """Set up sampling options with defaults."""
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2', 'dpm2_ancestral', 'dpmpp_2m_sde', 'dpm2_karras',
                'dpm2_ancestral_karras', 'dpmpp_2m_sde_karras') else False
        return schedule, discretization, seed, discard_penultimate_step

    def _get_solver_fn(self, solver):
        """Get the solver function based on solver name."""
        solver_fns = {
            'ddim': sample_ddim,
            'euler': sample_euler,
            'euler_ancestral': sample_euler_ancestral,
            'dpm_2': sample_dpm_2,
            'dpm_2_ancestral': sample_dpm_2_ancestral,
            'res_2s': sample_res_2s,
            'res_3s': sample_res_3s,
            'res_2m': sample_res_2m,
            'res_3m': sample_res_3m,
        }
        return solver_fns[solver]

    def _create_model_fn(self, model, model_kwargs, guide_scale, guide_rescale, clamp, percentile, return_intermediate):
        """Create the model function for denoising."""
        intermediates = []

        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).expand(len(xt)).round().long()
            x0 = self.denoise(xt, t, None, model, model_kwargs, guide_scale,
                              guide_rescale, clamp, percentile)[-2]

            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                intermediates.append(x0)
            return x0

        return model_fn, intermediates

    def _get_timesteps(self, steps, t_max, t_min, discretization, device):
        """Get discretized timesteps."""
        if isinstance(steps, int):
            steps += 1 if discretization == 'leading' else 0  # Wait, let's check original
            # Original has steps += 1 if discard_penultimate_step else 0, but that's later
            # For discretization, it's only if leading? No, original has steps += 1 if discard_penultimate_step else 0 before discretization
            # I need to adjust
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)  # type: ignore
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)  # type: ignore
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))  # type: ignore
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented')
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(steps,
                                dtype=torch.float32,
                                device=device)
        return steps

    def _get_sigmas(self, steps, schedule, sigmas, discard_penultimate_step):
        """Get sigma schedule for sampling."""
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])  # type: ignore
        if schedule == 'karras':
            if torch.isclose(sigmas[0], torch.tensor(1.0), atol=1e-6):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < 1.].max().item(),
                    rho=7.).to(sigmas)  # type: ignore
                sigmas = torch.cat(
                    [sigmas.new_ones([1]), sigmas,
                     sigmas.new_zeros([1])])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.).to(sigmas)  # type: ignore
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    @torch.no_grad()
    def sample(self, noise: torch.Tensor, model: Callable, config: SampleConfig) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Sample from the diffusion model.

        Args:
            noise: Initial noise tensor.
            model: Denoising model function.
            config: Sampling configuration.

        Returns:
            Sampled tensor or tuple of (tensor, intermediates) if return_intermediate is set.
        """
        # Validate inputs
        self._validate_sample_inputs(config.steps, config.t_max, config.t_min, config.discretization,
                                     config.discard_penultimate_step, config.return_intermediate, config.solver)

        # Setup options
        schedule, discretization, _, discard_penultimate_step = self._setup_sample_options(
            config.steps, config.solver, config.discretization, config.discard_penultimate_step, config.seed)

        # Create model function
        model_fn, intermediates = self._create_model_fn(
            model, config.model_kwargs, config.guide_scale, config.guide_rescale,
            config.clamp, config.percentile, config.return_intermediate)

        # Get timesteps
        steps = config.steps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
        timesteps = self._get_timesteps(steps, config.t_max, config.t_min, discretization, noise.device)

        # Get sigmas
        sigmas = self._t_to_sigma(timesteps)
        sigmas = self._get_sigmas(timesteps, schedule, sigmas, discard_penultimate_step)

        # Get solver
        solver_fn = self._get_solver_fn(config.solver)

        # Sampling
        x0 = solver_fn(noise, model_fn, sigmas, show_progress=config.show_progress, **(config.kwargs or {}))
        return (x0, intermediates) if config.return_intermediate is not None else x0

    def _get_model_output(self, xt, t, model, model_kwargs, guide_scale, guide_rescale):
        """Get model output with optional guidance."""
        if guide_scale is None:
            assert isinstance(model_kwargs, dict)
            out = model(xt, t=t, **model_kwargs)
        else:
            # classifier-free guidance (arXiv:2207.12598)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, t=t, **model_kwargs[0])
            if guide_scale is not None and torch.isclose(torch.tensor(guide_scale), torch.tensor(1.0), atol=1e-6):
                out = y_out
            else:
                u_out = model(xt, t=t, **model_kwargs[1])
                out = u_out + guide_scale * (y_out - u_out)

                # rescale the output according to arXiv:2305.08891
                if guide_rescale is not None:
                    assert guide_rescale >= 0 and guide_rescale <= 1
                    ratio = (y_out.flatten(1).std(dim=1) /
                             (out.flatten(1).std(dim=1) +
                              1e-12)).view((-1, ) + (1, ) * (y_out.ndim - 1))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0
        return out

    def _compute_x0(self, out, xt, sigmas, alphas):
        """Compute x0 from model output."""
        if self.prediction_type == 'x0':
            x0 = out
        elif self.prediction_type == 'eps':
            x0 = (xt - sigmas * out) / alphas
        elif self.prediction_type == 'v':
            x0 = alphas * xt - sigmas * out
        else:
            raise NotImplementedError(
                f'prediction_type {self.prediction_type} not implemented')
        return x0

    def _restrict_x0_range(self, x0, percentile, clamp):
        """Restrict the range of x0."""
        if percentile is not None:
            # NOTE: percentile should only be used when data is within range [-1, 1]
            assert percentile > 0 and percentile <= 1
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1, ) + (1, ) * (x0.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return x0

    # Backward compatibility - keep the old signature
    def sample_legacy(self,
                      noise,
                      model,
                      model_kwargs={},
                      guide_scale=None,
                      guide_rescale=None,
                      clamp=None,
                      percentile=None,
                      solver='ddim',
                      steps=20,
                      t_max=None,
                      t_min=None,
                      discretization=None,
                      discard_penultimate_step=None,
                      return_intermediate=None,
                      show_progress=False,
                      seed=-1,
                      **kwargs):
        """Legacy sample method for backward compatibility."""
        config = SampleConfig(
            model_kwargs=model_kwargs,
            guide_scale=guide_scale,
            guide_rescale=guide_rescale,
            clamp=clamp,
            percentile=percentile,
            solver=solver,
            steps=steps,
            t_max=t_max,
            t_min=t_min,
            discretization=discretization,
            discard_penultimate_step=discard_penultimate_step,
            return_intermediate=return_intermediate,
            show_progress=show_progress,
            seed=seed,
            **kwargs
        )
        return self.sample(noise, model, config)

    def _t_to_sigma(self, t):
        """Convert time steps (float) to sigmas by interpolating in the log-VE-
        sigma space."""
        t = t.float()
        i, j, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = vp_to_ve(self.sigmas).log().to(t)  # type: ignore
        log_sigma = (1 - w) * log_sigmas[i] + w * log_sigmas[j]
        log_sigma[torch.isnan(log_sigma)
                  | torch.isinf(log_sigma)] = float('inf')
        return ve_to_vp(log_sigma.exp())

    def _sigma_to_t(self, sigma):
        """Convert sigma to time step (float) by interpolating in the log-VE-
        sigma space."""
        if torch.isclose(sigma, torch.tensor(1.0), atol=1e-6):
            t = torch.full_like(sigma, self.num_timesteps - 1)
        else:
            log_sigmas = vp_to_ve(self.sigmas).log().to(sigma)  # type: ignore
            log_sigma = vp_to_ve(sigma).log()  # type: ignore

            # interpolation
            i = torch.where((log_sigma - log_sigmas).ge(0))[0][-1].clamp(
                max=self.num_timesteps - 2).unsqueeze(0)
            j = i + 1
            w = (log_sigmas[i] - log_sigma) / (log_sigmas[i] - log_sigmas[j])
            w = w.clamp(0, 1)
            t = (1 - w) * i + w * j
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t
