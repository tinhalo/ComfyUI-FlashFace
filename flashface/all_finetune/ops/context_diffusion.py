"""GaussianDiffusion wraps operators for denoising diffusion models, including
the diffusion and denoising processes, as well as the loss evaluation."""

import torch
import math
from typing import Dict, List, Optional, Tuple, Union, cast

from ldm.ops.diffusion import GaussianDiffusion


# Constants
EPSILON = 1e-12
MIN_LOG_VAR = -20
MAX_LOG_VAR = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CLASSIFIER_SCALE = 1.0
DEFAULT_STEP_TO_LAUNCH_FACE_GUIDANCE = 600
DEFAULT_SIMILARITY = 0.85
DEFAULT_LAMBDA_FEAT_BEFORE_REF_GUIDANCE = 0.85


def _i(tensor, t, x):
    """Index tensor using t and format the output according to x.

    Args:
        tensor: The tensor to index
        t: The indices to use
        x: Reference tensor for output shape

    Returns:
        Indexed tensor formatted to match x's dimensions
    """
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).float().to(x.device)


DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CLASSIFIER_SCALE = 1.0
DEFAULT_STEP_TO_LAUNCH_FACE_GUIDANCE = 600
DEFAULT_SIMILARITY = 0.85
DEFAULT_LAMBDA_FEAT_BEFORE_REF_GUIDANCE = 0.85

"""  """


class ContextGaussianDiffusion(GaussianDiffusion):
    """Gaussian diffusion with context guidance and classifier-free guidance support."""

    def __init__(self, sigmas, prediction_type="eps", use_mixed_precision=True):
        """Initialize a context-aware Gaussian diffusion model.

        Args:
            sigmas: Noise coefficients
            prediction_type: Type of prediction ('eps', 'x0', or 'v')
            use_mixed_precision: Whether to use automatic mixed precision
        """
        super().__init__(
            sigmas=sigmas,
            prediction_type=prediction_type,
            use_mixed_precision=use_mixed_precision,
        )
        # Initialize default values for guidance
        self.num_pairs = 4  # Default number of context pairs
        
        # The base class sets self.denoise to a compiled function, but we override it
        # Delete the instance attribute so our class method is found
        if hasattr(self, 'denoise'):
            delattr(self, 'denoise')

    def denoise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        model: Optional[torch.nn.Module] = None,
        model_kwargs: Optional[Union[Dict, List]] = None,
        guide_scale: Optional[float] = None,
        guide_rescale: Optional[float] = None,
        clamp: Optional[float] = None,
        percentile: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one step of denoising from the posterior distribution q(x_s | x_t, x0).

        Since x0 is not available, estimate the denoising results using the
        learned distribution p(x_s | x_t, x_hat_0 == f(x_t)).

        Args:
            xt: Input noisy tensor at timestep t
            t: Current timestep
            s: Target timestep (default: t-1)
            model: The diffusion model
            model_kwargs: Model keyword arguments
            guide_scale: Guidance scale for classifier-free guidance
            guide_rescale: Rescale factor for guided output
            clamp: Value to clamp x0 predictions
            percentile: Percentile for restricting x0 range

        Returns:
            Tuple of (mu, var, log_var, x0, eps)
        """
        if model is None:
            raise ValueError("Model cannot be None for denoising")

        if model_kwargs is None:
            model_kwargs = {}

        # Set target step if not provided
        s = t - 1 if s is None else s

        # Compute hyperparameters
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.0
        sigmas_s = torch.sqrt(1 - alphas_s**2)

        # Precompute variables for posterior distribution
        betas = 1 - (alphas / alphas_s) ** 2
        coef1 = betas * alphas_s / sigmas**2
        coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var = betas * (sigmas_s / sigmas) ** 2
        log_var = torch.log(var).clamp_(MIN_LOG_VAR, MAX_LOG_VAR)

        # Update progress if available
        if hasattr(self, "progress"):
            self.progress = 1 - float(t[0]) / 1000

        # Get classifier strength (default to 0 if not set)
        classifier = getattr(
            self, "classifier", DEFAULT_CLASSIFIER_SCALE
        )  # Handle face guidance based on time step
        if hasattr(model, "share_cache"):
            # Use getattr to safely access the dynamically added attribute
            share_cache = getattr(model, "share_cache", {})  # type: ignore[attr-defined]
            step_to_launch_face_guidance = cast(
                float,
                share_cache.get(  # type: ignore[attr-defined]
                    "step_to_launch_face_guidance", DEFAULT_STEP_TO_LAUNCH_FACE_GUIDANCE
                ),
            )

            # Adjust similarity based on current step
            if (t > step_to_launch_face_guidance).any():
                classifier = 0
                share_cache["similarity"] = cast(
                    float,
                    share_cache.get(  # type: ignore[assignment]
                        "lamda_feat_before_ref_guidance",
                        DEFAULT_LAMBDA_FEAT_BEFORE_REF_GUIDANCE,
                    ),
                )
            else:
                share_cache["similarity"] = cast(
                    float,
                    share_cache.get(  # type: ignore[assignment]
                        "ori_similarity", DEFAULT_SIMILARITY
                    ),
                )

        # Apply classifier-free guidance
        if classifier > 0:
            # Validate model_kwargs format
            if not isinstance(model_kwargs, list) or len(model_kwargs) != 2:
                raise ValueError(
                    "model_kwargs must be a list of length 2 when using classifier guidance"
                )

            # Prepare input for 3-way guidance (conditional with ref, conditional, non-conditional)
            cat_xt = xt.repeat(3, 1, 1, 1)
            conditional_embed = model_kwargs[0].get("context")
            non_conditional_embed = model_kwargs[1].get("context")

            if conditional_embed is None or non_conditional_embed is None:
                raise ValueError("Missing context embeddings in model_kwargs")

            text_embed = torch.cat(
                [conditional_embed, conditional_embed, non_conditional_embed], dim=0
            )

            # Get model output
            raw_out = model(cat_xt, t=t, context=text_embed)

            # Split the output
            y_out_with_ref, y_out, u_out = torch.split(
                raw_out, raw_out.size(0) // 3, dim=0
            )

            # Apply guidance
            if guide_scale is None:
                guide_scale = DEFAULT_GUIDANCE_SCALE

            out = (
                u_out
                + guide_scale * (y_out - u_out)
                + classifier * (y_out_with_ref - y_out)
            )

            # Apply rescaling if specified
            if guide_rescale is not None:
                if not (0 <= guide_rescale <= 1):
                    raise ValueError("guide_rescale must be between 0 and 1")

                ratio = (
                    y_out.flatten(1).to(torch.float32).std(dim=1)
                    / (out.flatten(1).to(torch.float32).std(dim=1) + EPSILON)
                ).view((-1,) + (1,) * (y_out.ndim - 1))
                out = out * (guide_rescale * ratio + (1 - guide_rescale) * 1.0)
        elif isinstance(model_kwargs, list) and len(model_kwargs) == 2:
            # Standard classifier-free guidance (conditional vs non-conditional)
            # Prepare input for 2-way guidance (conditional, non-conditional)
            cat_xt = xt.repeat(2, 1, 1, 1)
            conditional_embed = model_kwargs[0].get("context")
            non_conditional_embed = model_kwargs[1].get("context")

            if conditional_embed is None or non_conditional_embed is None:
                raise ValueError("Missing context embeddings in model_kwargs")

            text_embed = torch.cat([conditional_embed, non_conditional_embed], dim=0)

            # Get model output
            y_out = model(cat_xt, t=t, context=text_embed)
            y_out_with_text, y_out_no_text = torch.split(
                y_out, y_out.size(0) // 2, dim=0
            )

            # Apply guidance
            if guide_scale is None:
                guide_scale = DEFAULT_GUIDANCE_SCALE

            out = y_out_no_text + guide_scale * (y_out_with_text - y_out_no_text)

            # Apply rescaling if specified
            if guide_rescale is not None:
                if not (0 <= guide_rescale <= 1):
                    raise ValueError("guide_rescale must be between 0 and 1")

                ratio = (
                    y_out_with_text.flatten(1).to(torch.float32).std(dim=1)
                    / (out.flatten(1).to(torch.float32).std(dim=1) + EPSILON)
                ).view((-1,) + (1,) * (y_out_with_text.ndim - 1))
                out = out * (guide_rescale * ratio + (1 - guide_rescale) * 1.0)
        else:
            # Direct model output without guidance
            # Extract only the context from model_kwargs, not null_context
            if isinstance(model_kwargs, dict):
                context = model_kwargs.get('context')
                if context is not None:
                    out = model(xt, t=t, context=context)
                else:
                    out = model(xt, t=t)
            else:
                # Fallback for other model_kwargs formats
                out = model(xt, t=t, **model_kwargs)  # type: ignore

        # Compute x0 based on prediction type
        if self.prediction_type == "x0":
            x0 = out
        elif self.prediction_type == "eps":
            x0 = (xt - sigmas * out) / alphas
        elif self.prediction_type == "v":
            x0 = alphas * xt - sigmas * out
        else:
            raise ValueError(f"Prediction type {self.prediction_type} not implemented")

        # Apply range restriction if specified
        if percentile is not None:
            if not (0 < percentile <= 1):
                raise ValueError(
                    f"Percentile should be between 0 and 1, got {percentile}"
                )

            # Apply percentile-based clamping (assumes data in range [-1, 1])
            s = torch.quantile(x0.flatten(1).to(torch.float32).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1,) + (1,) * (xt.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        # Recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # Compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt

        return mu, var, log_var, x0, eps

    def set_classifier_scale(self, scale: float) -> None:
        """Set the classifier guidance scale.

        Args:
            scale: Classifier guidance scale value
        """
        self.classifier = scale

    def set_face_guidance_step(
        self,
        step: int,
        model: torch.nn.Module,
        similarity: float = DEFAULT_SIMILARITY,
        lamda_feat: float = DEFAULT_LAMBDA_FEAT_BEFORE_REF_GUIDANCE,
    ) -> None:
        """Set the face guidance step and related parameters.

        Args:
            step: Timestep to launch face guidance
            model: The model with share_cache
            similarity: Original similarity value
            lamda_feat: Lambda feature value before reference guidance
        """
        if not hasattr(model, "share_cache"):
            setattr(model, "share_cache", {})  # type: ignore[attr-defined]

        share_cache = getattr(model, "share_cache", {})  # type: ignore[attr-defined]
        share_cache["step_to_launch_face_guidance"] = step  # type: ignore[assignment]
        share_cache["ori_similarity"] = similarity  # type: ignore[assignment]
        share_cache["lamda_feat_before_ref_guidance"] = lamda_feat  # type: ignore[assignment]

    def sample(
        self,
        shape: Tuple[int, ...],
        model: torch.nn.Module,
        model_kwargs: Union[Dict, List],
        device: torch.device,
        guide_scale: float = DEFAULT_GUIDANCE_SCALE,
        guide_rescale: Optional[float] = None,
        clamp: Optional[float] = None,
        percentile: Optional[float] = None,
        clip_denoised: bool = True,
        repeat_noise: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            shape: Shape of the sample to generate
            model: The diffusion model
            model_kwargs: Model keyword arguments
            device: Device to run sampling on
            guide_scale: Guidance scale for classifier-free guidance
            guide_rescale: Rescale factor for guided output
            clamp: Value to clamp x0 predictions
            percentile: Percentile for restricting x0 range
            clip_denoised: Whether to clip denoised values to [-1, 1]
            repeat_noise: Whether to repeat noise across batch
            seed: Random seed for reproducibility

        Returns:
            Generated sample tensor
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize noise
        xt = torch.randn(shape, device=device)

        # Get timesteps for reverse process
        timesteps = list(range(0, self.num_timesteps))[::-1]

        # Loop through timesteps for reverse diffusion process
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t] * shape[0], device=device)

            # Add noise for stochasticity
            noise = torch.randn_like(xt) if i < len(timesteps) - 1 else 0
            if repeat_noise:
                noise = noise[0:1].repeat(shape[0], 1, 1, 1)

            # Apply denoise step
            mu, _, log_var, _, _ = ContextGaussianDiffusion.denoise(  # type: ignore
                self,
                xt=xt,
                t=t_tensor,
                s=None,
                model=model,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=guide_rescale,
                clamp=clamp,
                percentile=percentile,
            )

            # Compute the next step
            xt = mu + (0.5 * log_var).exp() * noise

            # Clip denoised values if requested
            if clip_denoised and i == len(timesteps) - 1:
                xt = xt.clamp(-1, 1)

        return xt
