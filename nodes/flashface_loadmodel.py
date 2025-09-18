import os
import torch
from typing import Tuple, Dict, Any, List, Optional

from ..flashface.all_finetune.config import cfg
from ..flashface.all_finetune.ops.context_diffusion import ContextGaussianDiffusion
from ..flashface.all_finetune.models import sd_v1_ref_unet
from ..ldm import sd_v1_vae, ops, models
from ..ldm.utils import load_model_weights
from ..flashface.models.model_registry import get_model_path as get_registry_model_path


class FlashFaceLoadModel:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_name": (["flashface"],),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            },
            "optional": {
                "custom_scheduler": ("SCHEDULER", {}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", )
    FUNCTION = "load_models"

    def load_models(self, model_name: str, device: str = "auto", custom_scheduler=None) -> Tuple[Tuple, Any, Any]:
        # Determine the appropriate device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"Using device: {device}")
        
        # Destructure needed configuration values
        clip_model_name = cfg.get('clip_model', 'clip_vit_l_14')
        num_pairs = cfg.get('num_pairs', 4)
        schedule = cfg.get('schedule', 'scaled_linear')
        num_timesteps = cfg.get('num_timesteps', 1000)
        scale_min = cfg.get('scale_min', 0.00085)
        scale_max = cfg.get('scale_max', 0.0120)
        prediction_type = cfg.get('prediction_type', 'eps')
        
        clip = getattr(models, clip_model_name)(pretrained=True).eval().requires_grad_(False).textual.to(device)
        autoencoder = sd_v1_vae(pretrained=True).eval().requires_grad_(False).to(device)
        flashface_model = sd_v1_ref_unet(pretrained=True, version='sd-v1-5_nonema', enable_encoder=False).to(device)
        flashface_model.replace_input_conv()
        flashface_model = flashface_model.eval().requires_grad_(False).to(device)
        flashface_model.share_cache['num_pairs'] = num_pairs

        # Get model path from registry with auto-download if needed
        model_path = get_registry_model_path(model_name, auto_download=True)
        if model_path is None:
            # Fallback to the old method if model registry fails
            model_path = os.path.join(get_model_path(), "flashface.safetensors")
            print(f"Warning: Using fallback model path: {model_path}")
        
        print(f"Loading FlashFace model from: {model_path}")
        model_weight = load_model_weights(model_path, device=device)
        msg = flashface_model.load_state_dict(model_weight, strict=True)
        print(msg)

        # Create noise schedule - use custom scheduler if provided
        if custom_scheduler is not None:
            # Use custom scheduler function
            sigmas = custom_scheduler(n=num_timesteps)
        else:
            # Use default schedule
            sigmas = ops.noise_schedule(schedule=schedule, n=num_timesteps, beta_min=scale_min, beta_max=scale_max)
        
        diffusion = ContextGaussianDiffusion(sigmas=sigmas, prediction_type=prediction_type)
        # Add num_pairs as a custom attribute to diffusion
        setattr(diffusion, 'num_pairs', num_pairs)

        return ((flashface_model, diffusion), clip, autoencoder, )

def get_model_path():
    """Get the absolute path to the models/flashface directory.
    
    This function checks multiple possible locations for the models directory,
    making it more robust across different ComfyUI installation configurations.
    """
    try:
        import pydash as _
    except ImportError:
        # Fallback to original implementation if pydash is not installed
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different possible locations for the models directory
        possible_paths = [
            # Standard location (ComfyUI/models/flashface)
            os.path.join(current_dir, "../../../models/flashface"),
            
            # Alternative location if installed as a custom node
            os.path.join(current_dir, "../../models/flashface"),
            
            # Inside the extension directory
            os.path.join(current_dir, "../models/flashface"),
            
            # Fallback to the cache directory in the repository
            os.path.join(current_dir, "../cache")
        ]
        
        # Check each path and use the first valid one
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.isdir(abs_path):
                return abs_path
                
        # If no valid path is found, create the default one
        default_path = os.path.abspath(os.path.join(current_dir, "../../../models/flashface"))
        os.makedirs(default_path, exist_ok=True)
        return default_path
    else:
        # Using pydash implementation for cleaner code
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        possible_paths = [
            os.path.join(current_dir, "../../../models/flashface"),
            os.path.join(current_dir, "../../models/flashface"),
            os.path.join(current_dir, "../models/flashface"),
            os.path.join(current_dir, "../cache")
        ]
        
        # Use pydash to find the first valid directory
        abs_paths = _.map_(possible_paths, os.path.abspath)
        valid_path = _.find(abs_paths, os.path.isdir)
        
        if valid_path:
            return valid_path
            
        # If no valid path is found, create the default one
        default_path = os.path.abspath(os.path.join(current_dir, "../../../models/flashface"))
        os.makedirs(default_path, exist_ok=True)
        return default_path