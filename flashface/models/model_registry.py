import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..utils.cache_utils import ensure_file_exists

logger = logging.getLogger("flashface.model_registry")

# Registry of available models with their URLs and MD5 checksums
MODEL_REGISTRY = {
    "flashface": {
        "url": "https://huggingface.co/mmeendez/ComfyUI-FlashFace/resolve/main/model.safetensors",
        "md5": "c2e0f6c0adcc5c68b6995568a8e30e2c",
        "filename": "flashface.safetensors",
        "description": "FlashFace base model"
    },
    "retinaface": {
        "url": "https://huggingface.co/mmeendez/ComfyUI-FlashFace/resolve/main/detection_Resnet50_Final.pth",
        "md5": "7c70c60a0702fd5e56b18ec46e2d3533",
        "filename": "retinaface_resnet50.pth",
        "description": "RetinaFace detection model"
    },
    # Add other models as needed
}

def get_model_dir() -> str:
    """Get the directory where models are stored."""
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the models directory (adjust as needed)
    models_dir = os.path.abspath(os.path.join(current_dir, "../../models"))
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_available_models() -> List[str]:
    """Return a list of available model names."""
    try:
        import pydash as _
        return _.keys(MODEL_REGISTRY)
    except ImportError:
        return list(MODEL_REGISTRY.keys())

def get_model_info(model_name: str) -> Optional[Dict]:
    """Get information about a specific model."""
    return MODEL_REGISTRY.get(model_name)

def get_model_path(model_name: str, auto_download: bool = True) -> Optional[str]:
    """
    Get the path to a model file, downloading it if necessary.
    
    Args:
        model_name: Name of the model in the registry
        auto_download: Whether to automatically download the model if not found
        
    Returns:
        Path to the model file or None if not available
    """
    try:
        import pydash as _
        
        # Get model info using pydash
        model_info = _.get(MODEL_REGISTRY, model_name)
        if model_info is None:
            logger.error(f"Model '{model_name}' not found in registry")
            return None
        
        models_dir = get_model_dir()
        model_path = os.path.join(models_dir, _.get(model_info, "filename"))
        
        if not os.path.exists(model_path) and auto_download:
            logger.info(f"Model '{model_name}' not found locally, attempting to download...")
            model_path = ensure_file_exists(
                url=_.get(model_info, "url"),
                dest_path=model_path,
                expected_md5=_.get(model_info, "md5"),
                description=_.get(model_info, "description")
            )
        
        if model_path is None or not os.path.exists(model_path):
            logger.error(f"Model '{model_name}' not found or could not be downloaded")
            return None
            
        return model_path
        
    except ImportError:
        # Fallback to original implementation
        model_info = MODEL_REGISTRY.get(model_name)
        if not model_info:
            logger.error(f"Model '{model_name}' not found in registry")
            return None
        
        models_dir = get_model_dir()
        model_path = os.path.join(models_dir, model_info["filename"])
        
        if not os.path.exists(model_path) and auto_download:
            logger.info(f"Model '{model_name}' not found locally, attempting to download...")
            model_path = ensure_file_exists(
                url=model_info["url"],
                dest_path=model_path,
                expected_md5=model_info.get("md5"),
                description=model_info.get("description")
            )
        
        if model_path is None or not os.path.exists(model_path):
            logger.error(f"Model '{model_name}' not found or could not be downloaded")
            return None
            
        return model_path