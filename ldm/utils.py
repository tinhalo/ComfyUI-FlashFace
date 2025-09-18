import os
import torch
from safetensors.torch import load_file as load_safetensors

def load_model_weights(model_path, device="cpu"):
    """
    Load model weights from a .pth or .safetensors file.
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to load the model on
        
    Returns:
        dict: The loaded model weights
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.endswith(".safetensors"):
        return load_safetensors(model_path, device=device)
    else:
        return torch.load(model_path, map_location=device)