"""Utility functions for model loading with SafeTensors support."""
import os
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

def load_model_weights(model_path, device='cpu'):
    """Load model weights with support for both PyTorch and SafeTensors formats.
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to load the model to
        
    Returns:
        dict: Model state dictionary
    """
    # Get the file extension
    ext = os.path.splitext(model_path)[1].lower()
    
    # Load based on file format
    if ext == '.safetensors':
        # Load using SafeTensors
        return load_safetensors(model_path)
    else:
        # Load using traditional PyTorch
        return torch.load(model_path, map_location=device)
        
def load_checkpoint(model, checkpoint_path, strict=True, device='cpu'):
    """Load checkpoint into model with support for both PyTorch and SafeTensors formats.
    
    Args:
        model (nn.Module): Model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        strict (bool): Whether to strictly enforce that the keys in state_dict match
        device (str): Device to load the model to
        
    Returns:
        str: Result message from load_state_dict
    """
    # Load the weights
    state_dict = load_model_weights(checkpoint_path, device)
    
    # Load into model
    return model.load_state_dict(state_dict, strict=strict)