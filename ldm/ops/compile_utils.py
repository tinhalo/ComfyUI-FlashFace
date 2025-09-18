"""Utility module for applying torch.compile to key computational functions."""
import torch
from typing import Any, Callable, Dict, Optional, Union

def _get_best_compile_settings() -> Dict[str, Any]:
    """
    Determine the best compilation settings based on PyTorch version and available hardware.
    
    Returns:
        Dict containing the optimal compilation settings
    """
    # Check PyTorch version
    pytorch_version = torch.__version__.split('.')
    major, minor = int(pytorch_version[0]), int(pytorch_version[1])
    
    # Set default settings based on PyTorch version
    if (major == 2 and minor >= 8) or major > 2:
        # PyTorch 2.8+ settings
        settings = {
            'mode': 'reduce-overhead',  # Best for PyTorch 2.8+
            'fullgraph': True,  # Enable full graph optimization
            'dynamic': True,  # Better support for dynamic shapes
        }
    elif major == 2:
        # PyTorch 2.0-2.7 settings
        settings = {
            'mode': 'max-autotune',  # Best for PyTorch 2.0-2.7
            'fullgraph': False,  # May not fully support fullgraph
        }
    else:
        # Fallback for earlier versions
        settings = {
            'mode': 'default',
        }
    
    # Add backend detection for PyTorch 2.4+
    if (major == 2 and minor >= 4) or major > 2:
        if torch.cuda.is_available():
            # Use inductor with CUDA if available
            settings['backend'] = 'inductor'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Use AOT eager with MPS
            settings['backend'] = 'aot_eager'
    
    return settings

def apply_torch_compile(module_function: Callable, *args: Any, **kwargs: Any) -> Callable:
    """
    Apply torch.compile to a module function with optimal settings for PyTorch 2.8.
    
    Args:
        module_function: The function to compile
        *args, **kwargs: Additional arguments to pass to torch.compile
        
    Returns:
        The compiled function or original function if torch.compile is unavailable
    """
    # Only apply if torch.compile is available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        # Get optimal settings based on PyTorch version and hardware
        compile_kwargs = _get_best_compile_settings()
        
        # Update with any user-provided kwargs
        compile_kwargs.update(kwargs)
        
        try:
            return torch.compile(module_function, *args, **compile_kwargs)
        except Exception as e:
            print(f"Warning: Failed to compile function with error: {e}")
            print("Falling back to uncompiled version")
            return module_function
    else:
        # Return the original function if torch.compile is not available
        return module_function