"""Utility module for automatic mixed precision."""
import torch
from contextlib import contextmanager
from typing import Optional, Generator

@contextmanager
def autocast_context(enabled: bool = True, device_type: Optional[str] = None) -> Generator:
    """
    Context manager for automatic mixed precision.
    
    Args:
        enabled: Whether to enable autocast
        device_type: The device type to use for autocast ("cuda", "cpu", "mps")
                    If None, will be automatically determined
                    
    Yields:
        A context manager that enables or disables autocast
    """
    # If device_type is not specified, try to detect it
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    
    # Only use autocast if enabled and device supports it
    # (CPU autocast requires PyTorch 1.10+, MPS support varies)
    if enabled and hasattr(torch, "autocast"):
        try:
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                yield
        except Exception as e:
            print(f"Autocast failed with error: {e}. Falling back to full precision.")
            yield
    else:
        # If autocast is not enabled or not available, just yield
        yield