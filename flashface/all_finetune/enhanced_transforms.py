"""Enhanced image processing utilities for FlashFace."""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Tuple, List, Optional, Union, Dict, Any

class EnhancedPadToSquare:
    """
    Enhanced version of PadToSquare using torchvision 0.23.0+ features.
    Resizes and pads an image to make it square with the specified size.
    
    Args:
        size (int): The desired output size
        padding_mode (str): Padding mode for border padding ('constant', 'edge', 'reflect', or 'symmetric')
        fill (int, tuple): Fill value for 'constant' padding mode
    """
    def __init__(self, size: int, padding_mode: str = 'constant', fill: int = 0):
        self.size = size
        self.padding_mode = padding_mode
        self.fill = fill
        
    def __call__(self, img: Image.Image) -> Image.Image:
        # First resize the image to fit within the target size
        scale = self.size / max(img.size)
        resized_img = TF.resize(
            img, 
            [round(img.height * scale), round(img.width * scale)],
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True  # New in torchvision 0.23.0
        )
        
        # Calculate padding
        w, h = resized_img.size
        padding_left = (self.size - w) // 2
        padding_right = self.size - w - padding_left
        padding_top = (self.size - h) // 2
        padding_bottom = self.size - h - padding_top
        
        # Apply padding
        padded_img = TF.pad(
            resized_img, 
            [padding_left, padding_top, padding_right, padding_bottom], 
            fill=self.fill,
            padding_mode=self.padding_mode
        )
        
        return padded_img

class EnhancedCompose:
    """
    Enhanced version of Compose with additional functionality.
    
    Args:
        transforms (list): List of transformations to apply
        preprocessors (list, optional): List of preprocessing transforms to apply before main transforms
        postprocessors (list, optional): List of postprocessing transforms to apply after main transforms
    """
    def __init__(
        self, 
        transforms: List[Any], 
        preprocessors: Optional[List[Any]] = None,
        postprocessors: Optional[List[Any]] = None
    ):
        self.transforms = transforms
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        
    def __call__(self, img: Any) -> Any:
        # Apply preprocessors
        for t in self.preprocessors:
            img = t(img)
            
        # Apply main transforms
        for t in self.transforms:
            img = t(img)
            
        # Apply postprocessors
        for t in self.postprocessors:
            img = t(img)
            
        return img

def get_enhanced_padding(w: int, h: int, size: int) -> Tuple[int, int, int, int]:
    """
    Calculate padding to make a rectangular image square.
    
    Args:
        w (int): Width of the image
        h (int): Height of the image
        size (int): Target size (width and height) of the square image
        
    Returns:
        Tuple[int, int, int, int]: Padding values (left, top, right, bottom)
    """
    if w > h:
        pad = size - h
        return (0, pad // 2, 0, pad - pad // 2)
    elif h > w:
        pad = size - w
        return (pad // 2, 0, pad - pad // 2, 0)
    else:
        return (0, 0, 0, 0)

def create_face_transforms(
    mean: List[float] = [0.5, 0.5, 0.5], 
    std: List[float] = [0.5, 0.5, 0.5],
    size: int = 224
) -> EnhancedCompose:
    """
    Create transforms for face processing with improved quality.
    
    Args:
        mean (List[float]): Normalization mean values
        std (List[float]): Normalization std values
        size (int): Target size for the face images
        
    Returns:
        EnhancedCompose: Composition of transforms
    """
    return EnhancedCompose([
        EnhancedPadToSquare(size=size, padding_mode='constant', fill=0),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])