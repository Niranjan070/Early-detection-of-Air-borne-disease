"""
Image preprocessing utilities for spore trap images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image (BGR or RGB)
        target_size: Target size (width, height)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image
    """
    # Resize
    processed = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    if normalize:
        processed = processed.astype(np.float32) / 255.0
    
    return processed


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.
    Useful for making spores more visible.
    
    Args:
        image: Input image
        clip_limit: CLAHE clip limit
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
    
    return enhanced


def remove_background(
    image: np.ndarray,
    threshold: int = 127,
    method: str = 'otsu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove background from spore trap image.
    
    Args:
        image: Input image
        threshold: Threshold value for binary segmentation
        method: Thresholding method ('otsu' or 'adaptive')
        
    Returns:
        Tuple of (foreground image, mask)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding
    if method == 'otsu':
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply mask to original image
    if len(image.shape) == 3:
        foreground = cv2.bitwise_and(image, image, mask=mask)
    else:
        foreground = cv2.bitwise_and(gray, gray, mask=mask)
    
    return foreground, mask


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply denoising to reduce noise in microscopy images.
    
    Args:
        image: Input image
        strength: Denoising strength
        
    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    return denoised


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Sharpen image to enhance spore edges.
    
    Args:
        image: Input image
        strength: Sharpening strength
        
    Returns:
        Sharpened image
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]) * strength
    
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened
