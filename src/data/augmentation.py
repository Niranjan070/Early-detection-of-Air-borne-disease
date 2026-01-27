"""
Data augmentation techniques for spore detection training.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import random


def augment_image(
    image: np.ndarray,
    annotations: Optional[np.ndarray] = None,
    augmentations: Optional[list] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply random augmentations to image and update annotations.
    
    Args:
        image: Input image
        annotations: YOLO format annotations (class, x, y, w, h)
        augmentations: List of augmentation names to apply
        
    Returns:
        Tuple of (augmented image, updated annotations)
    """
    if augmentations is None:
        augmentations = ['flip', 'rotate', 'brightness', 'blur']
    
    aug_image = image.copy()
    aug_annotations = annotations.copy() if annotations is not None else None
    
    for aug in augmentations:
        if random.random() > 0.5:
            if aug == 'flip':
                aug_image, aug_annotations = random_flip(aug_image, aug_annotations)
            elif aug == 'rotate':
                aug_image, aug_annotations = random_rotate(aug_image, aug_annotations)
            elif aug == 'brightness':
                aug_image = random_brightness(aug_image)
            elif aug == 'blur':
                aug_image = random_blur(aug_image)
            elif aug == 'noise':
                aug_image = add_noise(aug_image)
    
    return aug_image, aug_annotations


def random_flip(
    image: np.ndarray,
    annotations: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply random horizontal/vertical flip."""
    flip_type = random.choice([0, 1, -1])  # 0: vertical, 1: horizontal, -1: both
    
    flipped = cv2.flip(image, flip_type)
    
    if annotations is not None:
        flipped_annotations = annotations.copy()
        if flip_type in [1, -1]:  # Horizontal flip
            flipped_annotations[:, 1] = 1 - flipped_annotations[:, 1]
        if flip_type in [0, -1]:  # Vertical flip
            flipped_annotations[:, 2] = 1 - flipped_annotations[:, 2]
        return flipped, flipped_annotations
    
    return flipped, None


def random_rotate(
    image: np.ndarray,
    annotations: Optional[np.ndarray] = None,
    max_angle: int = 30
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply random rotation."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    # Note: For accurate annotation rotation, more complex transformation needed
    # This is a simplified version
    return rotated, annotations


def random_brightness(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.7, 1.3)
) -> np.ndarray:
    """Adjust image brightness randomly."""
    factor = random.uniform(*brightness_range)
    adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
    return adjusted


def random_blur(
    image: np.ndarray,
    kernel_range: Tuple[int, int] = (3, 7)
) -> np.ndarray:
    """Apply random Gaussian blur."""
    kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def add_noise(
    image: np.ndarray,
    noise_type: str = 'gaussian',
    intensity: float = 25
) -> np.ndarray:
    """Add random noise to image."""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        noisy = image.copy()
        salt_mask = np.random.random(image.shape[:2]) < 0.01
        pepper_mask = np.random.random(image.shape[:2]) < 0.01
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
    else:
        noisy = image
    
    return noisy


def mosaic_augmentation(
    images: list,
    annotations_list: list,
    output_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mosaic augmentation (combine 4 images).
    
    Args:
        images: List of 4 images
        annotations_list: List of 4 annotation arrays
        output_size: Output image size
        
    Returns:
        Tuple of (mosaic image, combined annotations)
    """
    assert len(images) == 4, "Mosaic requires exactly 4 images"
    
    h, w = output_size
    mosaic = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Calculate center point
    cx, cy = w // 2, h // 2
    
    # Place images in quadrants
    positions = [
        (0, 0, cx, cy),           # Top-left
        (cx, 0, w, cy),           # Top-right
        (0, cy, cx, h),           # Bottom-left
        (cx, cy, w, h)            # Bottom-right
    ]
    
    all_annotations = []
    
    for i, (x1, y1, x2, y2) in enumerate(positions):
        img = cv2.resize(images[i], (x2 - x1, y2 - y1))
        mosaic[y1:y2, x1:x2] = img
        
        if annotations_list[i] is not None:
            # Adjust annotations for the quadrant
            # This is a simplified version
            pass
    
    return mosaic, np.array(all_annotations) if all_annotations else None
