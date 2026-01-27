"""
Dataset utilities for spore trap image loading and processing.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset


class SporeDataset(Dataset):
    """
    Custom dataset for loading spore trap images and annotations.
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size for resizing
        transform: Optional transforms to apply
    """
    
    def __init__(
        self, 
        data_dir: str, 
        img_size: int = 640, 
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Get all image paths
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self) -> List[Path]:
        """Get all image file paths from the data directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        images_dir = self.data_dir / 'images'
        if images_dir.exists():
            for ext in extensions:
                images.extend(images_dir.glob(f'*{ext}'))
        
        return sorted(images)
    
    def _load_annotation(self, img_path: Path) -> Optional[np.ndarray]:
        """Load YOLO format annotation for an image."""
        label_path = self.data_dir / 'labels' / f'{img_path.stem}.txt'
        
        if not label_path.exists():
            return None
            
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append([class_id, x_center, y_center, width, height])
        
        return np.array(annotations) if annotations else None
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """
        Get image and annotation by index.
        
        Returns:
            tuple: (image, annotations, image_path)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Load annotations
        annotations = self._load_annotation(img_path)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, annotations, str(img_path)


def create_data_yaml(
    train_path: str,
    val_path: str,
    test_path: str,
    class_names: List[str],
    output_path: str = 'data.yaml'
) -> None:
    """
    Create YOLO data.yaml file for training.
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Path to test images
        class_names: List of class names
        output_path: Output path for data.yaml
    """
    content = f"""# Dataset configuration for YOLO training

train: {train_path}
val: {val_path}
test: {test_path}

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Created data.yaml at {output_path}")
