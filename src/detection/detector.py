"""
YOLO-based spore detection module.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO


class SporeDetector:
    """
    Spore detection using YOLOv8.
    
    Args:
        model_path: Path to trained YOLO model weights
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: Device to run inference on ('cpu', 'cuda', or device id)
    """
    
    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Load pretrained YOLOv8 nano as base
            self.model = YOLO('yolov8n.pt')
            print("Warning: Using pretrained YOLOv8n. Train on spore data for better results.")
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        return_image: bool = False
    ) -> Dict:
        """
        Detect spores in an image.
        
        Args:
            image: Image path or numpy array
            return_image: Whether to return annotated image
            
        Returns:
            Dictionary containing detection results
        """
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device
        )[0]
        
        # Parse results
        detections = []
        for box in results.boxes:
            detection = {
                'class_id': int(box.cls.item()),
                'class_name': results.names[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                'bbox_normalized': box.xywhn[0].tolist()  # [x_center, y_center, width, height] normalized
            }
            detections.append(detection)
        
        result = {
            'detections': detections,
            'num_detections': len(detections),
            'image_shape': results.orig_shape
        }
        
        if return_image:
            result['annotated_image'] = results.plot()
        
        return result
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Detect spores in multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            batch_size: Batch size for inference
            
        Returns:
            List of detection result dictionaries
        """
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self.model(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device
            )
            
            for result in batch_results:
                detections = []
                for box in result.boxes:
                    detection = {
                        'class_id': int(box.cls.item()),
                        'class_name': result.names[int(box.cls.item())],
                        'confidence': float(box.conf.item()),
                        'bbox': box.xyxy[0].tolist()
                    }
                    detections.append(detection)
                
                all_results.append({
                    'detections': detections,
                    'num_detections': len(detections)
                })
        
        return all_results
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int = 16,
        project: str = 'runs/train',
        name: str = 'spore_detector'
    ) -> None:
        """
        Train the YOLO model on spore dataset.
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Training batch size
            project: Project directory for saving results
            name: Experiment name
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project=project,
            name=name,
            device=self.device
        )
        
        print(f"Training complete. Results saved to {project}/{name}")
    
    def export(self, format: str = 'onnx') -> str:
        """
        Export model to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            
        Returns:
            Path to exported model
        """
        return self.model.export(format=format)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return list(self.model.names.values())
