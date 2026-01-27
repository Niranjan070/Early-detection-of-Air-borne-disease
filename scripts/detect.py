"""
Detection script for spore trap images.
"""

import argparse
import yaml
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.detection.detector import SporeDetector
from src.detection.counter import SporeCounter
from src.utils.visualization import visualize_detections
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Detect spores in images')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='models/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='outputs/predictions',
                        help='Output directory')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info(f"Processing: {args.image}")
    
    # Initialize detector
    detector = SporeDetector(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    # Run detection
    results = detector.detect(args.image, return_image=True)
    
    logger.info(f"Detected {results['num_detections']} spores")
    
    # Count spores
    counter = SporeCounter()
    counts = counter.count_spores(results['detections'])
    
    logger.info("Spore counts:")
    for spore_type, count in counts.items():
        if count > 0:
            logger.info(f"  {spore_type}: {count}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(args.image).stem
    output_path = output_dir / f"{image_name}_detected.jpg"
    
    cv2.imwrite(str(output_path), results['annotated_image'])
    logger.info(f"Saved result to: {output_path}")
    
    # Display if requested
    if args.show:
        cv2.imshow('Spore Detection', results['annotated_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results, counts


if __name__ == '__main__':
    main()
