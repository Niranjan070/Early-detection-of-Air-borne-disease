"""
Training script for spore detection model.
"""

import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.detection.detector import SporeDetector
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train spore detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger(
        log_level=config.get('logging', {}).get('level', 'INFO'),
        save_logs=config.get('logging', {}).get('save_logs', True)
    )
    
    logger.info("Starting training...")
    logger.info(f"Config: {args.config}")
    
    # Initialize detector
    detector = SporeDetector(
        model_path=args.resume,
        device=config.get('training', {}).get('device', 'auto')
    )
    
    # Get training parameters
    epochs = args.epochs or config.get('training', {}).get('epochs', 100)
    batch_size = args.batch_size or config.get('data', {}).get('batch_size', 16)
    img_size = config.get('data', {}).get('image_size', 640)
    
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Train model
    detector.train(
        data_yaml=args.data,
        epochs=epochs,
        img_size=img_size,
        batch_size=batch_size,
        project='runs/train',
        name='spore_detector'
    )
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
