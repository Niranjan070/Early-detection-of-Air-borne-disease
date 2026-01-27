"""
Disease prediction script.
"""

import argparse
import yaml
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.detection.detector import SporeDetector
from src.detection.counter import SporeCounter
from src.prediction.disease_predictor import DiseasePredictor
from src.prediction.risk_analyzer import RiskAnalyzer
from src.utils.visualization import create_summary_report
from src.utils.logger import setup_logger
import cv2


def main():
    parser = argparse.ArgumentParser(description='Predict plant diseases from spore trap image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to spore trap image')
    parser.add_argument('--model', type=str, default='models/weights/best.pt',
                        help='Path to detection model')
    parser.add_argument('--crop', type=str, default=None,
                        help='Crop type to filter predictions')
    parser.add_argument('--output', type=str, default='outputs/reports',
                        help='Output directory for reports')
    parser.add_argument('--save-report', action='store_true',
                        help='Save visual report')
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info(f"Analyzing image: {args.image}")
    
    # Initialize components
    detector = SporeDetector(model_path=args.model)
    counter = SporeCounter()
    predictor = DiseasePredictor(mapping_path='configs/disease_mapping.yaml')
    risk_analyzer = RiskAnalyzer()
    
    # Step 1: Detect spores
    logger.info("Detecting spores...")
    detection_results = detector.detect(args.image, return_image=True)
    logger.info(f"Detected {detection_results['num_detections']} spores")
    
    # Step 2: Count spores
    counts = counter.count_spores(detection_results['detections'])
    stats = counter.get_statistics(counts)
    
    logger.info("Spore Analysis:")
    logger.info(f"  Total spores: {counts['total']}")
    logger.info(f"  Dominant type: {stats['dominant_spore']}")
    
    # Step 3: Predict diseases
    logger.info("Predicting diseases...")
    predictions = predictor.predict(counts, crop_type=args.crop)
    
    logger.info(f"Found {predictions['total_diseases']} potential disease risks")
    logger.info(f"Highest risk level: {predictions['highest_risk']}")
    
    # Step 4: Analyze risk
    risk_analysis = risk_analyzer.analyze(predictions['predictions'])
    
    logger.info(f"Overall Risk: {risk_analysis['overall_risk'].upper()}")
    logger.info(f"Risk Score: {risk_analysis['risk_score']}/4.0")
    
    # Print recommendations
    logger.info("\nRecommendations:")
    for rec in risk_analysis['recommendations']:
        logger.info(f"  • {rec}")
    
    # Print detailed predictions
    if predictions['predictions']:
        logger.info("\nDetailed Predictions:")
        for pred in predictions['predictions']:
            logger.info(f"  - {pred['disease']} ({pred['risk_level']} risk)")
            logger.info(f"    Caused by: {pred['spore_type']} ({pred['spore_count']} spores)")
    
    # Save report
    if args.save_report:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(args.image).stem
        
        # Save visual report
        image = cv2.imread(args.image)
        report_path = output_dir / f"{image_name}_report.png"
        create_summary_report(
            image,
            detection_results['detections'],
            counts,
            predictions,
            save_path=str(report_path)
        )
        logger.info(f"Report saved to: {report_path}")
        
        # Save JSON report
        json_report = {
            'image': args.image,
            'spore_counts': counts,
            'statistics': stats,
            'predictions': predictions,
            'risk_analysis': risk_analysis
        }
        json_path = output_dir / f"{image_name}_report.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        logger.info(f"JSON report saved to: {json_path}")
    
    return predictions, risk_analysis


if __name__ == '__main__':
    main()
