"""
Disease prediction based on spore detection results.
"""

from typing import Dict, List, Optional
import yaml
from pathlib import Path


class DiseasePredictor:
    """
    Predict potential plant diseases based on detected spores.
    
    Args:
        mapping_path: Path to disease mapping YAML file
    """
    
    def __init__(self, mapping_path: str = None):
        self.mapping_path = mapping_path
        self.disease_mapping = self._load_mapping()
    
    def _load_mapping(self) -> Dict:
        """Load disease mapping from YAML file."""
        if self.mapping_path and Path(self.mapping_path).exists():
            with open(self.mapping_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default mapping
        return {
            'disease_mapping': {
                'magnaporthe_oryzae': {
                    'diseases': [
                        {'name': 'Rice Blast', 'crops': ['rice'], 'severity': 'critical'}
                    ],
                    'threshold_low': 5,
                    'threshold_high': 20
                },
                'botrytis': {
                    'diseases': [
                        {'name': 'Gray Mold', 'crops': ['strawberry', 'grape'], 'severity': 'high'}
                    ],
                    'threshold_low': 15,
                    'threshold_high': 60
                },
                'powdery_mildew': {
                    'diseases': [
                        {'name': 'Powdery Mildew', 'crops': ['cucumber', 'squash', 'grape'], 'severity': 'medium'}
                    ],
                    'threshold_low': 20,
                    'threshold_high': 80
                },
                'rust_spores': {
                    'diseases': [
                        {'name': 'Rust Disease', 'crops': ['wheat', 'corn', 'beans'], 'severity': 'high'}
                    ],
                    'threshold_low': 10,
                    'threshold_high': 40
                },
                'downy_mildew': {
                    'diseases': [
                        {'name': 'Downy Mildew', 'crops': ['grape', 'cucumber', 'lettuce'], 'severity': 'high'}
                    ],
                    'threshold_low': 15,
                    'threshold_high': 50
                }
            }
        }
    
    def predict(
        self,
        spore_counts: Dict[str, int],
        crop_type: Optional[str] = None,
        exposure_hours: Optional[float] = None,
    ) -> Dict:
        """
        Predict diseases based on spore counts.
        
        Args:
            spore_counts: Dictionary with spore type counts
            crop_type: Optional crop type to filter predictions
            
        Returns:
            Dictionary with predictions and risk levels
        """
        predictions = []

        use_rate = exposure_hours is not None and float(exposure_hours) > 0
        exposure = float(exposure_hours) if use_rate else None
        
        for spore_type, count in spore_counts.items():
            if spore_type == 'total' or count == 0:
                continue
            
            mapping = self.disease_mapping.get('disease_mapping', {}).get(spore_type)
            if not mapping:
                continue
            
            # Determine risk level (prefer per-hour thresholds when exposure is provided)
            threshold_low = mapping.get('threshold_low', 10)
            threshold_high = mapping.get('threshold_high', 50)

            metric = 'count'
            metric_value = count

            if use_rate and (
                'threshold_low_per_hour' in mapping or 'threshold_high_per_hour' in mapping
            ):
                metric = 'per_hour'
                metric_value = count / exposure
                threshold_low = mapping.get('threshold_low_per_hour', threshold_low)
                threshold_high = mapping.get('threshold_high_per_hour', threshold_high)

            if metric_value < threshold_low:
                risk_level = 'low'
            elif metric_value < threshold_high:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            # Get associated diseases
            for disease in mapping.get('diseases', []):
                # Filter by crop if specified
                if crop_type and crop_type.lower() not in [c.lower() for c in disease.get('crops', [])]:
                    continue
                
                prediction = {
                    'disease': disease['name'],
                    'spore_type': spore_type,
                    'spore_count': count,
                    'metric': metric,
                    'metric_value': round(float(metric_value), 4),
                    'risk_level': risk_level,
                    'severity': disease.get('severity', 'unknown'),
                    'affected_crops': disease.get('crops', [])
                }
                predictions.append(prediction)
        
        # Sort by risk level (high > medium > low)
        risk_order = {'high': 0, 'medium': 1, 'low': 2}
        predictions.sort(key=lambda x: risk_order.get(x['risk_level'], 3))
        
        return {
            'predictions': predictions,
            'total_diseases': len(predictions),
            'highest_risk': predictions[0]['risk_level'] if predictions else 'none',
            'recommendation': self._get_recommendation(predictions)
        }
    
    def _get_recommendation(self, predictions: List[Dict]) -> str:
        """Generate recommendation based on predictions."""
        if not predictions:
            return "No significant disease risk detected. Continue routine monitoring."
        
        high_risk = [p for p in predictions if p['risk_level'] == 'high']
        medium_risk = [p for p in predictions if p['risk_level'] == 'medium']
        
        if high_risk:
            diseases = [p['disease'] for p in high_risk]
            return f"HIGH ALERT: Immediate action required for {', '.join(diseases)}. Apply appropriate fungicides and increase monitoring frequency."
        elif medium_risk:
            diseases = [p['disease'] for p in medium_risk]
            return f"WARNING: Preventive measures recommended for {', '.join(diseases)}. Consider applying preventive treatments."
        else:
            return "Low risk detected. Monitor regularly and maintain good agricultural practices."
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific disease.
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Dictionary with disease information
        """
        for spore_type, mapping in self.disease_mapping.get('disease_mapping', {}).items():
            for disease in mapping.get('diseases', []):
                if disease['name'].lower() == disease_name.lower():
                    return {
                        'name': disease['name'],
                        'caused_by': spore_type,
                        'severity': disease.get('severity'),
                        'affected_crops': disease.get('crops', []),
                        'thresholds': {
                            'low': mapping.get('threshold_low'),
                            'high': mapping.get('threshold_high')
                        }
                    }
        return None
