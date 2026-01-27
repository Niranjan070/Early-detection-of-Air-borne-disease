"""
Risk analysis module for disease prediction.
"""

from typing import Dict, List, Optional
from datetime import datetime


class RiskAnalyzer:
    """
    Analyze risk levels and provide actionable insights.
    """
    
    RISK_LEVELS = {
        'low': {
            'color': 'green',
            'score': 1,
            'description': 'Low risk - Continue routine monitoring',
            'action': 'No immediate action required'
        },
        'medium': {
            'color': 'yellow',
            'score': 2,
            'description': 'Medium risk - Preventive measures recommended',
            'action': 'Apply preventive fungicides, increase monitoring'
        },
        'high': {
            'color': 'red',
            'score': 3,
            'description': 'High risk - Immediate action required',
            'action': 'Apply treatment immediately, isolate affected areas'
        },
        'critical': {
            'color': 'darkred',
            'score': 4,
            'description': 'Critical - Emergency response needed',
            'action': 'Emergency intervention, consider removing infected plants'
        }
    }
    
    def __init__(self):
        self.history = []
    
    def analyze(
        self,
        predictions: List[Dict],
        environmental_factors: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze overall risk from predictions.
        
        Args:
            predictions: List of disease predictions
            environmental_factors: Optional environmental data (humidity, temp, etc.)
            
        Returns:
            Risk analysis results
        """
        if not predictions:
            return {
                'overall_risk': 'low',
                'risk_score': 0,
                'diseases_at_risk': [],
                'recommendations': ['No significant risk detected.'],
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate risk score
        risk_scores = []
        diseases_at_risk = []
        
        for pred in predictions:
            risk_level = pred.get('risk_level', 'low')
            risk_info = self.RISK_LEVELS.get(risk_level, self.RISK_LEVELS['low'])
            
            score = risk_info['score']
            
            # Adjust score based on environmental factors
            if environmental_factors:
                score = self._adjust_for_environment(score, environmental_factors)
            
            risk_scores.append(score)
            diseases_at_risk.append({
                'disease': pred['disease'],
                'risk_level': risk_level,
                'spore_count': pred.get('spore_count', 0)
            })
        
        # Calculate overall risk
        avg_score = sum(risk_scores) / len(risk_scores)
        max_score = max(risk_scores)
        
        overall_risk = self._score_to_risk_level(max_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(predictions, overall_risk)
        
        result = {
            'overall_risk': overall_risk,
            'risk_score': round(avg_score, 2),
            'max_risk_score': max_score,
            'diseases_at_risk': diseases_at_risk,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.history.append(result)
        
        return result
    
    def _adjust_for_environment(self, score: float, env_factors: Dict) -> float:
        """Adjust risk score based on environmental factors."""
        adjusted_score = score
        
        # High humidity increases fungal disease risk
        humidity = env_factors.get('humidity', 50)
        if humidity > 80:
            adjusted_score *= 1.3
        elif humidity > 60:
            adjusted_score *= 1.1
        
        # Temperature affects spore germination
        temperature = env_factors.get('temperature', 25)
        if 20 <= temperature <= 30:  # Optimal range for many fungi
            adjusted_score *= 1.2
        
        # Recent rainfall increases risk
        if env_factors.get('recent_rain', False):
            adjusted_score *= 1.2
        
        return min(adjusted_score, 4)  # Cap at critical level
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert numerical score to risk level."""
        if score >= 3.5:
            return 'critical'
        elif score >= 2.5:
            return 'high'
        elif score >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(
        self,
        predictions: List[Dict],
        overall_risk: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        risk_action = self.RISK_LEVELS.get(overall_risk, {}).get('action', '')
        if risk_action:
            recommendations.append(risk_action)
        
        # Disease-specific recommendations
        high_risk_diseases = [p for p in predictions if p.get('risk_level') == 'high']
        
        for disease_pred in high_risk_diseases:
            disease = disease_pred.get('disease', '')
            spore_type = disease_pred.get('spore_type', '')
            
            if 'blight' in disease.lower():
                recommendations.append(f"For {disease}: Remove infected plant parts, apply copper-based fungicide")
            elif 'mildew' in disease.lower():
                recommendations.append(f"For {disease}: Improve air circulation, apply sulfur-based treatment")
            elif 'rust' in disease.lower():
                recommendations.append(f"For {disease}: Remove infected leaves, apply protective fungicide")
            elif 'wilt' in disease.lower():
                recommendations.append(f"For {disease}: Consider soil treatment, avoid overwatering")
        
        if not recommendations:
            recommendations.append("Continue regular monitoring and maintain good agricultural practices.")
        
        return recommendations
    
    def get_trend(self, days: int = 7) -> Dict:
        """
        Analyze risk trend over recent history.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Trend analysis
        """
        if len(self.history) < 2:
            return {'trend': 'insufficient_data', 'change': 0}
        
        recent = self.history[-min(days, len(self.history)):]
        scores = [h['risk_score'] for h in recent]
        
        if scores[-1] > scores[0]:
            trend = 'increasing'
        elif scores[-1] < scores[0]:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': scores[-1] - scores[0],
            'average_score': sum(scores) / len(scores)
        }
