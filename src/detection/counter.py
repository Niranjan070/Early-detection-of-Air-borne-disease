"""
Spore counting module.
"""

from typing import Dict, List
from collections import Counter


class SporeCounter:
    """
    Count spores by type from detection results.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize spore counter.
        
        Args:
            class_names: List of spore class names
        """
        self.class_names = class_names or [
            'alternaria', 'fusarium', 'botrytis',
            'powdery_mildew', 'rust_spores', 'downy_mildew'
        ]
    
    def count_spores(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count spores by type from detection results.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with spore type counts
        """
        # Extract class names from detections
        class_names = [d['class_name'] for d in detections]
        
        # Count occurrences
        counts = Counter(class_names)
        
        # Ensure all classes are represented
        result = {name: counts.get(name, 0) for name in self.class_names}
        result['total'] = sum(result.values())
        
        return result
    
    def calculate_density(
        self,
        counts: Dict[str, int],
        image_area_mm2: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate spore density (spores per mm²).
        
        Args:
            counts: Spore counts by type
            image_area_mm2: Image area in square millimeters
            
        Returns:
            Dictionary with spore densities
        """
        density = {}
        for spore_type, count in counts.items():
            if spore_type != 'total':
                density[f'{spore_type}_per_mm2'] = count / image_area_mm2
        
        density['total_per_mm2'] = counts['total'] / image_area_mm2
        
        return density
    
    def get_dominant_spore(self, counts: Dict[str, int]) -> str:
        """
        Get the most common spore type.
        
        Args:
            counts: Spore counts by type
            
        Returns:
            Name of dominant spore type
        """
        # Remove 'total' from consideration
        filtered_counts = {k: v for k, v in counts.items() if k != 'total'}
        
        if not filtered_counts or all(v == 0 for v in filtered_counts.values()):
            return None
        
        return max(filtered_counts, key=filtered_counts.get)
    
    def get_statistics(self, counts: Dict[str, int]) -> Dict:
        """
        Get statistics about spore distribution.
        
        Args:
            counts: Spore counts by type
            
        Returns:
            Dictionary with statistics
        """
        filtered_counts = {k: v for k, v in counts.items() if k != 'total'}
        values = list(filtered_counts.values())
        total = counts.get('total', sum(values))
        
        if total == 0:
            return {
                'dominant_spore': None,
                'diversity_index': 0,
                'percentages': {k: 0 for k in filtered_counts}
            }
        
        # Calculate percentages
        percentages = {k: (v / total) * 100 for k, v in filtered_counts.items()}
        
        # Calculate Shannon diversity index
        diversity = 0
        for count in values:
            if count > 0:
                p = count / total
                diversity -= p * (p if p > 0 else 1)
        
        return {
            'dominant_spore': self.get_dominant_spore(counts),
            'diversity_index': abs(diversity),
            'percentages': percentages,
            'total_count': total
        }
