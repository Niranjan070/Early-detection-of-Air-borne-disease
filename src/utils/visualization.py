"""
Visualization utilities for spore detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# Color palette for different spore types
SPORE_COLORS = {
    'alternaria': (255, 0, 0),      # Red
    'fusarium': (0, 255, 0),         # Green
    'botrytis': (0, 0, 255),         # Blue
    'powdery_mildew': (255, 255, 0), # Yellow
    'rust_spores': (255, 165, 0),    # Orange
    'downy_mildew': (128, 0, 128)    # Purple
}


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw detection boxes on image.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        line_thickness: Box line thickness
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for det in detections:
        class_name = det.get('class_name', 'unknown')
        confidence = det.get('confidence', 0)
        bbox = det.get('bbox', [0, 0, 0, 0])
        
        # Get color for this spore type
        color = SPORE_COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        if show_labels:
            label = class_name
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label size and position
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return annotated


def plot_spore_distribution(
    counts: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot bar chart of spore distribution.
    
    Args:
        counts: Spore counts by type
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Remove 'total' from counts
    filtered_counts = {k: v for k, v in counts.items() if k != 'total'}
    
    spore_types = list(filtered_counts.keys())
    values = list(filtered_counts.values())
    colors = [np.array(SPORE_COLORS.get(s, (128, 128, 128))) / 255 for s in spore_types]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(spore_types, values, color=colors)
    
    plt.xlabel('Spore Type')
    plt.ylabel('Count')
    plt.title('Spore Distribution')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def plot_risk_gauge(
    risk_score: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4)
) -> None:
    """
    Plot risk level gauge.
    
    Args:
        risk_score: Risk score (0-4)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create gauge background
    colors = ['green', 'yellow', 'orange', 'red']
    labels = ['Low', 'Medium', 'High', 'Critical']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.barh(0, 1, left=i, color=color, height=0.5, alpha=0.7)
        ax.text(i + 0.5, -0.4, label, ha='center', fontsize=10)
    
    # Draw needle
    needle_x = min(risk_score, 4)
    ax.plot([needle_x, needle_x], [-0.25, 0.25], 'k-', linewidth=3)
    ax.plot([needle_x], [0.25], 'ko', markersize=10)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    ax.set_title(f'Risk Level: {risk_score:.1f} / 4.0', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def create_summary_report(
    image: np.ndarray,
    detections: List[Dict],
    counts: Dict[str, int],
    predictions: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive visual report.
    
    Args:
        image: Original image
        detections: Detection results
        counts: Spore counts
        predictions: Disease predictions
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Annotated image
    ax1 = fig.add_subplot(2, 2, 1)
    annotated = visualize_detections(image, detections)
    ax1.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax1.set_title('Detected Spores')
    ax1.axis('off')
    
    # Spore distribution
    ax2 = fig.add_subplot(2, 2, 2)
    filtered_counts = {k: v for k, v in counts.items() if k != 'total'}
    ax2.bar(filtered_counts.keys(), filtered_counts.values())
    ax2.set_title('Spore Distribution')
    ax2.set_xlabel('Spore Type')
    ax2.set_ylabel('Count')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Predictions table
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    
    if predictions.get('predictions'):
        table_data = [
            [p['disease'], p['spore_type'], p['risk_level'], p['spore_count']]
            for p in predictions['predictions'][:5]  # Top 5
        ]
        table = ax3.table(
            cellText=table_data,
            colLabels=['Disease', 'Spore', 'Risk', 'Count'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    ax3.set_title('Disease Predictions')
    
    # Recommendations
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    recommendation = predictions.get('recommendation', 'No recommendations')
    ax4.text(0.1, 0.5, recommendation, wrap=True, fontsize=11)
    ax4.set_title('Recommendations')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
