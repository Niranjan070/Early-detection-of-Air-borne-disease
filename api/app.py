"""
FastAPI application for spore detection and disease prediction.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from src.detection.detector import SporeDetector
from src.detection.counter import SporeCounter
from src.prediction.disease_predictor import DiseasePredictor
from src.prediction.risk_analyzer import RiskAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Prediction API",
    description="API for predicting plant diseases from spore trap images",
    version="1.0.0"
)

# Initialize components
detector = SporeDetector(model_path='models/weights/best.pt')
counter = SporeCounter()
predictor = DiseasePredictor(mapping_path='configs/disease_mapping.yaml')
risk_analyzer = RiskAnalyzer()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Plant Disease Prediction API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/detect")
async def detect_spores(file: UploadFile = File(...)):
    """
    Detect spores in uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Detection results with spore counts
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect spores
        results = detector.detect(image)
        
        # Count spores
        counts = counter.count_spores(results['detections'])
        
        return {
            "success": True,
            "num_detections": results['num_detections'],
            "detections": results['detections'],
            "spore_counts": counts
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    crop_type: Optional[str] = None
):
    """
    Predict plant diseases from spore trap image.
    
    Args:
        file: Uploaded image file
        crop_type: Optional crop type to filter predictions
        
    Returns:
        Disease predictions and risk analysis
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect spores
        results = detector.detect(image)
        
        # Count spores
        counts = counter.count_spores(results['detections'])
        stats = counter.get_statistics(counts)
        
        # Predict diseases
        predictions = predictor.predict(counts, crop_type=crop_type)
        
        # Analyze risk
        risk_analysis = risk_analyzer.analyze(predictions['predictions'])
        
        return {
            "success": True,
            "spore_counts": counts,
            "statistics": stats,
            "predictions": predictions,
            "risk_analysis": risk_analysis
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Get list of detectable spore classes."""
    return {
        "classes": detector.get_class_names()
    }


# Run with: uvicorn api.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
