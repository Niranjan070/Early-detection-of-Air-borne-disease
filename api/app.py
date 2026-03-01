"""
FastAPI application for spore detection and disease prediction.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.detection.detector import SporeDetector
from src.detection.counter import SporeCounter
from src.prediction.disease_predictor import DiseasePredictor
from src.prediction.risk_analyzer import RiskAnalyzer
from src.storage.sample_store import SampleStore

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Prediction API",
    description="API for predicting plant diseases from spore trap images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"] ,
    allow_headers=["*"],
)

# Initialize components – prefer the trained spore model, fall back to generic YOLOv8n
_model_candidates = [
    'models/weights/best.pt',
    'runs/detect/runs/train/spore_detector2/weights/best.pt',
]
_preferred_model = next((p for p in _model_candidates if os.path.exists(p)), 'yolov8n.pt')
detector = SporeDetector(model_path=_preferred_model)
counter = SporeCounter()
predictor = DiseasePredictor(mapping_path='configs/disease_mapping.yaml')
risk_analyzer = RiskAnalyzer()

store = SampleStore(db_path='outputs/db/samples.sqlite3')


def _compute_frequency_per_hour(counts: dict, exposure_hours: float) -> dict:
    exposure = float(exposure_hours) if exposure_hours and float(exposure_hours) > 0 else 24.0
    freq = {}
    for k, v in counts.items():
        try:
            freq[f"{k}_per_hour"] = round(float(v) / exposure, 4)
        except Exception:
            freq[f"{k}_per_hour"] = None
    return freq


def _today_iso() -> str:
    return datetime.now().strftime('%Y-%m-%d')


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


@app.post("/samples")
async def create_or_update_daily_sample(
    file: UploadFile = File(...),
    farmer_id: str = Form(...),
    date: str = Form(...),
    crop_type: Optional[str] = Form(None),
    exposure_hours: float = Form(24.0),
):
    """Create/update the farmer's sample for a given day (one record per farmer per day)."""
    try:
        farmer_id = (farmer_id or '').strip()
        if not farmer_id:
            raise HTTPException(status_code=400, detail="farmer_id is required")

        date = (date or '').strip() or _today_iso()

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Save original upload (for audit/demo)
        uploads_dir = Path('outputs/uploads') / farmer_id
        uploads_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or '').suffix.lower() or '.jpg'
        safe_suffix = suffix if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} else '.jpg'
        image_path = uploads_dir / f"{date}{safe_suffix}"
        cv2.imwrite(str(image_path), image)

        # Analyze
        results = detector.detect(image)
        counts = counter.count_spores(results['detections'])
        stats = counter.get_statistics(counts)
        predictions = predictor.predict(counts, crop_type=crop_type, exposure_hours=exposure_hours)
        risk_analysis = risk_analyzer.analyze(predictions['predictions'])
        frequency = _compute_frequency_per_hour(counts, exposure_hours)

        payload = {
            "success": True,
            "farmer_id": farmer_id,
            "date": date,
            "crop_type": crop_type,
            "exposure_hours": float(exposure_hours),
            "image_path": str(image_path).replace('\\', '/'),
            "spore_counts": counts,
            "frequency": {
                "total_per_hour": frequency.get('total_per_hour'),
                "by_type_per_hour": {k: v for k, v in frequency.items() if k != 'total_per_hour'},
            },
            "statistics": stats,
            "predictions": predictions,
            "risk_analysis": risk_analysis,
        }

        record = store.upsert(
            farmer_id=farmer_id,
            date=date,
            crop_type=crop_type,
            exposure_hours=float(exposure_hours),
            image_path=str(image_path),
            payload=payload,
        )

        # return payload (already includes key info)
        payload["sample_id"] = record.id
        payload["updated_at"] = record.updated_at
        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/samples/today")
async def get_today_sample(farmer_id: str, date: Optional[str] = None):
    try:
        farmer_id = (farmer_id or '').strip()
        if not farmer_id:
            raise HTTPException(status_code=400, detail="farmer_id is required")
        date = (date or '').strip() or _today_iso()
        record = store.get(farmer_id=farmer_id, date=date)
        if not record:
            return {"success": False, "detail": "No sample for this day"}
        payload = dict(record.payload)
        payload["sample_id"] = record.id
        payload["updated_at"] = record.updated_at
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/samples/history")
async def get_sample_history(farmer_id: str, limit: int = 7):
    try:
        farmer_id = (farmer_id or '').strip()
        if not farmer_id:
            raise HTTPException(status_code=400, detail="farmer_id is required")
        rows = store.list_recent(farmer_id=farmer_id, limit=limit)
        return {
            "success": True,
            "farmer_id": farmer_id,
            "samples": [
                {
                    "id": r.id,
                    "date": r.date,
                    "crop_type": r.crop_type,
                    "exposure_hours": r.exposure_hours,
                    "overall_risk": (r.payload.get('risk_analysis') or {}).get('overall_risk'),
                    "total_spores": (r.payload.get('spore_counts') or {}).get('total'),
                    "total_per_hour": ((r.payload.get('frequency') or {}).get('total_per_hour')),
                    "updated_at": r.updated_at,
                }
                for r in rows
            ],
        }
    except HTTPException:
        raise
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
