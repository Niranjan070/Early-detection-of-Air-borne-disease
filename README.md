# 🌿 Airborne Spore Detection & Plant Disease Prediction System

An AI-powered system that predicts potential plant diseases by detecting and counting airborne fungal spores in spore trap images using YOLOv8 object detection. The system is designed to support multiple spore types — currently trained for *Magnaporthe oryzae* (Rice Blast), with more spore classes to be added.

## 📋 Project Overview

Airborne fungal spores are early indicators of plant disease outbreaks. By capturing and analyzing spore trap images, this system enables early detection and risk assessment:

- **Detect** airborne fungal spores in spore trap images using YOLOv8
- **Count** spore quantities to assess density
- **Predict** potential plant diseases based on spore-to-disease mapping
- **Alert** farmers before disease outbreaks occur

### Currently Supported Spores

| Spore Type | Associated Disease | Status |
|---|---|---|
| *Magnaporthe oryzae* | Rice Blast | ✅ Trained |
| *Alternaria* | Early Blight, Leaf Spot | 🔜 Planned |
| *Fusarium* | Fusarium Wilt, Root Rot | 🔜 Planned |
| *Botrytis* | Gray Mold | 🔜 Planned |
| Rust Spores | Rust Disease | 🔜 Planned |

## 📊 Training Results (*Magnaporthe oryzae*)

The model was trained for **100 epochs** on the [Spore M. Oryzae dataset](https://universe.roboflow.com/iowa-state-university-cwvqa/spore-m-oryzae-xzewf/dataset/6) from Iowa State University.

| Metric | Best (Epoch 91) | Final (Epoch 100) |
|---|---|---|
| **mAP50** | **0.779** | 0.740 |
| **mAP50-95** | **0.334** | 0.311 |
| **Precision** | 0.835 | 0.806 |
| **Recall** | 0.696 | 0.659 |

## 🏗️ Project Structure

```
MINI_PROJECT/
├── api/
│   ├── __init__.py
│   └── app.py                  # FastAPI application
│
├── configs/
│   ├── config.yaml             # Main configuration file
│   ├── data.yaml               # Dataset paths & class definitions
│   ├── spore_classes.yaml      # Spore class definitions
│   └── disease_mapping.yaml    # Spore-to-disease mapping rules
│
├── data/
│   ├── raw/                    # Original spore trap images
│   ├── processed/              # Preprocessed images
│   ├── annotations/            # YOLO format annotations
│   └── splits/
│       ├── train/              # Training dataset (images + labels)
│       ├── val/                # Validation dataset
│       └── test/               # Test dataset
│
├── models/
│   ├── weights/                # Trained model weights (.pt)
│   └── configs/                # Model configuration files
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset loading utilities
│   │   ├── preprocessing.py    # Image preprocessing
│   │   └── augmentation.py     # Data augmentation
│   ├── detection/
│   │   ├── detector.py         # YOLOv8 spore detection module
│   │   └── counter.py          # Spore counting logic
│   ├── prediction/
│   │   ├── disease_predictor.py    # Disease prediction engine
│   │   └── risk_analyzer.py        # Risk level analysis
│   └── utils/
│       ├── visualization.py    # Result visualization
│       └── logger.py           # Logging utilities
│
├── scripts/
│   ├── train.py                # Model training script
│   ├── detect.py               # Spore detection script
│   └── predict_disease.py      # Disease prediction pipeline
│
├── notebooks/                  # Jupyter notebooks for analysis
├── outputs/
│   ├── predictions/            # Detection output images
│   ├── reports/                # Disease prediction reports
│   ├── visualizations/         # Generated plots
│   └── logs/                   # Training & inference logs
│
├── runs/                       # YOLO training runs & checkpoints
├── tests/                      # Unit tests
├── requirements.txt
├── setup_rice_blast.py
├── .gitignore
└── README.md
```

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd MINI_PROJECT

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Training the Model

```bash
# Train from scratch
python scripts/train.py --config configs/config.yaml

# Resume interrupted training from checkpoint
python scripts/train.py --resume runs/detect/runs/train/spore_detector2/weights/last.pt
```

### Running Spore Detection

```bash
# Detect spores in an image (saves result to outputs/predictions/)
python scripts/detect.py --image path/to/spore_image.jpg --model runs/detect/runs/train/spore_detector2/weights/best.pt

# With display window
python scripts/detect.py --image path/to/spore_image.jpg --model runs/detect/runs/train/spore_detector2/weights/best.pt --show

# Adjust confidence threshold
python scripts/detect.py --image path/to/spore_image.jpg --model runs/detect/runs/train/spore_detector2/weights/best.pt --conf 0.4
```

### Predicting Disease Risk

```bash
# Full pipeline: detect → count → predict disease → analyze risk
python scripts/predict_disease.py --image path/to/spore_image.jpg --model runs/detect/runs/train/spore_detector2/weights/best.pt

# Save visual report
python scripts/predict_disease.py --image path/to/spore_image.jpg --model runs/detect/runs/train/spore_detector2/weights/best.pt --save-report

# Filter by crop type
python scripts/predict_disease.py --image path/to/spore_image.jpg --model runs/detect/runs/train/spore_detector2/weights/best.pt --crop rice
```

### Starting the API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Spore Detection Classes

| Class ID | Spore Type | Description | Associated Disease | Status |
|----------|------------|-------------|-------------------|--------|
| 0 | *Magnaporthe oryzae* | Pear-shaped (pyriform), usually 3-celled spores | Rice Blast | ✅ Trained |
| 1 | *Alternaria* | Dark, club-shaped, multicellular spores | Early Blight, Leaf Spot | 🔜 Planned |
| 2 | *Fusarium* | Canoe-shaped macroconidia | Fusarium Wilt, Root Rot | 🔜 Planned |
| 3 | *Botrytis* | Oval/elliptical, grape-like clusters | Gray Mold | 🔜 Planned |
| 4 | Rust Spores | Round/oval, orange-brown | Rust Disease | 🔜 Planned |

> **Note:** Currently only class 0 (*M. oryzae*) is trained. Additional spore classes will be added as annotated datasets become available.

## 🔮 Disease Prediction Logic

The system uses spore count thresholds to determine risk levels:

```
LOW RISK:     spore_count < 5      → Monitor crops
MEDIUM RISK:  5 ≤ spore_count < 20 → Consider preventive action
HIGH RISK:    spore_count ≥ 20     → Immediate treatment recommended
```

**Affected Crops:** Rice, Wheat, Barley

## 🛠️ Tech Stack

- **Detection Model:** YOLOv8n (Ultralytics)
- **Framework:** PyTorch
- **API:** FastAPI + Uvicorn
- **Image Processing:** OpenCV, Pillow
- **Dataset Source:** [Roboflow - Iowa State University](https://universe.roboflow.com/iowa-state-university-cwvqa/spore-m-oryzae-xzewf/dataset/6) (CC BY 4.0)

## 📈 Future Enhancements

- [ ] Add more spore classes (Alternaria, Fusarium, Botrytis, Rust, Downy Mildew)
- [ ] Multi-class detection in a single model
- [ ] Environmental data integration (humidity, temperature, wind)
- [ ] Time-series analysis for outbreak prediction
- [ ] Mobile app for in-field use
- [ ] Real-time monitoring dashboard with IoT sensor integration

## 📝 License

MIT License

## 👥 Contributors

- NIRANJAN

---
*Early detection saves crops! 🌱*
