# 🌿 Plant Disease Prediction System

An AI-powered system that predicts potential plant diseases by analyzing spore trap images using YOLO object detection to identify and count different spore types.

## 📋 Project Overview

This system captures and analyzes spore trap images to:
- **Detect** different types of fungal spores using YOLOv8
- **Count** spore quantities for each detected type
- **Predict** potential plant diseases based on spore analysis
- **Alert** farmers/users before disease outbreak occurs

## 🏗️ Project Structure

```
MINI_PROJECT/
├── data/
│   ├── raw/                    # Original spore trap images
│   ├── processed/              # Preprocessed images
│   ├── annotations/            # YOLO format annotations (.txt files)
│   └── splits/
│       ├── train/              # Training dataset
│       ├── val/                # Validation dataset
│       └── test/               # Test dataset
│
├── models/
│   ├── weights/                # Trained model weights (.pt files)
│   └── configs/                # Model configuration files
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset loading utilities
│   │   ├── preprocessing.py    # Image preprocessing functions
│   │   └── augmentation.py     # Data augmentation techniques
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py         # YOLO spore detection module
│   │   ├── counter.py          # Spore counting logic
│   │   └── tracker.py          # Spore tracking (optional)
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── disease_predictor.py    # Disease prediction based on spore data
│   │   ├── risk_analyzer.py        # Risk level analysis
│   │   └── spore_disease_map.py    # Spore type to disease mapping
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py    # Result visualization
│       ├── logger.py           # Logging utilities
│       └── helpers.py          # General helper functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA on spore images
│   ├── 02_model_training.ipynb         # YOLO training notebook
│   ├── 03_evaluation.ipynb             # Model evaluation
│   └── 04_disease_analysis.ipynb       # Disease prediction analysis
│
├── configs/
│   ├── config.yaml             # Main configuration file
│   ├── spore_classes.yaml      # Spore class definitions
│   └── disease_mapping.yaml    # Spore to disease mapping rules
│
├── scripts/
│   ├── train.py                # Training script
│   ├── detect.py               # Detection script
│   ├── predict_disease.py      # Disease prediction script
│   └── evaluate.py             # Evaluation script
│
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI/Flask application
│   └── routes.py               # API endpoints
│
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   ├── test_predictor.py
│   └── test_api.py
│
├── outputs/
│   ├── predictions/            # Prediction results
│   ├── visualizations/         # Generated visualizations
│   └── reports/                # Analysis reports
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🚀 Implementation Steps

### Phase 1: Data Collection & Preparation
1. **Collect spore trap images** from agricultural fields
2. **Annotate images** using tools like LabelImg or Roboflow (YOLO format)
3. **Split dataset** into train/val/test (70/20/10)
4. **Apply augmentation** to increase dataset diversity

### Phase 2: Model Development
1. **Select YOLO version** (YOLOv8 recommended for best performance)
2. **Configure model** for spore detection classes
3. **Train model** on annotated dataset
4. **Fine-tune** hyperparameters for optimal performance

### Phase 3: Disease Prediction Logic
1. **Map spore types** to associated plant diseases
2. **Define threshold rules** for disease risk levels
3. **Implement prediction algorithm** based on:
   - Spore type detected
   - Spore count/density
   - Environmental factors (optional)

### Phase 4: Integration & Deployment
1. **Build API** for easy integration
2. **Create user interface** (web/mobile)
3. **Deploy model** for real-time predictions
4. **Set up alerting system**

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd MINI_PROJECT

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

```
ultralytics>=8.0.0      # YOLOv8
torch>=2.0.0            # PyTorch
opencv-python>=4.8.0    # Image processing
numpy>=1.24.0           # Numerical operations
pandas>=2.0.0           # Data manipulation
matplotlib>=3.7.0       # Visualization
seaborn>=0.12.0         # Statistical plots
pyyaml>=6.0             # Configuration files
fastapi>=0.100.0        # API framework
uvicorn>=0.23.0         # ASGI server
pillow>=10.0.0          # Image handling
scikit-learn>=1.3.0     # ML utilities
```

## 🎯 Usage

### Training the Model
```bash
python scripts/train.py --config configs/config.yaml
```

### Running Detection
```bash
python scripts/detect.py --image path/to/spore_image.jpg
```

### Predicting Disease
```bash
python scripts/predict_disease.py --image path/to/spore_image.jpg
```

### Starting the API
```bash
uvicorn api.app:app --reload
```

## 📊 Spore Classes (Example)

| Class ID | Spore Type | Associated Diseases |
|----------|------------|---------------------|
| 0 | Alternaria | Early Blight, Leaf Spot |
| 1 | Fusarium | Fusarium Wilt, Root Rot |
| 2 | Botrytis | Gray Mold, Blossom Blight |
| 3 | Powdery Mildew | Powdery Mildew Disease |
| 4 | Rust Spores | Rust Disease |
| 5 | Downy Mildew | Downy Mildew Disease |

## 🔮 Disease Prediction Logic

```
Risk Level = f(spore_count, spore_type, threshold)

LOW RISK:     spore_count < threshold_low
MEDIUM RISK:  threshold_low <= spore_count < threshold_high  
HIGH RISK:    spore_count >= threshold_high
```

## 📈 Future Enhancements

- [ ] Multi-crop disease support
- [ ] Environmental data integration (humidity, temperature)
- [ ] Time-series analysis for outbreak prediction
- [ ] Mobile app for field use
- [ ] Integration with IoT sensors
- [ ] Real-time monitoring dashboard

## 📝 License

MIT License

## 👥 Contributors

- NIRANJAN

---
*Early detection saves crops! 🌱*
