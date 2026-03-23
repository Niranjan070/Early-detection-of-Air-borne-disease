# 🌾 Crop Disease Alert — Airborne Spore Detection & Plant Disease Prediction

An AI-powered system that predicts potential plant diseases by detecting and counting airborne fungal spores in spore trap images using YOLOv8 object detection. Designed for **farmers** — upload a spore trap photo, get an instant risk assessment, and know exactly what action to take.

---

## 📋 What It Does

Airborne fungal spores are early indicators of plant disease outbreaks. This system captures and analyzes spore trap images to help farmers act before it's too late:

1. **Detect** — Finds airborne fungal spores in microscope images using YOLOv8
2. **Count** — Measures spore density and calculates spores-per-hour frequency
3. **Predict** — Maps detected spores to potential crop diseases
4. **Alert** — Shows a clear risk level (Safe / Moderate / High / Critical) with actionable advice

### Currently Supported Spores

| Spore Type | Associated Disease | Status |
|---|---|---|
| *Magnaporthe oryzae* | Rice Blast | ✅ Trained |
| *Alternaria* | Early Blight, Leaf Spot | 🔜 Planned |
| *Fusarium* | Fusarium Wilt, Root Rot | 🔜 Planned |
| *Botrytis* | Gray Mold | 🔜 Planned |
| Rust Spores | Rust Disease | 🔜 Planned |

---

## 📊 Training Results (*Magnaporthe oryzae*)

Trained for **100 epochs** on the [Spore M. Oryzae dataset](https://universe.roboflow.com/iowa-state-university-cwvqa/spore-m-oryzae-xzewf/dataset/6) from Iowa State University.

| Metric | Best (Epoch 91) | Final (Epoch 100) |
|---|---|---|
| **mAP50** | **0.779** | 0.740 |
| **mAP50-95** | **0.334** | 0.311 |
| **Precision** | 0.835 | 0.806 |
| **Recall** | 0.696 | 0.659 |

---

## 🖥️ Screenshots

### Login Screen
Simple farmer-friendly login — just enter your name or ID, no email required.

### Dashboard — Upload Tab
Large tap-friendly upload area with image preview, crop selector, and exposure time input.

### Dashboard — Results Tab
Color-coded risk banner, spore count stats, breakdown by type, and plain-language recommendations ("What You Should Do").

### Dashboard — History Tab
See your last 10 daily checks at a glance with date, crop, spore count, and risk level.

---

## 🏗️ Project Structure

```
MINI_PROJECT/
├── api/
│   └── app.py                      # FastAPI backend (detect, predict, store)
│
├── frontend/                        # React + Vite farmer dashboard
│   ├── src/
│   │   ├── App.jsx                  # Main app component (login, tabs, results)
│   │   ├── styles.css               # Farmer-friendly green/earth theme
│   │   ├── main.jsx                 # Entry point
│   │   └── api.js                   # API helper utilities
│   ├── index.html
│   ├── vite.config.js               # Dev proxy → FastAPI backend
│   └── package.json
│
├── configs/
│   ├── config.yaml                  # Main configuration
│   ├── data.yaml                    # Dataset paths & class definitions
│   ├── spore_classes.yaml           # Spore class definitions
│   └── disease_mapping.yaml         # Spore → disease mapping rules
│
├── data/
│   ├── raw/                         # Original spore trap images
│   ├── processed/                   # Preprocessed images
│   ├── annotations/                 # YOLO format annotations
│   └── splits/
│       ├── train/                   # Training set (images + labels)
│       ├── val/                     # Validation set
│       └── test/                    # Test set
│
├── src/
│   ├── detection/
│   │   ├── detector.py              # YOLOv8 spore detection
│   │   └── counter.py               # Spore counting logic
│   ├── prediction/
│   │   ├── disease_predictor.py     # Disease prediction engine
│   │   └── risk_analyzer.py         # Risk level analysis
│   ├── storage/
│   │   └── sample_store.py          # SQLite sample storage
│   ├── data/
│   │   ├── dataset.py               # Dataset loading
│   │   ├── preprocessing.py         # Image preprocessing
│   │   └── augmentation.py          # Data augmentation
│   └── utils/
│       ├── visualization.py         # Result visualization
│       └── logger.py                # Logging utilities
│
├── scripts/
│   ├── train.py                     # Model training
│   ├── detect.py                    # Spore detection CLI
│   └── predict_disease.py           # Full prediction pipeline
│
├── notebooks/                       # Jupyter notebooks for analysis
├── outputs/
│   ├── db/samples.sqlite3           # SQLite database (per-farmer daily records)
│   ├── uploads/                     # Uploaded images
│   ├── predictions/                 # Detection output images
│   ├── reports/                     # Disease prediction reports
│   ├── visualizations/              # Generated plots
│   └── logs/                        # Training & inference logs
│
├── models/weights/                  # Trained model weights (.pt)
├── runs/                            # YOLO training runs & checkpoints
├── tests/                           # Unit tests
├── requirements.txt
├── setup_rice_blast.py
└── README.md
```

---

## 🔧 Installation

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the frontend)

### Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd MINI_PROJECT

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# Install Python dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

---

## 🚀 Running the Application

### Start Both Servers

```bash
# Terminal 1 — Backend API
venv\Scripts\activate
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend Dev Server
cd frontend
npm run dev
```

Open **http://localhost:5173/** in your browser.

### Daily Check-In Workflow (for Farmers)

1. **Login** — Enter your name or ID (e.g., "Ramesh" or "F-101")
2. **Select crop** — Rice, Wheat, or Barley
3. **Set trap exposure** — How many hours was the spore trap exposed (default: 24)
4. **Upload image** — Take a photo of the spore trap slide under a microscope
5. **View results** — See risk level, spore count, per-hour frequency, and recommendations
6. **Check history** — Track your daily checks over time

---

## 🧪 CLI Usage

### Train the Model

```bash
python scripts/train.py --config configs/config.yaml

# Resume from checkpoint
python scripts/train.py --resume runs/detect/runs/train/spore_detector2/weights/last.pt
```

### Detect Spores

```bash
python scripts/detect.py --image path/to/image.jpg \
    --model runs/detect/runs/train/spore_detector2/weights/best.pt

# With display window
python scripts/detect.py --image path/to/image.jpg \
    --model runs/detect/runs/train/spore_detector2/weights/best.pt --show

# Custom confidence threshold
python scripts/detect.py --image path/to/image.jpg \
    --model runs/detect/runs/train/spore_detector2/weights/best.pt --conf 0.4
```

### Predict Disease

```bash
python scripts/predict_disease.py --image path/to/image.jpg \
    --model runs/detect/runs/train/spore_detector2/weights/best.pt

# Filter by crop
python scripts/predict_disease.py --image path/to/image.jpg \
    --model runs/detect/runs/train/spore_detector2/weights/best.pt --crop rice

# Save visual report
python scripts/predict_disease.py --image path/to/image.jpg \
    --model runs/detect/runs/train/spore_detector2/weights/best.pt --save-report
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/health` | Health check |
| `POST` | `/detect` | Detect spores in an uploaded image |
| `POST` | `/predict` | Detect + predict disease from an image |
| `POST` | `/samples` | Create/update a farmer's daily sample |
| `GET`  | `/samples/today` | Get today's result for a farmer |
| `GET`  | `/samples/history` | Get a farmer's recent check history |
| `GET`  | `/classes` | List detectable spore classes |

---

## 🔮 Disease Prediction Logic

### Rice Blast (*Magnaporthe oryzae*)

**By Frequency (spores/hour):**

| Risk Level | Threshold | Action |
|---|---|---|
| ✅ Low | < 0.21 spores/hr | Monitor crops normally |
| ⚠️ Medium | 0.21 – 0.83 spores/hr | Consider preventive fungicide |
| 🔴 High | ≥ 0.83 spores/hr | Immediate treatment recommended |

**By Count (fallback when no exposure time):**

| Risk Level | Threshold | Action |
|---|---|---|
| ✅ Low | < 5 spores | Monitor crops normally |
| ⚠️ Medium | 5 – 20 spores | Consider preventive action |
| 🔴 High | ≥ 20 spores | Immediate treatment recommended |

**Affected Crops:** Rice, Wheat, Barley

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Detection Model** | YOLOv8n (Ultralytics) |
| **ML Framework** | PyTorch |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React 19 + Vite 7 |
| **Database** | SQLite |
| **Image Processing** | OpenCV, Pillow |
| **Dataset** | [Roboflow — Iowa State University](https://universe.roboflow.com/iowa-state-university-cwvqa/spore-m-oryzae-xzewf/dataset/6) (CC BY 4.0) |

---

## 📈 Future Enhancements

- [ ] Add more spore classes (Alternaria, Fusarium, Botrytis, Rust, Downy Mildew)
- [ ] Multi-class detection in a single model
- [ ] Environmental data integration (humidity, temperature, wind)
- [ ] Time-series analysis for outbreak prediction
- [ ] Mobile app for in-field use
- [ ] Real-time monitoring dashboard with IoT sensor integration
- [ ] Multi-language support (Hindi, Tamil, Telugu, etc.)

---

## 📝 License

MIT License

## 👥 Contributors

- NIRANJAN

---

*🌾 Early detection saves crops!*
