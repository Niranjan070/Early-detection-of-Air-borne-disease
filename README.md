# 🔬 SporeNet AI — Advanced Plant Disease Risk Detection

An advanced deep learning framework powered by YOLOv8 for detecting and classifying microscopic plant pathogen spores. Originally built for **Magnaporthe oryzae** (Rice Blast), the model has been upgraded to a robust multi-class architecture to isolate pathogens from benign background spores effectively.

---

## 🧠 Model Architecture & Training

Our core detection model is a fine-tuned **YOLOv8 Nano (`yolov8n.pt`)**, optimized for microscopic object detection at high inference speeds.

### 📊 Dataset Details
To strictly reduce False Positives and enhance the model's vocabulary, the dataset was constructed by **merging** our proprietary Rice Blast dataset with a broader Curvularia genus dataset. This acts as a robust mechanism for "negative example" isolation.

- **Total Classes**: 9 distinct spore variants.
- **Dataset Size**: ~2,000+ annotated microscopic images (Train/Val/Test splits).
- **Target Resolution**: 640x640 pixels
- **Bounding Boxes**: Polygon segmentations were converted dynamically to YOLO normalized bounding boxes for strict object localization.

### 🏷️ Detectable Spore Classes
1. **`magnaporthe_oryzae`** *(Primary Target: Rice Blast)* 
2. `alternaria` *(Background/Negative)*
3. `bipolaris` *(Background/Negative)*
4. `curvularia` *(Background/Negative)*
5. `curvularia_eragrostidis` *(Background/Negative)*
6. `exserohilum` *(Background/Negative)*
7. `fusarium` *(Background/Negative)*
8. `fusarium_microconidie` *(Background/Negative)*
9. `mycelium` *(Background/Negative)*

### ⚙️ Training Parameters (Latest Run: `spore_detector`)
- **Epochs**: 50
- **Batch Size**: 16
- **Optimizer**: Auto (AdamW/SGD depending on heuristics)
- **Base Weights**: Pretrained `yolov8n.pt`
- **Configuration**: `configs/data_merged.yaml`

---

## 📁 Repository Structure

```
MINI_PROJECT/
│
├── configs/                    # ⚙️ Configuration files
│   ├── data_merged.yaml        #    → Multi-class dataset mapping (9 classes)
│   ├── data.yaml               #    → Original single-class mapping
│   ├── config.yaml             #    → Training hyperparameters
│   ├── spore_classes.yaml      #    → Spore vocabulary
│   └── disease_mapping.yaml    #    → Disease correlation matrix
│
├── data/                       # 📊 Datasets (Raw, Processed, and Merged)
│   ├── raw/                    #    → Original microscopic captures
│   ├── splits/                 #    → Original dataset split
│   ├── new_dataset/            #    → Supplemental negative examples
│   └── merged/                 #    → The final 9-class composite training set
│       ├── train/, val/, test/
│
├── scripts/                    # 🚀 Execution Modules
│   ├── train.py                #    → Triggers training pipeline
│   ├── detect.py               #    → Standalone inference engine
│   ├── merge_datasets.py       #    → Dataset curation & class remapping
│   └── predict_disease.py      #    → Pipeline: Detection → Density Count → Risk
│
├── src/                        # 🧠 Core Application Logic
│   ├── data/                   #    → Dataloaders & augmentation
│   ├── detection/              #    → YOLO wrapper & SporeCounter algorithm
│   ├── prediction/             #    → Disease probability risk logic
│   └── utils/                  #    → Logging & visualizers
│
├── runs/                       # 📈 Auto-tracked Training Artifacts
│   └── detect/runs/train/
│       └── spore_detector/     #    → ⭐ ALIGN TO THIS RUN!
│           ├── weights/
│           │   └── best.pt     #    → Optimal 9-class weights
│           ├── results.png     #    → F1/Precision/Recall curves
│           └── args.yaml       #    → Immutable training record
│
├── outputs/                    # 📤 Processed visual artifacts & logs
├── requirements.txt            # 📦 Dependencies
└── README.md                   # 📖 Documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Inference on a Microscopic Slide
Test the multi-class model on an image to see its distinguishing capabilities.

```powershell
python scripts/detect.py --model "runs\detect\runs\train\spore_detector\weights\best.pt" --image "data\merged\test\images\YOUR_IMAGE.jpg" --show
```
*Note: Due to the 9-class structure, the model will accurately label benign spores rather than throwing false positive Rice Blast alerts.*

### 3. Generate Disease Risk Report
Pipe the detection results strictly into the risk assessment engine:

```powershell
python scripts/predict_disease.py --model "runs\detect\runs\train\spore_detector\weights\best.pt" --image "data\merged\test\images\YOUR_IMAGE.jpg" --save-report
```

### 4. Retrain the Network
To increment epochs or adjust augmentation logic on the merged dataset:

```powershell
python scripts/train.py --data configs/data_merged.yaml --epochs 50
```
