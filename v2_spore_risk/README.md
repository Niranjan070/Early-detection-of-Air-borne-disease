# Spore Risk V2

This folder is a separate `v2` pipeline built on top of your trained YOLO spore detector.

The idea is:

1. Run YOLO inference on trap images.
2. Extract per-image and per-class spore counts.
3. Build a feature table that can include metadata like site, timestamp, humidity, and trap area.
4. Train a separate risk model to predict the final rice blast risk.

## Folder Structure

```text
v2_spore_risk/
  configs/
    pipeline.yaml
  data/
    raw/
    interim/
    processed/
  models/
  outputs/
  scripts/
    01_yolo_inference.py
    02_build_feature_table.py
    03_train_risk_model.py
    04_score_risk.py
  src/
    spore_risk_v2/
      __init__.py
      config.py
      features.py
      inference.py
      risk_model.py
      utils.py
  templates/
    sample_metadata_template.csv
    risk_labels_template.csv
```

## Recommended Workflow

### 1. Update the config

Edit `configs/pipeline.yaml` and point `yolo.weights` to your trained YOLO model.

### 2. Prepare image data

Place trap images under:

```text
v2_spore_risk/data/raw/images
```

### 3. Optional metadata

If you have metadata, start from:

```text
templates/sample_metadata_template.csv
```

The pipeline can use:

- `sample_id`
- `image_name`
- `captured_at`
- `site_id`
- `trap_id`
- `trap_area_mm2`
- `humidity`
- `temperature_c`
- `rainfall_mm`
- `leaf_wetness_hours`

### 4. Run inference

```powershell
.\venv\Scripts\python.exe v2_spore_risk\scripts\01_yolo_inference.py --config v2_spore_risk\configs\pipeline.yaml
```

This creates:

- `data/interim/detections.csv`
- `data/interim/image_counts.csv`

### 5. Build the feature table

```powershell
.\venv\Scripts\python.exe v2_spore_risk\scripts\02_build_feature_table.py --config v2_spore_risk\configs\pipeline.yaml
```

This creates:

- `data/processed/feature_table.csv`

### 6. Add labels for supervised training

Start from:

```text
templates/risk_labels_template.csv
```

The target column in the default config is `blast_risk_label`.

Example labels:

- `low`
- `medium`
- `high`

### 7. Train the risk model

```powershell
.\venv\Scripts\python.exe v2_spore_risk\scripts\03_train_risk_model.py --config v2_spore_risk\configs\pipeline.yaml
```

This creates:

- `models/risk_model.pkl`
- `models/risk_metrics.json`

### 8. Score new samples

```powershell
.\venv\Scripts\python.exe v2_spore_risk\scripts\04_score_risk.py --config v2_spore_risk\configs\pipeline.yaml
```

This creates:

- `outputs/risk_predictions.csv`

## V2 Frontend

The v2 dashboard is a separate static frontend inside:

```text
v2_spore_risk/frontend
```

It is served by the same FastAPI backend and reads:

- `data/interim/image_counts.csv`
- `outputs/risk_predictions.csv`
- `models/risk_metrics.json`
- `data/raw/risk_labels.csv`

### Run the backend

```powershell
.\venv\Scripts\python.exe -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/v2
```

The dashboard fetches data from:

```text
/api/v2/dashboard
```

## Design Notes

- YOLO stays responsible only for detection.
- The second model handles final disease risk prediction.
- This separation is easier to debug and improves maintainability.
- The risk model becomes much better if you include time-based and environmental features.

## Best Practice For Your Use Case

For rice blast, use spore counts as one signal, not the only signal.

Strong predictors usually include:

- total blast-like spores
- ratio of blast spores vs other spores
- short-term count trend
- humidity
- temperature
- rainfall
- leaf wetness
- trap location and capture time

That makes this `v2` folder a good bridge between your object detector and a practical decision model.
