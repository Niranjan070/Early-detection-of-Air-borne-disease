from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spore_risk_v2.config import load_config, resolve_path
from spore_risk_v2.inference import YOLOInferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference and extract spore counts.")
    parser.add_argument("--config", required=True, help="Path to the v2 pipeline YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    yolo_cfg = config["yolo"]

    pipeline = YOLOInferencePipeline(
        weights_path=resolve_path(config, yolo_cfg["weights"]),
        confidence=float(yolo_cfg.get("confidence", 0.25)),
        iou=float(yolo_cfg.get("iou", 0.45)),
        max_det=int(yolo_cfg.get("max_det", 300)),
        device=str(yolo_cfg.get("device", "cpu")),
    )

    image_dir = resolve_path(config, paths["image_dir"])
    detections_csv = resolve_path(config, paths["detections_csv"])
    counts_csv = resolve_path(config, paths["counts_csv"])

    detections_df, counts_df = pipeline.run(
        image_dir=image_dir,
        detections_csv=detections_csv,
        counts_csv=counts_csv,
    )

    print(f"Images processed: {len(counts_df)}")
    print(f"Detections saved to: {detections_csv}")
    print(f"Counts saved to: {counts_csv}")
    if detections_df.empty:
        print("No detections were produced. Check the weights, thresholds, or image quality.")


if __name__ == "__main__":
    main()
