from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from .utils import ensure_parent_dir, list_images, slugify_label


class YOLOInferencePipeline:
    def __init__(
        self,
        weights_path: Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: str = "cpu",
    ) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.iou = iou
        self.max_det = max_det
        self.device = device

    def run(
        self,
        image_dir: Path,
        detections_csv: Path,
        counts_csv: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        images = list_images(image_dir)
        detection_rows: list[dict] = []
        count_rows: list[dict] = []
        class_columns: set[str] = set()

        for image_path in images:
            result = self.model.predict(
                source=str(image_path),
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )[0]

            image_height, image_width = result.orig_shape
            counts = Counter()
            confidences: list[float] = []
            areas_px: list[float] = []

            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(result.names[class_id])
                class_slug = slugify_label(class_name)
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                bbox_width = max(0.0, x2 - x1)
                bbox_height = max(0.0, y2 - y1)
                bbox_area = bbox_width * bbox_height

                counts[class_slug] += 1
                confidences.append(confidence)
                areas_px.append(bbox_area)
                class_columns.add(f"count__{class_slug}")

                detection_rows.append(
                    {
                        "sample_id": image_path.stem,
                        "image_name": image_path.name,
                        "image_path": str(image_path.resolve()),
                        "captured_at": datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(),
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_slug": class_slug,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "bbox_width_px": bbox_width,
                        "bbox_height_px": bbox_height,
                        "bbox_area_px": bbox_area,
                        "image_width_px": image_width,
                        "image_height_px": image_height,
                    }
                )

            count_row = {
                "sample_id": image_path.stem,
                "image_name": image_path.name,
                "image_path": str(image_path.resolve()),
                "captured_at": datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(),
                "total_count": int(sum(counts.values())),
                "hit_max_det": int(sum(counts.values()) >= self.max_det),
                "mean_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
                "max_confidence": max(confidences) if confidences else 0.0,
                "mean_bbox_area_px": float(sum(areas_px) / len(areas_px)) if areas_px else 0.0,
                "image_width_px": image_width,
                "image_height_px": image_height,
            }
            for class_slug, count in counts.items():
                count_row[f"count__{class_slug}"] = int(count)
            count_rows.append(count_row)

        detections_df = pd.DataFrame(detection_rows)
        counts_df = pd.DataFrame(count_rows)

        for column in sorted(class_columns):
            if column not in counts_df.columns:
                counts_df[column] = 0

        detection_columns = [
            "sample_id",
            "image_name",
            "image_path",
            "captured_at",
            "class_id",
            "class_name",
            "class_slug",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "bbox_width_px",
            "bbox_height_px",
            "bbox_area_px",
            "image_width_px",
            "image_height_px",
        ]
        if detections_df.empty:
            detections_df = pd.DataFrame(columns=detection_columns)
        else:
            detections_df = detections_df[detection_columns]

        ensure_parent_dir(detections_csv)
        ensure_parent_dir(counts_csv)
        detections_df.to_csv(detections_csv, index=False)
        counts_df.to_csv(counts_csv, index=False)
        return detections_df, counts_df
