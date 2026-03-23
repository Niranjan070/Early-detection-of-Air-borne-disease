from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spore_risk_v2.config import load_config, resolve_path
from spore_risk_v2.risk_model import train_risk_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v2 blast risk model.")
    parser.add_argument("--config", required=True, help="Path to the v2 pipeline YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("risk_model", {})
    paths = config["paths"]

    labels_csv = resolve_path(config, paths["labels_csv"])
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels file not found at {labels_csv}. Create it from templates/risk_labels_template.csv first."
        )

    metrics = train_risk_model(
        feature_table_csv=resolve_path(config, paths["feature_table_csv"]),
        labels_csv=labels_csv,
        model_output=resolve_path(config, paths["model_output"]),
        metrics_output=resolve_path(config, paths["metrics_output"]),
        target_column=str(model_cfg.get("target_column", "blast_risk_label")),
        join_key_priority=list(model_cfg.get("join_key_priority", ["sample_id", "image_name"])),
        drop_columns=list(model_cfg.get("drop_columns", [])),
        test_size=float(model_cfg.get("test_size", 0.2)),
        random_state=int(model_cfg.get("random_state", 42)),
        n_estimators=int(model_cfg.get("n_estimators", 300)),
    )

    print(f"Model saved to: {resolve_path(config, paths['model_output'])}")
    print(f"Metrics saved to: {resolve_path(config, paths['metrics_output'])}")
    print(f"Accuracy: {metrics.get('accuracy', 'n/a')}")


if __name__ == "__main__":
    main()
