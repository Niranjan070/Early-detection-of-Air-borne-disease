from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spore_risk_v2.config import load_config, resolve_path
from spore_risk_v2.features import build_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v2 risk feature table.")
    parser.add_argument("--config", required=True, help="Path to the v2 pipeline YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    features_cfg = config.get("features", {})

    metadata_csv = resolve_path(config, paths["metadata_csv"])
    if not metadata_csv.exists():
        metadata_csv = None

    feature_df = build_feature_table(
        counts_csv=resolve_path(config, paths["counts_csv"]),
        detections_csv=resolve_path(config, paths["detections_csv"]),
        feature_table_csv=resolve_path(config, paths["feature_table_csv"]),
        metadata_csv=metadata_csv,
        timestamp_column=str(features_cfg.get("timestamp_column", "captured_at")),
        preferred_group_columns=list(features_cfg.get("preferred_group_columns", [])),
        area_column_candidates=list(features_cfg.get("area_column_candidates", [])),
        rolling_window=int(features_cfg.get("rolling_window", 3)),
    )

    print(f"Feature rows written: {len(feature_df)}")
    print(f"Feature table saved to: {resolve_path(config, paths['feature_table_csv'])}")


if __name__ == "__main__":
    main()
