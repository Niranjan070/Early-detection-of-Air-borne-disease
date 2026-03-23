from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spore_risk_v2.config import load_config, resolve_path
from spore_risk_v2.risk_model import score_risk


def main() -> None:
    parser = argparse.ArgumentParser(description="Score risk for new samples using the trained v2 model.")
    parser.add_argument("--config", required=True, help="Path to the v2 pipeline YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    predictions_df = score_risk(
        feature_table_csv=resolve_path(config, paths["feature_table_csv"]),
        model_output=resolve_path(config, paths["model_output"]),
        predictions_output=resolve_path(config, paths["predictions_output"]),
    )

    print(f"Prediction rows written: {len(predictions_df)}")
    print(f"Predictions saved to: {resolve_path(config, paths['predictions_output'])}")


if __name__ == "__main__":
    main()
