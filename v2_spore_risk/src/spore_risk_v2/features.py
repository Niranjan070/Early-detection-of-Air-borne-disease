from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import choose_join_key, ensure_parent_dir, load_optional_csv, pick_first_present


def build_feature_table(
    counts_csv: Path,
    detections_csv: Path,
    feature_table_csv: Path,
    metadata_csv: Path | None = None,
    timestamp_column: str = "captured_at",
    preferred_group_columns: list[str] | None = None,
    area_column_candidates: list[str] | None = None,
    rolling_window: int = 3,
) -> pd.DataFrame:
    counts_df = pd.read_csv(counts_csv)
    detections_df = pd.read_csv(detections_csv)
    metadata_df = load_optional_csv(metadata_csv)

    if counts_df.empty:
        raise ValueError("Counts table is empty. Run the inference stage first.")

    feature_df = counts_df.copy()
    count_columns = sorted(column for column in feature_df.columns if column.startswith("count__"))
    if "total_count" not in feature_df.columns:
        feature_df["total_count"] = feature_df[count_columns].sum(axis=1)

    if count_columns:
        feature_df["non_zero_class_count"] = (feature_df[count_columns] > 0).sum(axis=1)
        feature_df["dominant_class"] = feature_df[count_columns].idxmax(axis=1).str.replace("count__", "", regex=False)
        feature_df["dominant_class_count"] = feature_df[count_columns].max(axis=1)
    else:
        feature_df["non_zero_class_count"] = 0
        feature_df["dominant_class"] = "none"
        feature_df["dominant_class_count"] = 0
    feature_df["dominant_class_ratio"] = np.where(
        feature_df["total_count"] > 0,
        feature_df["dominant_class_count"] / feature_df["total_count"],
        0.0,
    )
    feature_df["image_area_px"] = feature_df["image_width_px"] * feature_df["image_height_px"]
    feature_df["detections_per_megapixel"] = np.where(
        feature_df["image_area_px"] > 0,
        feature_df["total_count"] / (feature_df["image_area_px"] / 1_000_000.0),
        0.0,
    )

    for column in count_columns:
        ratio_column = column.replace("count__", "ratio__")
        feature_df[ratio_column] = np.where(
            feature_df["total_count"] > 0,
            feature_df[column] / feature_df["total_count"],
            0.0,
        )

    if not detections_df.empty:
        bbox_stats = (
            detections_df.groupby("sample_id")
            .agg(
                mean_detection_confidence=("confidence", "mean"),
                std_detection_confidence=("confidence", "std"),
                mean_detection_area_px=("bbox_area_px", "mean"),
                max_detection_area_px=("bbox_area_px", "max"),
            )
            .reset_index()
            .fillna(0.0)
        )
        feature_df = feature_df.merge(bbox_stats, on="sample_id", how="left")

    if metadata_df is not None and not metadata_df.empty:
        join_key = choose_join_key(feature_df, metadata_df, ["sample_id", "image_name"])
        feature_df = feature_df.merge(metadata_df, on=join_key, how="left")

    area_column = pick_first_present(area_column_candidates or [], feature_df)
    if area_column is not None:
        feature_df["spores_per_mm2"] = np.where(
            feature_df[area_column].fillna(0) > 0,
            feature_df["total_count"] / feature_df[area_column],
            0.0,
        )
        for column in count_columns:
            density_column = column.replace("count__", "density__")
            feature_df[density_column] = np.where(
                feature_df[area_column].fillna(0) > 0,
                feature_df[column] / feature_df[area_column],
                0.0,
            )

    if timestamp_column in feature_df.columns:
        feature_df[timestamp_column] = pd.to_datetime(feature_df[timestamp_column], errors="coerce")
        group_column = pick_first_present(preferred_group_columns or [], feature_df)
        sort_columns = [timestamp_column]
        if group_column is not None:
            sort_columns = [group_column, timestamp_column]
        feature_df = feature_df.sort_values(sort_columns).reset_index(drop=True)
        feature_df = _add_temporal_features(
            feature_df=feature_df,
            count_columns=["total_count", *count_columns],
            timestamp_column=timestamp_column,
            group_column=group_column,
            rolling_window=rolling_window,
        )
        feature_df[timestamp_column] = feature_df[timestamp_column].dt.strftime("%Y-%m-%dT%H:%M:%S")

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns
    feature_df[numeric_columns] = feature_df[numeric_columns].fillna(0)

    ensure_parent_dir(feature_table_csv)
    feature_df.to_csv(feature_table_csv, index=False)
    return feature_df


def _add_temporal_features(
    feature_df: pd.DataFrame,
    count_columns: list[str],
    timestamp_column: str,
    group_column: str | None,
    rolling_window: int,
) -> pd.DataFrame:
    previous_timestamp = (
        feature_df.groupby(group_column)[timestamp_column].shift(1)
        if group_column is not None
        else feature_df[timestamp_column].shift(1)
    )
    elapsed_hours = (feature_df[timestamp_column] - previous_timestamp).dt.total_seconds().div(3600)
    feature_df["hours_since_previous_sample"] = elapsed_hours.fillna(0).clip(lower=0)

    for column in count_columns:
        if group_column is not None:
            feature_df[f"prev__{column}"] = feature_df.groupby(group_column)[column].shift(1).fillna(0)
            rolling_series = (
                feature_df.groupby(group_column)[column]
                .rolling(window=rolling_window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            feature_df[f"prev__{column}"] = feature_df[column].shift(1).fillna(0)
            rolling_series = feature_df[column].rolling(window=rolling_window, min_periods=1).mean()

        feature_df[f"delta__{column}"] = feature_df[column] - feature_df[f"prev__{column}"]
        feature_df[f"rolling_mean__{column}"] = rolling_series.fillna(0)
        feature_df[f"growth_ratio__{column}"] = np.where(
            feature_df[f"prev__{column}"] > 0,
            feature_df[column] / feature_df[f"prev__{column}"],
            0.0,
        )

    return feature_df
