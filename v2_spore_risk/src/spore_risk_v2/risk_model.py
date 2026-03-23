from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .utils import choose_join_key, ensure_parent_dir


def train_risk_model(
    feature_table_csv: Path,
    labels_csv: Path,
    model_output: Path,
    metrics_output: Path,
    target_column: str,
    join_key_priority: list[str],
    drop_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
) -> dict[str, Any]:
    features_df = pd.read_csv(feature_table_csv)
    labels_df = pd.read_csv(labels_csv)

    if target_column not in labels_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in labels CSV.")

    join_key = choose_join_key(features_df, labels_df, join_key_priority)
    labels_subset = labels_df[[join_key, target_column]].drop_duplicates()
    dataset = features_df.merge(labels_subset, on=join_key, how="inner")
    dataset = dataset.dropna(subset=[target_column]).copy()
    if dataset.empty:
        raise ValueError("No labeled rows matched between the feature table and labels CSV.")

    excluded_columns = set(drop_columns)
    excluded_columns.add(target_column)
    excluded_columns.add(join_key)
    candidate_feature_columns = [column for column in dataset.columns if column not in excluded_columns]

    X = dataset[candidate_feature_columns].copy()
    y = dataset[target_column].astype(str)

    numeric_columns = [column for column in X.columns if is_numeric_dtype(X[column])]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "join_key": join_key,
        "target_column": target_column,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": sorted(y.unique().tolist()),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=sorted(y.unique())).tolist(),
        "top_feature_importance": _top_feature_importance(model, limit=20),
    }

    artifact = {
        "model": model,
        "join_key": join_key,
        "target_column": target_column,
        "feature_columns": candidate_feature_columns,
    }

    ensure_parent_dir(model_output)
    ensure_parent_dir(metrics_output)
    with model_output.open("wb") as file_obj:
        pickle.dump(artifact, file_obj)
    with metrics_output.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    return metrics


def score_risk(
    feature_table_csv: Path,
    model_output: Path,
    predictions_output: Path,
) -> pd.DataFrame:
    feature_df = pd.read_csv(feature_table_csv)
    with model_output.open("rb") as file_obj:
        artifact = pickle.load(file_obj)

    model: Pipeline = artifact["model"]
    feature_columns = artifact["feature_columns"]
    scoring_frame = feature_df.reindex(columns=feature_columns, fill_value=0)
    predictions = model.predict(scoring_frame)
    prediction_frame = _prediction_frame(model, scoring_frame, predictions)

    base_columns = [column for column in ["sample_id", "image_name", "captured_at"] if column in feature_df.columns]
    output_df = pd.concat([feature_df[base_columns].reset_index(drop=True), prediction_frame.reset_index(drop=True)], axis=1)

    ensure_parent_dir(predictions_output)
    output_df.to_csv(predictions_output, index=False)
    return output_df


def _prediction_frame(model: Pipeline, X: pd.DataFrame, predictions: Any) -> pd.DataFrame:
    prediction_frame = pd.DataFrame({"predicted_risk": predictions})
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(X)
        for index, class_name in enumerate(classifier.classes_):
            prediction_frame[f"probability__{class_name}"] = probabilities[:, index]
    return prediction_frame


def _top_feature_importance(model: Pipeline, limit: int = 20) -> list[dict[str, float]]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    classifier: RandomForestClassifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return []

    feature_names = preprocessor.get_feature_names_out()
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    )
    importance_frame = importance_frame.sort_values("importance", ascending=False).head(limit)
    return [
        {
            "feature": str(row["feature"]),
            "importance": float(row["importance"]),
        }
        for _, row in importance_frame.iterrows()
    ]
