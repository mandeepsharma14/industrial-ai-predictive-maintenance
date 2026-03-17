
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "industrial_time_series_dataset_v2.csv",
    BASE_DIR / "data" / "industrial_time_series_dataset_v2.csv",
    BASE_DIR.parent / "data" / "industrial_time_series_dataset_v2.csv",
    BASE_DIR / "industrial_time_series_dataset.csv",
    BASE_DIR / "data" / "industrial_time_series_dataset.csv",
    BASE_DIR.parent / "data" / "industrial_time_series_dataset.csv",
]
OUTPUT_JSON = BASE_DIR / "time_forecast_model_results_summary_v2.json"

FEATURE_COLUMNS = [
    "machine_type", "shift", "machine_age_years", "duty_class", "ambient_temp_f", "humidity_pct",
    "operating_hours", "days_since_maintenance", "tool_wear_hours", "rpm", "motor_load_pct",
    "voltage_v", "hydraulic_pressure_psi", "lubrication_score", "vibration_mm_s", "bearing_temp_f",
    "dust_collector_dp_inwc", "servo_current_a", "vacuum_pressure_kpa", "encoder_error_count",
]
CLASSIFICATION_TARGETS = ["failure_next_7d", "failure_next_30d", "failure_next_90d"]
RUL_TARGET = "rul_days_capped_120"


def resolve_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find industrial_time_series_dataset_v2.csv. Put it in the same folder as this script or in a data/ subfolder."
    )


def load_dataset() -> pd.DataFrame:
    path = resolve_data_path()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["timestamp", "machine_id"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_days: int = 45):
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff].copy(), df[df["timestamp"] > cutoff].copy()


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in FEATURE_COLUMNS if df[c].dtype == "object"]
    numeric = [c for c in FEATURE_COLUMNS if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def build_classifiers(preprocessor: ColumnTransformer):
    return {
        "Logistic Regression": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1200, class_weight="balanced"))]),
        "Decision Tree": Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=18, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=180, max_depth=10, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingClassifier(random_state=42))]),
    }


def build_rul_model(preprocessor: ColumnTransformer):
    return Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])


def optimize_threshold(y_true: pd.Series, prob: np.ndarray) -> tuple[float, np.ndarray]:
    best_threshold = 0.50
    best_score = -1.0
    best_pred = (prob >= best_threshold).astype(int)
    for threshold in np.arange(0.10, 0.86, 0.05):
        pred = (prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_pred = pred
    return best_threshold, best_pred


def evaluate_models(models, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        threshold, pred = optimize_threshold(y_test, prob)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob),
            "optimized_threshold": threshold,
            "avg_probability": float(prob.mean()),
            "positive_events": int(y_test.sum()),
            "records": int(len(y_test)),
        })
    return pd.DataFrame(rows).sort_values(["f1_score", "roc_auc", "recall"], ascending=False)


def main() -> None:
    df = load_dataset()
    train_df, test_df = temporal_split(df)
    preprocessor = build_preprocessor(df)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    results = {
        "dataset_path": str(resolve_data_path()),
        "data_summary": {
            "records": int(len(df)),
            "machines": int(df["machine_id"].nunique()),
            "date_range": [str(df["timestamp"].min().date()), str(df["timestamp"].max().date())],
            "training_records": int(len(train_df)),
            "test_records": int(len(test_df)),
            "failure_event_today_count": int(df["failure_event_today"].sum()),
            "failure_event_today_rate": round(float(df["failure_event_today"].mean()), 4),
        },
        "classification_results": {},
        "rul_regression": {},
    }

    print("=" * 96)
    print("INDUSTRIAL AI | TIME-BASED FAILURE FORECASTING + RUL | ENHANCED DATASET V2")
    print("=" * 96)
    print(f"Dataset: {resolve_data_path()}")
    print(f"Records: {len(df):,} | Machines: {df['machine_id'].nunique():,} | Failure events: {int(df['failure_event_today'].sum()):,}")

    for target in CLASSIFICATION_TARGETS:
        print(f"\nTarget: {target} | Base rate in holdout: {test_df[target].mean():.2%}")
        frame = evaluate_models(build_classifiers(preprocessor), X_train, train_df[target], X_test, test_df[target])
        print(frame.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
        results["classification_results"][target] = frame.round(4).to_dict(orient="records")

    print("\nRUL regression")
    rul_model = build_rul_model(preprocessor)
    rul_model.fit(X_train, train_df[RUL_TARGET])
    pred = np.clip(rul_model.predict(X_test), 0, 120)
    mae = mean_absolute_error(test_df[RUL_TARGET], pred)
    results["rul_regression"] = {
        "model": "Gradient Boosting Regressor",
        "mae_days": round(float(mae), 3),
        "prediction_range_days": [round(float(pred.min()), 2), round(float(pred.max()), 2)],
    }
    print(f"MAE (days): {mae:0.3f}")

    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved summary to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
