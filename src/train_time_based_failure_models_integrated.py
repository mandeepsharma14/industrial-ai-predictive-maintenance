
from __future__ import annotations

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "industrial_time_series_dataset_v2.csv",
    PROJECT_ROOT / "data" / "industrial_time_series_dataset.csv",
    PROJECT_ROOT / "industrial_time_series_dataset_v2.csv",
    PROJECT_ROOT / "industrial_time_series_dataset.csv",
]

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = MODELS_DIR / "model_training_summary.json"
STANDARDIZED_DATA_PATH = MODELS_DIR / "standardized_latest_snapshot.csv"

def resolve_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Put industrial_time_series_dataset_v2.csv in the project's data/ folder or project root."
    )

def load_raw_dataset() -> pd.DataFrame:
    path = resolve_data_path()
    return pd.read_csv(path, parse_dates=["timestamp"])

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        # expected simplified schema
        "Temperature_C": "temperature_c",
        "temperature_C": "temperature_c",
        "Motor_Load_pct": "motor_load_pct",
        "Tool_Wear_hrs": "tool_wear_hrs",
        "Hours_Since_Maintenance": "hours_since_maintenance",
        "Dust_Collector_DP_kPa": "dust_collector_dp_kpa",
        "Servo_Current_A": "servo_current_a",
        "Vacuum_Pressure_kPa": "vacuum_pressure_kpa",
        "Failure": "failure",
        "Machine_ID": "machine_id",
        "Vibration_mm_s": "vibration_mm_s",
        "Voltage_V": "voltage_v",
        "Pressure_bar": "pressure_bar",
        "RPM": "rpm",
        "Encoder_Error_Count": "encoder_error_count",
        # alternative detailed schema to simplified schema
        "bearing_temp_f": "temperature_c",
        "motor_load_pct": "motor_load_pct",
        "tool_wear_hours": "tool_wear_hrs",
        "days_since_maintenance": "hours_since_maintenance",
        "dust_collector_dp_inwc": "dust_collector_dp_kpa",
        "hydraulic_pressure_psi": "pressure_bar",
        "lubrication_score": "lubrication_score",
        "vacuum_pressure_kpa": "vacuum_pressure_kpa",
    }
    out = df.rename(columns=rename_map).copy()

    if "temperature_c" not in out.columns and "bearing_temp_f" in df.columns:
        out["temperature_c"] = (df["bearing_temp_f"] - 32.0) * 5.0 / 9.0
    if "pressure_bar" not in out.columns and "hydraulic_pressure_psi" in df.columns:
        out["pressure_bar"] = df["hydraulic_pressure_psi"] / 14.5038
    if "dust_collector_dp_kpa" not in out.columns and "dust_collector_dp_inwc" in df.columns:
        out["dust_collector_dp_kpa"] = df["dust_collector_dp_inwc"] * 0.249089
    if "hours_since_maintenance" not in out.columns and "days_since_maintenance" in df.columns:
        out["hours_since_maintenance"] = df["days_since_maintenance"] * 24.0
    if "tool_wear_hrs" not in out.columns and "tool_wear_hours" in df.columns:
        out["tool_wear_hrs"] = df["tool_wear_hours"]

    # fill expected optional columns
    if "lubrication_score" not in out.columns:
        # build a plausible proxy from degradation if absent
        vib = out.get("vibration_mm_s", pd.Series(np.nan, index=out.index)).fillna(
            out.get("Vibration_mm_s", pd.Series(5.0, index=out.index))
        )
        temp = out.get("temperature_c", pd.Series(70.0, index=out.index))
        out["lubrication_score"] = np.clip(100 - (vib * 7 + (temp - temp.median()).clip(lower=0) * 0.8), 5, 100)

    # derived forward labels if missing
    if "failure" not in out.columns:
        out["failure"] = 0

    # sort first for rolling/forward labels
    out = out.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    if "failure_next_7d" not in out.columns:
        out["failure_next_7d"] = 0
    if "failure_next_30d" not in out.columns:
        out["failure_next_30d"] = 0
    if "failure_next_90d" not in out.columns:
        out["failure_next_90d"] = 0
    if "rul_days" not in out.columns:
        out["rul_days"] = np.nan

    # derive from actual failure events if labels absent/empty
    labels_missing = (
        out[["failure_next_7d", "failure_next_30d", "failure_next_90d"]].sum().sum() == 0
        and out["failure"].sum() > 0
    )
    if labels_missing or "days_to_failure" in out.columns:
        out["failure_next_7d"] = 0
        out["failure_next_30d"] = 0
        out["failure_next_90d"] = 0
        out["rul_days"] = np.nan
        for machine_id, g in out.groupby("machine_id", sort=False):
            idx = g.index.to_list()
            fail_ts = g.loc[g["failure"] == 1, "timestamp"].tolist()
            if not fail_ts:
                continue
            ts = out.loc[idx, "timestamp"]
            future_days = []
            f7, f30, f90 = [], [], []
            for t in ts:
                future = [ft for ft in fail_ts if ft >= t]
                if future:
                    dt = (future[0] - t).days
                else:
                    dt = np.nan
                future_days.append(dt)
                f7.append(int(dt <= 7))
                f30.append(int(dt <= 30))
                f90.append(int(dt <= 90))
            out.loc[idx, "rul_days"] = future_days
            out.loc[idx, "failure_next_7d"] = f7
            out.loc[idx, "failure_next_30d"] = f30
            out.loc[idx, "failure_next_90d"] = f90

    # keep only sane range for known RUL values
    out["rul_days"] = out["rul_days"].where(out["rul_days"].isna(), out["rul_days"].clip(lower=0, upper=365))
    return out

FEATURES = [
    "vibration_mm_s",
    "temperature_c",
    "pressure_bar",
    "voltage_v",
    "rpm",
    "tool_wear_hrs",
    "motor_load_pct",
    "hours_since_maintenance",
    "dust_collector_dp_kpa",
    "encoder_error_count",
    "servo_current_a",
    "vacuum_pressure_kpa",
    "lubrication_score",
]

TARGETS = ["failure_next_7d", "failure_next_30d", "failure_next_90d"]

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in FEATURES if c in df.columns and df[c].dtype == "object"]
    numeric = [c for c in FEATURES if c in df.columns and c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )

def build_classifiers(preprocessor: ColumnTransformer) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))
        ]),
        "decision_tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=7, min_samples_leaf=20, class_weight="balanced", random_state=42))
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=260, max_depth=10, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42))
        ]),
    }

def build_rul_regressor(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

def temporal_split(df: pd.DataFrame, test_days: int = 45):
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff].copy(), df[df["timestamp"] > cutoff].copy()

def threshold_from_f1(y_true: pd.Series, prob: np.ndarray) -> float:
    thresholds = np.linspace(0.10, 0.80, 71)
    best_thr, best_f1 = 0.50, -1.0
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = score, float(thr)
    return best_thr

def evaluate_and_save(df: pd.DataFrame) -> dict:
    train_df, test_df = temporal_split(df)
    preprocessor = build_preprocessor(df)
    summary = {
        "dataset_path": str(resolve_data_path()),
        "rows": int(len(df)),
        "machines": int(df["machine_id"].nunique()),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "classification": {},
        "best_models": {},
        "rul": {},
    }

    X_train, X_test = train_df[FEATURES], test_df[FEATURES]

    for target in TARGETS:
        rows = []
        best = None
        best_f1 = -1.0
        for name, pipeline in build_classifiers(preprocessor).items():
            pipeline.fit(X_train, train_df[target])
            prob = pipeline.predict_proba(X_test)[:, 1]
            thr = threshold_from_f1(test_df[target], prob)
            pred = (prob >= thr).astype(int)
            row = {
                "model_key": name,
                "accuracy": float(accuracy_score(test_df[target], pred)),
                "precision": float(precision_score(test_df[target], pred, zero_division=0)),
                "recall": float(recall_score(test_df[target], pred, zero_division=0)),
                "f1": float(f1_score(test_df[target], pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(test_df[target], prob)) if test_df[target].nunique() > 1 else None,
                "pr_auc": float(average_precision_score(test_df[target], prob)),
                "threshold": thr,
                "base_rate": float(test_df[target].mean()),
                "positive_events_test": int(test_df[target].sum()),
            }
            rows.append(row)
            joblib.dump(pipeline, MODELS_DIR / f"{target}__{name}.joblib")
            if row["f1"] > best_f1:
                best_f1 = row["f1"]
                best = row

        rows = sorted(rows, key=lambda x: (x["f1"], x["recall"], x["precision"]), reverse=True)
        summary["classification"][target] = rows
        summary["best_models"][target] = rows[0]

    perf_rows = []
    for target, rows in summary["classification"].items():
        horizon_match = re.search(r"(\d+)", target)
        horizon_days = int(horizon_match.group(1)) if horizon_match else None
        for r in rows:
            perf_rows.append({
                "target": target,
                "horizon_days": horizon_days,
                "model": r["model_key"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1"],
                "roc_auc": r["roc_auc"],
                "pr_auc": r["pr_auc"],
                "threshold": r["threshold"],
                "base_rate": r["base_rate"],
                "positive_events_test": r["positive_events_test"],
            })
    pd.DataFrame(perf_rows).to_csv(MODELS_DIR / "model_performance_summary.csv", index=False)

    rul_model = build_rul_regressor(preprocessor)
    rul_train = train_df.dropna(subset=["rul_days"]).copy()
    rul_test = test_df.dropna(subset=["rul_days"]).copy()

    if len(rul_train) == 0:
        summary["rul"] = {
            "model_key": "gradient_boosting_regressor",
            "mae_days": None,
            "train_rows": 0,
            "test_rows": int(len(rul_test)),
            "note": "No rows with known RUL were available in the training window."
        }
    else:
        X_rul_train = rul_train[FEATURES]
        y_rul_train = rul_train["rul_days"]
        rul_model.fit(X_rul_train, y_rul_train)
        joblib.dump(rul_model, MODELS_DIR / "rul_regressor.joblib")

        if len(rul_test) == 0:
            summary["rul"] = {
                "model_key": "gradient_boosting_regressor",
                "mae_days": None,
                "train_rows": int(len(rul_train)),
                "test_rows": 0,
                "note": "No rows with known RUL were available in the test window."
            }
        else:
            X_rul_test = rul_test[FEATURES]
            y_rul_test = rul_test["rul_days"]
            rul_pred = np.clip(rul_model.predict(X_rul_test), 0, 365)
            summary["rul"] = {
                "model_key": "gradient_boosting_regressor",
                "mae_days": float(mean_absolute_error(y_rul_test, rul_pred)),
                "train_rows": int(len(rul_train)),
                "test_rows": int(len(rul_test))
            }

    # Save most recent snapshot for app convenience
    latest = df.sort_values("timestamp").groupby("machine_id").tail(1).copy()
    latest.to_csv(STANDARDIZED_DATA_PATH, index=False)

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    raw = load_raw_dataset()
    df = standardize_columns(raw)

    required = ["timestamp", "machine_id"] + FEATURES + TARGETS + ["rul_days"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is still missing required standardized columns: {missing}")

    summary = evaluate_and_save(df)
    print("Saved model artifacts to:", MODELS_DIR)
    print(json.dumps(summary["best_models"], indent=2))
    print("RUL:", summary["rul"])

if __name__ == "__main__":
    main()
