
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "industrial_time_series_dataset_v2.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

FEATURES = [
    "machine_type","shift","machine_age_years","duty_class","ambient_temp_f","humidity_pct",
    "operating_hours","days_since_maintenance","tool_wear_hours","rpm","motor_load_pct",
    "voltage_v","hydraulic_pressure_psi","lubrication_score","vibration_mm_s","bearing_temp_f",
    "dust_collector_dp_inwc","servo_current_a","vacuum_pressure_kpa","encoder_error_count",
]
TARGET = "failure_next_30d"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values(["machine_id","timestamp"]).reset_index(drop=True)
    return df

def temporal_split(df, test_days=45):
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff].copy(), df[df["timestamp"] > cutoff].copy()

def build_preprocessor(df):
    categorical = [c for c in FEATURES if df[c].dtype == "object"]
    numeric = [c for c in FEATURES if c not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

def train_failure_model(train_df, test_df):
    pre = build_preprocessor(train_df)
    model = Pipeline([
        ("preprocessor", pre),
        ("model", GradientBoostingClassifier(random_state=42))
    ])
    model.fit(train_df[FEATURES], train_df[TARGET])
    prob = model.predict_proba(test_df[FEATURES])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(test_df[TARGET], prob)),
        "pr_auc": float(average_precision_score(test_df[TARGET], prob)),
        "avg_probability": float(prob.mean()),
        "positive_events_test": int(test_df[TARGET].sum()),
        "records_test": int(len(test_df)),
    }
    joblib.dump(model, MODELS_DIR / "advanced_failure_30d_gb.joblib")
    return model, metrics

def train_anomaly_model(df):
    num_cols = [c for c in FEATURES if df[c].dtype != "object"]
    X = df[num_cols].copy().fillna(df[num_cols].median(numeric_only=True))
    model = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    score = -model.fit_predict(X)
    out = df[["timestamp","machine_id"] + num_cols].copy()
    out["anomaly_score"] = score
    out["anomaly_flag"] = (out["anomaly_score"] > 0).astype(int)
    out.to_csv(MODELS_DIR / "advanced_anomaly_scores.csv", index=False)
    joblib.dump(model, MODELS_DIR / "advanced_isolation_forest.joblib")
    return {
        "rows_scored": int(len(out)),
        "anomaly_flags": int(out["anomaly_flag"].sum()),
    }

def main():
    df = load_data()
    train_df, test_df = temporal_split(df)
    _, fail_metrics = train_failure_model(train_df, test_df)
    anomaly_metrics = train_anomaly_model(df)
    summary = {
        "dataset_rows": int(len(df)),
        "machines": int(df["machine_id"].nunique()),
        "failure_model": fail_metrics,
        "anomaly_model": anomaly_metrics,
    }
    (MODELS_DIR / "advanced_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
