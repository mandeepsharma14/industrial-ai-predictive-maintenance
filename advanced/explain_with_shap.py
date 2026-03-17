
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "industrial_time_series_dataset_v2.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "advanced_failure_30d_gb.joblib"
OUT_PATH = PROJECT_ROOT / "models" / "advanced_shap_summary.png"

FEATURES = [
    "machine_type","shift","machine_age_years","duty_class","ambient_temp_f","humidity_pct",
    "operating_hours","days_since_maintenance","tool_wear_hours","rpm","motor_load_pct",
    "voltage_v","hydraulic_pressure_psi","lubrication_score","vibration_mm_s","bearing_temp_f",
    "dust_collector_dp_inwc","servo_current_a","vacuum_pressure_kpa","encoder_error_count",
]

def main():
    try:
        import shap
    except Exception:
        print("shap is not installed. Install requirements-advanced.txt first.")
        return

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values(["timestamp","machine_id"])
    model = joblib.load(MODEL_PATH)
    sample = df[FEATURES].tail(250).copy()

    transformed = model.named_steps["preprocessor"].transform(sample)
    estimator = model.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, transformed, show=False)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved SHAP summary to {OUT_PATH}")

if __name__ == "__main__":
    main()
