
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

st.set_page_config(page_title="Industrial AI Factory Control Room", page_icon="🏭", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "industrial_time_series_dataset_v2.csv"
MODEL_DIR = PROJECT_ROOT / "models"

CANONICAL_FEATURES = [
    "machine_age_years","ambient_temp_f","humidity_pct","operating_hours",
    "hours_since_maintenance","tool_wear_hrs","rpm","motor_load_pct","voltage_v",
    "pressure_bar","lubrication_score","vibration_mm_s","temperature_c",
    "dust_collector_dp_kpa","servo_current_a","vacuum_pressure_kpa","encoder_error_count",
]

RENAME_MAP = {
    "bearing_temp_f": "temperature_c",
    "hydraulic_pressure_psi": "pressure_bar",
    "dust_collector_dp_inwc": "dust_collector_dp_kpa",
    "days_since_maintenance": "hours_since_maintenance",
    "tool_wear_hours": "tool_wear_hrs",
}

MODEL_OPTIONS = {
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
}

def inject_styles():
    st.markdown(
        """
        <style>
            .block-container {padding-top:1rem; padding-bottom:1rem; max-width:96rem;}
            div[data-testid="stMetricValue"] {font-size:1.6rem;}
            .hero {
                background: linear-gradient(120deg, #0b132b 0%, #1d4ed8 45%, #0f172a 100%);
                padding: 1.25rem 1.35rem;
                border-radius: 18px;
                color: white;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
                margin-bottom: 1rem;
            }
            .subhero {color: rgba(255,255,255,0.88); margin-top: 0.35rem; font-size: 0.98rem;}
            .legend-row {display:flex; gap:0.8rem; flex-wrap:wrap; margin-top:0.4rem; margin-bottom:0.75rem;}
            .legend-chip {padding:0.25rem 0.65rem; border-radius:999px; color:white; font-size:0.8rem; font-weight:700;}
            .glass {background:#f8fafc; border:1px solid #e2e8f0; border-radius:16px; padding:1rem; box-shadow:0 4px 16px rgba(15,23,42,0.05);}
            .footer-note {color:#64748b; font-size:0.84rem; margin-top:1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        st.error("Dataset not found. Please place industrial_time_series_dataset_v2.csv inside the data folder.")
        st.stop()

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.rename(columns=RENAME_MAP)

    required = {"timestamp", "machine_id", "machine_type", "shift"}
    missing = sorted(required - set(df.columns))
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.write("Found columns:", list(df.columns))
        st.stop()

    for col in CANONICAL_FEATURES:
        if col not in df.columns:
            if col == "lubrication_score":
                df[col] = 50.0
            elif col == "encoder_error_count":
                df[col] = 0
            else:
                df[col] = 0.0

    for col in ["failure_next_7d", "failure_next_30d", "failure_next_90d", "failure_event_today"]:
        if col not in df.columns:
            df[col] = 0
    if "rul_days" not in df.columns:
        df["rul_days"] = np.nan

    return df

@st.cache_resource
def load_models():
    models = {}
    for horizon in [7, 30, 90]:
        for name in MODEL_OPTIONS.values():
            path = MODEL_DIR / f"{name}_{horizon}d.joblib"
            if path.exists():
                models[f"{name}_{horizon}d"] = joblib.load(path)

    rul_path = MODEL_DIR / "rul_regressor.joblib"
    if rul_path.exists():
        models["rul_regressor"] = joblib.load(rul_path)

    perf_path = MODEL_DIR / "model_performance_summary.csv"
    perf = pd.read_csv(perf_path) if perf_path.exists() else None
    return models, perf

def normalize_score(series, invert=False):
    s = series.astype(float)
    if s.max() == s.min():
        out = pd.Series(np.full(len(s), 50.0), index=s.index)
    else:
        out = (s - s.min()) / (s.max() - s.min()) * 100.0
    if invert:
        out = 100.0 - out
    return out.clip(0, 100)

def compute_health_index(df):
    vib = normalize_score(df["vibration_mm_s"], True)
    temp = normalize_score(df["temperature_c"], True)
    load = normalize_score(df["motor_load_pct"], True)
    servo = normalize_score(df["servo_current_a"], True)
    dust = normalize_score(df["dust_collector_dp_kpa"], True)
    wear = normalize_score(df["tool_wear_hrs"], True)
    lube = normalize_score(df["lubrication_score"], False)
    maint = normalize_score(df["hours_since_maintenance"], True)
    health = 0.18*vib + 0.18*temp + 0.14*load + 0.12*servo + 0.10*dust + 0.10*wear + 0.10*lube + 0.08*maint
    return health.clip(0, 100)

def risk_band(risk_30d, rul_days):
    if risk_30d >= 0.70 or rul_days <= 10:
        return "Critical"
    if risk_30d >= 0.45 or rul_days <= 20:
        return "High"
    if risk_30d >= 0.25 or rul_days <= 35:
        return "Watch"
    return "Normal"

def risk_color(label):
    return {"Normal":"#16a34a","Watch":"#eab308","High":"#f97316","Critical":"#dc2626"}.get(label, "#334155")

def predict_latest(df_latest, models, selected_model_name):
    model_key = MODEL_OPTIONS[selected_model_name]
    X = df_latest[CANONICAL_FEATURES].copy()

    for horizon in [7, 30, 90]:
        model = models.get(f"{model_key}_{horizon}d")
        col = f"risk_{horizon}d"
        if model is not None:
            if hasattr(model, "predict_proba"):
                df_latest[col] = model.predict_proba(X)[:, 1]
            else:
                df_latest[col] = np.clip(model.predict(X), 0, 1)
        else:
            sev = (
                0.18*normalize_score(df_latest["vibration_mm_s"]) +
                0.18*normalize_score(df_latest["temperature_c"]) +
                0.12*normalize_score(df_latest["motor_load_pct"]) +
                0.10*normalize_score(df_latest["servo_current_a"]) +
                0.10*normalize_score(df_latest["dust_collector_dp_kpa"]) +
                0.10*normalize_score(df_latest["tool_wear_hrs"]) +
                0.10*normalize_score(df_latest["hours_since_maintenance"]) +
                0.06*normalize_score(df_latest["encoder_error_count"]) +
                0.06*normalize_score(df_latest["ambient_temp_f"])
            ) / 100.0
            mult = {7:0.65, 30:1.0, 90:1.25}[horizon]
            df_latest[col] = np.clip(sev*mult, 0.01, 0.99)

    rul_model = models.get("rul_regressor")
    if rul_model is not None:
        df_latest["predicted_rul_days"] = np.maximum(1, rul_model.predict(X))
    else:
        sev = df_latest["risk_30d"]*0.40 + df_latest["risk_90d"]*0.35 + (100-df_latest["health_score"])/100*0.25
        df_latest["predicted_rul_days"] = np.clip(95 - 85*sev, 3, 120)

    return df_latest

def make_status_table(df_latest):
    out = df_latest.copy()
    out["7D Risk (%)"] = (out["risk_7d"]*100).round(1)
    out["30D Risk (%)"] = (out["risk_30d"]*100).round(1)
    out["90D Risk (%)"] = (out["risk_90d"]*100).round(1)
    out["Predicted RUL (days)"] = out["predicted_rul_days"].round(1)
    out["Health Score"] = out["health_score"].round(1)
    out["Fleet Percentile"] = (out["risk_30d"].rank(pct=True)*100).round(1)
    return out

def plot_fleet_scatter(df_latest, risk_threshold):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = [risk_color(x) for x in df_latest["priority"]]
    ax.scatter(df_latest["risk_30d"]*100, df_latest["health_score"], s=120, c=colors, alpha=0.88, edgecolors="black", linewidths=0.35)
    ax.axvline(risk_threshold*100, linestyle="--", linewidth=1.2)
    for _, r in df_latest.nlargest(min(12, len(df_latest)), "risk_30d").iterrows():
        ax.annotate(r["machine_id"], (r["risk_30d"]*100, r["health_score"]), fontsize=8)
    ax.set_xlabel("30-Day Failure Risk (%)")
    ax.set_ylabel("Health Score")
    ax.set_title("Fleet Risk Prioritization")
    ax.grid(alpha=0.25)
    return fig

def plot_heatmap_tiles(df_latest):
    grid = df_latest.sort_values(["priority_rank","risk_30d"], ascending=[True, False]).reset_index(drop=True).copy()
    grid["tile_x"] = grid.index % 10
    grid["tile_y"] = grid.index // 10
    fig, ax = plt.subplots(figsize=(12, 6.7))
    for _, row in grid.iterrows():
        rect = plt.Rectangle((row["tile_x"], -row["tile_y"]), 0.94, -0.94, color=risk_color(row["priority"]), alpha=0.94)
        ax.add_patch(rect)
        ax.text(row["tile_x"]+0.47, -row["tile_y"]-0.34, row["machine_id"], ha="center", va="center", color="white", fontsize=7.8, weight="bold")
        ax.text(row["tile_x"]+0.47, -row["tile_y"]-0.63, f"{row['risk_30d']*100:.0f}%", ha="center", va="center", color="white", fontsize=7.8)
    ax.set_xlim(0, max(10, grid["tile_x"].max()+1))
    ax.set_ylim(-(grid["tile_y"].max()+1), 0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Factory Machine Map | Tile Color = Priority Band | Label = Machine & 30-Day Risk")
    return fig

def plot_machine_trends(machine_df):
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(machine_df["timestamp"], machine_df["vibration_mm_s"], label="Vibration")
    ax.plot(machine_df["timestamp"], machine_df["temperature_c"], label="Bearing Temp")
    ax.plot(machine_df["timestamp"], machine_df["motor_load_pct"], label="Motor Load")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title("Core Degradation Signals")
    return fig

def plot_support_trends(machine_df):
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(machine_df["timestamp"], machine_df["servo_current_a"], label="Servo Current")
    ax.plot(machine_df["timestamp"], machine_df["dust_collector_dp_kpa"], label="Dust Collector DP")
    ax.plot(machine_df["timestamp"], machine_df["vacuum_pressure_kpa"], label="Vacuum Pressure")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.set_title("Supporting System Signals")
    return fig

def plot_forecast_curve(row):
    horizons = np.array([7, 30, 90])
    risks = np.array([row["risk_7d"], row["risk_30d"], row["risk_90d"]]) * 100
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(horizons, risks, marker="o", linewidth=2.2)
    ax.fill_between(horizons, risks, alpha=0.15)
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("Failure Probability (%)")
    ax.set_title("Failure Probability Curve")
    ax.grid(alpha=0.25)
    return fig

def plot_model_metric(perf_df, metric_name):
    pivot = perf_df.pivot_table(index="model", columns="horizon_days", values=metric_name, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by Algorithm and Forecast Horizon")
    ax.grid(alpha=0.2, axis="y")
    return fig

def main():
    inject_styles()
    df = load_dataset()
    models, perf = load_models()

    st.sidebar.header("Control Panel")
    selected_model = st.sidebar.selectbox("Forecast Model", list(MODEL_OPTIONS.keys()), index=0)
    focus_machine = st.sidebar.selectbox("Focus Machine", ["Auto: Highest Risk"] + sorted(df["machine_id"].unique().tolist()), index=0)

    min_date = df["timestamp"].min().to_pydatetime()
    max_date = df["timestamp"].max().to_pydatetime()
    selected_dates = st.sidebar.slider("History Window", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    temp_min = float(df["ambient_temp_f"].min())
    temp_max = float(df["ambient_temp_f"].max())
    ambient_range = st.sidebar.slider("Ambient Temperature Filter (F)", temp_min, temp_max, (temp_min, temp_max))

    risk_threshold = st.sidebar.slider("30-Day Alert Threshold", 0.10, 0.90, 0.45, 0.05)

    df_view = df[(df["timestamp"] >= selected_dates[0]) & (df["timestamp"] <= selected_dates[1])].copy()
    df_view = df_view[(df_view["ambient_temp_f"] >= ambient_range[0]) & (df_view["ambient_temp_f"] <= ambient_range[1])].copy()

    latest = df_view.sort_values("timestamp").groupby("machine_id").tail(1).copy()
    latest["health_score"] = compute_health_index(latest)
    latest = predict_latest(latest, models, selected_model)
    latest["priority"] = [risk_band(r, rul) for r, rul in zip(latest["risk_30d"], latest["predicted_rul_days"])]
    latest["priority_rank"] = latest["priority"].map({"Critical":0,"High":1,"Watch":2,"Normal":3})
    latest["alert_flag"] = np.where(latest["risk_30d"] >= risk_threshold, "Alert", "OK")

    if latest.empty:
        st.error("No data remains after applying the current filters.")
        st.stop()

    if focus_machine == "Auto: Highest Risk":
        machine_id = latest.sort_values("risk_30d", ascending=False).iloc[0]["machine_id"]
    else:
        machine_id = focus_machine

    machine_df = df_view[df_view["machine_id"] == machine_id].sort_values("timestamp").copy()
    machine_latest = latest[latest["machine_id"] == machine_id].iloc[0]

    critical_count = int((latest["priority"] == "Critical").sum())
    high_count = int((latest["priority"] == "High").sum())
    avg_health = float(latest["health_score"].mean())
    avg_risk_30 = float(latest["risk_30d"].mean()*100)

    st.markdown(
        f"""
        <div class="hero">
            <div style="font-size:1.75rem;font-weight:800;">🏭 Industrial AI Factory Control Room</div>
            <div class="subhero">Digital twin operations dashboard for fleet health, failure forecasting, and maintenance prioritization.</div>
            <div class="subhero"><b>Model:</b> {selected_model} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Focus Machine:</b> {machine_id} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Window:</b> {selected_dates[0].date()} → {selected_dates[1].date()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Machines in View", f"{len(latest):,}")
    c2.metric("Critical", critical_count)
    c3.metric("High Priority", high_count)
    c4.metric("Avg Health", f"{avg_health:.1f}")
    c5.metric("Avg 30D Risk", f"{avg_risk_30:.1f}%")

    st.markdown(
        """
        <div class="legend-row">
            <span class="legend-chip" style="background:#dc2626;">Critical</span>
            <span class="legend-chip" style="background:#f97316;">High</span>
            <span class="legend-chip" style="background:#eab308;">Watch</span>
            <span class="legend-chip" style="background:#16a34a;">Normal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Control Room", "Factory Map", "Machine Digital Twin", "Model Intelligence"])

    with tab1:
        left, right = st.columns([1.3, 1.0])
        with left:
            st.subheader("Fleet Risk vs Health")
            st.pyplot(plot_fleet_scatter(latest, risk_threshold))
        with right:
            st.subheader("Maintenance Priority Queue")
            queue = make_status_table(latest).sort_values(["priority_rank","risk_30d","predicted_rul_days"], ascending=[True, False, True]).copy()
            queue = queue.rename(columns={"machine_id":"Machine","machine_type":"Type","shift":"Shift","priority":"Priority","alert_flag":"Alert"})
            st.dataframe(queue[["Machine","Type","Shift","Priority","30D Risk (%)","Predicted RUL (days)","Health Score","Alert"]].head(15), use_container_width=True, hide_index=True)

        st.subheader("Top Risk Machines")
        top = make_status_table(latest).sort_values("risk_30d", ascending=False).copy()
        top = top.rename(columns={"machine_id":"Machine","machine_type":"Type","shift":"Shift","priority":"Priority"})
        st.dataframe(top[["Machine","Type","Shift","Priority","7D Risk (%)","30D Risk (%)","90D Risk (%)","Predicted RUL (days)","Health Score","Fleet Percentile"]].head(20), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Factory Machine Heatmap")
        st.pyplot(plot_heatmap_tiles(latest))
        st.caption("Tiles are sorted by maintenance priority and 30-day risk. This gives a control-room style fleet map for daily operational review.")

    with tab3:
        a, b, c, d = st.columns(4)
        a.metric("Machine Health", f"{machine_latest['health_score']:.1f}")
        b.metric("7D Risk", f"{machine_latest['risk_7d']*100:.1f}%")
        c.metric("30D Risk", f"{machine_latest['risk_30d']*100:.1f}%")
        d.metric("Predicted RUL", f"{machine_latest['predicted_rul_days']:.1f} days")

        st.markdown(
            f"""
            <div class="glass">
                <div style="font-size:1.1rem;font-weight:700;margin-bottom:0.35rem;">Maintenance Recommendation</div>
                <div><b>Priority:</b> <span style="color:{risk_color(machine_latest['priority'])};font-weight:800;">{machine_latest['priority']}</span></div>
                <div><b>Action:</b> {"Immediate inspection and maintenance planning." if machine_latest['priority']=="Critical" else "Schedule maintenance during next window." if machine_latest['priority']=="High" else "Increase monitoring frequency and inspect degradation drivers." if machine_latest['priority']=="Watch" else "Continue routine monitoring."}</div>
                <div><b>Primary drivers:</b> vibration, bearing temperature, motor load, servo current, and maintenance interval.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        l1, l2 = st.columns(2)
        l1.pyplot(plot_machine_trends(machine_df))
        l2.pyplot(plot_support_trends(machine_df))

        ll, rr = st.columns([0.95, 1.05])
        ll.pyplot(plot_forecast_curve(machine_latest))
        snapshot = pd.DataFrame({
            "Signal": [
                "Machine Age (years)","Ambient Temp (F)","Humidity (%)","Operating Hours",
                "Hours Since Maintenance","Tool Wear Hours","RPM","Motor Load (%)",
                "Voltage (V)","Pressure","Lubrication Score","Vibration (mm/s)",
                "Bearing Temp","Dust Collector DP","Servo Current","Vacuum Pressure","Encoder Errors"
            ],
            "Value": [
                machine_latest["machine_age_years"], machine_latest["ambient_temp_f"], machine_latest["humidity_pct"],
                machine_latest["operating_hours"], machine_latest["hours_since_maintenance"], machine_latest["tool_wear_hrs"],
                machine_latest["rpm"], machine_latest["motor_load_pct"], machine_latest["voltage_v"], machine_latest["pressure_bar"],
                machine_latest["lubrication_score"], machine_latest["vibration_mm_s"], machine_latest["temperature_c"],
                machine_latest["dust_collector_dp_kpa"], machine_latest["servo_current_a"], machine_latest["vacuum_pressure_kpa"],
                machine_latest["encoder_error_count"]
            ]
        })
        snapshot["Value"] = snapshot["Value"].round(2)
        rr.subheader("Latest Condition Snapshot")
        rr.dataframe(snapshot, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Model Performance Intelligence")
        if perf is not None and len(perf):
            st.dataframe(perf, use_container_width=True, hide_index=True)
            metric_choices = [c for c in ["roc_auc","pr_auc","f1_score","recall","precision","accuracy"] if c in perf.columns]
            selected_metric = st.selectbox("Performance Metric", metric_choices, index=0)
            st.pyplot(plot_model_metric(perf, selected_metric))
        else:
            st.info("No model_performance_summary.csv found in models/. Run the training script first to populate model analytics.")

    st.markdown(
        """
        <div class="footer-note">
            Built specifically for the industrial_time_series_dataset_v2 schema and the associated trained model artifacts.
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
