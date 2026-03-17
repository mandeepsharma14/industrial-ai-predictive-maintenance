from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Industrial AI Failure Forecast", page_icon="🏭", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "industrial_time_series_dataset_v2.csv",
    BASE_DIR / "data" / "industrial_time_series_dataset_v2.csv",
    BASE_DIR.parent / "data" / "industrial_time_series_dataset_v2.csv",
    BASE_DIR / "industrial_time_series_dataset.csv",
    BASE_DIR / "data" / "industrial_time_series_dataset.csv",
    BASE_DIR.parent / "data" / "industrial_time_series_dataset.csv",
]
IMAGE_CANDIDATES = [
    BASE_DIR / "industrial_ai_banner.png",
    BASE_DIR / "images" / "industrial_ai_banner.png",
    BASE_DIR.parent / "images" / "industrial_ai_banner.png",
]

FEATURE_COLUMNS = [
    "machine_type", "shift", "machine_age_years", "duty_class", "ambient_temp_f", "humidity_pct",
    "operating_hours", "days_since_maintenance", "tool_wear_hours", "rpm", "motor_load_pct",
    "voltage_v", "hydraulic_pressure_psi", "lubrication_score", "vibration_mm_s", "bearing_temp_f",
    "dust_collector_dp_inwc", "servo_current_a", "vacuum_pressure_kpa", "encoder_error_count",
]
TARGETS = {
    "Next 7 Days": "failure_next_7d",
    "Next 30 Days": "failure_next_30d",
    "Next 90 Days": "failure_next_90d",
}
RUL_TARGET = "rul_days_capped_120"
DEGRADATION_METRICS = [
    "vibration_mm_s", "bearing_temp_f", "motor_load_pct", "servo_current_a",
    "dust_collector_dp_inwc", "lubrication_score"
]


def add_css() -> None:
    st.markdown(
        """
        <style>
        .main {background-color: #0f172a;}
        section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f172a 0%, #111827 100%);} 
        section[data-testid="stSidebar"] * {color: #f8fafc;}
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p {color: #f8fafc !important;}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
        section[data-testid="stSidebar"] [data-testid="stDateInputField"] {
            background: #ffffff !important;
            color: #111827 !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] span,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] [data-testid="stDateInputField"] input {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        .hero {padding: 1.15rem 1.35rem; border-radius: 18px; background: linear-gradient(135deg,#0b1220 0%,#102a43 45%,#0ea5e9 100%); color:white; border:1px solid rgba(255,255,255,0.12); box-shadow:0 10px 30px rgba(2,6,23,0.35); margin-bottom:1rem;}
        .hero h1 {font-size: 2.1rem; margin:0 0 .25rem 0;}
        .hero p {margin:0; opacity:.92;}
        .card {background:white; padding:1rem; border-radius:16px; border:1px solid #e5e7eb; box-shadow:0 6px 18px rgba(15,23,42,.08);} 
        .section-title {font-size:1.15rem; font-weight:700; margin:.3rem 0 .75rem 0; color:#e5e7eb;}
        .small-note {font-size:0.82rem; color:#94a3b8;}
        .selection-card {background:#ffffff; color:#111827; border-radius:14px; padding:.85rem 1rem; border:1px solid #dbe4f0; box-shadow:0 4px 14px rgba(15,23,42,.08); margin:.5rem 0 1rem 0;}
        .selection-card strong {color:#0f172a;}
        .pill {display:inline-block; background:#e0f2fe; color:#075985; border:1px solid #bae6fd; padding:.2rem .55rem; border-radius:999px; margin-right:.35rem; margin-top:.25rem; font-size:.8rem; font-weight:600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def resolve_existing_path(candidates: List[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = resolve_existing_path(DATA_CANDIDATES)
    if data_path is None:
        raise FileNotFoundError("Could not find industrial_time_series_dataset_v2.csv in the current folder or a data/ subfolder.")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    return df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_days: int = 45) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff].copy(), df[df["timestamp"] > cutoff].copy()


def preprocessor_for(df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in FEATURE_COLUMNS if df[c].dtype == "object"]
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1200, class_weight="balanced"))]),
        "Decision Tree": Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=260, max_depth=10, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingClassifier(random_state=42))]),
    }


def build_rul_model(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])


@st.cache_resource
def train_all_models():
    df = load_data()
    train_df, test_df = temporal_split(df)
    prep = preprocessor_for(df)
    X_train, X_test = train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]

    models_by_horizon = {}
    metrics_rows = []
    confusion_by_horizon = {}

    for horizon_label, target in TARGETS.items():
        horizon_models = build_models(prep)
        models_by_horizon[horizon_label] = {}
        confusion_by_horizon[horizon_label] = {}
        for model_name, pipe in horizon_models.items():
            pipe.fit(X_train, train_df[target])
            prob = pipe.predict_proba(X_test)[:, 1]
            best_threshold, pred = optimize_threshold(test_df[target], prob)
            base_rate = float(test_df[target].mean())
            metrics_rows.append({
                "Horizon": horizon_label,
                "Model": model_name,
                "Accuracy": accuracy_score(test_df[target], pred),
                "Precision": precision_score(test_df[target], pred, zero_division=0),
                "Recall": recall_score(test_df[target], pred, zero_division=0),
                "F1 Score": f1_score(test_df[target], pred, zero_division=0),
                "ROC AUC": roc_auc_score(test_df[target], prob),
                "PR AUC": average_precision_score(test_df[target], prob),
                "Optimized Threshold": best_threshold,
                "Average Risk Probability": float(prob.mean()),
                "Failure Base Rate": base_rate,
                "Positive Events": int(test_df[target].sum()),
                "Total Records": int(len(test_df)),
            })
            models_by_horizon[horizon_label][model_name] = pipe
            confusion_by_horizon[horizon_label][model_name] = confusion_matrix(test_df[target], pred)

    rul_model = build_rul_model(prep)
    rul_model.fit(X_train, train_df[RUL_TARGET])
    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "results_df": pd.DataFrame(metrics_rows),
        "models_by_horizon": models_by_horizon,
        "confusion_by_horizon": confusion_by_horizon,
        "rul_model": rul_model,
    }


def top_feature_frame(pipe: Pipeline) -> pd.DataFrame:
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    else:
        coef = getattr(model, "coef_", np.zeros((1, len(names))))
        scores = np.abs(coef[0])
    frame = pd.DataFrame({"Feature": names, "Score": scores})
    frame["Feature"] = frame["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    return frame.sort_values("Score", ascending=False).head(12)



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


def predict_single_row(row: pd.Series, trained: dict, model_name: str) -> dict:
    X = row[FEATURE_COLUMNS].to_frame().T
    forecast = {}
    for horizon_label in TARGETS:
        pipe = trained["models_by_horizon"][horizon_label][model_name]
        forecast[horizon_label] = float(pipe.predict_proba(X)[0][1])
    rul_days = float(np.clip(trained["rul_model"].predict(X)[0], 0, 120))
    return {"forecast": forecast, "rul_days": rul_days}


def maintenance_recommendation(risk_7: float, risk_30: float, rul_days: float, vibration: float, temp: float) -> str:
    if risk_7 >= 0.60 or rul_days <= 7:
        return "Immediate inspection recommended within 24 hours. Prioritize bearing, alignment, lubrication, and electrical checks."
    if risk_30 >= 0.55 or rul_days <= 21:
        return "Schedule maintenance in the next 1–2 weeks. Order likely replacement parts and plan a controlled downtime window."
    if vibration > 5.5 or temp > 185:
        return "Condition is elevated. Increase monitoring frequency and review load, cooling, and dust collection performance."
    return "Continue standard preventive maintenance. Machine is currently within a stable operating band."


def render_banner() -> None:
    image_path = resolve_existing_path(IMAGE_CANDIDATES)
    if image_path is not None:
        st.image(str(image_path), use_container_width=True)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Industrial AI Failure Forecasting Dashboard</h1>
            <p>Forecast failure risk across 7, 30, and 90-day horizons, estimate remaining useful life, visualize degradation curves,
            and convert machine signals into actionable maintenance decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_metric_trend(machine_hist: pd.DataFrame, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(machine_hist["timestamp"], machine_hist[y_col], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)


def main() -> None:
    add_css()
    render_banner()
    hero()

    trained = train_all_models()
    df = trained["df"]
    results_df = trained["results_df"]

    st.sidebar.title("Forecast Controls")
    horizon = st.sidebar.selectbox("Forecast horizon", list(TARGETS.keys()), index=1)
    model_name = st.sidebar.selectbox("ML algorithm", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"], index=2)
    machine_id = st.sidebar.selectbox("Focus machine", sorted(df["machine_id"].unique()))
    date_min = df["timestamp"].min().date()
    date_max = df["timestamp"].max().date()
    date_range = st.sidebar.date_input("Date window", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    probability_floor = st.sidebar.slider("Alert threshold", min_value=0.10, max_value=0.80, value=0.25, step=0.05)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = df[df["timestamp"].between(start_date, end_date)].copy()
    else:
        filtered = df.copy()

    st.markdown(
        f"""
        <div class="selection-card">
            <strong>Current Selection</strong><br>
            <span class="pill">Horizon: {horizon}</span>
            <span class="pill">Model: {model_name}</span>
            <span class="pill">Machine: {machine_id}</span>
            <span class="pill">Window: {pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}</span>
            <span class="pill">Alert threshold: {probability_floor:.0%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_rows = filtered.sort_values("timestamp").groupby("machine_id").tail(1).copy()
    for hz_label, target_col in TARGETS.items():
        risk_suffix = target_col.replace("failure_next_", "")
        latest_rows[f"risk_{risk_suffix}"] = trained["models_by_horizon"][hz_label][model_name].predict_proba(latest_rows[FEATURE_COLUMNS])[:, 1]
    latest_rows["rul_days_est"] = np.clip(trained["rul_model"].predict(latest_rows[FEATURE_COLUMNS]), 0, 120)
    latest_rows["risk_percentile_30d"] = latest_rows["risk_30d"].rank(pct=True)
    latest_rows["alert_flag"] = np.where(latest_rows["risk_30d"] >= probability_floor, "Review", "Normal")
    latest_rows["health_score"] = (100 - latest_rows["risk_percentile_30d"] * 100).clip(0, 100)

    machine_hist = filtered[filtered["machine_id"] == machine_id].copy()
    latest_machine = machine_hist.sort_values("timestamp").tail(1).squeeze()
    machine_pred = predict_single_row(latest_machine, trained, model_name)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machines monitored", f"{filtered['machine_id'].nunique():,}")
    c2.metric("Recorded failure events", f"{int(filtered['failure_event_today'].sum()):,}")
    c3.metric("Avg 30-day risk", f"{latest_rows['risk_30d'].mean()*100:0.1f}%")
    c4.metric("Median estimated RUL", f"{latest_rows['rul_days_est'].median():0.0f} days")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary", "Model Comparison", "Failure Timeline", "Degradation Curves", "Maintenance Planner"
    ])

    with tab1:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### Highest-risk machines")
            top_risk = latest_rows[["machine_id", "risk_7d", "risk_30d", "risk_90d", "rul_days_est", "health_score", "alert_flag"]].copy()
            top_risk.columns = ["Machine", "Risk 7d", "Risk 30d", "Risk 90d", "Est. RUL", "Health Score", "Alert"]
            st.dataframe(
                top_risk.sort_values("Risk 30d", ascending=False).head(15).style.format({
                    "Risk 7d": "{:.1%}", "Risk 30d": "{:.1%}", "Risk 90d": "{:.1%}", "Est. RUL": "{:.0f} d", "Health Score": "{:.0f}"
                }),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            sc = ax.scatter(latest_rows["bearing_temp_f"], latest_rows["vibration_mm_s"], c=latest_rows["risk_30d"], s=70)
            ax.set_title("Plant risk map")
            ax.set_xlabel("Bearing temperature (°F)")
            ax.set_ylabel("Vibration (mm/s)")
            plt.colorbar(sc, ax=ax, label="30-day risk")
            st.pyplot(fig)

            st.markdown("### Focus machine forecast")
            st.write(f"**Machine:** {machine_id}")
            selected_latest = latest_rows[latest_rows["machine_id"] == machine_id].iloc[0]
            rel_pct = float(selected_latest["risk_percentile_30d"])
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("7-day risk", f"{machine_pred['forecast']['Next 7 Days']:.1%}")
            f2.metric("30-day risk", f"{machine_pred['forecast']['Next 30 Days']:.1%}")
            f3.metric("90-day risk", f"{machine_pred['forecast']['Next 90 Days']:.1%}")
            f4.metric("Fleet risk percentile", f"{rel_pct:.0%}")
            f5.metric("Est. RUL", f"{machine_pred['rul_days']:.0f} d")
            st.caption("Raw probabilities can look low in rare-event datasets. The fleet percentile shows how risky the selected machine is relative to all other machines in the same time window.")

    with tab2:
        st.markdown("### Supervised ML model performance")
        selected_summary = results_df[(results_df["Horizon"] == horizon) & (results_df["Model"] == model_name)].iloc[0]
        st.info(
            f"Selected horizon holdout window contains {int(selected_summary['Positive Events'])} positive failure events out of "
            f"{int(selected_summary['Total Records'])} records ({selected_summary['Failure Base Rate']:.2%} base rate). "
            f"Low event counts make raw probabilities look smaller and precision/recall harder to optimize. "
            f"This dashboard uses class-balanced training and an F1-optimized alert threshold to make rare-event performance more realistic."
        )
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.1%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "F1 Score": "{:.1%}",
                "ROC AUC": "{:.3f}",
                "PR AUC": "{:.3f}",
                "Average Risk Probability": "{:.1%}",
                "Failure Base Rate": "{:.2%}",
                "Optimized Threshold": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        horizon_frame = results_df[results_df["Horizon"] == horizon].copy()
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(horizon_frame))
        width = 0.18
        for i, metric in enumerate(metrics):
            ax.bar(x + (i - 1.5) * width, horizon_frame[metric].values, width=width, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(horizon_frame["Model"], rotation=10)
        ax.set_ylim(0, 1)
        ax.set_title(f"Model comparison for {horizon} | alert threshold shown in table")
        ax.legend(ncol=4, fontsize=8)
        st.pyplot(fig)

        selected_pipe = trained["models_by_horizon"][horizon][model_name]
        feat_df = top_feature_frame(selected_pipe)
        left, right = st.columns([1, 1])
        with left:
            fig2, ax2 = plt.subplots(figsize=(7, 4.2))
            ax2.barh(feat_df["Feature"][::-1], feat_df["Score"][::-1])
            ax2.set_title(f"Top drivers | {model_name} | {horizon}")
            st.pyplot(fig2)
        with right:
            cm = trained["confusion_by_horizon"][horizon][model_name]
            fig3, ax3 = plt.subplots(figsize=(4.6, 4.2))
            im = ax3.imshow(cm)
            ax3.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax3.set_yticks([0, 1], ["Actual 0", "Actual 1"])
            ax3.set_title("Confusion matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax3.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
            plt.colorbar(im, ax=ax3)
            st.pyplot(fig3)
            st.caption(f"Confusion matrix uses the F1-optimized decision threshold from the model table. Current dashboard alert threshold for operational review is {probability_floor:.0%}.")

    with tab3:
        st.markdown("### Failure timeline and risk trend")
        machine_hist = machine_hist.sort_values("timestamp").copy()
        machine_hist["risk_7d"] = trained["models_by_horizon"]["Next 7 Days"][model_name].predict_proba(machine_hist[FEATURE_COLUMNS])[:, 1]
        machine_hist["risk_30d"] = trained["models_by_horizon"]["Next 30 Days"][model_name].predict_proba(machine_hist[FEATURE_COLUMNS])[:, 1]
        machine_hist["risk_90d"] = trained["models_by_horizon"]["Next 90 Days"][model_name].predict_proba(machine_hist[FEATURE_COLUMNS])[:, 1]
        machine_hist["rul_est"] = np.clip(trained["rul_model"].predict(machine_hist[FEATURE_COLUMNS]), 0, 120)

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(machine_hist["timestamp"], machine_hist["risk_7d"] * 100, label="7d risk", linewidth=2)
        ax.plot(machine_hist["timestamp"], machine_hist["risk_30d"] * 100, label="30d risk", linewidth=2)
        ax.plot(machine_hist["timestamp"], machine_hist["risk_90d"] * 100, label="90d risk", linewidth=2)
        failure_dates = machine_hist.loc[machine_hist["failure_event_today"] == 1, "timestamp"]
        for dt in failure_dates:
            ax.axvline(dt, linestyle="--", alpha=0.4)
        ax.set_title(f"Forecasted failure risk over time | {machine_id}")
        ax.set_ylabel("Probability (%)")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 3.8))
        ax2.plot(machine_hist["timestamp"], machine_hist["rul_est"], linewidth=2)
        ax2.set_title(f"Estimated remaining useful life over time | {machine_id}")
        ax2.set_ylabel("Days")
        ax2.tick_params(axis="x", rotation=30)
        st.pyplot(fig2)

        st.dataframe(
            machine_hist[["timestamp", "failure_event_today", "risk_7d", "risk_30d", "risk_90d", "rul_est"]].tail(20).style.format({
                "risk_7d": "{:.1%}", "risk_30d": "{:.1%}", "risk_90d": "{:.1%}", "rul_est": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab4:
        st.markdown("### Machine degradation curves")
        c_left, c_right = st.columns(2)
        with c_left:
            plot_metric_trend(machine_hist, "vibration_mm_s", f"Vibration trend | {machine_id}")
            plot_metric_trend(machine_hist, "bearing_temp_f", f"Bearing temperature trend | {machine_id}")
            plot_metric_trend(machine_hist, "motor_load_pct", f"Motor load trend | {machine_id}")
        with c_right:
            plot_metric_trend(machine_hist, "servo_current_a", f"Servo current trend | {machine_id}")
            plot_metric_trend(machine_hist, "dust_collector_dp_inwc", f"Dust collector differential pressure | {machine_id}")
            plot_metric_trend(machine_hist, "lubrication_score", f"Lubrication score trend | {machine_id}")

        rolling = machine_hist[["timestamp"] + DEGRADATION_METRICS].copy()
        for col in DEGRADATION_METRICS:
            if col == "lubrication_score":
                rolling[col] = 100 - rolling[col]
        rolling["degradation_index"] = rolling[[c for c in DEGRADATION_METRICS]].rank(pct=True).mean(axis=1) * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rolling["timestamp"], rolling["degradation_index"], linewidth=2.5)
        ax.set_title(f"Composite degradation index | {machine_id}")
        ax.set_ylabel("Index (0-100)")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

    with tab5:
        st.markdown("### Maintenance scheduling recommendation")
        plan_rows = latest_rows[["machine_id", "risk_7d", "risk_30d", "risk_90d", "rul_days_est", "vibration_mm_s", "bearing_temp_f", "days_since_maintenance"]].copy()
        plan_rows["Recommended Action"] = plan_rows.apply(
            lambda r: maintenance_recommendation(r["risk_7d"], r["risk_30d"], r["rul_days_est"], r["vibration_mm_s"], r["bearing_temp_f"]), axis=1
        )
        plan_rows["Priority"] = np.select(
            [plan_rows["risk_7d"] >= 0.60, (plan_rows["risk_30d"] >= 0.55) | (plan_rows["rul_days_est"] <= 21), plan_rows["bearing_temp_f"] >= 185],
            ["Immediate", "Plan Soon", "Monitor"],
            default="Routine",
        )
        st.write(f"**Focus machine recommendation:** {maintenance_recommendation(machine_pred['forecast']['Next 7 Days'], machine_pred['forecast']['Next 30 Days'], machine_pred['rul_days'], float(latest_machine['vibration_mm_s']), float(latest_machine['bearing_temp_f']))}")

        priority_order = pd.CategoricalDtype(["Immediate", "Plan Soon", "Monitor", "Routine"], ordered=True)
        plan_rows["Priority"] = plan_rows["Priority"].astype(priority_order)
        plan_rows = plan_rows.sort_values(["Priority", "risk_30d"], ascending=[True, False])
        out = plan_rows.rename(columns={
            "machine_id": "Machine", "risk_7d": "Risk 7d", "risk_30d": "Risk 30d", "risk_90d": "Risk 90d",
            "rul_days_est": "Est. RUL", "vibration_mm_s": "Vibration", "bearing_temp_f": "Bearing Temp", "days_since_maintenance": "Days Since Maint"
        })
        st.dataframe(
            out.style.format({"Risk 7d": "{:.1%}", "Risk 30d": "{:.1%}", "Risk 90d": "{:.1%}", "Est. RUL": "{:.0f} d", "Vibration": "{:.2f}", "Bearing Temp": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        counts = out["Priority"].value_counts().reindex(["Immediate", "Plan Soon", "Monitor", "Routine"]).fillna(0)
        ax.bar(counts.index, counts.values)
        ax.set_title("Maintenance queue by priority")
        ax.set_ylabel("Machines")
        st.pyplot(fig)

    st.caption("This dashboard uses an enhanced synthetic but time-based industrial dataset with stronger degradation signals and a moderately higher failure-event rate so forecasting behavior is easier to interpret in a portfolio setting. Forecasts and RUL estimates are still based on simulated historical machine behavior rather than confidential plant data.")


if __name__ == "__main__":
    main()
