
import streamlit as st
import pandas as pd
import time
from pathlib import Path

st.set_page_config(page_title="Industrial AI Live Operations Center", page_icon="🏭", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "industrial_time_series_dataset_v2.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values(["machine_id","timestamp"])

df = load_data()
machine_id = st.sidebar.selectbox("Machine", sorted(df["machine_id"].unique()))
speed = st.sidebar.slider("Playback Speed", 1, 20, 5)
window = st.sidebar.slider("Window Size", 20, 150, 40)

st.title("🏭 Real-Time Factory Operations Center")
st.caption("Simulated live streaming telemetry replay from historical industrial data.")

machine_df = df[df["machine_id"] == machine_id].reset_index(drop=True)
metric_box = st.empty()
chart_box = st.empty()

for i in range(window, len(machine_df), speed):
    current = machine_df.iloc[i-window:i+1]
    latest = current.iloc[-1]
    with metric_box.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Timestamp", str(latest["timestamp"]))
        c2.metric("Vibration", f"{latest['vibration_mm_s']:.2f}")
        c3.metric("Bearing Temp", f"{latest['bearing_temp_f']:.1f}")
        c4.metric("Motor Load", f"{latest['motor_load_pct']:.1f}%")
    chart_box.line_chart(current.set_index("timestamp")[["vibration_mm_s","bearing_temp_f","motor_load_pct"]])
    time.sleep(0.12)
