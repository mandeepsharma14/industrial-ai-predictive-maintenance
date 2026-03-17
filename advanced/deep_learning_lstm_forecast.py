
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    raise ImportError("TensorFlow is required. Install requirements-advanced.txt first.") from e

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "industrial_time_series_dataset_v2.csv"
OUT_PATH = PROJECT_ROOT / "models" / "advanced_lstm_failure_30d.keras"

SEQ_FEATURES = [
    "ambient_temp_f","humidity_pct","operating_hours","days_since_maintenance",
    "tool_wear_hours","rpm","motor_load_pct","voltage_v","hydraulic_pressure_psi",
    "lubrication_score","vibration_mm_s","bearing_temp_f","dust_collector_dp_inwc",
    "servo_current_a","vacuum_pressure_kpa","encoder_error_count",
]
TARGET = "failure_next_30d"
SEQ_LEN = 14

def make_sequences(df):
    X, y = [], []
    for _, g in df.sort_values(["machine_id","timestamp"]).groupby("machine_id"):
        g = g.reset_index(drop=True)
        feats = g[SEQ_FEATURES].copy().fillna(method="ffill").fillna(method="bfill").fillna(0).values
        target = g[TARGET].values
        if len(g) <= SEQ_LEN:
            continue
        for i in range(SEQ_LEN, len(g)):
            X.append(feats[i-SEQ_LEN:i])
            y.append(target[i])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    X, y = make_sequences(df)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=32,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)
    model.save(OUT_PATH)
    print(f"Saved LSTM model to {OUT_PATH}")

if __name__ == "__main__":
    main()
