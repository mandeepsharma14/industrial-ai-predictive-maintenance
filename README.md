
# Industrial AI Predictive Maintenance

End‑to‑end industrial AI project that forecasts machine failures and estimates
remaining useful life (RUL) using machine telemetry.

## Features

• Failure prediction for 7, 30, and 90 day horizons  
• Remaining Useful Life prediction  
• Streamlit dashboards for factory monitoring  
• Advanced AI extensions: anomaly detection, SHAP explainability, LSTM models  
• Real‑time factory dashboard simulation

## Repository Structure

```
data/
src/
dashboards/
advanced/
docs/
models/
notebooks/
images/
```

## Quick Start

Create environment

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run dashboard

```
streamlit run dashboards/industrial_ai_factory_control_room_pro.py
```

## Advanced Industrial AI Upgrade

The repository also includes an advanced upgrade path that expands the core predictive maintenance system into a stronger industrial AI platform.

### Advanced capabilities

- **Anomaly detection** using Isolation Forest to flag abnormal machine behavior
- **Model-driven live dashboard** connected to trained advanced model outputs
- **SHAP explainability** to identify the top drivers of predicted failures
- **LSTM deep learning starter** for sequence-based failure prediction
- **Real-time telemetry replay dashboard** to simulate a live factory operations center

### Advanced workflow

```text
Dataset
  -> preprocessing and time-based split
  -> advanced 30-day Gradient Boosting model
  -> anomaly detection model
  -> summary artifacts and model files
  -> model-driven live dashboard
```

### Advanced files

```text
advanced/
├── train_advanced_industrial_ai.py
├── explain_with_shap.py
├── deep_learning_lstm_forecast.py
├── real_time_factory_dashboard.py
├── real_time_factory_dashboard_pro.py
└── real_time_factory_dashboard_model_driven.py
```

### Run the advanced version

```bash
pip install -r requirements-advanced.txt
python advanced/train_advanced_industrial_ai.py
python advanced/explain_with_shap.py
python advanced/deep_learning_lstm_forecast.py
streamlit run advanced/real_time_factory_dashboard_model_driven.py
```

### Advanced artifacts generated

```text
models/
├── advanced_failure_30d_gb.joblib
├── advanced_isolation_forest.joblib
├── advanced_anomaly_scores.csv
├── advanced_summary.json
├── advanced_shap_summary.png
└── advanced_lstm_failure_30d.keras
```

### Why this matters

This advanced version makes the project look much closer to a real **Industry 4.0 / digital manufacturing AI solution**. It demonstrates not only predictive analytics, but also anomaly detection, explainability, and live operational monitoring.

