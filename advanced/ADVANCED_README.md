
# Advanced Industrial AI Upgrade

This package upgrades the core predictive maintenance project into a more impressive industrial AI system.

## Included modules

- `train_advanced_industrial_ai.py`  
  Trains an advanced 30-day failure model and anomaly detection model.

- `anomaly_detection_module.py`  
  Scores abnormal machine behavior using Isolation Forest.

- `explain_with_shap.py`  
  Generates SHAP feature importance for the advanced 30-day failure model.

- `deep_learning_lstm_forecast.py`  
  Sequence-based LSTM forecasting starter for 30-day failure prediction.

- `real_time_factory_dashboard.py`  
  Streamlit app that simulates a live factory telemetry operations center.

## Recommended run order

```bash
pip install -r requirements-advanced.txt
python advanced/train_advanced_industrial_ai.py
python advanced/explain_with_shap.py
python advanced/deep_learning_lstm_forecast.py
streamlit run advanced/real_time_factory_dashboard.py
```
