
# Advanced Upgrade Guide

## What changes in the advanced version

1. Adds **unsupervised anomaly detection**
2. Adds **SHAP explainability**
3. Adds **LSTM deep learning forecasting**
4. Adds a **real-time streaming operations dashboard**

## Where to place these files

Copy the contents of the `advanced/` folder into the `advanced/` folder of your GitHub repository.

## Commands

```bash
pip install -r requirements-advanced.txt
python advanced/train_advanced_industrial_ai.py
python advanced/explain_with_shap.py
python advanced/deep_learning_lstm_forecast.py
streamlit run advanced/real_time_factory_dashboard.py
```
