
# Industrial AI Control Tower
## Predictive Maintenance | Digital Twin | Smart Manufacturing

Transform factory data into real-time operational decisions

End‑to‑end industrial AI project that forecasts machine failures and estimates
remaining useful life (RUL) using machine telemetry.

# Highlights
• Predict machine failures before they happen; failure prediction for 7, 30, and 90 day horizons  
• Estimate Remaining Useful Life prediction (RUL) 
• Detect anomalies in machine behavior
• Prioritize maintenance across entire factory
• Monitor entire factory in real time (Streamlit Dashboards)
• AI-driven maintenance recommendations
• Advanced AI extensions: anomaly detection, SHAP explainability, LSTM models 

# Performance
• Model ROC AUC: ~0.84
• Monitored 60 machines with 14,400 data points
• Failure horizons 7/30/90 days

# Core Idea
This project simulates a factory AI Control Tower that prioritizes maintenance based on predictive 
risk and operational impact, not just downtime.  

Traditional manufacturing struggles with:
• Reactive maintenance
• Unplanned downtime
• Lack of actionable insights

This platform introduces:
• Predictive Intelligence
• Real-time decision-making
• AI-driven maintenance prioritization

# How It Works
1. Collect machine telemetry
2. Engineer predictive features
3. Train ML models
4. Generate failure risk + RUL
5. Visualize insights in dashboards
6. Recommend maintenance actions

# Why It Matters
This is not just analytics - it's an AI-powered decision system for manufacturing.

# ROI/Cost Savings Model
Assumptions:
* Plant has 60 machines
* Avg downtime per failure = 4 hours
* Cost per hour downtime = $5,000
* Failure per year = 100 events

  Current Cost: Total downtime cost = 100 x 4 x $5,000 = $2,000,000 per year

With AI System, assume:
* 25% failures prevented
* 20% downtime reduction

  Savings from prevention = $500,000
  Savings from reduction = $300,000
  Total savings = $800,000/year

# Even a modest 20-30% improvement in failure prediction can translate into hundreds
# of thousands of dollars in annual savings for a mid-size plant. 


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

