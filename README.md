
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
