
from sklearn.ensemble import IsolationForest
import pandas as pd

def run_anomaly_detection(df):
    model = IsolationForest(contamination=0.05)
    df["anomaly_score"] = model.fit_predict(df.select_dtypes("number"))
    return df
