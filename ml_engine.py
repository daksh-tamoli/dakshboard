import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(history_df):
    """
    Uses an Isolation Forest to detect physiological outliers in training history.
    Looks at the relationship between TRIMP load and Cardiac Drift.
    """
    if len(history_df) < 10:
        # We need at least 10 runs for the ML model to establish a baseline
        return history_df

    df = history_df.copy()
    
    # Isolate the features that indicate physiological strain
    features = df[['trimp', 'drift']].fillna(0)
    
    # Initialize the model (contamination = 0.1 means we expect ~10% of runs to be unusually stressful)
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    
    # Fit the model and predict (-1 is an anomaly, 1 is normal)
    df['anomaly'] = model.fit_predict(features)
    
    return df

def calculate_recovery_hours(trimp, atl):
    """
    Estimates required recovery time (EPOC proxy).
    Scales the base TRIMP stress by the athlete's current cumulative fatigue (ATL).
    """
    if trimp == 0:
        return 0
        
    # Base calculation: ~5 TRIMP points per hour of required recovery
    base_hours = trimp / 5.0
    
    # Fatigue multiplier: If ATL is high, recovery takes longer
    fatigue_multiplier = 1.0 + (atl / 100.0)
    
    recovery_hours = min(72, round(base_hours * fatigue_multiplier))
    return recovery_hours

def get_training_status(ctl, tsb):
    """
    Garmin-style training status classification based on the PMC model.
    """
    if ctl < 10:
        return "No Status (Build Base)"
        
    if tsb < -25:
        return "Overreaching ⚠️ (High injury risk)"
    elif -25 <= tsb < -10:
        return "Productive 📈 (Building fitness)"
    elif -10 <= tsb <= 5:
        return "Maintaining 🔄 (Balanced load)"
    elif 5 < tsb <= 20 and ctl > 30:
        return "Peaking 🚀 (Race ready)"
    elif tsb > 20:
        return "Detraining 📉 (Losing fitness)"
    else:
        return "Recovery 🔋"