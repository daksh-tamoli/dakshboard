import pandas as pd

def add_hr_zones(df, max_hr, rest_hr):
    """Calculates custom HR zones using the Karvonen (HRR) method and returns bins/labels."""
    if 'smoothed_heart_rate' not in df.columns:
        return df, []
        
    hrr = max_hr - rest_hr
    
    # We return these bins so our graphing tool knows exactly where to paint the background colors
    bins = [0, rest_hr + (hrr * 0.50), rest_hr + (hrr * 0.60), 
            rest_hr + (hrr * 0.70), rest_hr + (hrr * 0.80), 
            rest_hr + (hrr * 0.90), max_hr + 10]
            
    labels = ['Rest', 'Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
    df['hr_zone'] = pd.cut(df['smoothed_heart_rate'], bins=bins, labels=labels)
    
    return df, bins

def calculate_training_stress(df):
    """Calculates Edwards' TRIMP (Training Impulse) based on time in zones."""
    if 'hr_zone' not in df.columns:
        return 0
        
    # Get total seconds in each zone
    zone_counts = df['hr_zone'].value_counts()
    
    # Convert seconds to minutes and apply the Edwards multipliers
    stress_score = (
        (zone_counts.get('Zone 1', 0) / 60) * 1 +
        (zone_counts.get('Zone 2', 0) / 60) * 2 +
        (zone_counts.get('Zone 3', 0) / 60) * 3 +
        (zone_counts.get('Zone 4', 0) / 60) * 4 +
        (zone_counts.get('Zone 5', 0) / 60) * 5
    )
    return round(stress_score)

def classify_workout(df):
    """Determines the workout type based on zone distribution."""
    if 'hr_zone' not in df.columns:
        return "Unclassified"
        
    total_active_seconds = len(df[df['hr_zone'] != 'Rest'])
    if total_active_seconds == 0:
        return "Rest / No Data"
        
    zone_counts = df['hr_zone'].value_counts()
    
    # Calculate percentage of time spent in specific groupings
    z1_z2_pct = (zone_counts.get('Zone 1', 0) + zone_counts.get('Zone 2', 0)) / total_active_seconds
    z3_pct = zone_counts.get('Zone 3', 0) / total_active_seconds
    z4_z5_pct = (zone_counts.get('Zone 4', 0) + zone_counts.get('Zone 5', 0)) / total_active_seconds
    
    # Rule-based logic engine
    if z4_z5_pct > 0.15:
        return "Threshold / VO2 Max Interval" # Hard effort, high lactic acid
    elif z3_pct > 0.30:
        return "Tempo Run" # Sustained moderate-hard effort
    elif z1_z2_pct > 0.80:
        return "Recovery / Easy Aerobic" # Base building
    else:
        return "Mixed Aerobic Base"