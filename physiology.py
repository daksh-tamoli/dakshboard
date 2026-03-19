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
    
def calculate_cardiac_drift(df):
    """Calculates aerobic decoupling between the first and second half of a run."""
    if 'smoothed_speed_kmh' not in df.columns or 'smoothed_heart_rate' not in df.columns:
        return 0.0

    # Filter out stops - we only want active moving data
    moving_df = df[df['smoothed_speed_kmh'] > 0.5].copy()
    
    # If the run is too short, drift isn't scientifically valid
    if len(moving_df) < 120: 
        return 0.0
        
    # Calculate the strain ratio (Beats per minute per km/h)
    moving_df['hr_speed_ratio'] = moving_df['smoothed_heart_rate'] / moving_df['smoothed_speed_kmh']
    
    # Split the dataset in half chronologically
    midpoint = len(moving_df) // 2
    first_half = moving_df.iloc[:midpoint]
    second_half = moving_df.iloc[midpoint:]
    
    # Average the ratios
    ratio_1 = first_half['hr_speed_ratio'].mean()
    ratio_2 = second_half['hr_speed_ratio'].mean()
    
    # Calculate the percentage increase
    drift_percent = ((ratio_2 - ratio_1) / ratio_1) * 100
    return round(drift_percent, 2)

def get_basic_stats(df):
    """Calculates standard running metrics like total distance, time, and average pace."""
    if len(df) == 0:
        return {}
        
    # Safely get max distance (ignoring NaN values)
    total_distance = df['distance_km'].max() if 'distance_km' in df.columns else 0
    
    # Calculate Total Time
    total_seconds = (df.index[-1] - df.index[0]).total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    if hours > 0:
        formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        formatted_time = f"{minutes:02d}:{seconds:02d}"
        
    # Calculate Average Pace (min/km)
    if total_distance > 0:
        avg_pace_decimal = (total_seconds / 60) / total_distance
        pace_minutes = int(avg_pace_decimal)
        pace_seconds = int((avg_pace_decimal - pace_minutes) * 60)
        formatted_pace = f"{pace_minutes}:{pace_seconds:02d} /km"
    else:
        formatted_pace = "0:00 /km"
        
    # Calculate HR Averages
    avg_hr = int(df['smoothed_heart_rate'].mean()) if 'smoothed_heart_rate' in df.columns else 0
    max_hr = int(df['smoothed_heart_rate'].max()) if 'smoothed_heart_rate' in df.columns else 0
    
    return {
        "Distance": f"{total_distance:.2f} km",
        "Time": formatted_time,
        "Avg Pace": formatted_pace,
        "Avg HR": f"{avg_hr} bpm",
        "Max HR": f"{max_hr} bpm"
    }

def get_trimp_context(score):
    """Provides human-readable context for the TRIMP score."""
    if score < 50:
        return "Light Strain (Recovery, warm-up, or very short effort)"
    elif score < 120:
        return "Moderate Strain (Standard daily aerobic maintenance)"
    elif score < 200:
        return "High Strain (Hard workout, tempo, or long run)"
    else:
        return "Extreme Strain (Race day or grueling endurance event)"
    
import numpy as np

def generate_athlete_intelligence(df, stats, workout_type, drift, age, weight):
    """Generates a highly specific, lab-grade text analysis of the run."""
    if len(df) == 0 or 'smoothed_pace_min_km' not in df.columns:
        return "Not enough continuous data to generate DAKSHboard Intelligence."
        
    # --- 1. Pace Analysis (Kinematics) ---
    # We look at standard deviation to see how erratic the pacing was
    pace_std = df['smoothed_pace_min_km'].std()
    
    pace_text = f"**Pacing Kinematics:** You averaged {stats.get('Avg Pace', '0:00')} per kilometer. "
    if pace_std < 0.5:
        pace_text += f"Your pace variance was exceptionally tight (±{pace_std:.2f} min/km), indicating masterful mechanical efficiency and pacing discipline. You locked into your target rhythm and held it."
    elif pace_std < 1.0:
        pace_text += f"Your pace variance was moderate (±{pace_std:.2f} min/km). This is typical for rolling terrain or standard interval sessions where mechanical output naturally fluctuates."
    else:
        pace_text += f"Your pace showed high volatility (±{pace_std:.2f} min/km). If this wasn't a strict interval or hill workout, you may be bleeding kinetic energy through inconsistent effort pacing."

    # --- 2. Cardiovascular Analysis (Physiology) ---
    max_hr = df['smoothed_heart_rate'].max()
    hr_text = f"\n\n**Cardiovascular Load:** Peaking at {int(max_hr)} BPM, this session was classified as a *{workout_type}*. "
    
    if drift < 5.0:
        hr_text += f"With a cardiac drift of only {drift}%, your aerobic decoupling is practically non-existent. Your cardiovascular engine is highly adapted to this specific effort and duration; you did not experience significant late-stage fatigue."
    elif drift < 10.0:
        hr_text += f"Your cardiac drift hit {drift}%. As your core temperature rose and blood volume shifted, your heart had to work measurably harder to maintain the same mechanical output. This is a solid training stimulus."
    else:
        hr_text += f"A cardiac drift of {drift}% signals massive aerobic decoupling. Your physiological efficiency broke down significantly in the second half of the effort, likely due to dehydration, heat stress, or pushing beyond your current muscular endurance baseline."

    # --- 3. Biometric Context ---
    # Rough caloric estimate: duration (hours) * weight (kg) * METs (approx 10 for running)
    total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    est_calories = int(total_hours * weight * 10)
    
    bio_text = f"\n\n**Biometric Impact:** Based on your profile ({age} yrs, {weight} kg), this effort demanded approximately {est_calories} kilocalories of metabolic output. Prioritize glycogen replenishment and hydration to maximize supercompensation from this TRIMP load."

    return pace_text + hr_text + bio_text