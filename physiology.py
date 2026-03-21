import pandas as pd
import numpy as np

# ==========================================
# 1. CORE PHYSIOLOGICAL MATH
# ==========================================

def add_hr_zones(df, max_hr, rest_hr):
    if 'smoothed_heart_rate' not in df.columns:
        return df, []
    hrr = max_hr - rest_hr
    bins = [0, rest_hr + (hrr * 0.50), rest_hr + (hrr * 0.60), 
            rest_hr + (hrr * 0.70), rest_hr + (hrr * 0.80), 
            rest_hr + (hrr * 0.90), max_hr + 10]
    labels = ['Rest', 'Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
    df['hr_zone'] = pd.cut(df['smoothed_heart_rate'], bins=bins, labels=labels)
    return df, bins

def calculate_training_stress(df):
    if 'hr_zone' not in df.columns:
        return 0
    zone_counts = df['hr_zone'].value_counts()
    stress_score = (
        (zone_counts.get('Zone 1', 0) / 60) * 1 +
        (zone_counts.get('Zone 2', 0) / 60) * 2 +
        (zone_counts.get('Zone 3', 0) / 60) * 3 +
        (zone_counts.get('Zone 4', 0) / 60) * 4 +
        (zone_counts.get('Zone 5', 0) / 60) * 5
    )
    return round(stress_score)

def classify_workout(df):
    if 'hr_zone' not in df.columns: return "Unclassified"
    total_active_seconds = len(df[df['hr_zone'] != 'Rest'])
    if total_active_seconds == 0: return "Rest / No Data"
    zone_counts = df['hr_zone'].value_counts()
    z1_z2_pct = (zone_counts.get('Zone 1', 0) + zone_counts.get('Zone 2', 0)) / total_active_seconds
    z3_pct = zone_counts.get('Zone 3', 0) / total_active_seconds
    z4_z5_pct = (zone_counts.get('Zone 4', 0) + zone_counts.get('Zone 5', 0)) / total_active_seconds
    if z4_z5_pct > 0.15: return "Threshold / VO2 Max Interval"
    elif z3_pct > 0.30: return "Tempo Run"
    elif z1_z2_pct > 0.80: return "Recovery / Easy Aerobic"
    else: return "Mixed Aerobic Base"

def calculate_cardiac_drift(df):
    if 'smoothed_speed_kmh' not in df.columns or 'smoothed_heart_rate' not in df.columns: return 0.0
    moving_df = df[df['smoothed_speed_kmh'] > 0.5].copy()
    if len(moving_df) < 120: return 0.0
    moving_df['hr_speed_ratio'] = moving_df['smoothed_heart_rate'] / moving_df['smoothed_speed_kmh']
    midpoint = len(moving_df) // 2
    first_half = moving_df.iloc[:midpoint]
    second_half = moving_df.iloc[midpoint:]
    ratio_1 = first_half['hr_speed_ratio'].mean()
    ratio_2 = second_half['hr_speed_ratio'].mean()
    if ratio_1 == 0: return 0.0
    drift_percent = ((ratio_2 - ratio_1) / ratio_1) * 100
    return round(drift_percent, 2)

# ==========================================
# 2. DATA GRID EXTRACTION (GARMIN STYLE)
# ==========================================

def get_basic_stats(df, weight_kg):
    if len(df) == 0: return {}
    active_df = df.dropna(subset=['smoothed_speed_kmh']) if 'smoothed_speed_kmh' in df.columns else df
    total_sec = (df.index[-1] - df.index[0]).total_seconds()
    moving_sec = len(active_df)
    
    def fmt_time(seconds):
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
    def fmt_pace(speed_kmh):
        if pd.isna(speed_kmh) or speed_kmh <= 0.5: return "-- /km"
        pace_dec = 60 / speed_kmh
        m, s = int(pace_dec), int((pace_dec - int(pace_dec)) * 60)
        return f"{m}:{s:02d} /km"

    dist = df['distance_km'].max() if 'distance_km' in df.columns else 0
    avg_speed = active_df['smoothed_speed_kmh'].mean() if 'smoothed_speed_kmh' in df.columns else 0
    max_speed = active_df['smoothed_speed_kmh'].max() if 'smoothed_speed_kmh' in df.columns else 0
    avg_hr = int(df['smoothed_heart_rate'].mean()) if 'smoothed_heart_rate' in df.columns else 0
    max_hr = int(df['smoothed_heart_rate'].max()) if 'smoothed_heart_rate' in df.columns else 0
    avg_cadence = int(active_df['smoothed_spm'].mean()) if 'smoothed_spm' in df.columns else 0
    max_cadence = int(active_df['smoothed_spm'].max()) if 'smoothed_spm' in df.columns else 0
    avg_stride = ((avg_speed * 1000) / 60) / avg_cadence if avg_cadence > 0 else 0
    
    elev_gain = "--"
    min_elev, max_elev = "--", "--"
    if 'elevation_m' in df.columns:
        min_elev = f"{int(df['elevation_m'].min())} m"
        max_elev = f"{int(df['elevation_m'].max())} m"
        diffs = df['elevation_m'].diff()
        elev_gain = f"{int(diffs[diffs > 0].sum())} m"
        
    est_calories = int(weight_kg * dist * 1.036)
    est_sweat = int((moving_sec / 3600) * 800)
    avg_temp = f"{df['temperature'].mean():.1f} °C" if 'temperature' in df.columns else "--"

    return {
        "Distance": {"Distance": f"{dist:.2f} km"},
        "Timing": {"Time": fmt_time(total_sec), "Moving Time": fmt_time(moving_sec), "Elapsed Time": fmt_time(total_sec)},
        "Pace/Speed": {"Avg Pace": fmt_pace(avg_speed), "Best Pace": fmt_pace(max_speed)},
        "Heart Rate": {"Avg HR": f"{avg_hr} bpm", "Max HR": f"{max_hr} bpm"},
        "Running Dynamics": {"Avg Cadence": f"{avg_cadence} spm", "Max Cadence": f"{max_cadence} spm", "Avg Stride": f"{avg_stride:.2f} m"},
        "Elevation": {"Total Ascent": elev_gain, "Min Elev": min_elev, "Max Elev": max_elev},
        "Nutrition & Hydration": {"Est. Calories": f"{est_calories} kcal", "Est. Sweat Loss": f"{est_sweat} ml"},
        "Environment": {"Avg Temp": avg_temp}
    }

# ==========================================
# 3. FRAGMENTED COACHING INTELLIGENCE
# ==========================================

def get_pace_insight(df):
    if 'smoothed_pace_min_km' not in df.columns: return "Insufficient pacing data."
    std = df['smoothed_pace_min_km'].std()
    if std < 0.4: return f"**Kinematics:** Exceptionally tight pacing variance (±{std:.2f} min/km)."
    elif std < 0.8: return f"**Kinematics:** Moderate pacing variance (±{std:.2f} min/km)."
    return f"**Kinematics:** High pacing volatility (±{std:.2f} min/km)."

def get_hr_insight(drift, workout_type):
    if drift < 5.0: return f"**Physiology ({workout_type}):** Outstanding cardiovascular durability ({drift}% drift)."
    elif drift < 10.0: return f"**Physiology ({workout_type}):** Normal stress response ({drift}% drift)."
    return f"**Physiology ({workout_type}):** Significant aerobic decoupling ({drift}%)."

def get_cadence_insight(df):
    if 'smoothed_spm' not in df.columns: return "Cadence sensor data not detected."
    avg_spm = df['smoothed_spm'].mean()
    if avg_spm >= 170: return f"**Biomechanics:** Elite turnover rate ({int(avg_spm)} spm)."
    elif avg_spm >= 160: return f"**Biomechanics:** Solid turnover ({int(avg_spm)} spm)."
    return f"**Biomechanics:** Low turnover ({int(avg_spm)} spm). Consider increasing step frequency."

def get_trimp_context(score):
    if score < 50: return "Light Strain"
    elif score < 120: return "Moderate Strain"
    elif score < 200: return "High Strain"
    return "Extreme Strain"

# ==========================================
# 4. PMC METRICS (THE MISSING FUNCTION)
# ==========================================

def calculate_pmc_metrics(history_df):
    """Calculates CTL (Fitness), ATL (Fatigue), and TSB (Form) from the training log."""
    if history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date
    daily_trimp = df.groupby('date')['trimp'].sum().reset_index()
    daily_trimp.set_index('date', inplace=True)

    idx = pd.date_range(daily_trimp.index.min(), daily_trimp.index.max() + pd.Timedelta(days=14))
    daily_trimp.index = pd.DatetimeIndex(daily_trimp.index)
    daily_trimp = daily_trimp.reindex(idx, fill_value=0)

    pmc = pd.DataFrame(index=daily_trimp.index)
    pmc['Daily TRIMP'] = daily_trimp['trimp']
    pmc['CTL (Fitness)'] = daily_trimp['trimp'].ewm(span=42, adjust=False).mean()
    pmc['ATL (Fatigue)'] = daily_trimp['trimp'].ewm(span=7, adjust=False).mean()
    pmc['TSB (Form)'] = pmc['CTL (Fitness)'].shift(1) - pmc['ATL (Fatigue)'].shift(1)
    
    return pmc.fillna(0).round(1)