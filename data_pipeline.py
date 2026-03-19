import fitparse
import pandas as pd
import numpy as np

def extract_and_clean(file_path):
    """Parses a Garmin .fit file and outputs a mathematically smoothed time-series dataframe."""
    fitfile = fitparse.FitFile(file_path)
    
    data = []
    for record in fitfile.get_messages('record'):
        record_dict = {}
        for data_element in record:
            record_dict[data_element.name] = data_element.value
        data.append(record_dict)
        
    df = pd.DataFrame(data)
    
    # 1. Target extended Garmin columns
    cols_to_keep = ['timestamp', 'heart_rate', 'distance', 'enhanced_speed', 
                    'cadence', 'fractional_cadence', 'enhanced_altitude', 'altitude', 'temperature']
    available_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[available_cols].copy()
    
    # 2. Time-Series Indexing & Resampling (The "Smart Recording" Fix)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.resample('1s').asfreq()
    df = df.interpolate(method='linear')
    
    # 3. Physics & Smoothing
    if 'enhanced_speed' in df.columns:
        df['speed_kmh'] = df['enhanced_speed'] * 3.6
        df['moving_speed_kmh'] = df['speed_kmh'].replace(0, np.nan)
        df['smoothed_speed_kmh'] = df['moving_speed_kmh'].rolling(window='10s', min_periods=1).mean()
        df['smoothed_pace_min_km'] = np.where(df['smoothed_speed_kmh'] > 0.5, 
                                              60 / df['smoothed_speed_kmh'], np.nan)
                                              
    if 'distance' in df.columns:
        df['distance_km'] = df['distance'] / 1000

    if 'heart_rate' in df.columns:
        df['smoothed_heart_rate'] = df['heart_rate'].rolling(window='10s', min_periods=1).mean()
        
    # 4. Biomechanics (Cadence)
    if 'cadence' in df.columns:
        # Garmin sometimes logs single leg. If average is < 120, it's likely single leg.
        multiplier = 2 if df['cadence'].mean() < 120 else 1
        df['spm'] = df['cadence'] * multiplier
        df['smoothed_spm'] = df['spm'].replace(0, np.nan).rolling(window='10s', min_periods=1).mean()

    # 5. Environment (Elevation & Temp)
    if 'enhanced_altitude' in df.columns:
        df['elevation_m'] = df['enhanced_altitude']
    elif 'altitude' in df.columns:
        df['elevation_m'] = df['altitude']
        
    return df