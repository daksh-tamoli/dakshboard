import fitparse
import pandas as pd
import numpy as np

def extract_and_clean(file_path):
    """Parses a Garmin .fit file and outputs a mathematically smoothed time-series dataframe."""
    print(f"Loading {file_path} into the DAKSHboard data pipeline...")
    fitfile = fitparse.FitFile(file_path)
    
    data = []
    for record in fitfile.get_messages('record'):
        record_dict = {}
        for data_element in record:
            record_dict[data_element.name] = data_element.value
        data.append(record_dict)
        
    df = pd.DataFrame(data)
    
    # 1. Target your specific Garmin columns
    cols_to_keep = ['timestamp', 'heart_rate', 'distance', 'enhanced_speed']
    available_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[available_cols].copy()
    
    # 2. Time-Series Indexing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    # --- THE ARCHITECTURE FIX: 1-SECOND RESAMPLING ---
    # This forces the dataframe to create a row for every single second 
    # between your start time and end time.
    df = df.resample('1s').asfreq()

    # 3. Interpolate dead zones
    # Now that we have a perfect 1-second grid, this connects the dots
    # across any gaps left by Garmin's smart recording.
    df = df.interpolate(method='linear')
    
    # 4. The Physics Translation & Time-Based Smoothing
    if 'enhanced_speed' in df.columns:
        # Convert m/s to km/h
        df['speed_kmh'] = df['enhanced_speed'] * 3.6
        
        # Replace 0.0 speeds with NaN so stops don't drag down the moving average
        df['moving_speed_kmh'] = df['speed_kmh'].replace(0, np.nan)
        
        # Smooth the speed using a 10-second time-based window
        df['smoothed_speed_kmh'] = df['moving_speed_kmh'].rolling(window='10s', min_periods=1).mean()
        
        # Calculate Pace (minutes per kilometer) on the smoothed speed
        df['smoothed_pace_min_km'] = np.where(df['smoothed_speed_kmh'] > 0.5, 
                                              60 / df['smoothed_speed_kmh'], 
                                              np.nan)
                                              
    if 'distance' in df.columns:
        # Convert meters to kilometers
        df['distance_km'] = df['distance'] / 1000

    if 'heart_rate' in df.columns:
        # Smooth heart rate with a 10-second time window to remove static
        df['smoothed_heart_rate'] = df['heart_rate'].rolling(window='10s', min_periods=1).mean()
        
    return df