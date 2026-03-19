import matplotlib.pyplot as plt

def plot_hr_with_zones(df, bins):
    """Plots the HR curve over time with color-coded background zones."""
    if 'smoothed_heart_rate' not in df.columns or len(bins) < 7:
        print("Missing data for HR plotting.")
        return
        
    plt.figure(figsize=(12, 6))
    
    # --- FIX 2: X-Axis Elapsed Time ---
    # Convert datetime index to strictly elapsed minutes starting from 0
    elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
    
    # Plot the line using the new elapsed time
    plt.plot(elapsed_minutes, df['smoothed_heart_rate'], color='#1C2833', linewidth=2.5, label='Heart Rate')
    
    # --- FIX 1 & 3: Saturated Colors & Better Labels ---
    # Using highly saturated colors: Gray, Vibrant Blue, Emerald, Yellow, Orange, Crimson
    colors = ['#BDC3C7', '#2980B9', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C']
    labels = ['< Z1 (Warmup)', 'Z1 (Recovery)', 'Z2 (Aerobic)', 'Z3 (Tempo)', 'Z4 (Threshold)', 'Z5 (Anaerobic)']
    
    for i in range(len(bins)-1):
        plt.axhspan(bins[i], bins[i+1], facecolor=colors[i], alpha=0.5, label=labels[i])
        
    # Formatting
    plt.title('DAKSHboard: Heart Rate Dynamics & Zones', fontsize=16, fontweight='bold')
    plt.xlabel('Elapsed Time (Minutes)', fontsize=12)
    plt.ylabel('Heart Rate (BPM)', fontsize=12)
    
    # --- FIX 3: Axis Limits and Grid Density ---
    # Set X-axis to start strictly at 0
    plt.xlim(0, elapsed_minutes.max())
    
    # Set Y-axis to start 15 beats below your lowest recorded HR so you see the whole curve
    min_hr = df['smoothed_heart_rate'].min()
    plt.ylim(max(30, min_hr - 15), bins[-1]) 
    
    # Turn on minor ticks for high-resolution grid lines
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    # Move legend outside
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plt.show()