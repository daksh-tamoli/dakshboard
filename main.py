from data_pipeline import extract_and_clean
from physiology import add_hr_zones, calculate_training_stress, classify_workout
from visuals import plot_hr_with_zones

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

def run_dakshboard():
    print("\n--- DAKSHboard Setup ---")
    
    # 1. Ask for the file dynamically
    # The 'or "test_run.fit"' part is a neat Python trick. 
    # If you just hit Enter without typing anything, it defaults to your original file.
    file_name = input("Enter the name of your .fit file (e.g., hard_run.fit) [Default: test_run.fit]: ") or "test_run.fit"
    
    # 2. Ingest Data
    try:
        df = extract_and_clean(file_name)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{file_name}'. Make sure it is in the same folder as main.py!")
        return # Stops the script gracefully
    
    # 3. Setup Baseline
    user_max_hr = int(input("Enter Maximum HR: "))
    user_rest_hr = int(input("Enter Resting HR: "))
    
    # 4. Physiology Math
    df, hr_bins = add_hr_zones(df, user_max_hr, user_rest_hr)
    stress_score = calculate_training_stress(df)
    workout_type = classify_workout(df)
    stress_context = get_trimp_context(stress_score)
    
    # 5. Output Text Insights
    print("\n--- DAKSHboard Analytics ---")
    print(f"File Analyzed: {file_name}")
    print(f"Algorithm Classification: ** {workout_type.upper()} **")
    print(f"Training Stress Score (TRIMP): {stress_score}")
    print(f"TRIMP Analysis: {stress_context}")
    
    # 6. Render Graphics
    print("\nRendering physiological graph...")
    plot_hr_with_zones(df, hr_bins)

if __name__ == "__main__":
    run_dakshboard()