import streamlit as st
import tempfile
import os
import plotly.graph_objects as go

# Import your custom DAKSHboard engine
from data_pipeline import extract_and_clean
from physiology import add_hr_zones, calculate_training_stress, classify_workout, calculate_cardiac_drift

# --- PAGE SETUP ---
st.set_page_config(page_title="DAKSHboard", page_icon="⚡", layout="wide")

st.title("⚡ DAKSHboard")
st.markdown("*Premium Endurance Analytics & Physiological Modeling*")
st.divider()

# --- SIDEBAR: USER SETTINGS ---
with st.sidebar:
    st.header("Athlete Profile")
    user_max_hr = st.number_input("Maximum Heart Rate", min_value=150, max_value=220, value=200)
    user_rest_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=100, value=50)
    st.divider()
    st.markdown("**Upload Workout**")
    uploaded_file = st.file_uploader("Drop a Garmin .fit file here", type=['fit'])

# --- MAIN DASHBOARD LOGIC ---
if uploaded_file is not None:
    # Streamlit needs to save the uploaded file temporarily so fitparse can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Crunching physiological data..."):
        try:
            # 1. Run the Data Pipeline
            df = extract_and_clean(tmp_file_path)
            
            # 2. Run the Math Engine
            df, hr_bins = add_hr_zones(df, user_max_hr, user_rest_hr)
            stress_score = calculate_training_stress(df)
            workout_type = classify_workout(df)
            drift = calculate_cardiac_drift(df)

            # --- TOP METRICS ROW ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Workout Classification", workout_type)
            col2.metric("TRIMP (Stress Score)", stress_score)
            
            # Highlight Drift: Green if under 5%, Red if over
            drift_color = "normal" if drift < 5.0 else "inverse"
            col3.metric("Cardiac Drift", f"{drift}%", delta_color=drift_color)
            
            st.divider()

            # --- INTERACTIVE GRAPH (PLOTLY) ---
            st.subheader("Heart Rate Dynamics")
            
            # Convert index to elapsed minutes for the X-axis
            elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
            
            fig = go.Figure()
            
            # Add the Heart Rate line
            fig.add_trace(go.Scatter(
                x=elapsed_minutes, 
                y=df['smoothed_heart_rate'], 
                mode='lines', 
                name='Heart Rate',
                line=dict(color='white', width=2)
            ))

            # Draw the background zones
            colors = ['#2980B9', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C']
            labels = ['Z1 (Recovery)', 'Z2 (Aerobic)', 'Z3 (Tempo)', 'Z4 (Threshold)', 'Z5 (Anaerobic)']
            
            for i in range(1, len(hr_bins)-1):
                fig.add_hrect(
                    y0=hr_bins[i], y1=hr_bins[i+1], 
                    fillcolor=colors[i-1], opacity=0.3, 
                    layer="below", line_width=0,
                    annotation_text=labels[i-1], annotation_position="top left"
                )

            fig.update_layout(
                xaxis_title="Elapsed Time (Minutes)",
                yaxis_title="Heart Rate (BPM)",
                height=500,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing file: {e}")
        finally:
            # Clean up the temp file
            os.remove(tmp_file_path)
else:
    st.info("👈 Upload a .fit file in the sidebar to generate your DAKSHboard insights.")