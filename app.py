import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import your custom DAKSHboard engine
from data_pipeline import extract_and_clean
from physiology import (add_hr_zones, calculate_training_stress, 
                        classify_workout, calculate_cardiac_drift, 
                        get_basic_stats, get_trimp_context)

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Crunching physiological data..."):
        try:
            # 1. Run the Engines
            df = extract_and_clean(tmp_file_path)
            df, hr_bins = add_hr_zones(df, user_max_hr, user_rest_hr)
            
            # 2. Extract Metrics
            stats = get_basic_stats(df)
            stress_score = calculate_training_stress(df)
            workout_type = classify_workout(df)
            drift = calculate_cardiac_drift(df)
            stress_context = get_trimp_context(stress_score)

            # --- CREATE TABS ---
            tab1, tab2 = st.tabs(["📊 Primary Dashboard", "🔬 Detailed Insights"])

            # ==========================================
            # TAB 1: PRIMARY DASHBOARD
            # ==========================================
            with tab1:
                # Row 1: The Basic Stats (Strava Style)
                st.subheader(f"Run Classification: {workout_type}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Distance", stats.get("Distance", "0 km"))
                col2.metric("Elapsed Time", stats.get("Time", "0:00"))
                col3.metric("Avg Pace", stats.get("Avg Pace", "0:00 /km"))
                col4.metric("Avg Heart Rate", stats.get("Avg HR", "0 bpm"))
                
                st.divider()

                # The Interactive Graph
                st.subheader("Heart Rate Dynamics")
                elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=elapsed_minutes, y=df['smoothed_heart_rate'], 
                    mode='lines', name='Heart Rate', line=dict(color='white', width=2)
                ))

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
                    xaxis_title="Elapsed Time (Minutes)", yaxis_title="Heart Rate (BPM)",
                    height=450, margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

            # ==========================================
            # TAB 2: DETAILED INSIGHTS
            # ==========================================
            with tab2:
                st.subheader("Physiological Stress Analysis")
                
                # Premium Metrics Row
                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric("TRIMP (Stress Score)", stress_score)
                drift_color = "normal" if drift < 5.0 else "inverse"
                p_col2.metric("Cardiac Drift", f"{drift}%", delta_color=drift_color)
                p_col3.metric("Max Heart Rate", stats.get("Max HR", "0 bpm"))
                
                st.info(f"**TRIMP Analysis:** {stress_context}")
                st.divider()
                
                # Zone Distribution Bar Chart
                st.subheader("Time in Zones")
                if 'hr_zone' in df.columns:
                    # Calculate minutes per zone
                    zone_counts = df['hr_zone'].value_counts().sort_index() / 60.0
                    zone_df = pd.DataFrame({'Zone': zone_counts.index, 'Minutes': zone_counts.values})
                    
                    # Create a colorful bar chart
                    fig_zones = px.bar(
                        zone_df, x='Minutes', y='Zone', orientation='h',
                        color='Zone', 
                        color_discrete_sequence=['#BDC3C7', '#2980B9', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C']
                    )
                    fig_zones.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_zones, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing file: {e}")
        finally:
            os.remove(tmp_file_path)
else:
    st.info("👈 Upload a .fit file in the sidebar to generate your DAKSHboard insights.")