import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_pipeline import extract_and_clean
from physiology import (add_hr_zones, calculate_training_stress, 
                        classify_workout, calculate_cardiac_drift, 
                        get_basic_stats, get_trimp_context, generate_athlete_intelligence)

# --- PAGE SETUP ---
st.set_page_config(page_title="DAKSHboard", page_icon="⚡", layout="wide")

# --- INITIALIZE SESSION STATE (Profile Defaults) ---
if 'max_hr' not in st.session_state: st.session_state.max_hr = 200
if 'rest_hr' not in st.session_state: st.session_state.rest_hr = 50
if 'age' not in st.session_state: st.session_state.age = 19
if 'weight' not in st.session_state: st.session_state.weight = 71
if 'height' not in st.session_state: st.session_state.height = 175

# --- NAVIGATION ---
st.sidebar.title("⚡ DAKSHboard")
page = st.sidebar.radio("Navigation", ["📊 Analytics Dashboard", "👤 Athlete Profile"])
st.sidebar.divider()

if page == "👤 Athlete Profile":
    st.title("Athlete Profile Settings")
    st.markdown("Calibrate your physiological baselines to ensure hyper-accurate algorithmic analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Biometrics")
        st.session_state.age = st.number_input("Age", min_value=10, max_value=100, value=st.session_state.age)
        st.session_state.weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=float(st.session_state.weight))
        st.session_state.height = st.number_input("Height (cm)", min_value=100, max_value=250, value=st.session_state.height)
    with col2:
        st.subheader("Cardiovascular Engine")
        st.session_state.max_hr = st.number_input("Maximum Heart Rate", min_value=150, max_value=220, value=st.session_state.max_hr)
        st.session_state.rest_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=100, value=st.session_state.rest_hr)
        
    st.success("Profile updated in local session state. Return to the Dashboard to analyze data.")

elif page == "📊 Analytics Dashboard":
    st.title("Performance Analytics")
    
    uploaded_file = st.sidebar.file_uploader("Upload Garmin .fit file", type=['fit'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner("Crunching physiological data..."):
            try:
                # 1. Pipeline & Engine
                df = extract_and_clean(tmp_file_path)
                df, hr_bins = add_hr_zones(df, st.session_state.max_hr, st.session_state.rest_hr)
                
                # 2. Extract Metrics
                stats = get_basic_stats(df)
                stress_score = calculate_training_stress(df)
                workout_type = classify_workout(df)
                drift = calculate_cardiac_drift(df)
                
                # Generate AI Text
                intelligence_report = generate_athlete_intelligence(
                    df, stats, workout_type, drift, 
                    st.session_state.age, st.session_state.weight
                )

                # --- TOP ROW STATS ---
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Distance", stats.get("Distance", "0 km"))
                col2.metric("Time", stats.get("Time", "0:00"))
                col3.metric("Avg Pace", stats.get("Avg Pace", "0:00 /km"))
                col4.metric("TRIMP", stress_score)
                col5.metric("Drift", f"{drift}%", delta_color="normal" if drift < 5.0 else "inverse")
                
                st.divider()

                # --- TABS FOR GRAPHS & INSIGHTS ---
                tab1, tab2, tab3 = st.tabs(["📉 Visual telemetry", "🧠 DAKSHboard Intelligence", "🧬 Zone Breakdown"])

                # TAB 1: VISUALS (HR & PACE)
                with tab1:
                    elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
                    
                    # PACE CHART (Inverted Y-Axis so faster pace is higher)
                    st.subheader("Pace Kinematics")
                    fig_pace = go.Figure()
                    # Filter out NaN/stopped times for a cleaner pace graph
                    active_df = df.dropna(subset=['smoothed_pace_min_km'])
                    active_minutes = (active_df.index - df.index[0]).total_seconds() / 60.0
                    
                    fig_pace.add_trace(go.Scatter(
                        x=active_minutes, y=active_df['smoothed_pace_min_km'], 
                        mode='lines', line=dict(color='#3498DB', width=2)
                    ))
                    fig_pace.update_layout(
                        xaxis_title="Elapsed Time (Minutes)", yaxis_title="Pace (min/km)",
                        yaxis=dict(autorange="reversed"), # CRITICAL: Faster pace at top
                        height=350, margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_pace, use_container_width=True)

                    # HR CHART
                    st.subheader("Cardiovascular Response")
                    fig_hr = go.Figure()
                    fig_hr.add_trace(go.Scatter(
                        x=elapsed_minutes, y=df['smoothed_heart_rate'], 
                        mode='lines', line=dict(color='white', width=2)
                    ))
                    colors = ['#2980B9', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C']
                    labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
                    for i in range(1, len(hr_bins)-1):
                        fig_hr.add_hrect(
                            y0=hr_bins[i], y1=hr_bins[i+1], 
                            fillcolor=colors[i-1], opacity=0.3, layer="below", line_width=0,
                            annotation_text=labels[i-1], annotation_position="top left"
                        )
                    fig_hr.update_layout(
                        xaxis_title="Elapsed Time (Minutes)", yaxis_title="Heart Rate (BPM)",
                        height=350, margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_hr, use_container_width=True)

                # TAB 2: AI ANALYSIS
                with tab2:
                    st.subheader("Algorithmic Assessment")
                    st.info(f"**Primary Classification:** {workout_type.upper()}")
                    st.markdown(intelligence_report)

                # TAB 3: ZONES
                with tab3:
                    st.subheader("Time in Zones")
                    if 'hr_zone' in df.columns:
                        zone_counts = df['hr_zone'].value_counts().sort_index() / 60.0
                        zone_df = pd.DataFrame({'Zone': zone_counts.index, 'Minutes': zone_counts.values})
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
        st.info("👈 Configure your profile and upload a .fit file to begin.")