import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_pipeline import extract_and_clean
from physiology import (add_hr_zones, calculate_training_stress, classify_workout, 
                        calculate_cardiac_drift, get_basic_stats, get_trimp_context,
                        get_pace_insight, get_hr_insight, get_cadence_insight)

st.set_page_config(page_title="DAKSHboard", page_icon="⚡", layout="wide")

# --- SESSION STATE ---
if 'max_hr' not in st.session_state: st.session_state.max_hr = 200
if 'rest_hr' not in st.session_state: st.session_state.rest_hr = 50
if 'weight' not in st.session_state: st.session_state.weight = 71

# --- NAVIGATION ---
st.sidebar.title("⚡ DAKSHboard")
page = st.sidebar.radio("Navigation", ["📊 Analytics Dashboard", "👤 Athlete Profile"])
st.sidebar.divider()

if page == "👤 Athlete Profile":
    st.title("Athlete Profile Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=float(st.session_state.weight))
    with col2:
        st.session_state.max_hr = st.number_input("Maximum Heart Rate", min_value=150, max_value=220, value=st.session_state.max_hr)
        st.session_state.rest_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=100, value=st.session_state.rest_hr)

elif page == "📊 Analytics Dashboard":
    st.title("Performance Analytics")
    uploaded_file = st.sidebar.file_uploader("Upload Garmin .fit file", type=['fit'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner("Crunching physiological data..."):
            try:
                df = extract_and_clean(tmp_file_path)
                df, hr_bins = add_hr_zones(df, st.session_state.max_hr, st.session_state.rest_hr)
                
                stats = get_basic_stats(df, st.session_state.weight)
                stress_score = calculate_training_stress(df)
                workout_type = classify_workout(df)
                drift = calculate_cardiac_drift(df)

                # --- TOP LEVEL SUMMARY ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Classification", workout_type)
                c2.metric("TRIMP Score", stress_score)
                c3.metric("Distance", stats["Distance"]["Distance"])
                c4.metric("Total Time", stats["Timing"]["Elapsed Time"])
                st.divider()

                # --- NEW TABS ---
                tab1, tab2 = st.tabs(["📉 Telemetry & Coaching", "📋 Comprehensive Stats & Zones"])

                with tab1:
                    elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
                    active_df = df.dropna(subset=['smoothed_pace_min_km'])
                    active_minutes = (active_df.index - df.index[0]).total_seconds() / 60.0

                    # 1. PACE
                    st.subheader("Pace Kinematics")
                    fig_pace = go.Figure()
                    fig_pace.add_trace(go.Scatter(x=active_minutes, y=active_df['smoothed_pace_min_km'], mode='lines', line=dict(color='#3498DB', width=2)))
                    fig_pace.update_layout(height=300, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_pace, use_container_width=True)
                    st.info(get_pace_insight(df))
                    st.write("")

                    # 2. HEART RATE
                    st.subheader("Cardiovascular Response")
                    fig_hr = go.Figure()
                    fig_hr.add_trace(go.Scatter(x=elapsed_minutes, y=df['smoothed_heart_rate'], mode='lines', line=dict(color='white', width=2)))
                    colors = ['#2980B9', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C']
                    for i in range(1, len(hr_bins)-1):
                        fig_hr.add_hrect(y0=hr_bins[i], y1=hr_bins[i+1], fillcolor=colors[i-1], opacity=0.3, layer="below", line_width=0)
                    fig_hr.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_hr, use_container_width=True)
                    st.info(get_hr_insight(drift, workout_type))
                    st.write("")

                    # 3. CADENCE
                    if 'smoothed_spm' in df.columns:
                        st.subheader("Running Dynamics (Cadence)")
                        fig_cad = go.Figure()
                        fig_cad.add_trace(go.Scatter(x=active_minutes, y=active_df['smoothed_spm'], mode='lines', line=dict(color='#9B59B6', width=2)))
                        fig_cad.update_layout(height=300, yaxis_title="Steps per Minute", margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_cad, use_container_width=True)
                        st.info(get_cadence_insight(df))

                with tab2:
                    # --- THE GARMIN DATA GRID ---
                    st.subheader("Run Overview")
                    
                    # We create 3 columns to organize the dictionary groups cleanly
                    grid_cols = st.columns(3)
                    groups = list(stats.keys())
                    
                    for idx, group_name in enumerate(groups):
                        col = grid_cols[idx % 3] # Distribute evenly across the 3 columns
                        with col:
                            st.markdown(f"**{group_name}**")
                            for key, val in stats[group_name].items():
                                st.markdown(f"<span style='color:gray; font-size:0.9em'>{key}</span><br><span style='font-size:1.2em; font-weight:600'>{val}</span>", unsafe_allow_html=True)
                            st.markdown("<hr style='margin-top:0.5em; margin-bottom:1em'>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # --- ZONE BREAKDOWN ---
                    st.subheader("Intensity Distribution")
                    if 'hr_zone' in df.columns:
                        zone_counts = df['hr_zone'].value_counts().sort_index() / 60.0
                        zone_df = pd.DataFrame({'Zone': zone_counts.index, 'Minutes': zone_counts.values})
                        fig_zones = px.bar(zone_df, x='Minutes', y='Zone', orientation='h', color='Zone', color_discrete_sequence=colors + ['#E74C3C'])
                        fig_zones.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig_zones, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing file: {e}")
            finally:
                os.remove(tmp_file_path)
    else:
        st.info("👈 Upload a .fit file to view comprehensive analytics.")