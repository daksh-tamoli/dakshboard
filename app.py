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
from database import init_db, run_exists, save_run, load_history

st.set_page_config(page_title="DAKSHboard", page_icon="⚡", layout="wide")

# --- INITIALIZE DATABASE & SESSION STATE ---
init_db()

if 'max_hr' not in st.session_state: st.session_state.max_hr = 200
if 'rest_hr' not in st.session_state: st.session_state.rest_hr = 50
if 'age' not in st.session_state: st.session_state.age = 19
if 'weight' not in st.session_state: st.session_state.weight = 71.0
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

elif page == "📊 Analytics Dashboard":
    st.title("Performance Analytics")
    
    # --- MULTI-FILE UPLOADER ---
    st.sidebar.markdown("**Upload Training Data**")
    uploaded_files = st.sidebar.file_uploader("Upload Garmin .fit files", type=['fit'], accept_multiple_files=True)

    # Process new files into the database
    if uploaded_files:
        with st.spinner("Syncing new runs to database..."):
            for uploaded_file in uploaded_files:
                if not run_exists(uploaded_file.name):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Extract core data just to save to DB
                        df_temp = extract_and_clean(tmp_file_path)
                        df_temp, _ = add_hr_zones(df_temp, st.session_state.max_hr, st.session_state.rest_hr)
                        
                        run_date = df_temp.index[0].strftime("%Y-%m-%d %H:%M")
                        stats_temp = get_basic_stats(df_temp, st.session_state.weight)
                        trimp = calculate_training_stress(df_temp)
                        workout = classify_workout(df_temp)
                        drift = calculate_cardiac_drift(df_temp)
                        
                        # Save to SQLite
                        save_run(uploaded_file.name, run_date, stats_temp["Distance"]["Distance"], 
                                 trimp, workout, stats_temp["Pace/Speed"]["Avg Pace"], drift)
                    except Exception as e:
                        st.sidebar.error(f"Failed to process {uploaded_file.name}")
                    finally:
                        os.remove(tmp_file_path)

    # Load the historical database
    history_df = load_history()

    if not history_df.empty:
        # Let the user select which run to view in detail
        selected_filename = st.selectbox("Select a run to analyze:", history_df['filename'].tolist())
        
        # Find the actual file object from the uploaded list
        selected_file = next((f for f in uploaded_files if f.name == selected_filename), None)
        
        if selected_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
                tmp_file.write(selected_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # 1. Pipeline & Engine
                df = extract_and_clean(tmp_file_path)
                df, hr_bins = add_hr_zones(df, st.session_state.max_hr, st.session_state.rest_hr)
                
                # 2. Extract Metrics
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

                # --- TABS ---
                tab1, tab2, tab3 = st.tabs(["📉 Telemetry & Coaching", "📋 Comprehensive Stats & Zones", "📚 Training Log"])

                with tab1:
                    elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
                    active_df = df.dropna(subset=['smoothed_pace_min_km'])
                    active_minutes = (active_df.index - df.index[0]).total_seconds() / 60.0

                    # PACE
                    st.subheader("Pace Kinematics")
                    fig_pace = go.Figure()
                    fig_pace.add_trace(go.Scatter(x=active_minutes, y=active_df['smoothed_pace_min_km'], mode='lines', line=dict(color='#3498DB', width=2)))
                    fig_pace.update_layout(height=300, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_pace, use_container_width=True)
                    st.info(get_pace_insight(df))
                    st.write("")

                    # HEART RATE
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

                    # CADENCE
                    if 'smoothed_spm' in df.columns:
                        st.subheader("Running Dynamics (Cadence)")
                        fig_cad = go.Figure()
                        fig_cad.add_trace(go.Scatter(x=active_minutes, y=active_df['smoothed_spm'], mode='lines', line=dict(color='#9B59B6', width=2)))
                        fig_cad.update_layout(height=300, yaxis_title="Steps per Minute", margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_cad, use_container_width=True)
                        st.info(get_cadence_insight(df))

                with tab2:
                    st.subheader("Run Overview")
                    grid_cols = st.columns(3)
                    groups = list(stats.keys())
                    for idx, group_name in enumerate(groups):
                        col = grid_cols[idx % 3] 
                        with col:
                            st.markdown(f"**{group_name}**")
                            for key, val in stats[group_name].items():
                                st.markdown(f"<span style='color:gray; font-size:0.9em'>{key}</span><br><span style='font-size:1.2em; font-weight:600'>{val}</span>", unsafe_allow_html=True)
                            st.markdown("<hr style='margin-top:0.5em; margin-bottom:1em'>", unsafe_allow_html=True)
                    st.divider()
                    st.subheader("Intensity Distribution")
                    if 'hr_zone' in df.columns:
                        zone_counts = df['hr_zone'].value_counts().sort_index() / 60.0
                        zone_df = pd.DataFrame({'Zone': zone_counts.index, 'Minutes': zone_counts.values})
                        fig_zones = px.bar(zone_df, x='Minutes', y='Zone', orientation='h', color='Zone', color_discrete_sequence=colors + ['#E74C3C'])
                        fig_zones.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig_zones, use_container_width=True)

                with tab3:
                    st.subheader("Historical Training Log")
                    st.markdown("This database stores your aggregate metrics for the Race Predictor model.")
                    st.dataframe(history_df, use_container_width=True, hide_index=True)

            finally:
                os.remove(tmp_file_path)
    else:
        st.info("👈 Upload your .fit files to begin building your database.")