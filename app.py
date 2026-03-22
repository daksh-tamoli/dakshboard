import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_pipeline import extract_and_clean
from physiology import (add_hr_zones, calculate_training_stress, classify_workout, 
                        calculate_cardiac_drift, get_basic_stats, get_trimp_context,
                        get_pace_insight, get_hr_insight, get_cadence_insight, 
                        get_elevation_insight, calculate_pmc_metrics) # Added elevation insight
from database import init_db, run_exists, save_run, load_history, get_user_profile, save_user_profile
from ml_engine import detect_anomalies, calculate_recovery_hours, get_training_status

st.set_page_config(page_title="DAKSHboard", page_icon="⚡", layout="wide")

# --- CUSTOM UI / UX CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    [data-testid="stMetric"] {
        background-color: #191C24;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        border: 1px solid #2D313A;
        transition: transform 0.2s ease-in-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00FFCC;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0E1117;
        padding: 10px 10px 0 10px;
        border-radius: 12px 12px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #1E2127;
        border-radius: 8px 8px 0 0;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2D313A;
        border-top: 2px solid #00FFCC !important;
        color: #00FFCC !important;
    }
    </style>
""", unsafe_allow_html=True)

init_db()
st.session_state.email = "local_athlete"

if 'profile_loaded' not in st.session_state:
    profile = get_user_profile(st.session_state.email)
    if profile:
        st.session_state.age, st.session_state.weight, st.session_state.height, st.session_state.max_hr, st.session_state.rest_hr = profile
    else:
        st.session_state.age, st.session_state.weight, st.session_state.height, st.session_state.max_hr, st.session_state.rest_hr = 19, 71.0, 175, 200, 50
    st.session_state.profile_loaded = True

st.sidebar.title("⚡ DAKSHboard")
page = st.sidebar.radio("Navigation", ["📊 Analytics Dashboard", "⚙️ Athlete Profile"])
st.sidebar.divider()

if page == "⚙️ Athlete Profile":
    st.title("Athlete Profile Settings")
    col1, col2 = st.columns(2)
    with col1:
        new_age = st.number_input("Age", min_value=10, max_value=100, value=st.session_state.age)
        new_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=float(st.session_state.weight))
        new_height = st.number_input("Height (cm)", min_value=100, max_value=250, value=st.session_state.height)
    with col2:
        new_max_hr = st.number_input("Maximum Heart Rate", min_value=150, max_value=220, value=st.session_state.max_hr)
        new_rest_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=100, value=st.session_state.rest_hr)
        
    if st.button("Save Profile"):
        st.session_state.update({"age": new_age, "weight": new_weight, "height": new_height, "max_hr": new_max_hr, "rest_hr": new_rest_hr})
        save_user_profile(st.session_state.email, new_age, new_weight, new_height, new_max_hr, new_rest_hr)
        st.success("Profile permanently synced to database.")

elif page == "📊 Analytics Dashboard":
    st.title("Performance Analytics")
    uploaded_files = st.sidebar.file_uploader("Upload Garmin .fit files", type=['fit'], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Syncing runs to your database..."):
            for uploaded_file in uploaded_files:
                if not run_exists(st.session_state.email, uploaded_file.name):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    try:
                        df_temp = extract_and_clean(tmp_file_path)
                        df_temp, _ = add_hr_zones(df_temp, st.session_state.max_hr, st.session_state.rest_hr)
                        run_date = df_temp.index[0].strftime("%Y-%m-%d %H:%M")
                        stats_temp = get_basic_stats(df_temp, st.session_state.weight)
                        trimp = calculate_training_stress(df_temp)
                        workout = classify_workout(df_temp)
                        drift = calculate_cardiac_drift(df_temp)
                        save_run(st.session_state.email, uploaded_file.name, run_date, stats_temp["Distance"]["Distance"], trimp, workout, stats_temp["Pace/Speed"]["Avg Pace"], drift)
                    except Exception:
                        pass
                    finally:
                        os.remove(tmp_file_path)

    history_df = load_history(st.session_state.email)

    if not history_df.empty:
        selected_filename = st.selectbox("Select a run to analyze:", history_df['filename'].tolist())
        selected_file = next((f for f in uploaded_files if f.name == selected_filename), None)
        
        if selected_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
                tmp_file.write(selected_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                df = extract_and_clean(tmp_file_path)
                df, hr_bins = add_hr_zones(df, st.session_state.max_hr, st.session_state.rest_hr)
                
                stats = get_basic_stats(df, st.session_state.weight)
                stress_score = calculate_training_stress(df)
                workout_type = classify_workout(df)
                drift = calculate_cardiac_drift(df)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Classification", workout_type)
                c2.metric("TRIMP Score", stress_score)
                c3.metric("Distance", stats["Distance"]["Distance"])
                c4.metric("Total Time", stats["Timing"]["Elapsed Time"])
                st.divider()

                tab1, tab2, tab3, tab4 = st.tabs(["📉 Telemetry", "📋 Comprehensive Stats", "📚 Training Log", "📈 Machine Learning Insights"])

                with tab1:
                    elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
                    active_df = df.dropna(subset=['smoothed_pace_min_km'])
                    active_minutes = (active_df.index - df.index[0]).total_seconds() / 60.0

                    # --- ELEVATION CHART & INSIGHT ---
                    if 'elevation_m' in df.columns and not df['elevation_m'].isnull().all():
                        st.subheader("Topographical Profile")
                        fig_elev = go.Figure(go.Scatter(x=elapsed_minutes, y=df['elevation_m'], mode='lines', fill='tozeroy', line=dict(color='#27AE60', width=2)))
                        fig_elev.update_layout(
                            height=250, yaxis_title="Elevation (m)", margin=dict(l=0, r=0, t=10, b=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#A0AEC0'),
                            yaxis=dict(showgrid=True, gridcolor='#2D313A', zeroline=False), xaxis=dict(showgrid=False, zeroline=False)
                        )
                        st.plotly_chart(fig_elev, use_container_width=True)
                        st.info(get_elevation_insight(df))
                        st.write("")

                    # PACE
                    st.subheader("Pace Kinematics")
                    fig_pace = go.Figure(go.Scatter(x=active_minutes, y=active_df['smoothed_pace_min_km'], mode='lines', line=dict(color='#3498DB', width=2)))
                    fig_pace.update_layout(
                        height=250, yaxis=dict(autorange="reversed", showgrid=True, gridcolor='#2D313A', zeroline=False),
                        xaxis=dict(showgrid=False, zeroline=False), margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#A0AEC0')
                    )
                    st.plotly_chart(fig_pace, use_container_width=True)
                    st.info(get_pace_insight(df))
                    st.write("")

                    # HEART RATE
                    st.subheader("Cardiovascular Response")
                    fig_hr = go.Figure(go.Scatter(x=elapsed_minutes, y=df['smoothed_heart_rate'], mode='lines', line=dict(color='white', width=2)))
                    colors = ['#2980B9', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C']
                    for i in range(1, len(hr_bins)-1):
                        fig_hr.add_hrect(y0=hr_bins[i], y1=hr_bins[i+1], fillcolor=colors[i-1], opacity=0.3, layer="below", line_width=0)
                    fig_hr.update_layout(
                        height=250, yaxis=dict(showgrid=True, gridcolor='#2D313A', zeroline=False),
                        xaxis=dict(showgrid=False, zeroline=False), margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#A0AEC0')
                    )
                    st.plotly_chart(fig_hr, use_container_width=True)
                    st.info(get_hr_insight(drift, workout_type))
                    st.write("")

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
                            st.markdown("<hr style='margin-top:0.5em; margin-bottom:1em; border-color:#2D313A'>", unsafe_allow_html=True)

                with tab3:
                    st.subheader("Historical Training Log")
                    st.dataframe(history_df, use_container_width=True, hide_index=True)

                with tab4:
                    st.subheader("Performance Management & Recovery")
                    pmc_df = calculate_pmc_metrics(history_df)
                    
                    if not pmc_df.empty:
                        # --- FIX: Grab the EXACT date of the last recorded run, NOT 14 days in the future
                        last_run_date = pd.to_datetime(history_df['date']).dt.date.max()
                        last_timestamp = pd.Timestamp(last_run_date)
                        
                        current_ctl = pmc_df.loc[last_timestamp, 'CTL (Fitness)']
                        current_atl = pmc_df.loc[last_timestamp, 'ATL (Fatigue)']
                        current_tsb = pmc_df.loc[last_timestamp, 'TSB (Form)']
                        latest_trimp = pmc_df.loc[last_timestamp, 'Daily TRIMP']
                        
                        status = get_training_status(current_ctl, current_tsb)
                        recovery = calculate_recovery_hours(latest_trimp, current_atl)
                        
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Training Status", status)
                        sc2.metric("Est. Recovery Time", f"{recovery} Hours")
                        sc3.metric("Current Form (TSB)", round(current_tsb, 1))
                        
                        st.divider()
                        st.markdown("**Long-Term Fitness (PMC)**")
                        fig_pmc = go.Figure()
                        fig_pmc.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df['TSB (Form)'], mode='lines', fill='tozeroy', name='Form (TSB)', line=dict(color='rgba(241, 196, 15, 0.4)', width=0)))
                        fig_pmc.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df['ATL (Fatigue)'], mode='lines', name='Fatigue (ATL)', line=dict(color='#E74C3C', width=2)))
                        fig_pmc.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df['CTL (Fitness)'], mode='lines', name='Fitness (CTL)', line=dict(color='#3498DB', width=3)))
                        fig_pmc.update_layout(
                            height=350, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0), 
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#A0AEC0'),
                            yaxis=dict(showgrid=True, gridcolor='#2D313A'), xaxis=dict(showgrid=False)
                        )
                        st.plotly_chart(fig_pmc, use_container_width=True)
                        
                        st.divider()
                        st.subheader("🤖 Algorithmic Anomaly Detection")
                        anomaly_df = detect_anomalies(history_df)
                        
                        if 'anomaly' in anomaly_df.columns:
                            anomalies = anomaly_df[anomaly_df['anomaly'] == -1]
                            if not anomalies.empty:
                                st.warning(f"**{len(anomalies)} Runs Flagged.** The model detected unusual Cardiac Drift ratios. Ensure adequate hydration and recovery.")
                                st.dataframe(anomalies[['date', 'distance', 'workout_type', 'trimp', 'drift']], hide_index=True)
                            else:
                                st.success("All recent telemetry falls within expected physiological parameters.")
                        else:
                            st.info("The ML engine requires at least 10 logged runs to establish a baseline for anomaly detection.")
                            
                    else:
                        st.info("Upload more runs over consecutive days to generate your fitness curve and ML insights.")

            finally:
                os.remove(tmp_file_path)
    else:
        st.info("👈 Upload your .fit files to build your database.")