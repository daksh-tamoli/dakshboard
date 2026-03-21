import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from physiology import calculate_pmc_metrics

from data_pipeline import extract_and_clean
from physiology import (add_hr_zones, calculate_training_stress, classify_workout, 
                        calculate_cardiac_drift, get_basic_stats, get_trimp_context,
                        get_pace_insight, get_hr_insight, get_cadence_insight)
from database import init_db, run_exists, save_run, load_history, get_user_profile, save_user_profile

st.set_page_config(page_title="DAKSHboard", page_icon="⚡", layout="wide")
init_db()
# --- CUSTOM UI / UX CSS ---
st.markdown("""
    <style>
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style the Metric Cards */
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

    /* Make the metric values pop */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
    }
    
    /* Style the Tabs for a cleaner look */
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
    
    /* Soften the expanders and borders */
    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #2D313A;
    }
    </style>
""", unsafe_allow_html=True)

# --- BACKGROUND DATA SYNC (Bypassing Login) ---
st.session_state.email = "local_athlete"

if 'profile_loaded' not in st.session_state:
    profile = get_user_profile(st.session_state.email)
    if profile:
        st.session_state.age, st.session_state.weight, st.session_state.height, st.session_state.max_hr, st.session_state.rest_hr = profile
    else:
        st.session_state.age, st.session_state.weight, st.session_state.height, st.session_state.max_hr, st.session_state.rest_hr = 19, 71.0, 175, 200, 50
    st.session_state.profile_loaded = True

# --- NAVIGATION ---
st.sidebar.title("⚡ DAKSHboard")
page = st.sidebar.radio("Navigation", ["📊 Analytics Dashboard", "🤖 AI Training Coach", "⚙️ Athlete Profile"])
st.sidebar.divider()

if page == "⚙️ Athlete Profile":
    st.title("Athlete Profile Settings")
    st.markdown("Calibrate your baselines. Data is permanently saved to your local database.")
    
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

elif page == "🤖 AI Training Coach":
    st.title("🤖 DAKSHboard AI Coach")
    st.markdown("Your personal endurance coach, architected to analyze your biometric telemetry.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm the DAKSHboard analytical engine. Let's optimize your physiological load to ensure you lock in that sub-3.5 hour finish at the NMDC Hyderabad Marathon. How is the cardiovascular strain feeling after the latest tempo interval?"}
        ]

    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # React to user input
    if prompt := st.chat_input("Ask about TRIMP, cardiac drift, or race pacing..."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Placeholder response until API is connected
        bot_response = f"I've logged your query regarding '{prompt}'. To process this against your historical `.fit` database, you'll need to drop an LLM API key into the Streamlit secrets manager. Until then, keep an eye on your Zone 2 aerobic base!"
        st.chat_message("assistant").write(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

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
                        
                        save_run(st.session_state.email, uploaded_file.name, run_date, 
                                 stats_temp["Distance"]["Distance"], trimp, workout, 
                                 stats_temp["Pace/Speed"]["Avg Pace"], drift)
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

                tab1, tab2, tab3, tab4 = st.tabs(["📉 Telemetry & Environment", "📋 Comprehensive Stats & Zones", "📚 Training Log", "📈 Long-Term Fitness (PMC)"])
                with tab1:
                    elapsed_minutes = (df.index - df.index[0]).total_seconds() / 60.0
                    active_df = df.dropna(subset=['smoothed_pace_min_km'])
                    active_minutes = (active_df.index - df.index[0]).total_seconds() / 60.0

                    # --- NEW: MAPBOX GPS ROUTE ---
                    if 'lat' in df.columns and 'lon' in df.columns:
                        st.subheader("Geospatial Route")
                        map_df = df.dropna(subset=['lat', 'lon'])
                        if not map_df.empty:
                            fig_map = px.line_mapbox(map_df, lat="lat", lon="lon", zoom=13, height=400)
                            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                            st.plotly_chart(fig_map, use_container_width=True)
                            st.write("")

                    # --- NEW: ELEVATION PROFILE ---
                    if 'elevation_m' in df.columns:
                        st.subheader("Elevation Profile")
                        fig_elev = go.Figure(go.Scatter(x=elapsed_minutes, y=df['elevation_m'], mode='lines', fill='tozeroy', line=dict(color='#27AE60', width=2)))
                        fig_elev.update_layout(height=250, yaxis_title="Elevation (m)", margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_elev, use_container_width=True)
                        st.write("")

                    st.subheader("Pace Kinematics")
                    fig_pace = go.Figure(go.Scatter(x=active_minutes, y=active_df['smoothed_pace_min_km'], mode='lines', line=dict(color='#3498DB', width=2)))
                    fig_pace.update_layout(height=300, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_pace, use_container_width=True)
                    st.info(get_pace_insight(df))
                    st.write("")

                    st.subheader("Cardiovascular Response")
                    fig_hr = go.Figure(go.Scatter(x=elapsed_minutes, y=df['smoothed_heart_rate'], mode='lines', line=dict(color='white', width=2)))
                    colors = ['#2980B9', '#27AE60', '#F1C40F', '#E67E22', '#E74C3C']
                    for i in range(1, len(hr_bins)-1):
                        fig_hr.add_hrect(y0=hr_bins[i], y1=hr_bins[i+1], fillcolor=colors[i-1], opacity=0.3, layer="below", line_width=0)
                    fig_hr.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_hr, use_container_width=True)
                    st.info(get_hr_insight(drift, workout_type))
                    st.write("")

                    if 'smoothed_spm' in df.columns:
                        st.subheader("Running Dynamics (Cadence)")
                        fig_cad = go.Figure(go.Scatter(x=active_minutes, y=active_df['smoothed_spm'], mode='lines', line=dict(color='#9B59B6', width=2)))
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
                    st.dataframe(history_df, use_container_width=True, hide_index=True)

                with tab4:
                    st.subheader("Performance Management Chart (PMC)")
                    st.markdown("Track your Fitness (CTL), Fatigue (ATL), and Form (TSB) over time.")
                    
                    pmc_df = calculate_pmc_metrics(history_df)
                    
                    if not pmc_df.empty:
                        fig_pmc = go.Figure()
                        
                        # Add TSB (Form) as a filled bar/area chart in the background
                        fig_pmc.add_trace(go.Scatter(
                            x=pmc_df.index, y=pmc_df['TSB (Form)'], 
                            mode='lines', fill='tozeroy', name='Form (TSB)',
                            line=dict(color='rgba(241, 196, 15, 0.4)', width=0)
                        ))
                        
                        # Add ATL (Fatigue)
                        fig_pmc.add_trace(go.Scatter(
                            x=pmc_df.index, y=pmc_df['ATL (Fatigue)'], 
                            mode='lines', name='Fatigue (ATL)',
                            line=dict(color='#E74C3C', width=2)
                        ))
                        
                        # Add CTL (Fitness)
                        fig_pmc.add_trace(go.Scatter(
                            x=pmc_df.index, y=pmc_df['CTL (Fitness)'], 
                            mode='lines', name='Fitness (CTL)',
                            line=dict(color='#3498DB', width=3)
                        ))
                        
                        fig_pmc.update_layout(
                            height=450, hovermode="x unified",
                            margin=dict(l=0, r=0, t=10, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_pmc, use_container_width=True)
                    else:
                        st.info("Upload more runs over consecutive days to generate your fitness curve.")

            finally:
                os.remove(tmp_file_path)
    else:
        st.info("👈 Upload your .fit files to build your database.")
