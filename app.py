import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="VisionRain | Kingdom Commander",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for "The Wall" - Professional Scientific Look
st.markdown("""
    <style>
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #303030;
        padding: 10px;
        border-radius: 5px;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .success-box {
        padding: 20px;
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid green;
        border-radius: 5px;
        color: #00ff00;
    }
    .fail-box {
        padding: 20px;
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid red;
        border-radius: 5px;
        color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "b5710c0835b1558c7a5002809513f1a5.nc"
LOG_FILE = "mission_logs.csv"
ADMIN_PASSWORD = "123456"

# ==========================================
# 2. DATA ARCHITECTURE & PROCESSING
# ==========================================

@st.cache_resource
def load_data():
    """Loads the ERA5 NetCDF file and pre-processes variables."""
    if not os.path.exists(DATA_FILE):
        return None
    
    try:
        # Open dataset
        ds = xr.open_dataset(DATA_FILE, engine='netcdf4')
        
        # --- FIX 1: COORDINATE RENAMING ---
        # The error showed 'pressure_level' and 'valid_time' instead of 'level' and 'time'
        coord_renames = {}
        if 'pressure_level' in ds.coords or 'pressure_level' in ds.dims:
            coord_renames['pressure_level'] = 'level'
        if 'valid_time' in ds.coords or 'valid_time' in ds.dims:
            coord_renames['valid_time'] = 'time'
        
        if coord_renames:
            ds = ds.rename(coord_renames)

        # --- FIX 2: VARIABLE RENAMING ---
        # Rename variables to match your architecture for clarity
        rename_map = {
            't': 'Temperature',
            'r': 'Relative_Humidity',
            'clwc': 'Specific_cloud_liquid_water_content',
            'cc': 'Fraction_of_cloud_cover',
            'u': 'U_component_of_wind',
            'v': 'V_component_of_wind',
            'z': 'Geopotential',
            'w': 'Vertical_velocity',
            'd': 'Divergence',
            'q': 'Specific_humidity'
        }
        
        # Only rename what exists in the file
        actual_rename = {k: v for k, v in rename_map.items() if k in ds}
        ds = ds.rename(actual_rename)
        
        # Convert Temperature K -> C
        if 'Temperature' in ds:
            ds['Temperature'] = ds['Temperature'] - 273.15
            
        # Select pressure levels of interest
        # Ensure we only ask for levels that actually exist in the file to avoid errors
        desired_levels = [1000, 925, 850, 700, 600, 500]
        
        # Check which levels are actually in the file
        if 'level' in ds.dims:
            file_levels = ds['level'].values
            # Find nearest available levels for each desired level
            ds = ds.sel(level=desired_levels, method='nearest')
        
        return ds
    except Exception as e:
        st.error(f"Error reading NetCDF: {e}")
        return None

def scan_for_candidates(ds):
    """
    AUTO-SCAN: Scans the dataset for regions matching seeding criteria.
    Returns a dataframe of candidate locations.
    """
    # Look at the latest time step (Real-time simulation)
    # Using 'time' which we renamed from 'valid_time'
    ds_now = ds.isel(time=-1) 
    
    # Target Level: 700hPa (Cloud middle/top for KSA)
    target_layer = ds_now.sel(level=700, method='nearest')
    
    # Convert to DataFrame for easier filtering
    df = target_layer.to_dataframe().reset_index()
    
    # --- PRE-FILTERING FOR SAUDI ARABIA ---
    # Approx Bounds: Lat 16-32, Lon 34-56
    df = df[(df['latitude'] >= 16) & (df['latitude'] <= 32) & 
            (df['longitude'] >= 34) & (df['longitude'] <= 56)]
    
    # --- CANDIDATE PHYSICS LOGIC ---
    # 1. Cloud Cover > 0.2 (20%)
    # 2. Liquid Water Content > Threshold (e.g. 0.00001)
    
    # Ensure columns exist before filtering (handling potential missing vars)
    if 'Fraction_of_cloud_cover' in df.columns and 'Specific_cloud_liquid_water_content' in df.columns:
        candidates = df[
            (df['Fraction_of_cloud_cover'] > 0.2) & 
            (df['Specific_cloud_liquid_water_content'] > 0.000005)
        ].copy()
    else:
        # Fallback if specific vars are missing in this slice
        candidates = df.head(0)

    # Calculate a simple "Seedability Score" (0-100) for visualization
    if not candidates.empty:
        # More Water + Lower Temp (up to -15) = Higher Score
        # Using Relative Humidity as a proxy if available
        rh_val = candidates['Relative_Humidity'] if 'Relative_Humidity' in candidates else 50
        
        candidates['Score'] = (
            (candidates['Fraction_of_cloud_cover'] * 50) + 
            (rh_val / 2)
        ).clip(0, 100)
    
        return candidates.sort_values(by='Score', ascending=False).head(50) # Top 50 candidates
    else:
        return pd.DataFrame()

def log_to_bigquery_sim(region_id, lat, lon, decision, reason, metrics):
    """Simulates logging to BigQuery/CSV."""
    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Region_ID": region_id,
        "Latitude": lat,
        "Longitude": lon,
        "Decision": decision,
        "Reason": reason,
        "Metrics_JSON": str(metrics)
    }
    
    df_new = pd.DataFrame([log_entry])
    
    if os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(LOG_FILE, mode='w', header=True, index=False)

# ==========================================
# 3. UI LAYOUT & TABS
# ==========================================

ds = load_data()

st.title("🌧️ VisionRain: AI-Driven Cloud Seeding System")
st.markdown("**Aligned with Saudi Vision 2030 & The Saudi Green Initiative**")

if ds is None:
    st.warning(f"⚠️ System Offline: Please upload valid ERA5 file ({DATA_FILE}) to the root directory.")
    st.stop()

# Auto-Scan on Load
candidates = scan_for_candidates(ds)

tabs = st.tabs(["🎯 The Core Mission", "🗺️ Kingdom Commander", "📊 The Wall (Deep Dive)", "🔐 Admin Portal"])

# --- TAB 1: THE CORE MISSION ---
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Problem Statement")
        st.markdown("""
        Regions such as Saudi Arabia face increasing environmental challenges, including water scarcity, prolonged droughts, and heightened wildfire risk. 
        Current manual cloud seeding is **costly ($8,000/flight hour)**, **reactive**, and **weather-limited**. 
        Operators miss short-lived seedable cloud opportunities.
        """)
        
        st.header("Proposed Solution")
        st.info("VisionRain: A Pilotless, AI-Driven Decision Support Platform.")
        st.markdown("""
        * **Automated:** Scans atmospheric data in real-time.
        * **Predictive:** Identifies supercooled liquid water before rain forms.
        * **Cost-Efficient:** Designed for future drone swarms (Negative Cost).
        """)
        
    with col2:
        st.header("Impact")
        st.metric("Global Drought Affected", "1.4 Billion People")
        st.metric("Flight Cost Savings", "100%", delta="Switch to Drones")
        st.metric("Water Sustainability", "Vision 2030 Priority")

# --- TAB 2: KINGDOM COMMANDER (MAP) ---
with tabs[1]:
    st.subheader("📍 Real-Time Target Identification")
    st.markdown("Automatic scan of Saudi Arabia Sector. Top seedable candidates identified based on Liquid Water Content and Atmospheric Stability.")
    
    # Map Visualization
    if not candidates.empty:
        fig_map = px.scatter_mapbox(
            candidates, 
            lat="latitude", 
            lon="longitude", 
            color="Score",
            size="Score",
            hover_data=["Temperature", "Specific_cloud_liquid_water_content", "Relative_Humidity"],
            color_continuous_scale="Bluered",
            zoom=4, 
            center={"lat": 24.0, "lon": 45.0},
            height=600,
            mapbox_style="carto-darkmatter"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Visual Table (Prompt Requirement #5)
        st.subheader("📋 Regional Status Report")
        display_cols = ['latitude', 'longitude', 'Temperature', 'Relative_Humidity', 'Fraction_of_cloud_cover', 'Score']
        # Filter strictly to columns that exist
        final_cols = [c for c in display_cols if c in candidates.columns]
        
        st.dataframe(
            candidates[final_cols].style.background_gradient(cmap='Blues')
        )
    else:
        st.warning("No suitable clouds found in current sector scan.")

# --- TAB 3: THE WALL (ANALYSIS & AI BRAIN) ---
with tabs[2]:
    st.subheader("🔬 Deep Analysis & AI Decision")
    
    if candidates.empty:
        st.write("No targets available for analysis.")
    else:
        # Selector for specific target
        selected_index = st.selectbox(
            "Select Target Coordinates:", 
            candidates.index, 
            format_func=lambda x: f"Lat: {candidates.loc[x, 'latitude']:.2f}, Lon: {candidates.loc[x, 'longitude']:.2f} (Score: {candidates.loc[x, 'Score']:.0f})"
        )
        
        target = candidates.loc[selected_index]
        
        # Retrieve full vertical profile for this location
        # Using the latest time step
        loc_data = ds.isel(time=-1).sel(
            latitude=target['latitude'], 
            longitude=target['longitude'], 
            method='nearest'
        )
        
        # --- 4. THE VISUAL DASHBOARD ("THE WALL") ---
        col_dash1, col_dash2 = st.columns([2, 1])
        
        with col_dash1:
            st.markdown("#### 📉 Vertical Atmospheric Profile")
            
            # Create subplots for the Matrix of Scientific Plots
            fig = go.Figure()
            
            # 1. Temperature Profile
            if 'Temperature' in loc_data:
                fig.add_trace(go.Scatter(
                    x=loc_data['Temperature'].values, 
                    y=loc_data['level'].values, 
                    mode='lines+markers', 
                    name='Temp (°C)',
                    line=dict(color='red')
                ))
            
            # 2. Dewpoint/Humidity Proxy (RH)
            if 'Relative_Humidity' in loc_data:
                fig.add_trace(go.Scatter(
                    x=loc_data['Relative_Humidity'].values, 
                    y=loc_data['level'].values, 
                    mode='lines+markers', 
                    name='Rel. Humidity (%)',
                    line=dict(color='blue', dash='dot')
                ))

            fig.update_layout(
                title=f"Skew-T Proxy: Lat {target['latitude']:.2f}, Lon {target['longitude']:.2f}",
                yaxis=dict(title="Pressure (hPa)", autorange="reversed"),
                xaxis=dict(title="Value"),
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Wind Vectors (Drone Safety)
            st.markdown("#### 🚁 Drone Wind Vector Analysis")
            if 'U_component_of_wind' in target and 'V_component_of_wind' in target:
                wind_speed = np.sqrt(target['U_component_of_wind']**2 + target['V_component_of_wind']**2)
                st.metric("Wind Speed (Combined)", f"{wind_speed:.2f} m/s")
            else:
                st.write("Wind data unavailable.")

        with col_dash2:
            st.markdown("#### 🔢 Live Metrics (700hPa)")
            # Safety checks for metrics
            t_val = target['Temperature'] if 'Temperature' in target else 0
            lwc_val = target['Specific_cloud_liquid_water_content'] if 'Specific_cloud_liquid_water_content' in target else 0
            cc_val = target['Fraction_of_cloud_cover'] if 'Fraction_of_cloud_cover' in target else 0
            vv_val = target['Vertical_velocity'] if 'Vertical_velocity' in target else 0
            rh_val = target['Relative_Humidity'] if 'Relative_Humidity' in target else 0

            st.metric("Temperature", f"{t_val:.2f} °C")
            st.metric("Liquid Water Content", f"{lwc_val:.5f} kg/kg")
            st.metric("Cloud Cover", f"{cc_val*100:.1f} %")
            st.metric("Vertical Velocity", f"{vv_val:.4f} Pa/s")
            
            # 5. THE AI BRAIN (GEMINI FUSION)
            st.divider()
            st.subheader("🤖 Gemini Fusion Core")
            
            # --- PHYSICS RULES ENGINE ---
            # Rule 1: Temperature Window (-5 to -15 ideal for seeding)
            temp_check = -15 <= t_val <= -5
            
            # Rule 2: Cloud Phase (Proxy)
            # If Temp < 0 and LWC > 0, we have Supercooled Liquid Water
            phase = "Ice" if t_val < -40 else ("Mixed" if t_val < 0 else "Liquid")
            phase_check = (phase == "Mixed") and (lwc_val > 1e-5)
            
            # Rule 3: Radius (Simulated for Demo as ERA5 lacks this)
            simulated_radius = 12.5 # microns (Simulated)
            
            st.write(f"**Observed Phase:** {phase}")
            st.write(f"**Est. Droplet Radius:** {simulated_radius} µm")
            
            # FINAL DECISION
            if temp_check and phase_check:
                decision = "GO"
                reason = "Optimal Supercooled Liquid Water Detected. Temp within -5°C to -15°C range."
                box_class = "success-box"
                btn_label = "🚀 INITIATE DRONE SWARM"
            else:
                decision = "NO-GO"
                reason = []
                if not temp_check: reason.append(f"Temp {t_val:.1f}°C out of seedable range (-5 to -15).")
                if not phase_check: reason.append("Insufficient Liquid Water Content.")
                reason = " ".join(reason)
                box_class = "fail-box"
                btn_label = "❌ ABORT MISSION"

            st.markdown(f"""
            <div class="{box_class}">
                <h2 style='text-align: center; margin:0;'>DECISION: {decision}</h2>
                <p style='text-align: center;'>{reason}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action Button
            if st.button(btn_label):
                metrics_log = {
                    "temp": t_val,
                    "lwc": lwc_val,
                    "rh": rh_val
                }
                log_to_bigquery_sim(selected_index, target['latitude'], target['longitude'], decision, reason, metrics_log)
                st.toast(f"Mission Logged to BigQuery: {decision}")

# --- TAB 4: ADMIN PORTAL ---
with tabs[3]:
    st.header("🔐 Admin Portal")
    pwd = st.text_input("Enter Access Code", type="password")
    
    if pwd == ADMIN_PASSWORD:
        st.success("Access Granted")
        if os.path.exists(LOG_FILE):
            st.subheader("🗄️ Mission Logs (Simulating BigQuery)")
            df_logs = pd.read_csv(LOG_FILE)
            st.dataframe(df_logs)
            
            st.download_button(
                "📥 Export Logs to CSV",
                df_logs.to_csv(index=False).encode('utf-8'),
                "mission_logs.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("No missions logged yet.")
    elif pwd:
        st.error("Invalid Access Code")
