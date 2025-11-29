import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import csv
import requests
from io import BytesIO
import folium
from streamlit_folium import st_folium
import random

# --- SAFELY IMPORT SCIENTIFIC LIBS ---
try:
    import xarray as xr
    import cfgrib
except ImportError:
    st.error("⚠️ Scientific Libraries Missing! Update requirements.txt")
    xr = None

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"

# FILES (Must be in same folder)
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="🛰️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 15px;}
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    .success-box {background-color: rgba(0, 255, 128, 0.1); border: 1px solid #00ff80; color: #00ff80; padding: 15px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADERS ---
@st.cache_resource
def load_data():
    """Loads both datasets once"""
    if xr is None: return None, None
    
    ds_sat, ds_era = None, None
    if os.path.exists(NETCDF_FILE):
        try: ds_sat = xr.open_dataset(NETCDF_FILE, engine='netcdf4')
        except: pass
    if os.path.exists(ERA5_FILE):
        try: ds_era = xr.open_dataset(ERA5_FILE, engine='cfgrib')
        except: pass
        
    return ds_sat, ds_era

# --- 2. KINGDOM SCANNER (Pandas Logic) ---
def scan_saudi_sector(ds_sat, ds_era):
    """
    Scans the Saudi Arabia Bounding Box for seedable clouds.
    Returns a DataFrame of targets.
    """
    # 1. Define Saudi BBox (Approx Pixel Coords for Meteosat Disk)
    # Center of Saudi is approx 2300, 750 in the 3712x3712 grid
    y_slice = slice(2100, 2500)
    x_slice = slice(600, 900)
    
    targets = []
    
    if ds_sat:
        # Extract Raw Arrays
        prob = ds_sat['cloud_probability'].isel(y=y_slice, x=x_slice).values
        press = ds_sat['cloud_top_pressure'].isel(y=y_slice, x=x_slice).values
        
        # 2. FILTER LOGIC (The "Seedability" Algorithm)
        # We want: High Probability (>80%) AND Mid-Level Altitude (400-700 hPa)
        # This avoids low fog (>800) and high ice cirrus (<300)
        # mask returns indices where conditions are met
        y_idxs, x_idxs = np.where((prob > 80) & (press > 40000) & (press < 70000))
        
        # 3. Sample Targets (Don't take every single pixel, just clusters)
        if len(y_idxs) > 0:
            # Pick 5 random distinct storm cells
            indices = np.linspace(0, len(y_idxs)-1, 5, dtype=int)
            
            for i in indices:
                y, x = y_idxs[i], x_idxs[i]
                
                # Convert Matrix Coords to Lat/Lon (Approximate for Demo)
                # Meteosat 0,0 is Lat 0, Lon 0. 
                # 1 pixel ~ 3km.
                # This transformation approximates the view over Saudi
                lat_approx = 24.0 + (2300 - (y + 2100)) * 0.03
                lon_approx = 45.0 + ((x + 600) - 750) * 0.03
                
                # Get ERA5 Context (Simulated lookup for specific point)
                era_humid = 65 + random.randint(-10, 10) # Simulated look up
                
                targets.append({
                    "ID": f"TGT-{random.randint(1000,9999)}",
                    "Lat": round(lat_approx, 4),
                    "Lon": round(lon_approx, 4),
                    "Cloud Prob": int(prob[y, x]),
                    "Pressure": int(press[y, x] / 100), # Pa to hPa
                    "Humidity": era_humid,
                    "Status": "SEEDABLE"
                })
    
    # Fallback if no file or no clouds found (Simulation Mode)
    if not targets:
        return pd.DataFrame([
            {"ID": "SIM-ALPHA", "Lat": 24.7136, "Lon": 46.6753, "Cloud Prob": 92, "Pressure": 650, "Humidity": 72, "Status": "SEEDABLE"},
            {"ID": "SIM-BRAVO", "Lat": 21.5433, "Lon": 39.1728, "Cloud Prob": 88, "Pressure": 580, "Humidity": 68, "Status": "SEEDABLE"},
            {"ID": "SIM-CHARLIE", "Lat": 26.4207, "Lon": 50.0888, "Cloud Prob": 45, "Pressure": 900, "Humidity": 30, "Status": "UNSTABLE"}
        ])
        
    return pd.DataFrame(targets)

# --- 3. VISUALIZER ---
def plot_target_visuals(ds_sat, ds_era, lat, lon):
    """Generates the scientific plots for the specific target"""
    
    # Create Buffers
    buf_sat = BytesIO()
    
    # 1. Satellite Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0e1117')
    
    if ds_sat:
        # Generate local crop
        data = ds_sat['cloud_top_pressure'].isel(y=slice(2200,2400), x=slice(700,900)).values
        im = ax.imshow(data, cmap='gray_r')
        ax.set_title(f"Meteosat Target: {lat:.2f}, {lon:.2f}", color="white")
    else:
        # Simulation
        data = np.random.rand(100,100)
        ax.imshow(data, cmap='gray_r')
        ax.set_title("Simulated Target Feed", color="white")
        
    ax.axis('off')
    plt.savefig(buf_sat, format="png", facecolor='#0e1117')
    buf_sat.seek(0)
    
    return Image.open(buf_sat)

# --- 4. LOGGING ---
def log_mission(target_id, lat, lon, decision, reason):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["Timestamp", "Target_ID", "Location", "Decision", "Reason"])
        writer.writerow([ts, target_id, f"{lat},{lon}", decision, reason])

def load_logs():
    if os.path.isfile(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v8.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Regional Scanner")
    
    if 'scan_results' not in st.session_state:
        st.session_state['scan_results'] = None
        
    if st.button("SCAN SAUDI SECTOR"):
        ds_sat, ds_era = load_data()
        with st.spinner("Scanning 2.15 Million km² for Seedable Clouds..."):
            st.session_state['scan_results'] = scan_saudi_sector(ds_sat, ds_era)
            
    # Target Selection
    selected_row = None
    if st.session_state['scan_results'] is not None:
        df = st.session_state['scan_results']
        st.success(f"{len(df)} Potential Targets Identified")
        target_id = st.selectbox("Select Target to Engage:", df['ID'])
        selected_row = df[df['ID'] == target_id].iloc[0]
        
    # Admin
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Pitch & Strategy", "🗺️ Live Threat Map", "🧠 Gemini Fusion"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b> face escalating environmental crises: water scarcity and drought. 
    Current cloud seeding operations are <b>manual, expensive, and reactive</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - An AI-driven decision support platform.")
    with c2:
        st.success("**Impact:** Supports Saudi Green Initiative & Water Security.")

# --- TAB 2: MAP & SENSORS ---
with tab2:
    if selected_row is not None:
        lat, lon = selected_row['Lat'], selected_row['Lon']
        st.header(f"Target Analysis: {selected_row['ID']}")
        
        c_map, c_data = st.columns([2, 1])
        
        with c_map:
            # Dynamic Map with Pins
            m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
            
            # Plot ALL targets
            for _, row in st.session_state['scan_results'].iterrows():
                color = 'green' if row['Status'] == 'SEEDABLE' else 'red'
                folium.Marker(
                    [row['Lat'], row['Lon']], 
                    popup=f"{row['ID']}: {row['Cloud Prob']}%",
                    icon=folium.Icon(color=color, icon='cloud')
                ).add_to(m)
            
            # Highlight Selected
            folium.CircleMarker(
                [lat, lon], radius=20, color='#00e5ff', fill=False
            ).add_to(m)
            
            st_folium(m, height=400, width=700)

        with c_data:
            st.subheader("Telemetry")
            st.metric("Cloud Probability", f"{selected_row['Cloud Prob']}%")
            st.metric("Cloud Top Pressure", f"{selected_row['Pressure']} hPa")
            st.metric("Humidity (ERA5)", f"{selected_row['Humidity']}%")
            st.metric("Status", selected_row['Status'])
            
    else:
        st.info("👈 Please run a **SCAN** from the sidebar to identify targets.")

# --- TAB 3: GEMINI ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if selected_row is not None:
        # Generate visual for this specific target
        ds_sat, ds_era = load_data()
        plot_img = plot_target_visuals(ds_sat, ds_era, lat, lon)
        
        st.image(plot_img, caption=f"Satellite Recon: {selected_row['ID']}", width=400)
        
        if st.button("AUTHORIZE DRONE SWARM", type="primary"):
            if not api_key:
                st.error("🔑 API Key Missing")
            else:
                genai.configure(api_key=api_key)
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    ACT AS A MISSION COMMANDER. Analyze this Target for Cloud Seeding.
                    
                    --- TARGET PROFILE ---
                    ID: {selected_row['ID']}
                    Location: {lat}, {lon}
                    
                    --- SENSOR DATA ---
                    - Cloud Probability: {selected_row['Cloud Prob']}%
                    - Cloud Top Pressure: {selected_row['Pressure']} hPa
                    - Humidity: {selected_row['Humidity']}%
                    
                    --- LOGIC ---
                    1. IF Probability > 80% AND Pressure < 700hPa -> HIGH PRIORITY TARGET.
                    2. IF Humidity < 30% -> ABORT (Dry Air).
                    
                    --- VISUALS ---
                    See attached satellite scan.
                    
                    --- COMMAND ---
                    1. **Situation Report:** Describe the target.
                    2. **Decision:** **GO** or **NO-GO**?
                    3. **Orders:** If GO, dispatch drone swarm to coords.
                    """
                    
                    with st.spinner("Verifying Target Parameters..."):
                        res = model.generate_content([prompt, plot_img])
                        
                        decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                        log_mission(selected_row['ID'], lat, lon, decision, "AI Authorized")
                        
                        st.markdown("### 🛰️ Mission Directive")
                        st.write(res.text)
                        
                        if decision == "GO":
                            st.balloons()
                            st.success(f"✅ DRONES DISPATCHED TO {lat}, {lon}")
                            
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.warning("Select a target from the Regional Scanner first.")
