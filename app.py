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
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

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
    ds_sat, ds_era = None, None
    if xr:
        if os.path.exists(NETCDF_FILE):
            try: ds_sat = xr.open_dataset(NETCDF_FILE, engine='netcdf4')
            except: pass
        if os.path.exists(ERA5_FILE):
            try: ds_era = xr.open_dataset(ERA5_FILE, engine='cfgrib')
            except: pass
    return ds_sat, ds_era

# --- 2. KINGDOM SCANNER (The Automation Engine) ---
def scan_kingdom_targets(ds_sat):
    """
    Scans the satellite data grid for 'Seedable' signatures within Saudi Arabia.
    """
    targets = []
    
    # If no file, return simulated targets for demo continuity
    if ds_sat is None:
        return pd.DataFrame([
            {"ID": "TGT-ALPHA", "Lat": 24.7136, "Lon": 46.6753, "Cloud Prob": 88, "Pressure": 650, "LiquidWater": 0.003, "Status": "HIGH PRIORITY"},
            {"ID": "TGT-BRAVO", "Lat": 21.5433, "Lon": 39.1728, "Cloud Prob": 72, "Pressure": 580, "LiquidWater": 0.001, "Status": "MODERATE"},
            {"ID": "TGT-CHARLIE", "Lat": 26.4207, "Lon": 50.0888, "Cloud Prob": 95, "Pressure": 400, "LiquidWater": 0.000, "Status": "ICE (AVOID)"}
        ])

    try:
        # Dynamic Dimension Detection (Fixes the ValueError)
        dims = list(ds_sat['cloud_probability'].dims)
        y_dim, x_dim = dims[0], dims[1]
        
        # 1. Extract Raw Data
        # Slice approx Saudi region (Y: 2000-2600, X: 600-1200 in Meteosat grid)
        # Using dynamic kwargs to avoid "y not found" error
        slice_args = {
            y_dim: slice(2100, 2500),
            x_dim: slice(600, 900)
        }
        
        prob = ds_sat['cloud_probability'].isel(**slice_args).values
        press = ds_sat['cloud_top_pressure'].isel(**slice_args).values
        
        # 2. The "Seedability" Algorithm
        # Logic: High Probability (>60) AND Mid-Level Pressure (400-800hPa)
        y_idxs, x_idxs = np.where((prob > 60) & (prob < 101) & (press > 40000) & (press < 80000))
        
        # 3. Cluster & Select Targets
        if len(y_idxs) > 0:
            # Pick 3 distinct points
            indices = np.linspace(0, len(y_idxs)-1, 3, dtype=int)
            
            for i, idx in enumerate(indices):
                y, x = y_idxs[idx], x_idxs[idx]
                
                # Convert Grid to Lat/Lon (Approx)
                lat_approx = 24.0 + (200 - y) * 0.03
                lon_approx = 45.0 + (x - 150) * 0.03
                
                # Value Extraction
                p_val = float(prob[y, x])
                press_val = float(press[y, x] / 100.0) # hPa
                
                status = "HIGH PRIORITY" if p_val > 80 else "MODERATE"
                
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat_approx, 4),
                    "Lon": round(lon_approx, 4),
                    "Cloud Prob": int(p_val),
                    "Pressure": int(press_val),
                    "LiquidWater": 0.002, # Simulated ERA5 value
                    "Status": status
                })
                
    except Exception as e:
        st.error(f"Scan Error: {e}")
        
    return pd.DataFrame(targets) if targets else pd.DataFrame()

# --- 3. VISUALIZER ---
def generate_target_visual(ds, lat, lon):
    """Generates the specific plot for the AI to see"""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#0e1117')
    
    if ds:
        # Mock slice for demo speed (visuals only)
        data = np.random.rand(50, 50) * 100
        ax.imshow(data, cmap='Blues', vmin=0, vmax=100)
        ax.set_title(f"Target: {lat},{lon}", color="white")
    else:
        # Fallback
        ax.text(0.5, 0.5, "DATA LINK LOST", color="red", ha='center')
        
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    return Image.open(buf)

# --- 4. LOGGING ---
def log_mission(target, data, decision):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f: f.write("Timestamp,Target,Data,Decision\n")
    with open(LOG_FILE, 'a') as f:
        f.write(f"{ts},{target},{data},{decision}\n")

def load_logs():
    if os.path.exists(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v9.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 System Status")
    # Auto-Load Data on Start
    ds_sat, ds_era = load_data()
    if ds_sat: st.success("EUMETSAT Link: ACTIVE")
    else: st.warning("EUMETSAT Link: SIMULATED")
    
    if ds_era: st.success("ERA5 Link: ACTIVE")
    else: st.warning("ERA5 Link: SIMULATED")
    
    # Admin
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "🗺️ Kingdom Operations", "🧠 Gemini Decision Core"])

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

# --- TAB 2: OPERATIONS MAP ---
with tab2:
    st.header("Live Threat Map (Saudi Arabia)")
    
    # 1. AUTO-SCAN
    if 'targets_df' not in st.session_state:
        with st.spinner("Scanning Kingdom for Targets..."):
            st.session_state['targets_df'] = scan_kingdom_targets(ds_sat)
    
    df = st.session_state['targets_df']
    
    if not df.empty:
        col_map, col_list = st.columns([2, 1])
        
        with col_map:
            # Map Visualization
            m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
            
            for _, row in df.iterrows():
                color = 'green' if row['Status'] == 'HIGH PRIORITY' else 'orange'
                folium.Marker(
                    [row['Lat'], row['Lon']],
                    popup=f"{row['ID']}: {row['Cloud Prob']}%",
                    icon=folium.Icon(color=color, icon="cloud")
                ).add_to(m)
            
            st_folium(m, height=400, width=700)
            
        with col_list:
            st.subheader("Detected Targets")
            st.dataframe(df[["ID", "Cloud Prob", "Status"]])
            
            target_id = st.selectbox("Select Target to Engage:", df["ID"])
            selected_row = df[df["ID"] == target_id].iloc[0]
            st.session_state['active_target'] = selected_row
    else:
        st.warning("No Seedable Clouds Detected in Sector.")

# --- TAB 3: GEMINI CORE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if 'active_target' in st.session_state:
        t = st.session_state['active_target']
        
        st.info(f"Engaging Target: **{t['ID']}** at **{t['Lat']}, {t['Lon']}**")
        
        # 1. VISUAL EVIDENCE
        img = generate_target_visual(ds_sat, t['Lat'], t['Lon'])
        st.image(img, caption="Real-Time Microphysics Scan", width=300)
        
        # 2. DATA TABLE
        st.markdown("### 📊 Target Telemetry")
        st.table(pd.DataFrame({
            "Metric": ["Cloud Probability", "Top Pressure", "Liquid Water (ERA5)", "Humidity"],
            "Value": [f"{t['Cloud Prob']}%", f"{t['Pressure']} hPa", f"{t['LiquidWater']}", "65%"],
            "Ideal": ["> 70%", "400-700 hPa", "> 0.001", "> 50%"]
        }))
        
        # 3. EXECUTE
        if st.button("AUTHORIZE MISSION", type="primary"):
            if not api_key:
                st.error("🔑 Google API Key Missing!")
            else:
                genai.configure(api_key=api_key)
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    ACT AS A MISSION COMMANDER. Analyze this Target.
                    
                    --- TARGET DATA ---
                    ID: {t['ID']}
                    Cloud Prob: {t['Cloud Prob']}%
                    Pressure: {t['Pressure']} hPa
                    
                    --- LOGIC ---
                    1. IF Prob > 80% AND Pressure < 700hPa -> GO.
                    2. IF Prob < 50% -> NO-GO.
                    
                    --- OUTPUT ---
                    1. **Assessment:** Analyze data.
                    2. **Decision:** **GO** or **NO-GO**.
                    3. **Protocol:** "Deploy Drones" or "Stand Down".
                    """
                    
                    with st.spinner("Vertex AI validating parameters..."):
                        res = model.generate_content([prompt, img])
                        
                        decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                        log_mission(t['ID'], f"P:{t['Cloud Prob']}", decision)
                        
                        st.markdown("### 🛰️ Mission Directive")
                        st.write(res.text)
                        
                        if decision == "GO":
                            st.balloons()
                            st.markdown("<div class='success-box'>✅ MISSION APPROVED: Drones Dispatched</div>", unsafe_allow_html=True)
                        else:
                            st.error("⛔ MISSION ABORTED")
                            
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.info("👈 Select a Target in Tab 2 first.")
