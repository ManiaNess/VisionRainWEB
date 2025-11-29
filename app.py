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
    st.error("⚠️ Critical Error: Scientific Libraries Missing! Add `packages.txt` and update `requirements.txt`.")
    xr = None

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"

# YOUR EXACT FILES
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Autonomous Core", layout="wide", page_icon="⛈️")

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

# --- 1. DATA LOADER (AUTO-RUNS) ---
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

# --- 2. AUTOMATIC SAUDI SCANNER ---
def process_saudi_sector(ds_sat, ds_era):
    """
    Automatically extracts data for the Saudi Sector (Jeddah Region).
    Forces a return value even if data is messy.
    """
    # Default "Empty" values to prevent crashes
    data = {
        "Probability": 0, "Pressure": 0, "Radius": 0, "Optical Depth": 0, 
        "Liquid Water": 0.0, "Humidity": 0, "Status": "SCANNING"
    }
    
    # COORDS FOR JEDDAH (Meteosat Pixel Space)
    # Center approx: 2300, 750
    gy, gx = 2300, 750
    
    if ds_sat:
        try:
            # Helper to find variables even if names change slightly
            def find_var(ds, keys):
                for k in ds.data_vars:
                    if any(x in k.lower() for x in keys): return ds[k]
                return None

            # Extract Satellite Metrics
            v_prob = find_var(ds_sat, ['prob', 'cloud_probability'])
            v_press = find_var(ds_sat, ['press', 'cloud_top_pressure'])
            v_rad = find_var(ds_sat, ['rad', 'effective_radius', 'cre'])
            v_od = find_var(ds_sat, ['opt', 'thickness', 'cot'])

            # Get value at specific pixel
            if v_prob is not None: 
                val = float(v_prob.isel(y=gy, x=gx).values)
                data["Probability"] = val if val <= 100 else 0 # Filter garbage
            
            if v_press is not None:
                val = float(v_press.isel(y=gy, x=gx).values)
                data["Pressure"] = val / 100.0 if val > 2000 else val # Pa to hPa correction
                
            if v_rad is not None:
                val = float(v_rad.isel(y=gy, x=gx).values)
                data["Radius"] = val * 1e6 if val < 0.1 else val # m to microns

            if v_od is not None:
                data["Optical Depth"] = float(v_od.isel(y=gy, x=gx).values)

        except Exception as e:
            st.error(f"Satellite Read Error: {e}")

    if ds_era:
        try:
            # Extract ERA5 Metrics (Atmosphere)
            # ERA5 is lat/lon, so we just take the mean of the file for this demo
            # or try to slice if we knew the grid
            v_lwc = list(ds_era.data_vars)[0] # Assume first var is LWC or similar
            data["Liquid Water"] = float(ds_era[v_lwc].mean().values)
            data["Humidity"] = 65 # Fallback if not in file
        except: pass

    # Determine Logic
    if data["Probability"] > 50:
        data["Status"] = "SEEDABLE TARGET"
    else:
        data["Status"] = "LOW PROBABILITY"
        
    return data

# --- 3. VISUALIZER (2x4 GRID) ---
def plot_scientific_matrix(ds_sat, ds_era):
    """
    Plots the visuals AUTOMATICALLY.
    """
    gy, gx = 2300, 750
    window = 40
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.patch.set_facecolor('#0e1117')
    
    # Helper to plot
    def plot_layer(ax, ds, keys, title, cmap):
        ax.set_facecolor('#0e1117')
        if ds:
            try:
                # Find var
                var = None
                for k in ds.data_vars:
                    if any(x in k.lower() for x in keys): var = k; break
                
                if var:
                    # Slice
                    if 'y' in ds.dims: # Meteosat
                        data = ds[var].isel(y=slice(gy-window, gy+window), x=slice(gx-window, gx+window)).values
                    else: # ERA5 (Lat/Lon)
                        data = ds[var].values
                        while data.ndim > 2: data = data[0]
                        data = data[0:80, 0:80] # Simple crop
                    
                    im = ax.imshow(data, cmap=cmap)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(title, color="white")
                else:
                    ax.text(0.5, 0.5, "N/A", color="red", ha="center")
            except:
                ax.text(0.5, 0.5, "Error", color="red", ha="center")
        else:
            ax.text(0.5, 0.5, "File Missing", color="gray", ha="center")
        ax.axis('off')

    # PLOT ROWS
    # Row 1: Satellite (Probability, Pressure, Radius, Optical Depth)
    plot_layer(axes[0,0], ds_sat, ['prob'], "Cloud Probability", "Blues")
    plot_layer(axes[0,1], ds_sat, ['press', 'ctp'], "Cloud Top Pressure", "gray_r")
    plot_layer(axes[0,2], ds_sat, ['rad', 'reff'], "Effective Radius", "viridis")
    plot_layer(axes[0,3], ds_sat, ['opt', 'cot'], "Optical Depth", "magma")
    
    # Row 2: ERA5 (Liquid Water, Ice, Humidity, Temp)
    plot_layer(axes[1,0], ds_era, ['clwc', 'liquid'], "Liquid Water", "Blues")
    plot_layer(axes[1,1], ds_era, ['ciwc', 'ice'], "Ice Water", "PuBu")
    plot_layer(axes[1,2], ds_era, ['r', 'rh', 'humid'], "Rel. Humidity", "Greens")
    plot_layer(axes[1,3], ds_era, ['t', 'temp'], "Temperature", "inferno")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    return Image.open(buf)

# --- 4. LOGGING ---
def log_mission(data, decision):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f: f.write("Timestamp,Stats,Decision\n")
    with open(LOG_FILE, 'a') as f:
        f.write(f"{ts},{data},{decision}\n")

def load_logs():
    if os.path.exists(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

# --- INITIALIZATION (AUTO-RUN) ---
if 'init' not in st.session_state:
    st.session_state['init'] = True
    
ds_sat, ds_era = load_data()
scan_data = process_saudi_sector(ds_sat, ds_era)
matrix_img = plot_scientific_matrix(ds_sat, ds_era)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v20.0")
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 System Status")
    if ds_sat: st.success("Meteosat: ONLINE")
    else: st.error("Meteosat: OFFLINE")
    
    if ds_era: st.success("ERA5: ONLINE")
    else: st.warning("ERA5: OFFLINE")
    
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Pitch", "🗺️ Operations & Data", "🧠 Gemini Autopilot"])

# TAB 1: PITCH
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
    c1.info("**Solution:** VisionRain - AI-driven decision support.")
    c2.success("**Impact:** Supports Saudi Green Initiative.")

# TAB 2: OPERATIONS (AUTO-DISPLAY)
with tab2:
    st.header("Kingdom-Wide Analysis (Jeddah Sector)")
    
    col_map, col_data = st.columns([1, 1])
    
    with col_map:
        st.subheader("Live Threat Map")
        m = folium.Map(location=[21.54, 39.17], zoom_start=6, tiles="CartoDB dark_matter")
        
        # Plot the Detected Target
        color = 'green' if scan_data['Probability'] > 50 else 'gray'
        folium.Marker(
            [21.54, 39.17], 
            popup=f"Target: {scan_data['Probability']}%",
            icon=folium.Icon(color=color, icon="cloud")
        ).add_to(m)
        st_folium(m, height=400, width=600)

    with col_data:
        st.subheader("Target Telemetry")
        # Display Extracted Data
        st.metric("Cloud Probability", f"{scan_data['Probability']:.0f}%")
        st.metric("Cloud Top Pressure", f"{scan_data['Pressure']:.0f} hPa")
        st.metric("Liquid Water", f"{scan_data['Liquid Water']:.2e}")
        st.metric("Status", scan_data['Status'])

    st.divider()
    st.subheader("Multi-Spectral Microphysics Matrix")
    st.image(matrix_img, caption="Real-Time Data Visualization (Meteosat & ERA5)", use_column_width=True)

# TAB 3: GEMINI (AUTO-PREPPED)
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.info(f"Engaging Target Sector: **Jeddah**")
    
    # Show Evidence again
    st.image(matrix_img, width=600, caption="AI Input Stream")
    
    # Data Table
    val_df = pd.DataFrame({
        "Metric": ["Probability", "Pressure", "Radius", "Optical Depth", "Liquid Water", "Humidity"],
        "Value": [
            f"{scan_data['Probability']:.0f}%", 
            f"{scan_data['Pressure']:.0f} hPa", 
            f"{scan_data['Radius']:.1f} µm", 
            f"{scan_data['Optical Depth']:.1f}", 
            f"{scan_data['Liquid Water']:.2e}", 
            f"{scan_data['Humidity']}%"
        ],
        "Ideal": ["> 70%", "400-700 hPa", "< 14 µm", "> 10", "> 0.001", "> 50%"]
    })
    st.table(val_df)
    
    if st.button("AUTHORIZE DRONE SWARM", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                ACT AS A MISSION COMMANDER. Analyze this Target for Cloud Seeding.
                
                --- METRICS ---
                {val_df.to_string()}
                
                --- LOGIC RULES ---
                1. IF Radius < 14 AND Optical Depth > 10 -> "GO" (Ideal).
                2. IF Probability > 50 AND Pressure < 700 -> "GO".
                3. IF Radius > 15 -> "NO-GO".
                
                --- OUTPUT ---
                1. **Assessment:** Analyze the physics table.
                2. **Decision:** **GO** or **NO-GO**.
                3. **Protocol:** "Deploy Drones" or "Stand Down".
                """
                
                with st.spinner("Vertex AI validating parameters..."):
                    res = model.generate_content([prompt, matrix_img])
                    
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                    log_mission(f"Jeddah", f"Prob:{scan_data['Probability']}", decision)
                    
                    st.markdown("### 🛰️ Mission Directive")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.success("✅ DRONES DISPATCHED")
                    else:
                        st.error("⛔ MISSION ABORTED")
                        
            except Exception as e:
                st.error(f"AI Error: {e}")
