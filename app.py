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
import random

# --- SAFELY IMPORT SCIENTIFIC LIBS ---
try:
    import xarray as xr
except ImportError:
    st.error("⚠️ Scientific Libraries Missing! Please update requirements.txt")
    xr = None

try:
    import cfgrib
except ImportError:
    pass # Grib engine check

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
LOG_FILE = "mission_logs.csv"

# DATA FILES
SAT_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Scientific Twin", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {
        background-color: #1a1a1a; 
        border: 1px solid #333; 
        border-radius: 12px; 
        padding: 15px;
    }
    .pitch-box {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #00e5ff;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: rgba(0, 255, 128, 0.1); 
        border: 1px solid #00ff80; 
        color: #00ff80; 
        padding: 15px; 
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADERS ---
@st.cache_resource
def load_satellite_data():
    """Loads EUMETSAT NetCDF"""
    if xr is None: return None
    if os.path.exists(SAT_FILE):
        try:
            return xr.open_dataset(SAT_FILE, engine='netcdf4')
        except: return None
    return None

@st.cache_resource
def load_era5_data():
    """Loads ERA5 GRIB"""
    if xr is None: return None
    if os.path.exists(ERA5_FILE):
        try:
            # Try opening with cfgrib engine
            return xr.open_dataset(ERA5_FILE, engine='cfgrib')
        except: return None
    return None

# --- 2. VISUALIZERS (Scientific Plots) ---
def generate_satellite_plot(ds, center_y, center_x, window):
    """Plots Meteosat Data"""
    if ds is None: return None, 0, 0, 0, 0
    
    try:
        # Slicing Logic
        y_slice = slice(max(0, center_y - window), center_y + window)
        x_slice = slice(max(0, center_x - window), center_x + window)
        
        # Extract
        press = ds['cloud_top_pressure'].isel(y=y_slice, x=x_slice).values
        prob = ds['cloud_probability'].isel(y=y_slice, x=x_slice).values
        
        # Metrics
        val_press = np.nanmean(press)
        val_prob = np.nanmean(prob)
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0e1117')
        im = ax.imshow(press, cmap='gray_r')
        ax.set_title("Meteosat: Cloud Top Pressure", color="white")
        ax.axis('off')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor='#0e1117')
        buf.seek(0)
        return Image.open(buf), val_press, val_prob
        
    except: return None, 0, 0

def generate_era5_plot(ds):
    """Plots ERA5 Atmospheric Data"""
    if ds is None: return None, 0, 0
    
    try:
        # Try to find Liquid Water Content or Temperature
        # Common GRIB names: 'clwc' (Cloud Liquid Water Content) or 't' (Temperature)
        target_var = 'clwc' if 'clwc' in ds else 't' if 't' in ds else list(ds.data_vars)[0]
        
        data = ds[target_var].values
        
        # If 3D (time/levels), take first slice
        if data.ndim > 2: data = data[0]
        if data.ndim > 2: data = data[0] # Handle 4D
        
        # Metrics
        val_mean = np.nanmean(data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0e1117')
        im = ax.imshow(data, cmap='viridis') # Viridis for scientific data
        ax.set_title(f"ERA5 Analysis: {target_var.upper()}", color="white")
        ax.axis('off')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor='#0e1117')
        buf.seek(0)
        
        return Image.open(buf), val_mean, target_var
        
    except: return None, 0, "Error"

# --- 3. LOGGING ---
def log_mission(location, conditions, decision, reason):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["Timestamp", "Location", "Conditions", "Decision", "Reason"])
        writer.writerow([ts, location, conditions, decision, reason])

def load_logs():
    if os.path.isfile(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Location", "Conditions", "Decision", "Reason"])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Scientific Twin | v6.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📍 Mission Target")
    st.info("Locked Sector: **Jeddah Storm**")
    st.caption("Coordinates: 21.54, 39.17")
    lat, lon = 21.54, 39.17
    
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown("### *Data Source: EUMETSAT (Optics) + ERA5 (Physics)*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Sensor Array", "🧠 Gemini Fusion"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b> face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    These issues are intensifying due to climate change and unstable precipitation patterns.</p>
    <p>Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>. Pilots often fly blind, missing critical seeding windows.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - An AI-driven decision support platform analyzing satellite microphysics for precision seeding.")
    with c2:
        st.success("**Impact:** Enables low-cost, safer deployment and scales globally to support emergency climate-response.")
        
    st.markdown("""
    <div class="pitch-box">
    <h3>🚀 Implementation Plan</h3>
    <ul>
    <li><b>Phase 1 (Ground Truth):</b> Ingest Satellite (Meteosat) & Atmospheric Models (ERA5).</li>
    <li><b>Phase 2 (AI Fusion):</b> Use Vertex AI to correlate cloud texture with internal liquid water content.</li>
    <li><b>Phase 3 (Action):</b> Automated GO/NO-GO signals for drone swarms.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# --- DATA PROCESSING ---
ds_sat = load_satellite_data()
ds_era = load_era5_data()

# 1. Generate Satellite Plot (Meteosat)
if ds_sat:
    sat_img, press, prob = generate_satellite_plot(ds_sat, 2300, 750, 150)
else:
    # Fallback simulation if file missing
    st.error("⚠️ Satellite File Missing.")
    sat_img, press, prob = None, 0, 0

# 2. Generate Atmospheric Plot (ERA5)
if ds_era:
    era_img, era_val, era_var = generate_era5_plot(ds_era)
else:
    st.warning("⚠️ ERA5 GRIB File Missing. Using fallback simulation.")
    # Simulate ERA5 Plot if missing
    era_img = sat_img # Reuse visual style for demo
    era_val = 0.5 # kg/kg
    era_var = "Specific Cloud Liquid Water"

# Live Telemetry (Fallback)
humidity = 65
temp = 28

# --- TAB 2: SENSORS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    col_sat, col_era = st.columns(2)
    
    with col_sat:
        st.subheader("A. Satellite Optics (Meteosat)")
        if sat_img: st.image(sat_img, caption="Cloud Top Pressure (Visual Proxy)", use_column_width=True)
        
    with col_era:
        st.subheader("B. Atmospheric Physics (ERA5)")
        if era_img: st.image(era_img, caption=f"Internal Structure: {era_var}", use_column_width=True)

    st.divider()
    
    # TELEMETRY TABLE
    st.subheader("Microphysical Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Probability", f"{prob:.1f}%", "AI Confidence")
    c2.metric("Cloud Top Pressure", f"{press:.0f} hPa", "Altitude")
    c3.metric("Liquid Water (ERA5)", f"{era_val:.2e}", "Fuel")
    c4.metric("Seeding Status", "ANALYZING", delta="Standby", delta_color="off")

# --- TAB 3: GEMINI FUSION ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    # 1. MASTER TABLE
    st.markdown("### 🔬 Physics-Informed Logic (The Master Table)")
    table_data = {
        "Parameter": ["Cloud Phase", "Liquid Water Content", "Cloud Probability", "Top Pressure"],
        "Ideal Range": ["Liquid/Mixed", "> 1e-5 kg/kg", "> 70%", "< 700 hPa"],
        "Current Value": ["Analyzing...", f"{era_val:.2e}", f"{prob:.1f}%", f"{press:.0f} hPa"]
    }
    st.table(pd.DataFrame(table_data))

    # 2. VISUAL EVIDENCE
    st.caption("Visual Evidence Sent to Vertex AI:")
    if sat_img and era_img:
        c1, c2 = st.columns(2)
        c1.image(sat_img, caption="Satellite View")
        c2.image(era_img, caption="ERA5 Internal Physics")

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not sat_img:
            st.error("⚠️ No Data.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # --- THE SUPER PROMPT ---
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST. Analyze this Multi-Modal Data (Satellite + ERA5).
                
                --- MISSION CONTEXT ---
                Location: Jeddah Sector
                Objective: Hygroscopic Cloud Seeding.
                
                --- INPUT DATA ---
                1. Satellite (Meteosat):
                   - Cloud Probability: {prob:.1f}% (0-100)
                   - Cloud Top Pressure: {press:.0f} hPa
                   
                2. ERA5 Atmospheric Model:
                   - Variable: {era_var}
                   - Value: {era_val:.2e}
                
                --- VISUALS (Attached) ---
                - Image 1: Satellite View (Cloud Texture/Height).
                - Image 2: ERA5 Physics Map (Internal Water/Temp Structure).
                
                --- LOGIC RULES ---
                1. IF Cloud Probability > 60% AND Pressure < 700hPa -> Potential Cloud.
                2. IF ERA5 Liquid Water is High -> Cloud has "Fuel" for seeding.
                3. IF both match -> "GO".
                
                --- OUTPUT ---
                1. **Visual Analysis:** Compare the Satellite view with the ERA5 Physics map. Do they align?
                2. **Microphysics Check:** Is there enough liquid water?
                3. **Decision:** **GO** or **NO-GO**?
                4. **Reasoning:** Scientific justification.
                """
                
                with st.spinner("Vertex AI is fusing Satellite & ERA5 streams..."):
                    # Send BOTH images
                    inputs = [prompt, sat_img]
                    if era_img: inputs.append(era_img)
                    
                    res = model.generate_content(inputs)
                    
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                    log_mission(f"{lat},{lon}", f"ERA5:{era_val:.2e}", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
