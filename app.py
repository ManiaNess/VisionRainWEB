import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import datetime
import os
import csv
import requests
from io import BytesIO

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
LOG_FILE = "mission_logs.csv"
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"

st.set_page_config(page_title="VisionRain Scientific Core", layout="wide", page_icon="⛈️")

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

# --- 1. DATA LOADER (EUMETSAT NetCDF) ---
@st.cache_resource
def load_netcdf_data():
    if os.path.exists(NETCDF_FILE):
        try:
            return xr.open_dataset(NETCDF_FILE)
        except Exception as e:
            return None
    return None

# --- 2. VISUALIZER (Scientific Plots) ---
def generate_scientific_plots(ds, center_y, center_x, window):
    """
    Generates Visuals & Extracts Microphysics Data
    """
    # 1. Extract Variables (Handle missing vars gracefully)
    try:
        # Data Arrays
        press = ds['cloud_top_pressure'].values if 'cloud_top_pressure' in ds else np.zeros((3712, 3712))
        phase = ds['cloud_phase'].values if 'cloud_phase' in ds else np.zeros((3712, 3712))
        reff = ds['cloud_effective_radius'].values if 'cloud_effective_radius' in ds else np.zeros((3712, 3712))
        cot = ds['cloud_optical_thickness'].values if 'cloud_optical_thickness' in ds else np.zeros((3712, 3712))
        
        # Masking "Space" (Values usually > 10000 or NaN)
        press = np.where(press < 2000, press, np.nan)
        
        # 2. Zoom Slicing
        y1, y2 = center_y - window, center_y + window
        x1, x2 = center_x - window, center_x + window
        zoom_press = press[y1:y2, x1:x2]
        
        # 3. Calculate Mean Values for the Target Sector
        # (Using simple mean of valid pixels in the zoom window)
        val_press = np.nanmean(zoom_press) if not np.isnan(np.nanmean(zoom_press)) else 0
        val_phase = np.nanmean(phase[y1:y2, x1:x2]) 
        val_reff = np.nanmean(reff[y1:y2, x1:x2]) 
        val_cot = np.nanmean(cot[y1:y2, x1:x2])

        # Interpretation
        phase_str = "Liquid" if val_phase < 1.5 else "Ice" # Simplified threshold
        
        # 4. Generate Plots
        fig = plt.figure(figsize=(10, 12))
        fig.patch.set_facecolor('#0e1117')
        
        # Top: Full Disk
        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(press[::10, ::10], cmap='gray_r') # Subsampled
        rect = plt.Rectangle((center_x/10 - window/10, center_y/10 - window/10), window/5, window/5, linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title("1. Global Context (Meteosat Full Disk)", color="white")
        ax1.axis('off')
        
        # Bottom: Zoomed Sector
        ax2 = plt.subplot(2, 1, 2)
        im2 = ax2.imshow(zoom_press, cmap='turbo') # Turbo is great for scientific data
        ax2.set_title("2. Target Sector Microphysics (Jeddah)", color="white")
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor='#0e1117')
        buf.seek(0)
        
        return Image.open(buf), val_press, val_reff, val_cot, phase_str
        
    except Exception as e:
        st.error(f"Plotting Error: {e}")
        return None, 0, 0, 0, "Error"

# --- 3. OWM TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 4. LOGGING ---
def log_mission(location, humidity, decision, reason):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["Timestamp", "Location", "Humidity", "Decision", "Reason"])
        writer.writerow([ts, location, humidity, decision, reason])

def load_logs():
    if os.path.isfile(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Location", "Humidity", "Decision", "Reason"])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Scientific Core v5.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("### 📍 Target Sector")
    st.info("Locked: **Jeddah (Meteosat)**\nPixel Coords: 2300, 750")
    lat, lon = 21.54, 39.17
    
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown("### *Data Source: EUMETSAT NetCDF (Raw) + OpenWeatherMap*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Sensor Array", "🧠 Gemini Fusion"])

# --- TAB 1: STRATEGY ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b>, California, Australia, and the Mediterranean are facing increasing environmental challenges including water scarcity, prolonged droughts, and wildfire escalation. 
    These issues are intensifying due to climate change, rising temperatures, and unstable precipitation patterns.</p>
    <p>While cloud seeding is an established method for rainfall enhancement, current operations are <b>manual, expensive, and rely on reactive decision-making</b>. 
    Existing systems do not leverage AI, real-time satellite analysis, or autonomous deployment technologies.</p>
    <p>This challenge is critically aligned with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - An AI-driven decision support platform analyzing satellite microphysics for precision seeding.")
    with c2:
        st.success("**Impact:** Enables low-cost, safer deployment and scales globally to support emergency climate-response.")

# --- DATA PROCESSING ---
ds = load_netcdf_data()
if ds:
    plot_img, press, reff, cot, phase = generate_scientific_plots(ds, 2300, 750, 150)
else:
    st.error("⚠️ NetCDF File Missing. Upload to GitHub.")
    plot_img, press, reff, cot, phase = None, 0, 0, 0, "N/A"

# Get Live OWM Data
w = get_weather_telemetry(lat, lon, weather_key)
humidity = w['main']['humidity'] if w else "N/A"

# --- TAB 2: SENSORS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    if plot_img:
        st.image(plot_img, caption="Meteosat-9 Analysis: Full Disk (Top) & Jeddah Sector (Bottom)", use_column_width=True)

    st.divider()
    
    # TELEMETRY TABLE
    st.subheader("Microphysical Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Phase", phase, "Liquid/Ice")
    c2.metric("Effective Radius", f"{reff:.1f} µm", "Target < 14µm")
    c3.metric("Optical Thickness", f"{cot:.1f}", "Density")
    c4.metric("Surface Humidity", f"{humidity}%", "Station Data")

# --- TAB 3: GEMINI FUSION ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    # 1. MASTER TABLE
    st.markdown("### 🔬 AI Training Data: The Master Table")
    table_data = {
        "Parameter": ["Cloud Phase", "Effective Radius", "Optical Thickness", "Cloud Top Pressure", "Surface Humidity"],
        "Seedable Sweet Spot": ["Liquid", "< 14 µm", "High / Opaque", "> 0°C (Warm) / -5°C (Cold)", "> 50%"],
        "Current Reading": [phase, f"{reff:.1f} µm", f"{cot:.1f}", f"{press:.0f} hPa", f"{humidity}%"]
    }
    st.table(pd.DataFrame(table_data))

    # 2. VISUAL EVIDENCE
    if plot_img:
        st.caption("Visual Evidence (Sent to Vertex AI):")
        st.image(plot_img, width=400)

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not plot_img:
            st.error("⚠️ No Data.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # --- THE SUPER PROMPT ---
                prompt = f"""
                ROLE: You are the VisionRain Decision Support Engine.
                Analyze this Satellite Data and Telemetry for Cloud Seeding.
                
                --- INPUT DATA ---
                - Cloud Phase: {phase}
                - Effective Radius: {reff:.1f} microns
                - Optical Thickness: {cot:.1f}
                - Cloud Top Pressure: {press:.0f} hPa
                - Surface Humidity: {humidity}%
                
                --- LOGIC RULES (The Physics) ---
                1. IF Cloud Phase is "Ice" -> "NO-GO" (Already glaciated).
                2. IF Effective Radius < 14 microns AND Phase is "Liquid" -> "GO" (Stuck cloud, needs nuclei).
                3. IF Optical Thickness is Low (< 5) -> "NO-GO" (Too thin).
                4. IF Humidity < 30% -> "NO-GO" (Rain will evaporate).
                
                --- VISUALS ---
                The attached image shows the Cloud Top Pressure. Darker areas are lower/warmer clouds.
                
                --- OUTPUT ---
                1. **Microphysical Analysis:** Assess the phase and droplet size.
                2. **Thermodynamic Check:** Is the humidity sufficient?
                3. **Decision:** **PRIORITY 1 (GO)** or **ABORT (NO-GO)**.
                4. **Reasoning:** Explain using the logic rules above.
                """
                
                with st.spinner("Vertex AI is calculating microphysics..."):
                    res = model.generate_content([prompt, plot_img])
                    
                    decision = "GO" if "PRIORITY" in res.text.upper() else "NO-GO"
                    log_mission(f"{lat},{lon}", f"Hum:{humidity}%", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
