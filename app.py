import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import datetime
import os
import csv
import requests
from io import BytesIO
import random

# --- FIX MATPLOTLIB CRASH ON WEB ---
matplotlib.use('Agg')

# --- SAFELY IMPORT SCIENTIFIC LIBS ---
try:
    import xarray as xr
except ImportError:
    st.error("⚠️ Scientific Libraries Missing! Please update requirements.txt with: xarray, netCDF4, h5netcdf")
    xr = None

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
LOG_FILE = "mission_logs.csv"
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"

st.set_page_config(page_title="VisionRain | Scientific Core", layout="wide", page_icon="🛰️")

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

# --- 1. DATA LOADER ---
@st.cache_resource
def load_netcdf_data():
    if xr is None: return None
    # Use os.path.join to ensure it finds the file on Cloud/Windows
    file_path = os.path.join(os.getcwd(), NETCDF_FILE)
    if os.path.exists(file_path):
        try:
            return xr.open_dataset(file_path, engine='netcdf4')
        except Exception as e:
            st.error(f"Error reading NetCDF: {e}")
            return None
    return None

# --- 2. SCIENTIFIC VISUALIZER ---
def generate_scientific_plots(ds, center_y, center_x, window, title_prefix="Target"):
    """
    Generates the Matplotlib visualization using the user's exact logic.
    """
    if ds is None: return None, 0, 0, 0, 0 

    # 1. Dynamic Dimension Finder
    try:
        dims = list(ds['cloud_probability'].dims)
        y_dim_name = dims[0]
        x_dim_name = dims[1]
    except:
        return None, 0, 0, 0, 0

    # 2. Slicing
    # Ensure we don't go out of bounds
    max_y = ds.sizes[y_dim_name]
    max_x = ds.sizes[x_dim_name]
    
    y_start = max(0, center_y - window)
    y_end = min(max_y, center_y + window)
    x_start = max(0, center_x - window)
    x_end = min(max_x, center_x + window)

    slice_dict = {
        y_dim_name: slice(y_start, y_end),
        x_dim_name: slice(x_start, x_end)
    }

    # 3. Extract Data
    sat_image = ds['cloud_top_pressure'].isel(**slice_dict)
    ai_mask = (ds['cloud_probability'].isel(**slice_dict))/100
    
    # --- 3a. Radius Extraction ---
    if 'cloud_particle_effective_radius' in ds:
        rad_slice = ds['cloud_particle_effective_radius'].isel(**slice_dict)
        avg_rad = float(rad_slice.mean()) * 1e6 
    else:
        avg_rad = 0.0
        
    # --- 3b. Optical Depth Extraction ---
    if 'cloud_optical_depth_log' in ds:
        cod_slice = ds['cloud_optical_depth_log'].isel(**slice_dict)
        # File stores as Log10. Convert to Linear: 10^x
        avg_cod = float((10**cod_slice).mean())
    else:
        avg_cod = 0.0

    # Calculate Stats for Telemetry
    avg_press = float(sat_image.mean()) / 100.0 if sat_image.size > 0 else 0
    avg_prob = float(ai_mask.mean()) * 100.0 if ai_mask.size > 0 else 0

    # 4. PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0e1117')

    # Plot 1: Satellite Feed (Pressure)
    im1 = ax1.imshow(sat_image, cmap='gray_r', origin='upper')
    ax1.set_title(f"{title_prefix} Satellite Feed (Cloud Top Pressure)", fontsize=12, color="white")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label="Pressure (Pa)").ax.yaxis.set_tick_params(color='white')

    # Plot 2: AI Detection (Probability)
    im2 = ax2.imshow(ai_mask, cmap='Blues', vmin=0, vmax=1, origin='upper')
    ax2.set_title(f"{title_prefix} AI Identification (Cloud Probability)", fontsize=12, color="white")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label="Probability (0-1)").ax.yaxis.set_tick_params(color='white')

    # Add Crosshair
    mid_y = (y_end - y_start) // 2
    mid_x = (x_end - x_start) // 2
    ax1.plot(mid_x, mid_y, 'c+', markersize=20, markeredgewidth=3)
    ax2.plot(mid_x, mid_y, 'r+', markersize=20, markeredgewidth=3)

    plt.suptitle(f"System Lock: {center_x}X / {center_y}Y", fontsize=16, fontweight='bold', color="#00e5ff")
    plt.tight_layout()
    
    # Save to Buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf), avg_press, avg_prob, avg_rad, avg_cod

# --- 3. OWM TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 4. LOGGING ---
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
    st.caption("Scientific Core | EUMETSAT")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📍 Mission Target")
    st.info("Locked Sector: **Jeddah Storm**")
    lat, lon = 21.54, 39.17
    
    # Admin
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())
            if st.button("Clear Logs"):
                if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
                st.rerun()

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown("### *Data Source: EUMETSAT NetCDF (Raw Analysis)*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Sensor Array", "🧠 Gemini Fusion"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b> face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>.</p>
    <p>This aligns critically with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b>.</p>
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
    # 1. Generate Full Disk (Global Context)
    full_img, _, _, _, _ = generate_scientific_plots(ds, 1856, 1856, 1800, title_prefix="Global")
    
    # 2. Generate Zoomed Sector (Jeddah)
    zoom_img, pressure, prob, radius, opt_depth = generate_scientific_plots(ds, 2300, 750, 100, title_prefix="Jeddah Sector")
else:
    st.error("⚠️ NetCDF File Missing. Please upload 'W_XX...nc' to GitHub.")
    full_img, zoom_img, pressure, prob, radius, opt_depth = None, None, 0, 0, 0, 0

# Get Live OWM Data
w = get_weather_telemetry(lat, lon, WEATHER_API_KEY)
humidity = w['main']['humidity'] if w else 65 

# --- TAB 2: SENSORS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    # VISUALS
    if full_img:
        st.image(full_img, caption="1. Global Context (Meteosat Full Disk)", use_column_width=True)
    
    st.write("---")
    
    if zoom_img:
        st.image(zoom_img, caption="2. Target Sector Analysis (Jeddah)", use_column_width=True)

    st.divider()
    
    # TELEMETRY TABLE
    st.subheader("Microphysical Telemetry")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cloud Probability", f"{prob:.1f}%", "AI Confidence")
    c2.metric("Cloud Top Pressure", f"{pressure:.0f} hPa", "Altitude Proxy")
    c3.metric("Droplet Radius", f"{radius:.1f} µm", "Stalled Growth")
    c4.metric("Optical Depth", f"{opt_depth:.1f}", "Water Volume")
    c5.metric("Humidity", f"{humidity}%", "Atmospheric Water")

# --- TAB 3: GEMINI FUSION ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    # 1. MASTER TABLE
    st.markdown("### 🔬 Physics-Informed Logic (The Master Table)")
    table_data = {
        "Parameter": ["Cloud Probability", "Cloud Top Pressure", "Droplet Radius", "Optical Depth", "Humidity"],
        "Ideal Range": ["> 70%", "< 700 hPa", "< 14 µm", "> 10", "> 50%"],
        "Current Value": [f"{prob:.1f}%", f"{pressure:.0f} hPa", f"{radius:.1f} µm", f"{opt_depth:.1f}", f"{humidity}%"]
    }
    st.table(pd.DataFrame(table_data))

    # 2. VISUAL EVIDENCE
    if zoom_img:
        st.caption("Visual Evidence Sent to Vertex AI:")
        st.image(zoom_img, width=500)

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not zoom_img:
            st.error("⚠️ No Data.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # --- THE SUPER PROMPT ---
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST. Analyze this EUMETSAT Satellite Data.
                
                --- MISSION CONTEXT ---
                Location: Jeddah Sector (Meteosat-9)
                Objective: Hygroscopic Cloud Seeding.
                
                --- INPUT DATA ---
                1. AI Cloud Probability: {prob:.1f}% (0-100 scale)
                2. Cloud Top Pressure: {pressure:.0f} hPa
                3. Droplet Effective Radius: {radius:.1f} microns
                4. Optical Depth (Thickness): {opt_depth:.1f}
                5. Surface Humidity: {humidity}%
                
                --- VISUALS (Attached) ---
                The image contains two plots:
                - Left: Cloud Top Pressure (Darker/Grey = Lower/Warmer Clouds).
                - Right: AI Probability Mask (Blue = High Probability).
                
                --- LOGIC ---
                1. IF Probability > 60% AND Pressure < 800hPa -> "GO" (Cloud is substantial).
                2. IF Droplet Radius < 14 microns -> "GO" (Cloud is stuck, needs seeding).
                3. IF Droplet Radius > 14 microns -> "NO-GO" (Already raining).
                4. IF Optical Depth < 5 -> "NO-GO" (Cloud too thin).
                5. IF Humidity < 30% -> "NO-GO" (Too dry).
                
                --- OUTPUT ---
                1. **Analysis:** Describe the cloud density seen in the zoomed sector plots and the microphysics.
                2. **Decision:** **GO** or **NO-GO**?
                3. **Reasoning:** Scientific justification based on the 5 metrics.
                """
                
                with st.spinner("Vertex AI is calculating microphysics..."):
                    res = model.generate_content([prompt, zoom_img])
                    
                    # --- FIXED LOGIC HERE ---
                    text_response = res.text.upper()
                    
                    if "NO-GO" in text_response:
                        decision = "NO-GO"
                    elif "GO" in text_response:
                        decision = "GO"
                    else:
                        decision = "NO-GO" # Fallback
                    # ------------------------
                    
                    # Logging
                    log_str = f"P:{prob:.0f}% R:{radius:.1f}um OD:{opt_depth:.1f} H:{humidity}%"
                    log_mission(f"{lat},{lon}", log_str, decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
