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

# --- FIX MATPLOTLIB CRASH ON WEB ---
matplotlib.use('Agg')

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
LOG_FILE = "mission_logs.csv"
# EXACT FILE NAME YOU UPLOADED
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

# --- 1. DATA LOADER (ROBUST) ---
@st.cache_resource
def load_netcdf_data():
    import xarray as xr
    # Dynamic path handling
    file_path = os.path.join(os.getcwd(), NETCDF_FILE)
    
    if os.path.exists(file_path):
        try:
            # TRY ENGINE 1: Standard NetCDF4 (Best for this file type)
            return xr.open_dataset(file_path, engine='netcdf4')
        except Exception as e1:
            try:
                # TRY ENGINE 2: H5NetCDF (Fallback)
                return xr.open_dataset(file_path, engine='h5netcdf')
            except Exception as e2:
                st.error(f"Failed to open file. \nError 1: {e1} \nError 2: {e2}")
                return None
    return None

# --- 2. SCIENTIFIC VISUALIZER ---
def generate_scientific_plots(ds, center_y, center_x, window, title_prefix="Target"):
    if ds is None: return None, 0, 0, 0

    # 1. Dynamic Dimension Finder
    # EUMETSAT files vary, we try to auto-detect dimensions
    try:
        dims = list(ds.dims)
        # Usually dims are something like 'xc', 'yc' or 'number_of_lines'
        y_dim = [d for d in dims if 'line' in d or 'y' in d][0]
        x_dim = [d for d in dims if 'pixel' in d or 'x' in d][0]
    except:
        return None, 0, 0, 0

    # 2. Slicing
    max_y = ds.sizes[y_dim]
    max_x = ds.sizes[x_dim]
    
    y_start = max(0, center_y - window)
    y_end = min(max_y, center_y + window)
    x_start = max(0, center_x - window)
    x_end = min(max_x, center_x + window)

    slice_dict = {
        y_dim: slice(y_start, y_end),
        x_dim: slice(x_start, x_end)
    }

    # 3. Extract Data (With error handling for missing vars)
    try:
        # Pressure
        sat_image = ds['cloud_top_pressure'].isel(**slice_dict)
        
        # Probability
        if 'cloud_probability' in ds:
            ai_mask = (ds['cloud_probability'].isel(**slice_dict))/100
        else:
            ai_mask = sat_image * 0 # Dummy empty mask

        # Radius (The new requested variable)
        if 'cloud_particle_effective_radius' in ds:
            rad_grid = ds['cloud_particle_effective_radius'].isel(**slice_dict)
            avg_rad = float(rad_grid.mean()) * 1e6 # Convert to microns
        else:
            avg_rad = 0.0

        # Calculate Stats
        avg_press = float(sat_image.mean()) / 100.0 if sat_image.size > 0 else 0
        avg_prob = float(ai_mask.mean()) * 100.0 if ai_mask.size > 0 else 0

    except KeyError as e:
        st.error(f"Variable Missing in NetCDF: {e}")
        return None, 0, 0, 0

    # 4. PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0e1117')

    # Plot 1: Pressure
    im1 = ax1.imshow(sat_image, cmap='gray_r', origin='upper')
    ax1.set_title(f"{title_prefix} Cloud Top Pressure (Altitude)", fontsize=12, color="white")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label="Pressure (Pa)").ax.yaxis.set_tick_params(color='white')

    # Plot 2: Probability
    im2 = ax2.imshow(ai_mask, cmap='Blues', vmin=0, vmax=1, origin='upper')
    ax2.set_title(f"{title_prefix} AI Cloud Probability", fontsize=12, color="white")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label="Probability").ax.yaxis.set_tick_params(color='white')

    plt.suptitle(f"Target Lock: {center_x}X / {center_y}Y", fontsize=16, fontweight='bold', color="#00e5ff")
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf), avg_press, avg_prob, avg_rad

# --- 3. LOGGING & HELPERS ---
def get_weather_telemetry(lat, lon, key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

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
    # Coordinates for demo
    lat, lon = 21.54, 39.17
    
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

# --- TAB 1 ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Saudi Arabia faces critical water scarcity. Current seeding is manual and reactive.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("**Solution:** VisionRain analyzes satellite microphysics (Radius + Pressure) for precision seeding.")

# --- DATA PROCESSING ---
ds = load_netcdf_data()

if ds:
    # Generate Plots
    zoom_img, pressure, prob, radius = generate_scientific_plots(ds, 2300, 750, 100, title_prefix="Jeddah Sector")
else:
    st.error("⚠️ NetCDF File Missing. Please upload 'W_XX...nc' to GitHub.")
    zoom_img, pressure, prob, radius = None, 0, 0, 0

# OWM Data
w = get_weather_telemetry(lat, lon, WEATHER_API_KEY)
humidity = w['main']['humidity'] if w else 65 

# --- TAB 2 ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    if zoom_img:
        st.image(zoom_img, caption="Target Sector Analysis", use_column_width=True)
    
    st.divider()
    
    # METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Probability", f"{prob:.1f}%")
    c2.metric("Cloud Top Pressure", f"{pressure:.0f} hPa")
    c3.metric("Droplet Radius", f"{radius:.1f} µm", help="Target < 14 microns")
    c4.metric("Seeding Status", "ANALYZING", delta="Standby", delta_color="off")

# --- TAB 3 ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.markdown("### 🔬 Physics-Informed Logic")
    table_data = {
        "Parameter": ["Cloud Probability", "Cloud Top Pressure", "Droplet Radius", "Humidity"],
        "Ideal Range": ["> 70%", "< 700 hPa", "< 14 µm (Stuck)", "> 50%"],
        "Current Value": [f"{prob:.1f}%", f"{pressure:.0f} hPa", f"{radius:.1f} µm", f"{humidity}%"]
    }
    st.table(pd.DataFrame(table_data))
    
    if zoom_img:
        st.image(zoom_img, width=400)

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not zoom_img:
            st.error("⚠️ No Data.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST. Analyze this EUMETSAT Data.
                
                --- INPUT DATA ---
                - Cloud Probability: {prob:.1f}%
                - Pressure: {pressure:.0f} hPa
                - Droplet Radius: {radius:.1f} microns
                - Humidity: {humidity}%
                
                --- LOGIC ---
                1. IF Radius < 14 microns -> "GO" (Cloud is stuck).
                2. IF Radius > 14 microns -> "NO-GO" (Already raining).
                3. IF Humidity < 30% -> "NO-GO".
                
                --- OUTPUT ---
                1. **Analysis:** Describe the microphysics.
                2. **Decision:** **GO** or **NO-GO**?
                3. **Reasoning:** Justify.
                """
                
                with st.spinner("Gemini is analyzing..."):
                    res = model.generate_content([prompt, zoom_img])
                    
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                    log_mission(f"{lat},{lon}", f"Rad:{radius:.1f}um", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
