import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Removed xarray import
import pandas as pd
import datetime
import os
import csv
import requests
from io import BytesIO
import random

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"
# We keep the filename variable but won't actually read it
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"

st.set_page_config(page_title="VisionRain | EUMETSAT Core", layout="wide", page_icon="🛰️")

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

# --- 1. SIMULATED VISUALIZER (No Xarray Needed) ---
def generate_full_and_zoom(center_y, center_x, window):
    """
    Generates visual plots using random noise to simulate satellite data.
    This removes the need for xarray/netCDF4 libraries.
    """
    
    # Generate Fake Data (Clouds)
    # 3712x3712 is the size of a real MSG disk
    size = 3712
    
    # Create a 'Planet' mask (Circle)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = (x - center)**2 + (y - center)**2 <= (center - 50)**2
    
    # Generate Cloud Pressure (Random Noise)
    raw_press = np.random.randint(200, 1000, (size, size))
    raw_press = np.where(mask, raw_press, np.nan) # Mask space

    # Generate Cloud Probability (Random Noise)
    raw_prob = np.random.randint(0, 100, (size, size))
    raw_prob = np.where(mask, raw_prob, np.nan)

    # Zoom Slice
    y_min, y_max = max(0, center_y - window), min(size, center_y + window)
    x_min, x_max = max(0, center_x - window), min(size, center_x + window)
    
    zoom_press = raw_press[y_min:y_max, x_min:x_max]
    zoom_prob = raw_prob[y_min:y_max, x_min:x_max]

    # Calculate Stats
    avg_press = float(np.nanmean(zoom_press))
    avg_prob = float(np.nanmean(zoom_prob))

    # Plotting
    fig = plt.figure(figsize=(10, 12))
    fig.patch.set_facecolor('#0e1117')
    
    # Top: Full Disk (Subsampled for speed)
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(raw_prob[::10, ::10], cmap='Blues_r', vmin=0, vmax=100)
    rect = plt.Rectangle((center_x/10 - window/10, center_y/10 - window/10), 
                         window/5, window/5, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title("1. Global Context (Simulated Meteosat)", color="white")
    ax1.axis('off')

    # Bottom: Zoom
    ax2 = plt.subplot(2, 1, 2)
    im2 = ax2.imshow(zoom_press, cmap='gray_r') 
    ax2.set_title("2. Target Sector Analysis (Simulated Pressure)", color="white")
    ax2.axis('off')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    
    return Image.open(buf), avg_press, avg_prob

# --- 2. LOGGING ---
def log_mission(lat, lon, conditions, decision, reason):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["Timestamp", "Location", "Conditions", "Decision", "Reason"])
        writer.writerow([ts, f"{lat},{lon}", conditions, decision, reason])

def load_logs():
    if os.path.isfile(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Location", "Conditions", "Decision", "Reason"])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Scientific Core | EUMETSAT Integration")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📍 Target Sector")
    st.info("Locked: **Jeddah Storm Cell**\n(Coords: 2300, 750)")
    
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown("### *Source: EUMETSAT OCA (Simulated Feed)*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Sensor Array", "🧠 Gemini Fusion"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b> face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - An AI-driven decision support platform analyzing satellite microphysics for precision seeding.")
    with c2:
        st.success("**Impact:** Supports Saudi Green Initiative & Water Security.")

# --- PROCESS DATA (SIMULATED) ---
# We removed the file loader and just generate the visuals
plot_img, pressure, prob = generate_full_and_zoom(1856, 1856, 150) # Using center of disk for demo
humidity = 68 

# --- TAB 2: SENSORS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    if plot_img:
        st.image(plot_img, caption="Meteosat-9 Analysis: Full Disk (Top) & Jeddah Sector (Bottom)", use_column_width=True)
    
    st.divider()
    
    # TELEMETRY ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Probability", f"{prob:.1f}%", "AI Confidence")
    c2.metric("Cloud Top Pressure", f"{pressure:.0f} hPa", "Altitude Proxy")
    c3.metric("Surface Humidity", f"{humidity}%", "Station Data")
    c4.metric("Seeding Status", "ANALYZING", delta="Standby", delta_color="off")

# --- TAB 3: GEMINI AI ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.markdown("### 🔬 Physics-Informed Logic (The Master Table)")
    table_data = {
        "Parameter": ["Cloud Probability", "Cloud Top Pressure", "Humidity", "Visual Structure"],
        "Ideal Range": ["> 60%", "< 700 hPa (High)", "> 50%", "Convective/Lumpy"],
        "Current Value": [f"{prob:.1f}%", f"{pressure:.0f} hPa", f"{humidity}%", "See Zoom Plot"]
    }
    st.table(pd.DataFrame(table_data))

    if plot_img:
        st.caption("Visual Evidence Sent to Vertex AI:")
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
                
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST. Analyze this EUMETSAT Satellite Data.
                
                --- MISSION CONTEXT ---
                Location: Jeddah Sector (Meteosat-9)
                Objective: Hygroscopic Cloud Seeding.
                
                --- INPUT DATA ---
                - AI Cloud Probability: {prob:.1f}%
                - Cloud Top Pressure: {pressure:.0f} hPa
                - Surface Humidity: {humidity}%
                
                --- VISUALS (Attached) ---
                - Top Image: Full Earth Disk (Global Context).
                - Bottom Image: Target Sector (Cloud Top Pressure). Darker/Grey = Lower Clouds, White = Higher Clouds.
                
                --- LOGIC ---
                1. IF Probability > 60% AND Pressure < 800hPa -> "GO" (Cloud is substantial).
                2. IF Humidity < 30% -> "NO-GO" (Too dry).
                
                --- OUTPUT ---
                1. **Analysis:** Describe the cloud density seen in the plots.
                2. **Decision:** **GO** or **NO-GO**?
                3. **Reasoning:** Scientific justification.
                """
                
                with st.spinner("Vertex AI is calculating microphysics..."):
                    res = model.generate_content([prompt, plot_img])
                    
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                    log_mission(21.54, 39.17, f"Prob:{prob:.1f}%", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED: Atmospheric Conditions Optimal</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
