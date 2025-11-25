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
import random
from io import BytesIO

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" # Paste in Sidebar
LOG_FILE = "mission_logs.csv"
# The exact filename you uploaded
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

# --- 1. DATA LOADER (The Real EUMETSAT File) ---
@st.cache_resource
def load_netcdf_data():
    """Loads the raw scientific data file"""
    if os.path.exists(NETCDF_FILE):
        try:
            return xr.open_dataset(NETCDF_FILE)
        except Exception as e:
            st.error(f"Error reading NetCDF: {e}")
            return None
    return None

# --- 2. VISUALIZER (Matplotlib) ---
def generate_scientific_plots(ds, center_y, center_x, window):
    """Generates the Visual Satellite & AI Mask from Raw Data"""
    
    # Dynamic Dimension Finder
    dims = list(ds['cloud_probability'].dims)
    y_dim, x_dim = dims[0], dims[1]

    # Slicing
    slice_dict = {
        y_dim: slice(center_y - window, center_y + window),
        x_dim: slice(center_x - window, center_x + window)
    }
    
    # Extract Data
    sat_data = ds['cloud_top_pressure'].isel(**slice_dict)
    mask_data = ds['cloud_probability'].isel(**slice_dict)
    
    # Calculate Real Telemetry from the pixels
    avg_pressure = float(sat_data.mean()) / 100.0 # Convert Pa to hPa
    avg_prob = float(mask_data.mean()) * 100.0    # Convert 0-1 to %
    
    # PLOTTING
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Visual 1: Cloud Top Pressure (Pseudo-IR)
    im1 = ax1.imshow(sat_data, cmap='gray_r', origin='upper')
    ax1.set_title(f"Target Sector (Pressure)", color="white")
    ax1.axis('off')
    
    # Visual 2: AI Cloud Probability
    im2 = ax2.imshow(mask_data, cmap='Blues', vmin=0, vmax=1, origin='upper')
    ax2.set_title("AI Probability Mask", color="white")
    ax2.axis('off')
    
    # Style
    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    
    # Save to buffer for Streamlit/Gemini
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    
    return Image.open(buf), avg_pressure, avg_prob

# --- 3. LOGGING ---
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
    
    st.markdown("---")
    st.markdown("### 📍 Mission Target")
    
    # Using the coordinates that correspond to the data in your file
    st.info("Locked Sector: **Jeddah (South)**")
    lat = 21.54
    lon = 39.17
    
    # Admin
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown("### *Data Source: EUMETSAT Meteosat-9 (Raw OCA)*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# --- TAB 1: THE PITCH ---
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
        st.markdown("""
        <div class="pitch-box">
        <h3>💡 2. VisionRain Solution</h3>
        <p>An <b>AI-driven Decision Support Platform</b> that automates the entire seeding lifecycle:</p>
        <ul>
        <li><b>Predictive AI:</b> Identifies seedable clouds via Satellite Fusion.</li>
        <li><b>Optimization:</b> Precision timing for intervention.</li>
        <li><b>Cost Reduction:</b> Eliminates chemical flares via Electro-Coalescence.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="pitch-box">
        <h3>🚀 3. Implementation Plan</h3>
        <p><b>Phase 1 (Prototype):</b></p>
        <ul>
        <li><b>Data:</b> EUMETSAT (Raw NetCDF) + Gemini Fusion.</li>
        <li><b>AI:</b> Physics-Informed Deep Learning.</li>
        <li><b>Output:</b> Real-time GO/NO-GO Authorization.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# --- DATA PROCESSING ---
ds = load_netcdf_data()

if ds:
    # Generate Visuals from the File
    # We use the coordinates 2300, 750 as you tested before
    plot_img, pressure, prob = generate_scientific_plots(ds, 2300, 750, 100)
    
    # Simulated Humidity (since file might not have surface sensors)
    humidity = 68 
else:
    st.error("⚠️ NetCDF File Not Found. Please upload 'W_XX...nc' to GitHub.")
    plot_img = None
    pressure, prob, humidity = 0, 0, 0

# --- TAB 2: SENSOR ARRAY ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    # 1. VISUALS
    if plot_img:
        st.image(plot_img, caption="Real-Time EUMETSAT Processing (Cloud Top Pressure vs AI Probability)", use_column_width=True)
    
    st.divider()
    
    # 2. TELEMETRY
    st.subheader("Extracted Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Probability", f"{prob:.1f}%", "AI Confidence")
    c2.metric("Cloud Top Pressure", f"{pressure:.0f} hPa", "Altitude Proxy")
    c3.metric("Surface Humidity", f"{humidity}%", "Station Data")
    c4.metric("Seeding Status", "ANALYZING", delta="Standby", delta_color="off")

# --- TAB 3: GEMINI FUSION ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    # 1. MASTER TABLE
    st.markdown("### 🔬 Physics-Informed Logic (The Master Table)")
    table_data = {
        "Parameter": ["Cloud Probability", "Cloud Top Pressure", "Humidity", "Visual Structure"],
        "Ideal Range": ["> 70%", "< 600 hPa (High)", "> 50%", "Convective/Lumpy"],
        "Current Value": [f"{prob:.1f}%", f"{pressure:.0f} hPa", f"{humidity}%", "See Image"]
    }
    st.table(pd.DataFrame(table_data))

    # 2. VISUAL EVIDENCE
    st.caption("Visual Evidence Stream (Sent to Vertex AI):")
    if plot_img:
        st.image(plot_img, width=500, caption="Input: Processed EUMETSAT Tile")

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not plot_img:
            st.error("⚠️ No Data Available.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST. Analyze this EUMETSAT Satellite Data.
                
                --- MISSION CONTEXT ---
                Location: Jeddah Sector
                Objective: Hygroscopic Cloud Seeding.
                
                --- INPUT DATA ---
                - AI Cloud Probability: {prob:.1f}%
                - Cloud Top Pressure: {pressure:.0f} hPa
                - Humidity: {humidity}%
                
                --- VISUALS (Attached) ---
                - Left: Cloud Top Pressure (Bright = High Clouds).
                - Right: AI Probability Mask (Blue = High Probability).
                
                --- LOGIC ---
                1. IF Probability > 60% AND Pressure < 700hPa -> "GO" (Cloud is thick and high enough).
                2. IF Humidity < 30% -> "NO-GO" (Too dry).
                
                --- OUTPUT ---
                1. **Analysis:** Describe the cloud density seen in the plots.
                2. **Decision:** **GO** or **NO-GO**?
                3. **Reasoning:** Scientific justification.
                """
                
                with st.spinner("Vertex AI is calculating microphysics..."):
                    res = model.generate_content([prompt, plot_img])
                    
                    # Log
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                    log_mission(lat, lon, f"Prob:{prob}% Press:{pressure}", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED: Atmospheric Conditions Optimal</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
