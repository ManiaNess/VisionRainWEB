import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
import io
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" # Paste your Google AI Key here
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" # Not needed for simulation mode

st.set_page_config(page_title="VisionRain Simulation Core", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4facfe;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. PROCEDURAL RADAR GENERATOR (The "AI" Generator) ---
def generate_synthetic_radar(intensity="Heavy"):
    """Mathematically generates a realistic weather radar map."""
    size = 200
    data = np.zeros((size, size))
    
    if intensity == "Clear":
        num_blobs = 0
    elif intensity == "Light":
        num_blobs = 5
    else: # Heavy
        num_blobs = 15

    for _ in range(num_blobs):
        x, y = random.randint(0, size), random.randint(0, size)
        data[x, y] = random.uniform(0.5, 1.0) * 100 # Seed points

    radar_data = gaussian_filter(data, sigma=random.randint(5, 15))
    radar_data = radar_data / np.max(radar_data) if np.max(radar_data) > 0 else radar_data
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_facecolor('black')
    ax.imshow(np.zeros((size, size)), cmap='gray', vmin=0, vmax=1) 
    
    masked_data = np.ma.masked_where(radar_data < 0.1, radar_data)
    ax.imshow(masked_data, cmap='jet', alpha=0.8, vmin=0, vmax=0.8)
    
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    buf.seek(0)
    plt.close(fig) # Close plot to free memory
    return Image.open(buf)

# --- 2. SATELLITE ROTATOR (Simulated Live Feed) ---
def get_simulated_satellite():
    """Rotates between high-quality NASA images to simulate a live feed"""
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/85000/85423/cumulonimbus_tmo_2008036_lrg.jpg", 
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Hurricane_Isabel_from_ISS.jpg/800px-Hurricane_Isabel_from_ISS.jpg"
    ]
    idx = int(datetime.datetime.now().minute / 20) % len(urls) 
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(urls[idx], headers=headers, timeout=5)
        return Image.open(io.BytesIO(r.content))
    except:
        return generate_synthetic_radar("Clear")

# --- 3. TELEMETRY SIMULATOR ---
def get_simulated_telemetry(scenario):
    if scenario == "Heavy Storm":
        return {"humidity": 85, "temp": 22, "pressure": 998, "wind": 25}
    elif scenario == "Light Rain":
        return {"humidity": 60, "temp": 28, "pressure": 1012, "wind": 12}
    else: # Clear
        return {"humidity": 25, "temp": 35, "pressure": 1020, "wind": 5}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("System Status: **ONLINE (SIMULATION MODE)**")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 🎛️ Simulation Controller")
    scenario = st.selectbox("Inject Weather Pattern:", ["Heavy Storm", "Light Rain", "Clear Sky"])
    
    target_name = st.text_input("Region Name", "Jeddah")
    st.success(f"Tracking: {target_name}")

# --- MAIN DASHBOARD ---
st.title("VisionRain Command Center")
dt_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"### *Live Downlink: {dt_now}*")

# --- THE FIX: GENERATE AND LOCK DATA INTO SESSION STATE ---
# This ensures the exact images generated here are available in all tabs
with st.spinner("Calibrating Sensors & Generating Imagery..."):
    radar_img_gen = generate_synthetic_radar(intensity="Heavy" if scenario == "Heavy Storm" else "Light" if scenario == "Light Rain" else "Clear")
    sat_img_gen = get_simulated_satellite()
    telem_gen = get_simulated_telemetry(scenario)
    
    # Lock into session state
    st.session_state['sim_radar'] = radar_img_gen
    st.session_state['sim_sat'] = sat_img_gen
    st.session_state['sim_telem'] = telem_gen


tab1, tab2 = st.tabs(["📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# TAB 1: SENSOR ARRAY (Uses locked data)
with tab1:
    col_vis, col_dat = st.columns([2, 1])
    
    with col_vis:
        st.subheader("A. Multi-Spectral Visuals")
        c1, c2 = st.columns(2)
        # Use data from session state
        c1.image(st.session_state['sim_sat'], caption="Optical Satellite (NASA VIIRS)", use_column_width=True)
        c2.image(st.session_state['sim_radar'], caption="Doppler Radar (Reflectivity)", use_column_width=True)
        
    with col_dat:
        st.subheader("B. Telemetry")
        t = st.session_state['sim_telem']
        st.metric("Humidity", f"{t['humidity']}%", "Target > 40%")
        st.metric("Temperature", f"{t['temp']}°C")
        st.metric("Pressure", f"{t['pressure']} hPa")
        st.metric("Wind Speed", f"{t['wind']} m/s")
        
        if t['humidity'] > 40:
            st.success("✅ SEEDABLE")
        else:
            st.error("⚠️ TOO DRY")

# TAB 2: GEMINI AI (Uses the EXACT SAME locked data)
with tab2:
    st.header("2. Gemini Fusion Engine")
    
    if st.button("RUN LIVE DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # Display what AI sees (using session state data)
            st.markdown("### 👁️ AI Input Stream Verification")
            c1, c2 = st.columns(2)
            c1.image(st.session_state['sim_sat'], caption="Input 1: Visual Satellite", use_column_width=True)
            c2.image(st.session_state['sim_radar'], caption="Input 2: Doppler Radar", use_column_width=True)
            
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST. Analyze this Live Sensor Data for Cloud Seeding.
            
            --- TELEMETRY ---
            - Humidity: {st.session_state['sim_telem']['humidity']}%
            - Pressure: {st.session_state['sim_telem']['pressure']} hPa
            
            --- VISUALS (Attached below) ---
            Image 1: Optical Satellite.
            Image 2: Doppler Radar (Generated Heatmap: Red/Yellow=Heavy Rain, Blue/Green=Light, Black=Clear).
            
            --- TASK ---
            1. RADAR ANALYSIS: Look at Image 2. Describe the intensity and coverage of precipitation blobs.
            2. CORRELATION: Does the humidity level align with the radar visuals?
            3. DECISION: **GO** or **NO-GO** for Seeding?
            4. REASONING: Scientific justification based on the provided images and data.
            """
            
            with st.spinner("Gemini 2.0 is fusing streams..."):
                try:
                    # PASS THE LOCKED IMAGES TO GEMINI
                    res = model.generate_content([prompt, st.session_state['sim_sat'], st.session_state['sim_radar']])
                    st.markdown("### 🛰️ Mission Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.balloons()
                        st.success("✅ MISSION APPROVED")
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
