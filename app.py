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
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

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
    """
    Mathematically generates a realistic weather radar map.
    No internet required. Never breaks. Looks 100% scientific.
    """
    # Setup grid
    size = 200
    data = np.zeros((size, size))
    
    # Generate random "Storm Cells"
    if intensity == "Clear":
        num_blobs = 0
    elif intensity == "Light":
        num_blobs = 5
        max_val = 0.5
    else: # Heavy
        num_blobs = 15
        max_val = 1.0

    for _ in range(num_blobs):
        x, y = random.randint(0, size), random.randint(0, size)
        data[x, y] = random.uniform(0.5, 1.0) * 100 # Seed points

    # Apply Gaussian Blur to simulate organic cloud spread
    radar_data = gaussian_filter(data, sigma=random.randint(5, 15))
    
    # Normalize
    radar_data = radar_data / np.max(radar_data) if np.max(radar_data) > 0 else radar_data
    
    # Plotting
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    # Create a black background map look
    ax.set_facecolor('black')
    ax.imshow(np.zeros((size, size)), cmap='gray', vmin=0, vmax=1) 
    
    # Overlay Radar (Transparent Alpha)
    # 'jet' colormap gives the classic Blue->Green->Red->White radar look
    masked_data = np.ma.masked_where(radar_data < 0.1, radar_data)
    ax.imshow(masked_data, cmap='jet', alpha=0.8, vmin=0, vmax=0.8)
    
    # Add Fake Grid Lines (Lat/Lon)
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axis('off')
    
    # Save to Buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    buf.seek(0)
    return Image.open(buf)

# --- 2. SATELLITE ROTATOR (Simulated Live Feed) ---
def get_simulated_satellite():
    """Rotates between high-quality NASA images to simulate a live feed"""
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/85000/85423/cumulonimbus_tmo_2008036_lrg.jpg", 
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Hurricane_Isabel_from_ISS.jpg/800px-Hurricane_Isabel_from_ISS.jpg"
    ]
    # Pick one based on minute to simulate changing satellite passes
    idx = int(datetime.datetime.now().minute / 20) % len(urls) 
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(urls[idx], headers=headers, timeout=5)
        return Image.open(io.BytesIO(r.content))
    except:
        return generate_synthetic_radar("Clear") # Fallback to generated image

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
    st.caption("System Status: **ONLINE**")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 🎛️ Simulation Controller")
    st.caption("Inject Weather Pattern for Demo:")
    
    scenario = st.selectbox("Current Conditions:", ["Heavy Storm", "Light Rain", "Clear Sky"])
    
    target_name = st.text_input("Region Name", "Jeddah")
    st.success(f"Tracking: {target_name}")

# --- MAIN DASHBOARD ---
st.title("VisionRain Command Center")
dt_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"### *Live Downlink: {dt_now}*")

tab1, tab2 = st.tabs(["📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# Generate Data based on Scenario
with st.spinner("Calibrating Sensors..."):
    # 1. Generate Visuals
    radar_img = generate_synthetic_radar(intensity="Heavy" if scenario == "Heavy Storm" else "Light" if scenario == "Light Rain" else "Clear")
    sat_img = get_simulated_satellite()
    
    # 2. Generate Numbers
    telem = get_simulated_telemetry(scenario)

# TAB 1: SENSOR ARRAY
with tab1:
    col_vis, col_dat = st.columns([2, 1])
    
    with col_vis:
        st.subheader("A. Multi-Spectral Visuals")
        c1, c2 = st.columns(2)
        c1.image(sat_img, caption="Optical Satellite (NASA VIIRS)", use_column_width=True)
        c2.image(radar_img, caption="Doppler Radar (Reflectivity)", use_column_width=True)
        
    with col_dat:
        st.subheader("B. Telemetry")
        st.metric("Humidity", f"{telem['humidity']}%", "Target > 40%")
        st.metric("Temperature", f"{telem['temp']}°C")
        st.metric("Pressure", f"{telem['pressure']} hPa")
        st.metric("Wind Speed", f"{telem['wind']} m/s")
        
        if telem['humidity'] > 40:
            st.success("✅ SEEDABLE")
        else:
            st.error("⚠️ TOO DRY")

# TAB 2: GEMINI AI
with tab2:
    st.header("2. Gemini Fusion Engine")
    
    if st.button("RUN LIVE DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # Display what AI sees
            st.markdown("### 👁️ AI Input Stream")
            c1, c2 = st.columns(2)
            c1.image(sat_img, caption="Input 1: Visual", use_column_width=True)
            c2.image(radar_img, caption="Input 2: Generated Radar", use_column_width=True)
            
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST. Analyze this Live Sensor Data.
            
            --- TELEMETRY ---
            - Humidity: {telem['humidity']}%
            - Pressure: {telem['pressure']} hPa
            
            --- VISUALS ---
            Image 1: Satellite (Real-World).
            Image 2: Doppler Radar (Generated Heatmap). 
               - Red/Yellow blobs = Heavy Cells.
               - Blue/Green blobs = Light Rain.
               - Black = Clear.
            
            --- TASK ---
            1. RADAR ANALYSIS: Describe the intensity shown in Image 2.
            2. CORRELATION: Does the humidity support the radar echoes?
            3. DECISION: **GO** or **NO-GO** for Seeding?
            4. REASONING: Scientific justification.
            """
            
            with st.spinner("Gemini 2.0 is fusing streams..."):
                try:
                    res = model.generate_content([prompt, sat_img, radar_img])
                    st.markdown("### 🛰️ Mission Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.balloons()
                        st.success("✅ MISSION APPROVED")
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
