import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components
import requests
import datetime
from io import BytesIO
import os

# --- CONFIGURATION ---
DEFAULT_API_KEY = "AIzaSyA7Yk4WRdSu976U4EpHZN47m-KA8JbJ5do"
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain Platform", layout="wide", page_icon="⛈️")

# --- PRO STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #60a5fa;}
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: LIVE NASA FEED ---
def get_nasa_feed(lat, lon):
    """Fetches satellite imagery for the USER-SELECTED location"""
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bbox = f"{lat-5},{lon-5},{lat+5},{lon+5}" 
    
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "800", "HEIGHT": "800", "TIME": today
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200: return Image.open(BytesIO(r.content)), today
    except: return None, None

# --- HELPER: WEATHER ---
def get_weather_telemetry(lat, lon):
    if WEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            return requests.get(url).json()['main']
        except: pass
    return {"humidity": 65, "temp": 32, "pressure": 1012} 

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Intelligent Planet Initiative")
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Mission Target")
    
    target_name = st.text_input("Region Name", "Riyadh")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=24.7, step=0.1)
    with col2:
        lon = st.number_input("Longitude", value=46.7, step=0.1)
    
    st.map({"lat": [lat], "lon": [lon]})
    st.success(f"Locked: **{target_name}**")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Sector Analysis: {target_name} ({lat}, {lon})*")

tab1, tab2, tab3 = st.tabs(["📡 Data Fusion", "🧠 Gemini Fusion Core", "⚡ Drone Swarm"])

# TAB 1: DATA FUSION
with tab1:
    st.header("1. Hydro-Meteorological Ingestion")
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader(f"A. Satellite Downlink ({target_name})")
        option = st.radio("Source:", ["Live Feed (Real-Time)", "Archive (Simulation)"], horizontal=True)
        
        if option == "Live Feed (Real-Time)":
            if st.button("📡 ACQUIRE LIVE SIGNAL"):
                with st.spinner(f"Re-orienting Satellite to {lat}, {lon}..."):
                    img, date = get_nasa_feed(lat, lon)
                    if img:
                        st.image(img, caption=f"Live Feed: {date} | VIIRS/Suomi NPP", use_column_width=True)
                        st.session_state['analysis_img'] = img
                    else:
                        st.warning("Orbit Offline. Switching to Backup.")
                        st.session_state['analysis_img'] = Image.open("data/satellite/saudi_storm_visual.jpg")
                        st.image(st.session_state['analysis_img'], caption="Backup Archive")
        else:
            # Backup Image
            st.session_state['analysis_img'] = Image.open("data/satellite/saudi_storm_visual.jpg")
            st.image(st.session_state['analysis_img'], caption="Archive: Convective System", use_column_width=True)

    with col2:
        st.subheader("B. Atmospheric Telemetry")
        w = get_weather_telemetry(lat, lon)
        c1, c2 = st.columns(2)
        c1.metric("Humidity", f"{w['humidity']}%", "Target > 40%")
        c2.metric("Temperature", f"{w['temp']}°C")
        
        # --- FIXED RADAR SECTION ---
        st.subheader("C. Precipitation Radar")
        # We use the LOCAL file we downloaded in setup_assets.py
        radar_path = "data/radar/saudi_rain_map.jpg"
        
        if os.path.exists(radar_path):
            st.image(radar_path, caption="NASA GPM IMERG (Scientific Data)", use_column_width=True)
        else:
            # Guaranteed Internet Fallback if file is missing
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Radar_reflectivity.jpg/600px-Radar_reflectivity.jpg", caption="Radar Reflectivity")

# TAB 2: AI BRAIN
with tab2:
    st.header("2. Gemini Fusion Engine")
    st.caption("Model: **gemini-2.0-flash** (Multimodal)")
    
    if st.button("RUN DIAGNOSTICS"):
        if not api_key or "PASTE" in api_key:
            st.error("🔑 API Key Missing!")
        elif 'analysis_img' not in st.session_state:
            st.error("📡 No Data! Load Satellite Image in Tab 1.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze this satellite image of {target_name} ({lat}, {lon}).
            Telemetry: Humidity {w['humidity']}%.
            Task: DECISION: Is this suitable for 'Electro-Coalescence' seeding?
            Return JSON: {{Decision: GO/NO-GO, Confidence: %, Reasoning: text}}
            """
            
            with st.spinner("Fusion Engine Processing..."):
                try:
                    res = model.generate_content([prompt, st.session_state['analysis_img']])
                    st.markdown(res.text)
                    if "GO" in res.text.upper():
                        st.session_state['mission'] = "GO"
                        st.balloons()
                    else:
                        st.session_state['mission'] = "NO"
                except Exception as e:
                    st.error(f"AI Error: {e}")

# TAB 3: DRONE PHYSICS
with tab3:
    st.header("3. Digital Twin: Electro-Coalescence")
    
    if st.button("🚀 LAUNCH SWARM SIMULATION"):
        status_text = st.empty()
        status_text.info("Initializing Physics Engine...")
        
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        particles = np.random.rand(80, 2) * 60 + 20 
        drones = np.array([[20.0, 20.0], [80.0, 20.0], [50.0, 80.0]]) 
        
        def update(frame):
            ax.clear()
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.axis('off')
            
            if frame > 15:
                center = np.array([50.0, 50.0])
                particles[:] += (center - particles[:]) * 0.04 
                color = '#3b82f6' 
                size = 100
                status_text.success(f"⚡ IONIZATION ACTIVE: {frame*100} Volts")
            else:
                color = '#93c5fd'
                size = 30
                status_text.warning(f"Swarm Positioning over {target_name}...")

            ax.scatter(particles[:,0], particles[:,1], c=color, s=size, alpha=0.6, edgecolors='none')
            
            for i in range(3):
                if frame < 15:
                    drones[i] += (np.array([50.0, 50.0]) - drones[i]) * 0.05
                
                ax.scatter(drones[i,0], drones[i,1], c='#fbbf24', s=250, marker='^', edgecolors='white', zorder=10)
                if frame > 15:
                    circle = plt.Circle((drones[i,0], drones[i,1]), 15 + (frame%5)*2, color='yellow', fill=False, alpha=0.4, linewidth=2)
                    ax.add_patch(circle)

        ani = FuncAnimation(fig, update, frames=50, interval=80)
        components.html(ani.to_jshtml(), height=600)
