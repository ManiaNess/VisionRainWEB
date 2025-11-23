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
import random
import os

# --- CONFIGURATION ---
# API Keys (It's safer to use st.secrets for production, but this works for a hackathon demo)
DEFAULT_GOOGLE_KEY = "AIzaSyA7Yk4WRdSu976U4EpHZN47m-KA8JbJ5do"
DEFAULT_WEATHER_KEY = "11b260a4212d29eaccbd9754da459059"

st.set_page_config(page_title="VisionRain Data Core", layout="wide", page_icon="🛰️")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 10px;}
    h1, h2, h3 {color: #60a5fa;}
    .stSuccess {background-color: #064e3b; color: #6ee7b7;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. DATA INGESTION LAYER (The "Readers")
# ==========================================

def get_nasa_satellite(lat, lon):
    """Reads Cloud Structure from NASA GIBS (Free/Open)"""
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    # Bounding Box Calculation (2 degree zoom)
    bbox = f"{lon-2},{lat-2},{lon+2},{lat+2}"
    
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "600", "HEIGHT": "600", "TIME": today
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200: return Image.open(BytesIO(r.content)), "Live Feed (NASA GIBS)"
    except: pass
    return None, "Connection Failed"

def get_weather_data(lat, lon, api_key):
    """Reads Atmospheric Data (OpenWeatherMap or Simulation)"""
    if api_key:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            data = requests.get(url).json()
            
            if data.get('cod') != 200: # Check for API error
                 raise Exception(data.get('message'))

            main = data['main']
            wind = data['wind']
            clouds = data['clouds']
            
            # Hack: Calculate Cloud Base using Spread
            spread = (100 - main['humidity']) / 5 
            cloud_base_ft = spread * 1000
            
            return {
                "temp": main['temp'],
                "humidity": main['humidity'],
                "pressure": main['pressure'],
                "wind_speed": wind['speed'],
                "cloud_cover": clouds['all'],
                "cloud_base_est_ft": round(cloud_base_ft),
                "source": "OpenWeatherMap API (Live)"
            }
        except Exception as e: 
            pass # Fallback to simulation if API fails
    
    # FALLBACK SIMULATION
    return {
        "temp": 32.5,
        "humidity": random.randint(30, 70),
        "pressure": 1012,
        "wind_speed": 5.4,
        "cloud_cover": random.randint(20, 80),
        "cloud_base_est_ft": 4500,
        "source": "Telemetry Simulation (Offline Mode)"
    }

def get_radar_tile():
    """Reads Radar Data (RainViewer)"""
    try:
        ts = requests.get("https://api.rainviewer.com/public/weather-maps.json").json()[0]
        # Tile for Saudi Arabia (Zoom 6)
        return f"https://tile.rainviewer.com/{ts}/256/6/40/26/2/1_1.png"
    except:
        return "https://tile.rainviewer.com/1700000000/256/6/40/26/2/1_1.png" # Backup

# ==========================================
# 2. UI LAYOUT
# ==========================================

with st.sidebar:
    st.title("⛈️ VisionRain Data Core")
    st.markdown("---")
    google_key = st.text_input("Google API Key", value=DEFAULT_GOOGLE_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=DEFAULT_WEATHER_KEY, type="password")
    
    st.markdown("### 📍 Target Selection")
    loc_name = st.text_input("Region", "Riyadh, SA")
    lat = st.number_input("Latitude", 24.7136)
    lon = st.number_input("Longitude", 46.6753)
    
    st.success("Status: **DATA LINK ACTIVE**")

st.title(f"📡 Mission Target: {loc_name}")
st.caption("Real-Time Hydro-Meteorological Data Fusion")

col1, col2, col3 = st.columns([1, 1, 1.2])

# --- COLUMN 1: VISUAL (SATELLITE) ---
with col1:
    st.header("1. Visual Data")
    st.caption("Source: NASA VIIRS / Worldview")
    
    with st.spinner("Downlinking Satellite Feed..."):
        sat_img, status = get_nasa_satellite(lat, lon)
        
    if sat_img:
        st.image(sat_img, caption=f"{status} | {datetime.date.today()}", use_column_width=True)
    else:
        st.error("Satellite Offline (Night Orbit)")
        # Valid Backup Image
        try:
            backup_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/640px-Cumulonimbus_cloud_over_Singapore.jpg"
            sat_img = Image.open(requests.get(backup_url, stream=True).raw)
            st.image(sat_img, caption="Archive Backup", use_column_width=True)
        except:
            st.error("Backup image unavailable")

# --- COLUMN 2: NUMERICAL (WEATHER) ---
with col2:
    st.header("2. Atmospheric Data")
    st.caption("Source: Weather Stations / Telemetry")
    
    # Fetch Data
    w = get_weather_data(lat, lon, weather_key)
    
    st.info(f"Stream: {w['source']}")
    
    # Display Metrics
    c1, c2 = st.columns(2)
    c1.metric("Humidity", f"{w['humidity']}%", help="Target > 40%")
    c2.metric("Temperature", f"{w['temp']} °C")
    
    c3, c4 = st.columns(2)
    c3.metric("Cloud Cover", f"{w['cloud_cover']}%")
    c4.metric("Wind Speed", f"{w['wind_speed']} m/s")
    
    st.metric("Est. Cloud Base", f"{w['cloud_base_est_ft']} ft", delta="Seeding Altitude")

# --- COLUMN 3: INTELLIGENCE (GEMINI) ---
with col3:
    st.header("3. AI Analysis")
    st.caption("Engine: Google Gemini 1.5 Flash")
    
    st.write("Fusion of Visual + Numerical Data:")
    
    if st.button("RUN AI DIAGNOSTICS", type="primary"):
        if not google_key or "PASTE" in google_key:
            st.error("🔑 Please enter Google API Key!")
        elif sat_img is None:
             st.error("No satellite image available for analysis.")
        else:
            genai.configure(api_key=google_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # THE PROMPT STRATEGY
            prompt = f"""
            You are an AI Meteorologist for the Saudi Cloud Seeding Program.
            
            INPUT DATA:
            1. Visual: See attached satellite image.
            2. Telemetry:
               - Humidity: {w['humidity']}%
               - Cloud Cover: {w['cloud_cover']}%
               - Cloud Base: {w['cloud_base_est_ft']} ft
               - Temp: {w['temp']} C
            
            TASK:
            1. Analyze the cloud texture in the image (Convective vs Stratiform).
            2. Evaluate Telemetry (Is humidity > 40%? Is cloud base reachable?).
            3. DECISION: 'GO' or 'NO-GO' for Hygroscopic Seeding.
            4. REASONING: 1 sentence scientific explanation.
            
            Output Format: JSON.
            """
            
            with st.spinner("Gemini is fusing data streams..."):
                try:
                    response = model.generate_content([prompt, sat_img])
                    st.success("Analysis Complete")
                    st.write(response.text) # Using write() as json() expects a dict
                    
                    if "GO" in response.text:
                        st.balloons()
                except Exception as e:
                    st.error(f"AI Error: {e}")

# --- BOTTOM: RADAR & CONTEXT ---
st.markdown("---")
st.header("4. Regional Context (Radar)")
st.caption("Source: RainViewer API (Precipitation Intensity)")

radar_col, text_col = st.columns([2, 1])
with radar_col:
    st.image(get_radar_tile(), caption="Regional Radar Composite (Latest)", width=600)
with text_col:
    st.info("""
    **Data Pipeline:**
    1. **NASA GIBS:** Provides cloud texture verification.
    2. **OpenWeather:** Provides thermodynamic state.
    3. **RainViewer:** Verifies existing precipitation.
    4. **Gemini AI:** Synthesizes all 3 to make a decision.
    """)

# --- TAB 3: DRONE PHYSICS (MOVED TO BOTTOM FOR LAYOUT) ---
st.markdown("---")
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
            status_text.warning(f"Swarm Positioning over {loc_name}...")

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
