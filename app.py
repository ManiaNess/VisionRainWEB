import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import urllib.parse
import time
import folium
from streamlit_folium import st_folium
import pandas as pd
import random

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
# No Screenshot Key needed for Free Microlink Tier

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
    </style>
    """, unsafe_allow_html=True)

# --- 1. MICROLINK AGENT (The Fast One) ---
def get_microlink_capture(target_url, delay=4000):
    """
    Uses Microlink.io to capture complex JS websites.
    """
    # Microlink API URL construction
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(target_url)}&screenshot=true&meta=false&waitFor={delay}&viewport.width=1000&viewport.height=800"
    
    try:
        r = requests.get(api_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

def get_windy_layer(lat, lon, layer):
    """Captures Windy.com Dynamics"""
    url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=1000&height=800&zoom=5&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    return get_microlink_capture(url, delay=4000)

def get_rammb_microphysics(lat, lon):
    """Captures CIRA RAMMB 'Day Microphysics' RGB"""
    # Simplified RAMMB URL for the Middle East Sector
    url = f"https://rammb-slider.cira.colostate.edu/?sat=meteosat-9&sec=full_disk&x={12000}&y={12000}&z=2&angle_rotate=0&im=12&ts=1&st=0&et=0&speed=130&motion=loop&map=1&lat={lat}&p%5B0%5D=cira_day_microphysics&opacity%5B0%5D=1&hidden%5B0%5D=0&pause=0&slider=-1&hide_controls=1&mouse_draw=0&s=rammb-slider"
    return get_microlink_capture(url, delay=6000) # RAMMB is slower, needs 6s

# --- 2. TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 3. GEOCODING ---
def get_coordinates(city_name, api_key):
    if not api_key: return None, None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        data = requests.get(url).json()
        if data: return data[0]['lat'], data[0]['lon']
    except: pass
    return None, None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Scientific Core v4.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    target_input = st.text_input("Search Region", "Jeddah")
    
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728
    if 'target' not in st.session_state: st.session_state['target'] = "Jeddah"

    if st.button("Locate Target"):
        if weather_key:
            lat, lon = get_coordinates(target_input, weather_key)
            if lat:
                st.session_state['lat'] = lat
                st.session_state['lon'] = lon
                st.session_state['target'] = target_input
                st.session_state['data_fetched'] = False
                st.rerun()

    lat, lon = st.session_state['lat'], st.session_state['lon']
    target = st.session_state['target']
    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Sector Analysis: {target}*")

tab1, tab2, tab3 = st.tabs(["🌍 Pitch & Strategy", "📡 Live Sensor Fusion", "🧠 VisionRain AI Core"])

# --- TAB 1: STRATEGY ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions like <b>Saudi Arabia</b> face water scarcity and drought. Current cloud seeding is <b>manual, expensive ($8k/hr), and reactive</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Proposed Solution:** VisionRain - An AI-driven decision support platform using satellite microphysics and autonomous drone deployment logic.")
    with c2:
        st.success("**Impact:** Aligned with Saudi Vision 2030 & Green Initiative. Reduces chemical usage via Electro-Coalescence.")

# --- AUTO-CAPTURE LOGIC ---
if 'data_fetched' not in st.session_state or st.session_state.get('last_coords') != (lat, lon):
    with st.spinner("🛰️ Calibrating Sensors (Capturing RAMMB & Windy via Microlink)..."):
        # 1. Scientific Layer (RAMMB)
        st.session_state['img_rammb'] = get_rammb_microphysics(lat, lon)
        
        # 2. Dynamic Layers (Windy)
        st.session_state['img_radar'] = get_windy_layer(lat, lon, "radar")
        st.session_state['img_wind'] = get_windy_layer(lat, lon, "wind")
        
        # 3. Telemetry
        st.session_state['w_data'] = get_weather_telemetry(lat, lon, weather_key)
        
        st.session_state['last_coords'] = (lat, lon)
        st.session_state['data_fetched'] = True
        st.rerun()

# --- TAB 2: SENSOR FUSION ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    w = st.session_state.get('w_data')
    
    if w:
        c1, c2, c3 = st.columns(3)
        c1.metric("Humidity", f"{w['main']['humidity']}%")
        c2.metric("Temp", f"{w['main']['temp']}°C")
        c3.metric("Wind", f"{w['wind']['speed']} m/s")

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**A. Microphysics RGB (RAMMB)**")
        st.caption("Scientific view for Cloud Phase (Ice vs Water)")
        if st.session_state.get('img_rammb'): 
            st.image(st.session_state['img_rammb'], use_column_width=True)
        else: st.info("Waiting for Microlink...")

    with col2:
        st.markdown("**B. Doppler Radar (Windy)**")
        st.caption("Precipitation Intensity & Storm Cells")
        if st.session_state.get('img_radar'): 
            st.image(st.session_state['img_radar'], use_column_width=True)
        else: st.info("Waiting for Microlink...")

# --- TAB 3: VISIONRAIN AI CORE ---
with tab3:
    st.header("VisionRain Decision Support Engine")
    
    # 1. PHYSICS TABLE (Transparency)
    st.markdown("### 🔬 Physics-Informed Logic (The Master Table)")
    st.table(pd.DataFrame({
        "Parameter": ["Cloud Top Temp", "Effective Radius", "Cloud Phase", "Visual Texture"],
        "Seedable Sweet Spot": ["-5°C to -15°C", "< 14 microns", "Liquid (Supercooled)", "Cauliflower (Convective)"],
        "Physics Reason": ["Supercooled water is reactive", "Droplets need charge to grow", "Ice is dead; Water is alive", "Strong updraft indicator"]
    }))

    # 2. EVIDENCE
    st.caption("Visual Evidence Stream (Sent to Vertex AI):")
    imgs = [st.session_state.get('img_rammb'), st.session_state.get('img_radar')]
    valid_imgs = [i for i in imgs if i is not None]
    
    if valid_imgs:
        st.image(valid_imgs, width=150, caption=["Microphysics RGB", "Radar"])
    
    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not valid_imgs:
            st.error("⚠️ No Data Captured.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # --- THE VISIONRAIN SYSTEM PROMPT ---
            prompt = f"""
            ROLE: You are the VisionRain Decision Support Engine. 
            Your job is to analyze satellite data and recommend drone deployment for cloud seeding.
            
            --- INPUT TELEMETRY ---
            - Surface Humidity: {w['main']['humidity'] if w else 'N/A'}%
            - Wind Speed: {w['wind']['speed'] if w else 'N/A'} m/s
            
            --- VISUAL ANALYSIS INSTRUCTIONS ---
            Look at Image 1 (RAMMB Microphysics RGB):
            - **Target A (Yellow):** Strong Updrafts?
            - **Target B (Orange/Red):** Supercooled Liquid Water (SLW)? (SEEDABLE)
            - **Avoid (Cyan/Blue):** Pure Ice/Cirrus? (ABORT)
            
            Look at Image 2 (Radar):
            - Red/Yellow blobs = Active Heavy Rain.
            
            --- LOGIC RULES ---
            1. IF Cloud is Orange/Red (SLW) AND Humidity > 40% -> "PRIORITY 1: LAUNCH DRONES"
               (Reason: Ideal Supercooled Liquid Water. Electric charge will trigger coalescence.)
            
            2. IF Cloud has Yellow Updrafts AND Visual Texture is "Cauliflower" -> "PRIORITY 2: WARM SEEDING"
               (Reason: Active convective core. Ionization needed.)
            
            3. IF Cloud is Cyan (Ice) OR Radar shows massive Red zones (Already raining hard) -> "ABORT"
               (Reason: Cloud is glaciated or self-precipitating. No intervention needed.)
            
            --- OUTPUT FORMAT ---
            1. **Microphysical Analysis:** Describe the colors you see in Image 1.
            2. **Radar Correlation:** Is it raining already?
            3. **Decision:** **PRIORITY 1** / **PRIORITY 2** / **ABORT**.
            4. **Scientific Reasoning:** Explain using the physics rules above.
            """
            
            with st.spinner("Vertex AI is calculating Cloud Microphysics..."):
                try:
                    res = model.generate_content([prompt] + valid_imgs)
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if "PRIORITY" in res.text.upper():
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION AUTHORIZED: Drones Deployed</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED: Conditions Unsuitable")
                except Exception as e:
                    st.error(f"AI Error: {e}")
