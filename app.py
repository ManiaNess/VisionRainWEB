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
import csv
import os

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
LOG_FILE = "mission_logs.csv"

st.set_page_config(page_title="VisionRain | Intelligent Planet", layout="wide", page_icon="⛈️")

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
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .pitch-card {
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

# --- 1. CIRA RAMMB URL GENERATOR ---
def get_rammb_url(lat, lon, product):
    """
    Constructs the CIRA Slider URL based on location and requested product.
    """
    # Select Satellite based on Longitude
    if -140 <= lon < -50:
        sat = "goes-16" # Americas East
    elif -50 <= lon < 20:
        sat = "meteosat-11" # Atlantic/Europe
    elif 20 <= lon < 90:
        sat = "meteosat-9" # Middle East / Indian Ocean (Jeddah is here)
    else:
        sat = "himawari-9" # Asia

    # Map readable product names to CIRA URL parameters
    product_map = {
        "Day Convection (Yellow Updrafts)": "cira_day_convection",
        "Day Microphysics (Orange Water)": "cira_day_microphysics",
        "GeoColor (True Visual)": "geocolor",
        "Severe Storm (Convection)": "severe_storm_rgb"
    }
    p_code = product_map.get(product, "geocolor")

    # Construct URL (Zoomed to Sector if possible, or Full Disk)
    # We use a simplified URL structure that Microlink can render
    # Note: Exact X/Y centering on RAMMB is complex, so we default to the satellite's main sector for stability
    return f"https://rammb-slider.cira.colostate.edu/?sat={sat}&sec=full_disk&x=11000&y=11000&z=2&angle=0&im=12&ts=1&st=0&et=0&speed=130&motion=loop&maps%5Bborders%5D=white&p%5B0%5D={p_code}&opacity%5B0%5D=1&pause=0&slider=-1&hide_controls=1&mouse_draw=0"

# --- 2. MICROLINK AGENT (Captures RAMMB) ---
def capture_rammb_visual(url):
    """Uses Microlink to screenshot the CIRA tool."""
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(url)}&screenshot=true&meta=false&waitFor=8000&viewport.width=1000&viewport.height=800"
    try:
        r = requests.get(api_url, timeout=25)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

# --- 3. TELEMETRY (OpenWeatherMap Only) ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 4. GEOCODING ---
def get_coordinates(city_name, api_key):
    if not api_key: return None, None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        data = requests.get(url).json()
        if data: return data[0]['lat'], data[0]['lon']
    except: pass
    return None, None

# --- 5. BIGQUERY LOGGING ---
def log_mission_data(location, humidity, decision, reasoning):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Location", "Humidity", "Decision", "Reasoning"])
        writer.writerow([timestamp, location, humidity, decision, reasoning])

def load_mission_logs():
    if os.path.isfile(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Location", "Humidity", "Decision", "Reasoning"])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("CIRA/RAMMB Scientific Core")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("### 📍 Target Selector")
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

    m = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB dark_matter")
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.session_state['target'] = "Map Pin"
        st.session_state['data_fetched'] = False
        st.rerun()

    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

    # Admin
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_mission_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Sector: {target} | Source: CIRA RAMMB (Meteosat-9)*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Satellite Intelligence", "🧠 Gemini Fusion Core"])

# TAB 1: STRATEGY
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-card">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions like <b>Saudi Arabia</b> face water scarcity. Current cloud seeding is <b>manual, expensive, and reactive</b>. 
    We need a solution that leverages real-time satellite microphysics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("**Solution:** VisionRain uses **RGB Composite Analysis** (Microphysics & Convection) to detect seedable Supercooled Liquid Water.")

# TAB 2: SENSOR DATA
with tab2:
    st.header("Real-Time Microphysics Analysis")
    
    col_ctrl, col_disp = st.columns([1, 2])
    
    with col_ctrl:
        st.info("Select RGB Product for Analysis:")
        product = st.radio("Satellite Product", [
            "Day Microphysics (Orange Water)", 
            "Day Convection (Yellow Updrafts)", 
            "GeoColor (True Visual)"
        ])
        
        st.markdown("---")
        # TELEMETRY
        w = get_weather_telemetry(lat, lon, weather_key)
        if w:
            st.metric("Surface Humidity", f"{w['main']['humidity']}%", "Target > 40%")
            st.metric("Temp", f"{w['main']['temp']}°C")
            st.metric("Pressure", f"{w['main']['pressure']} hPa")
            st.session_state['ai_humid'] = w['main']['humidity']
        else:
            st.warning("Telemetry Offline")

    with col_disp:
        if st.button("📡 ACQUIRE SATELLITE FEED", type="primary"):
            with st.spinner(f"Downlinking {product} from CIRA RAMMB..."):
                rammb_url = get_rammb_url(lat, lon, product)
                img = capture_rammb_visual(rammb_url)
                
                if img:
                    st.session_state['ai_img'] = img
                    st.session_state['ai_prod'] = product
                    st.image(img, caption=f"Live Feed: {product}", use_column_width=True)
                else:
                    st.error("Feed Error. Check Internet/Key.")
        
        elif st.session_state.get('ai_img'):
             st.image(st.session_state['ai_img'], caption=f"Cached: {st.session_state['ai_prod']}", use_column_width=True)

# TAB 3: GEMINI CORE
with tab3:
    st.header("Gemini Decision Support Engine")
    
    st.markdown("### 🔬 AI Training Data: The Master Table")
    st.table(pd.DataFrame({
        "Parameter": ["Cloud Top Temp", "Effective Radius", "Visual Target", "Action"],
        "Ideal Range": ["-5°C to -15°C", "< 14 microns", "Orange/Red (Microphysics)", "Priority 1: Launch"],
        "Avoid": ["> 0°C or < -40°C", "> 15 microns", "Cyan/Blue (Ice)", "Abort"]
    }))
    
    st.divider()
    
    if st.button("RUN PHYSICS DIAGNOSTICS", type="primary"):
        if not api_key: st.error("Missing Google API Key")
        elif not st.session_state.get('ai_img'): st.error("No Satellite Image. Go to Tab 2.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # --- THE VISIONRAIN PHYSICS PROMPT ---
                prompt = f"""
                ROLE: You are the VisionRain Decision Support Engine.
                Analyze this CIRA RAMMB Satellite Image ({st.session_state['ai_prod']}) and Telemetry.
                
                --- INPUT DATA ---
                - Surface Humidity: {st.session_state.get('ai_humid', 'N/A')}%
                - Product Type: {st.session_state['ai_prod']}
                
                --- VISUAL ANALYSIS PROTOCOL ---
                
                1. IF "Day Microphysics":
                   - Scan for **ORANGE/RED** colors (Supercooled Liquid Water). This is SEEDABLE.
                   - Scan for **CYAN/BLUE** colors (Ice Crystals). This is NOT seedable.
                
                2. IF "Day Convection":
                   - Scan for **YELLOW** pixels (Strong Updrafts). This is SEEDABLE (Priority 1).
                   - Look for "Cauliflower" texture (Active Convection).
                
                --- DECISION LOGIC ---
                - IF (Orange/Red OR Yellow) AND Humidity > 40% -> **PRIORITY 1: LAUNCH DRONES**
                - IF (Cyan/Blue) OR Humidity < 30% -> **ABORT**
                
                --- OUTPUT ---
                1. **Visual Observation:** Describe the colors and textures you see.
                2. **Microphysical State:** Liquid vs Ice? Updrafts present?
                3. **Decision:** **PRIORITY 1** / **PRIORITY 2** / **ABORT**.
                4. **Reasoning:** Explain using the Master Table logic.
                """
                
                with st.spinner("Gemini is analyzing Cloud Microphysics..."):
                    res = model.generate_content([prompt, st.session_state['ai_img']])
                    
                    # Log & Display
                    decision = "GO" if "PRIORITY" in res.text.upper() else "NO-GO"
                    log_mission_data(target, f"{w['main']['humidity']}%", decision, "AI Analysis")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if "PRIORITY" in res.text.upper():
                        st.balloons()
                        st.success("✅ MISSION AUTHORIZED")
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
