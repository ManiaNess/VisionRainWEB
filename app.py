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

st.set_page_config(page_title="VisionRain | Sentinel Core", layout="wide", page_icon="⛈️")

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
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.1);
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

# --- 1. VISUAL AGENTS (Microlink) ---

def get_sentinel_capture(lat, lon):
    """
    Captures the Sentinel Hub EO Browser for High-Res Optical Imagery.
    """
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    # EO Browser URL: Centered on Lat/Lon, Zoom 11, Sentinel-2 True Color
    # We use a simplified URL that forces the latest available data
    sentinel_url = f"https://apps.sentinel-hub.com/eo-browser/?zoom=11&lat={lat}&lng={lon}&themeId=DEFAULT-THEME&datasetId=S2L1C&layerId=1_TRUE_COLOR&time=2020-01-01%7C{today}"
    
    # Microlink API (Free Tier)
    # waitFor=8000 (8s) because Sentinel Hub is heavy
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(sentinel_url)}&screenshot=true&meta=false&waitFor=8000&viewport.width=800&viewport.height=600"
    
    try:
        r = requests.get(api_url, timeout=25)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

def get_windy_radar_capture(lat, lon):
    """
    Captures Windy.com specifically for RADAR (Precipitation).
    """
    target_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=800&height=600&zoom=6&level=surface&overlay=radar&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(target_url)}&screenshot=true&meta=false&waitFor=4000&viewport.width=800&viewport.height=600"
    
    try:
        r = requests.get(api_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

# --- 2. TELEMETRY (OpenWeatherMap) ---
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

# --- 4. BIGQUERY SIMULATION ---
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
    st.caption("Sentinel Core | v6.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Target Acquisition")
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

    m = folium.Map(location=[lat, lon], zoom_start=9, tiles="CartoDB dark_matter")
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.session_state['target'] = "Map Pin"
        st.session_state['data_fetched'] = False 
        st.rerun()

    st.info(f"Coords: {lat:.4f}, {lon:.4f}")
    
    # --- ADMIN PORTAL ---
    st.markdown("---")
    with st.expander("🔒 Admin Portal (BigQuery)"):
        password = st.text_input("Password", type="password")
        if password == "123456":
            st.success("Access Granted")
            logs = load_mission_logs()
            st.dataframe(logs)

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Mission Target: {target}*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# --- TAB 1: THE PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-card">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b>, California, and Australia face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    These issues are intensifying due to climate change and unstable precipitation patterns.</p>
    <p>Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>. Pilots often fly blind, missing critical seeding windows.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="pitch-card">
        <h3>💡 2. VisionRain Solution</h3>
        <p>An <b>AI-driven Decision Support Platform</b> that automates the entire seeding lifecycle:</p>
        <ul>
        <li><b>Predictive AI:</b> Identifies seedable clouds via Satellite Fusion.</li>
        <li><b>Optimization:</b> Precision timing for intervention.</li>
        <li><b>Cost Reduction:</b> Eliminates chemical flares via Electro-Coalescence logic.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="pitch-card">
        <h3>🚀 3. Implementation Plan</h3>
        <p><b>Phase 1 (Prototype):</b></p>
        <ul>
        <li><b>Data:</b> Sentinel-2 (Optical) + Windy (Radar) + OpenWeatherMap.</li>
        <li><b>AI:</b> Gemini Multimodal Fusion.</li>
        <li><b>Output:</b> Real-time GO/NO-GO Authorization.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# --- AUTOMATIC DATA COLLECTION ---
if 'data_fetched' not in st.session_state or st.session_state.get('last_coords') != (lat, lon):
    with st.spinner("🛰️ Deploying Agents to Sentinel & Windy... (Wait ~15s)"):
        
        # 1. SENTINEL HUB (Visuals)
        st.session_state['img_sentinel'] = get_sentinel_capture(lat, lon)
        
        # 2. WINDY RADAR (Dynamics)
        st.session_state['img_radar'] = get_windy_radar_capture(lat, lon)
        
        # 3. OPENWEATHER (Numbers)
        st.session_state['w_data'] = get_weather_telemetry(lat, lon, weather_key)
        
        st.session_state['last_coords'] = (lat, lon)
        st.session_state['data_fetched'] = True
        st.rerun()

# --- TAB 2: THE WALL OF SCREENS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    # Telemetry Row (OpenWeather)
    w = st.session_state.get('w_data')
    if w:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Humidity", f"{w['main']['humidity']}%", "Target > 45%")
        c2.metric("Temperature", f"{w['main']['temp']}°C")
        c3.metric("Pressure", f"{w['main']['pressure']} hPa")
        c4.metric("Wind", f"{w['wind']['speed']} m/s")
    
    st.divider()

    # Visuals Grid
    c_sat, c_rad = st.columns(2)
    
    with c_sat:
        st.subheader("A. Optical Satellite (Sentinel-2)")
        st.caption("Source: Sentinel Hub EO Browser (High-Res)")
        if st.session_state.get('img_sentinel'): 
            st.image(st.session_state['img_sentinel'], use_column_width=True)
        else: 
            st.warning("Waiting for Sentinel Agent...")

    with c_rad:
        st.subheader("B. Doppler Radar (Windy)")
        st.caption("Source: Windy.com (Precipitation Intensity)")
        if st.session_state.get('img_radar'): 
            st.image(st.session_state['img_radar'], use_column_width=True)
        else: 
            st.warning("Waiting for Windy Agent...")

# --- TAB 3: GEMINI INTELLIGENCE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    # 1. TRANSPARENCY TABLE
    st.markdown("### 🔍 Operational Criteria Check")
    if w:
        comp_df = pd.DataFrame({
            "Parameter": ["Humidity", "Wind Speed", "Visual Texture (Sentinel)", "Radar Activity (Windy)"],
            "Ideal Condition": ["> 45%", "< 15 m/s", "Convective Structure", "Developing Cells"],
            "Actual Data": [
                f"{w['main']['humidity']}%", 
                f"{w['wind']['speed']} m/s", 
                "Analyzing Image A...", 
                "Analyzing Image B..."
            ]
        })
        st.table(comp_df)

    # 2. EVIDENCE REPLAY
    st.caption("Visual Evidence Stream (Sent to Gemini):")
    img_list = [st.session_state.get('img_sentinel'), st.session_state.get('img_radar')]
    valid_imgs = [i for i in img_list if i is not None]
    
    if valid_imgs:
        st.image(valid_imgs, width=150, caption=["Sentinel-2", "Windy Radar"])
    
    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not valid_imgs:
            st.error("⚠️ Waiting for Agents to finish...")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # --- THE SUPER PROMPT ---
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST.
            Analyze this Multi-Modal Sensor Data for the Saudi Cloud Seeding Program.
            
            --- TARGET ---
            Location: {target} ({lat}, {lon})
            
            --- ACTUAL TELEMETRY (OpenWeatherMap) ---
            - Humidity: {w['main']['humidity']}%
            - Wind: {w['wind']['speed']} m/s
            
            --- VISUALS (Attached) ---
            Image 1: SENTINEL-2 SATELLITE (High-Res Optical). Look for cloud texture.
            Image 2: WINDY RADAR. Look for rain intensity (Colors).
            
            --- TASK ---
            1. **Visual Analysis:** - Does Sentinel show clouds? Are they puffy (Convective) or flat (Stratus)?
               - Does Windy Radar show active rain (Red/Yellow/Green)?
            
            2. **Data Correlation:** - Does the visual evidence match the Humidity reading?
            
            3. **Decision:** **GO** or **NO-GO**?
               - GO if: Humidity > 45% AND Clouds are visible/convective.
               - NO-GO if: Sky is clear OR Humidity < 30%.
            
            4. **Scientific Reasoning:** Explain why using the provided data.
            """
            
            with st.spinner("Vertex AI is fusing visual streams + telemetry..."):
                try:
                    res = model.generate_content([prompt] + valid_imgs)
                    
                    # LOGGING TO BIGQUERY
                    decision = "GO" if "GO" in res.text.upper() and "NO-GO" not in res.text.upper() else "NO-GO"
                    log_mission_data(target, f"{w['main']['humidity']}%", decision, "AI Decision")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED: Atmospheric Conditions Optimal</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED: Conditions Unsuitable")
                except Exception as e:
                    st.error(f"AI Error: {e}")
