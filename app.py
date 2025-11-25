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
# LEAVE BLANK FOR GITHUB (Paste in Sidebar)
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
LOG_FILE = "mission_logs.csv"

st.set_page_config(page_title="VisionRain | Intelligent Planet", layout="wide", page_icon="⛈️")

# --- ULTRA-MODERN STYLING ---
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
    .pitch-box {
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

# --- 1. MICROLINK AGENT (Captures Windy) ---
def get_windy_capture(lat, lon, layer):
    """
    Uses Microlink to take a screenshot of the Windy Embed.
    """
    # Windy Embed URL
    target_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=600&height=400&zoom=5&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    
    # Microlink API Call (Free Tier)
    # waitFor=4s ensures the particles load
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(target_url)}&screenshot=true&meta=false&waitFor=4000&viewport.width=600&viewport.height=400"
    
    try:
        r = requests.get(api_url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

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

# --- 4. BIGQUERY SIMULATION (ADMIN LOGS) ---
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
    st.caption("v5.0 | Autonomous AI Core")
    
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
                st.session_state['data_fetched'] = False # Force refresh
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
        st.session_state['data_fetched'] = False # Force refresh
        st.rerun()

    st.info(f"Coords: {lat:.4f}, {lon:.4f}")
    
    # --- ADMIN PORTAL (BIGQUERY) ---
    st.markdown("---")
    with st.expander("🔒 Admin Portal (BigQuery)"):
        password = st.text_input("Password", type="password")
        if password == "123456":
            st.success("Access Granted")
            logs = load_mission_logs()
            st.dataframe(logs)
            if not logs.empty:
                st.download_button("Export Logs", logs.to_csv(index=False), "mission_logs.csv")
        elif password:
            st.error("Invalid Password")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Mission Target: {target}*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision (Pitch)", "📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# --- TAB 1: THE PITCH ---
with tab1:
    st.header("Strategic Framework")
    
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b>, California, and Australia face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    These issues are intensifying due to climate change and unstable precipitation patterns.</p>
    <p>Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>. Pilots often fly blind, missing critical seeding windows.</p>
    <p>This aligns critically with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b> for water sustainability.</p>
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
        <li><b>Cost Reduction:</b> Eliminates chemical flares via Electro-Coalescence logic.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="pitch-box">
        <h3>🚀 3. Implementation Plan</h3>
        <p><b>Phase 1 (Prototype):</b></p>
        <ul>
        <li><b>Data:</b> Windy.com (Multi-Layer) + OpenWeatherMap + Radar.</li>
        <li><b>AI:</b> Gemini Multimodal Fusion.</li>
        <li><b>Output:</b> Real-time GO/NO-GO Authorization.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="pitch-box">
    <h3>📈 4. Impact & Priority</h3>
    <b>Why Now?</b> To support the 10 Billion Trees initiative, we need a sustainable water source.
    <br><b>Economic:</b> Reduces reliance on expensive desalination.
    <br><b>Innovation:</b> Enables future autonomous drone swarms.
    </div>
    """, unsafe_allow_html=True)

# --- AUTOMATIC DATA COLLECTION (Runs once per location change) ---
if 'data_fetched' not in st.session_state or st.session_state.get('last_coords') != (lat, lon):
    
    progress_text = "📡 Initializing Global Sensor Array..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # 1. CAPTURE WINDY LAYERS
        my_bar.progress(20, text="📡 Capturing Windy Radar...")
        st.session_state['img_radar'] = get_windy_capture(lat, lon, "radar")
        
        my_bar.progress(40, text="🌪️ Capturing Windy Wind Velocity...")
        st.session_state['img_wind'] = get_windy_capture(lat, lon, "wind")
        
        my_bar.progress(60, text="☁️ Capturing Windy Clouds...")
        st.session_state['img_clouds'] = get_windy_capture(lat, lon, "clouds")
        st.session_state['img_lclouds'] = get_windy_capture(lat, lon, "lclouds") # Low
        st.session_state['img_hclouds'] = get_windy_capture(lat, lon, "hclouds") # High
        
        my_bar.progress(80, text="💧 Capturing Windy Rain Accumulation...")
        st.session_state['img_rain'] = get_windy_capture(lat, lon, "rain")
        
        # 2. NUMBERS
        my_bar.progress(90, text="📊 Syncing Telemetry...")
        st.session_state['w_data'] = get_weather_telemetry(lat, lon, weather_key)
        
        st.session_state['last_coords'] = (lat, lon)
        st.session_state['data_fetched'] = True
        my_bar.empty()
        st.rerun() # Refresh to show images
        
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")

# --- TAB 2: THE WALL OF SCREENS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    w = st.session_state.get('w_data')
    if w:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Humidity", f"{w['main']['humidity']}%", "Target > 40%")
        c2.metric("Temperature", f"{w['main']['temp']}°C")
        c3.metric("Pressure", f"{w['main']['pressure']} hPa")
        c4.metric("Wind", f"{w['wind']['speed']} m/s")
    
    st.divider()

    # Visuals Grid (3 Rows)
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.caption("A. Doppler Radar (Windy)")
        if st.session_state.get('img_radar'): st.image(st.session_state['img_radar'], use_column_width=True)
        else: st.info("Loading Agent...")
    with r1c2:
        st.caption("B. Wind Velocity (Windy)")
        if st.session_state.get('img_wind'): st.image(st.session_state['img_wind'], use_column_width=True)
        else: st.info("Loading Agent...")
    with r1c3:
        st.caption("C. Rain Accumulation (Windy)")
        if st.session_state.get('img_rain'): st.image(st.session_state['img_rain'], use_column_width=True)
        else: st.info("Loading Agent...")

    st.subheader("Cloud Stratification Analysis")
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.caption("D. Total Cloud Cover")
        if st.session_state.get('img_clouds'): st.image(st.session_state['img_clouds'], use_column_width=True)
        else: st.info("Loading Agent...")
    with r2c2:
        st.caption("E. Low Clouds (Seedable Target)")
        if st.session_state.get('img_lclouds'): st.image(st.session_state['img_lclouds'], use_column_width=True)
        else: st.info("Loading Agent...")
    with r2c3:
        st.caption("F. High Clouds (Ice/Unseedable)")
        if st.session_state.get('img_hclouds'): st.image(st.session_state['img_hclouds'], use_column_width=True)
        else: st.info("Loading Agent...")

# --- TAB 3: GEMINI INTELLIGENCE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.markdown("### 🔍 Operational Criteria Check")
    if w:
        comp_df = pd.DataFrame({
            "Parameter": ["Humidity", "Wind Speed", "Low Clouds (Seedable)", "High Clouds (Ice)"],
            "Ideal Condition": ["> 45%", "< 15 m/s", "High Density", "Low Density"],
            "Actual Data": [
                f"{w['main']['humidity']}%", 
                f"{w['wind']['speed']} m/s", 
                "Analyzing Visuals...", 
                "Analyzing Visuals..."
            ]
        })
        st.table(comp_df)

    st.caption("Visual Evidence Stream (Sent to Gemini):")
    img_list = [
        st.session_state.get('img_radar'), 
        st.session_state.get('img_lclouds'),
        st.session_state.get('img_hclouds')
    ]
    valid_imgs = [i for i in img_list if i is not None]
    
    if valid_imgs:
        st.image(valid_imgs, width=150, caption=["Radar", "Low Clouds", "High Clouds"])
    
    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not valid_imgs:
            st.error("⚠️ Waiting for Agents...")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # --- THE VISIONRAIN SUPER PROMPT ---
            prompt = f"""
            ROLE: You are the VisionRain Decision Support Engine. 
            Your job is to analyze satellite data and recommend drone deployment for cloud seeding.
            
            --- INPUT TELEMETRY ---
            - Surface Humidity: {w['main']['humidity'] if w else 'N/A'}%
            - Wind Speed: {w['wind']['speed'] if w else 'N/A'} m/s
            
            --- VISUAL ANALYSIS INSTRUCTIONS ---
            Look at Image 1 (Radar):
            - Red/Yellow blobs = Active Heavy Rain.
            
            Look at Image 2 (Low Clouds):
            - High density of low clouds = Good seedable targets (Water).
            
            Look at Image 3 (High Clouds):
            - High density of high clouds = Bad targets (Ice).
            
            --- LOGIC RULES ---
            1. IF Low Clouds exist AND Humidity > 40% -> "PRIORITY 1: LAUNCH DRONES"
               (Reason: Ideal Supercooled Liquid Water. Electric charge will trigger coalescence.)
            
            2. IF Radar shows massive Red zones (Already raining hard) OR High Clouds dominate -> "ABORT"
               (Reason: Cloud is glaciated or self-precipitating. No intervention needed.)
            
            --- OUTPUT FORMAT ---
            1. **Microphysical Analysis:** Describe the cloud layers you see.
            2. **Radar Correlation:** Is it raining already?
            3. **Decision:** **PRIORITY 1** / **PRIORITY 2** / **ABORT**.
            4. **Scientific Reasoning:** Explain using the physics rules above.
            """
            
            with st.spinner("Vertex AI is calculating Cloud Microphysics..."):
                try:
                    res = model.generate_content([prompt] + valid_imgs)
                    
                    # LOGGING TO BIGQUERY
                    decision = "GO" if "PRIORITY" in res.text.upper() else "NO-GO"
                    log_mission_data(target, f"{w['main']['humidity']}%", decision, "AI Decision")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if "PRIORITY" in res.text.upper():
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION AUTHORIZED: Drones Deployed</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED: Conditions Unsuitable")
                except Exception as e:
                    st.error(f"AI Error: {e}")
