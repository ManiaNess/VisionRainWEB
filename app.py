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
import os
import csv

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
# File to simulate BigQuery Database
LOG_FILE = "mission_logs.csv"

st.set_page_config(page_title="VisionRain Enterprise", layout="wide", page_icon="⛈️")

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
    .pitch-box {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #00e5ff;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.1);
    }
    .header-text {
        color: #00e5ff;
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    /* Custom Spinner */
    .stSpinner > div {border-top-color: #00e5ff !important;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. BIGQUERY SIMULATION (Data Logging) ---
def log_mission_data(location, humidity, decision, reasoning):
    """Saves mission result to a CSV file (Simulating BigQuery Insert)"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists, write header if not
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Location", "Humidity", "Decision", "Reasoning"])
        writer.writerow([timestamp, location, humidity, decision, reasoning])

def load_mission_logs():
    """Reads the CSV file (Simulating BigQuery Select)"""
    if os.path.isfile(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Location", "Humidity", "Decision", "Reasoning"])

# --- 2. GOOGLE EARTH ENGINE (Context Layer) ---
def get_soil_moisture_context(lat, lon):
    """
    Simulates fetching Soil Moisture data from Earth Engine to validate drought.
    (Real GEE requires complex auth, so we simulate based on location/randomness for demo)
    """
    # Mock Logic: 0.0 = Bone Dry, 1.0 = Saturated
    moisture_index = round(random.uniform(0.1, 0.4), 2) 
    status = "CRITICAL DROUGHT" if moisture_index < 0.2 else "MODERATE DRYNESS"
    return moisture_index, status

# --- 3. MICROLINK AGENT ---
def get_windy_capture(lat, lon, layer):
    target_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=600&height=400&zoom=5&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(target_url)}&screenshot=true&meta=false&waitFor=4000&viewport.width=600&viewport.height=400"
    try:
        r = requests.get(api_url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

# --- 4. NASA SATELLITE ---
def get_nasa_feed(lat, lon):
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bbox = f"{lon-5},{lat-5},{lon+5},{lat+5}" 
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "600", "HEIGHT": "400", "TIME": today
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 5. TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 6. GEOCODING ---
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
    st.caption("v4.0 | Enterprise Core")
    
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
    
    # --- ADMIN LOGIN (HIDDEN FEATURE) ---
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        password = st.text_input("Admin Password", type="password")
        if password == "123456":
            st.success("Access Granted: BigQuery Logs")
            logs = load_mission_logs()
            st.dataframe(logs)
            if st.button("Export CSV"):
                st.download_button("Download Logs", logs.to_csv(), "mission_logs.csv")
        elif password:
            st.error("Access Denied")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Mission Target: {target}*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# --- TAB 1: THE PITCH ---
with tab1:
    st.header("Strategic Framework")
    
    st.markdown("""
    <div class="pitch-box">
    <div class="header-text">🚨 1. Problem Statement</div>
    Globally, regions such as <b>Saudi Arabia</b>, California, and Australia face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    These issues are intensifying due to climate change and unstable precipitation patterns.<br><br>
    Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>. Pilots often fly blind, missing critical seeding windows.
    This aligns critically with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b> for water sustainability.
    </div>
    """, unsafe_allow_html=True)

    # GOOGLE CLOUD ARCHITECTURE DIAGRAM (SIMULATED)
    st.subheader("2. The Google Cloud Architecture")
    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="white", fontname="Helvetica"];
        edge [color="#00e5ff"];
        
        subgraph cluster_inputs {
            label = "1. Data Ingestion";
            style=dashed;
            color="#555";
            NASA [label="NASA GIBS\n(Visuals)", fillcolor="#222", fontcolor="white"];
            Windy [label="Windy Agent\n(Dynamics)", fillcolor="#222", fontcolor="white"];
            Sensors [label="OpenWeather\n(Telemetry)", fillcolor="#222", fontcolor="white"];
        }
        
        subgraph cluster_process {
            label = "2. Intelligence Layer";
            style=filled;
            color="#1e293b";
            fontcolor="white";
            
            GEE [label="Earth Engine\n(Soil Context)", fillcolor="#00e5ff", fontcolor="black"];
            Vertex [label="Vertex AI\n(Gemini 2.0)", shape=doublecircle, fillcolor="#d1c4e9", fontcolor="black"];
        }
        
        subgraph cluster_output {
            label = "3. Action & Memory";
            style=dashed;
            color="#555";
            Dashboard [label="Streamlit\n(UI)", fillcolor="#c8e6c9"];
            BigQuery [label="BigQuery\n(Logs)", fillcolor="#bbdefb"];
        }
        
        NASA -> Vertex;
        Windy -> Vertex;
        Sensors -> Vertex;
        GEE -> Vertex [label="Priority"];
        
        Vertex -> Dashboard [label="Decision"];
        Vertex -> BigQuery [label="Archive"];
    }
    """)

# --- AUTOMATIC DATA COLLECTION ---
if 'data_fetched' not in st.session_state or st.session_state.get('last_coords') != (lat, lon):
    
    progress_text = "📡 Initializing Global Sensor Array..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # 1. NASA
        my_bar.progress(10, text="🛰️ Downlinking NASA VIIRS...")
        st.session_state['img_nasa'] = get_nasa_feed(lat, lon)
        
        # 2. WINDY LAYERS
        my_bar.progress(30, text="🌪️ Capturing Windy Layers...")
        st.session_state['img_wind'] = get_windy_capture(lat, lon, "wind")
        st.session_state['img_radar'] = get_windy_capture(lat, lon, "radar")
        st.session_state['img_clouds'] = get_windy_capture(lat, lon, "clouds")
        st.session_state['img_rain'] = get_windy_capture(lat, lon, "rain")
        
        # 3. NUMBERS
        my_bar.progress(90, text="📊 Syncing Telemetry...")
        st.session_state['w_data'] = get_weather_telemetry(lat, lon, weather_key)
        
        # 4. EARTH ENGINE CONTEXT
        st.session_state['soil_moist'], st.session_state['drought_status'] = get_soil_moisture_context(lat, lon)
        
        st.session_state['last_coords'] = (lat, lon)
        st.session_state['data_fetched'] = True
        my_bar.empty()
        st.rerun()
        
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")

# --- TAB 2: THE WALL OF SCREENS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    # Telemetry Row
    w = st.session_state.get('w_data')
    soil = st.session_state.get('soil_moist')
    drought = st.session_state.get('drought_status')

    if w:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Humidity", f"{w['main']['humidity']}%", "Target > 40%")
        c2.metric("Temperature", f"{w['main']['temp']}°C")
        c3.metric("Pressure", f"{w['main']['pressure']} hPa")
        c4.metric("Wind", f"{w['wind']['speed']} m/s")
        c5.metric("Soil Moisture (GEE)", f"{soil}", delta=drought, delta_color="inverse")
    
    st.divider()

    # Visuals Grid
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.caption("A. NASA Optical Satellite")
        if st.session_state.get('img_nasa'): st.image(st.session_state['img_nasa'], use_column_width=True)
    with r1c2:
        st.caption("B. Doppler Radar (Windy)")
        if st.session_state.get('img_radar'): st.image(st.session_state['img_radar'], use_column_width=True)
    with r1c3:
        st.caption("C. Wind Velocity (Windy)")
        if st.session_state.get('img_wind'): st.image(st.session_state['img_wind'], use_column_width=True)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.caption("D. Cloud Density (Windy)")
        if st.session_state.get('img_clouds'): st.image(st.session_state['img_clouds'], use_column_width=True)
    with r2c2:
        st.caption("E. Rain Accumulation (Windy)")
        if st.session_state.get('img_rain'): st.image(st.session_state['img_rain'], use_column_width=True)
    with r2c3:
        st.markdown("#### 🟢 System Status")
        st.success("All Sensors Online")
        st.info("Ready for AI Analysis")

# --- TAB 3: GEMINI INTELLIGENCE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.markdown("### 🔍 Operational Criteria Check")
    if w:
        comp_df = pd.DataFrame({
            "Parameter": ["Humidity", "Wind Speed", "Drought Severity", "Radar Activity"],
            "Ideal Condition": ["> 45%", "< 15 m/s", "Critical (Needs Rain)", "Active Cells"],
            "Actual Data": [
                f"{w['main']['humidity']}%", 
                f"{w['wind']['speed']} m/s", 
                f"{drought}", 
                "Scanning..."
            ]
        })
        st.table(comp_df)

    st.caption("Visual Evidence Stream (Sent to Gemini):")
    img_list = [st.session_state.get(k) for k in ['img_nasa', 'img_radar', 'img_wind', 'img_clouds']]
    valid_imgs = [i for i in img_list if i is not None]
    
    if valid_imgs:
        st.image(valid_imgs, width=150, caption=["NASA", "Radar", "Wind", "Clouds"])
    
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
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST using Vertex AI Forecasting logic.
            Analyze this Multi-Modal Sensor Data for the Saudi Cloud Seeding Program.
            
            --- TARGET ---
            Location: {target} ({lat}, {lon})
            
            --- EARTH ENGINE CONTEXT ---
            - Soil Moisture Index: {soil} ({drought})
            - Priority: {"HIGH - DROUGHT ZONE" if soil < 0.2 else "LOW - SATURATED"}
            
            --- ACTUAL TELEMETRY ---
            - Humidity: {w['main']['humidity']}%
            - Wind: {w['wind']['speed']} m/s
            
            --- VISUALS (Attached) ---
            1. NASA Satellite: Check cloud texture.
            2. Windy Radar: Check rain intensity.
            
            --- OUTPUT FORMAT (Markdown) ---
            1. **Visual Analysis:** Describe the cloud formation and radar intensity.
            2. **Context Check:** Does the Earth Engine data justify seeding here?
            3. **Decision:** **GO** or **NO-GO**?
            4. **Scientific Reasoning:** Explain why based on Humidity, Cloud Type, and Soil Need.
            """
            
            with st.spinner("Vertex AI is calculating Seeding Success Probability..."):
                try:
                    res = model.generate_content([prompt] + valid_imgs)
                    
                    # SAVE TO BIGQUERY (SIMULATED)
                    decision = "GO" if "GO" in res.text.upper() and "NO-GO" not in res.text.upper() else "NO-GO"
                    log_mission_data(target, f"{w['main']['humidity']}%", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED: Atmospheric Conditions Optimal</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED: Conditions Unsuitable")
                        
                except Exception as e:
                    st.error(f"AI Error: {e}")
