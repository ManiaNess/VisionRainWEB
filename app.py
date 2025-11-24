import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import math
import time

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain | Intelligent Planet", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4facfe;}
    .data-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST IMAGE LOADER ---
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 1. GEOCODING (Smart City Search) ---
def get_coordinates(city_name, api_key):
    if not api_key: return None, None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        data = requests.get(url).json()
        if data: return data[0]['lat'], data[0]['lon']
    except: pass
    return None, None

# --- 2. NASA SATELLITE (Zoomed In) ---
def get_nasa_feed(lat, lon):
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    # TIGHT ZOOM (2 degrees = ~200km view)
    bbox = f"{lon-2},{lat-2},{lon+2},{lat+2}" 
    
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "800", "HEIGHT": "800", "TIME": today
    }
    try:
        full_url = requests.Request('GET', url, params=params).prepare().url
        return load_image_from_url(full_url), full_url
    except: return None, None

# --- 3. STATIC RADAR TILE (For AI) ---
def get_radar_tile_image(lat, lon, api_key):
    if not api_key: return None
    try:
        # Calculate Tile X/Y for Zoom Level 6
        zoom = 6
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        url = f"https://tile.openweathermap.org/map/precipitation_new/{zoom}/{xtile}/{ytile}.png?appid={api_key}"
        return load_image_from_url(url)
    except: return None

# --- 4. TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Intelligent Planet Initiative")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Target Selector")
    
    # SMART SEARCH
    target_name = st.text_input("Region Name", "Jeddah")
    
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728

    if st.button("Find Location"):
        if weather_key:
            new_lat, new_lon = get_coordinates(target_name, weather_key)
            if new_lat:
                st.session_state['lat'] = new_lat
                st.session_state['lon'] = new_lon
                st.success(f"Locked: {target_name}")
                st.rerun()
            else:
                st.error("City not found.")
        else:
            st.warning("Need Weather Key to search!")

    lat, lon = st.session_state['lat'], st.session_state['lon']
    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Sector Analysis: {target_name}*")

tab1, tab2, tab3 = st.tabs(["🌍 Mission & Impact", "📡 Data Fusion", "🧠 Gemini Fusion Core"])

# TAB 1: PITCH
with tab1:
    st.header("1. Strategic Imperatives")
    st.markdown("""
    <div class="pitch-box" style="background-color:#1e293b;padding:20px;border-radius:10px;border-left:5px solid #4facfe;">
    <h3>🚨 The Problem: Water Scarcity</h3>
    <p>Regions like Saudi Arabia face extreme water scarcity. Current cloud seeding is <b>manual and reactive</b>. 
    We need a data-driven, predictive solution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current Gap:**\n- ❌ Reactive Decision Making\n- ❌ High Operational Costs\n- ❌ No Real-time AI Validation")
    with c2:
        st.markdown("**VisionRain Solution:**\n- ✅ **Predictive AI:** Identifies seedable clouds.\n- ✅ **Multimodal Fusion:** Satellite + Radar + Telemetry.\n- ✅ **Automated Go/No-Go.**")

# TAB 2: DATA FUSION
with tab2:
    st.header("2. Real-Time Environmental Monitoring")
    
    # --- SECTION A: VISUALS (TOP) ---
    col_sat, col_radar = st.columns(2)
    
    # LEFT: NASA SATELLITE (ZOOMED IN)
    with col_sat:
        st.subheader("A. Optical Satellite (NASA VIIRS)")
        with st.spinner("Downlinking..."):
            img, url = get_nasa_feed(lat, lon)
            if img:
                st.image(img, caption=f"Real-Time Feed: {target_name}", use_column_width=True)
                st.session_state['ai_sat'] = img
            else:
                st.warning("Orbit Offline. Using Backup.")
                backup_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg"
                st.session_state['ai_sat'] = load_image_from_url(backup_url)
                st.image(st.session_state['ai_sat'], caption="Archive Backup")

    # RIGHT: LIVE RADAR MAP (INTERACTIVE)
    with col_radar:
        st.subheader("B. Precipitation Radar")
        if weather_key:
            # Leaflet Map with Precipitation Layer
            html_map = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
                <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
                <style>#map {{ height: 300px; width: 100%; border-radius: 10px; }}</style>
            </head>
            <body>
                <div id="map"></div>
                <script>
                    var map = L.map('map').setView([{lat}, {lon}], 6);
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© OpenStreetMap'
                    }}).addTo(map);
                    L.tileLayer('https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={weather_key}', {{
                        opacity: 0.8
                    }}).addTo(map);
                </script>
            </body>
            </html>
            """
            components.html(html_map, height=300)
            
            # FETCH STATIC TILE FOR AI (Hidden from user, used for Gemini)
            st.session_state['ai_rad'] = get_radar_tile_image(lat, lon, weather_key)
        else:
            st.error("⚠️ Enter OpenWeatherMap Key to load Radar.")

    # --- SECTION B: NUMERICAL DATA (BOTTOM) ---
    st.markdown("---")
    st.subheader("C. Atmospheric Telemetry (OpenWeatherMap)")
    
    w = get_weather_telemetry(lat, lon, weather_key)
    
    if w:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Relative Humidity", f"{w['main']['humidity']}%", "Target > 40%")
        c2.metric("Temperature", f"{w['main']['temp']}°C")
        c3.metric("Pressure", f"{w['main']['pressure']} hPa")
        c4.metric("Wind Speed", f"{w['wind']['speed']} m/s")
        
        st.session_state['ai_humid'] = w['main']['humidity']
        st.session_state['ai_press'] = w['main']['pressure']
        st.session_state['ai_wind'] = w['wind']['speed']
    else:
        st.warning("Waiting for API Key...")

# TAB 3: GEMINI FUSION
with tab3:
    st.header("3. Gemini Fusion Engine")
    
    if st.button("RUN DEEP DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # Ensure Asset (Fail-Safe)
            if 'ai_sat' not in st.session_state or st.session_state['ai_sat'] is None:
                st.session_state['ai_sat'] = load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg")

            # SHOW USER WHAT AI SEES (Transparency)
            st.markdown("### 👁️ AI Input Stream")
            c1, c2 = st.columns(2)
            c1.image(st.session_state['ai_sat'], caption="1. Visual Satellite (Cloud Texture)", use_column_width=True)
            
            if st.session_state.get('ai_rad'):
                c2.image(st.session_state['ai_rad'], caption="2. Radar Reflectivity (Rain Intensity)", use_column_width=True)
            else:
                c2.warning("Radar Tile Unavailable (Analysis will proceed with Telemetry)")

            # EXECUTE AI
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # THE SUPER PROMPT
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST. Analyze this Multi-Modal Sensor Data.
            
            --- MISSION CONTEXT ---
            Location: {target_name} ({lat}, {lon})
            Objective: Hygroscopic Cloud Seeding.
            
            --- TELEMETRY (Numerical) ---
            - Humidity: {st.session_state.get('ai_humid', 'N/A')}%
            - Pressure: {st.session_state.get('ai_press', 'N/A')} hPa
            - Wind Speed: {st.session_state.get('ai_wind', 'N/A')} m/s
            
            --- VISUALS (Attached) ---
            Image 1: VISUAL SATELLITE. Look for "Convective Towers" (lumpy/cauliflower texture).
            Image 2: PRECIPITATION RADAR. Colored pixels = Rain. Transparent = Dry.
            
            --- ANALYSIS TASK ---
            1. VISUAL OBSERVATION: 
               - Describe the cloud texture in Image 1. Is it deep convective or thin cirrus?
               - Describe the Radar in Image 2. Do you see any storm cells (Red/Yellow)?
            
            2. DATA CORRELATION: 
               - Does the Humidity ({st.session_state.get('ai_humid')}%) support the visual evidence?
            
            3. DECISION: **GO** or **NO-GO**?
            
            4. REASONING: Provide a scientific justification citing Updrafts and Coalescence.
            """
            
            inputs = [prompt, st.session_state['ai_sat']]
            if st.session_state.get('ai_rad'): inputs.append(st.session_state['ai_rad'])
            
            with st.spinner("Gemini 2.0 is fusing streams..."):
                try:
                    res = model.generate_content(inputs)
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.success("✅ MISSION APPROVED")
                        st.balloons()
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
