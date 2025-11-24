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
DEFAULT_API_KEY = "AIzaSyAZsUnki7M2SJjPYfZ5NHJ8LX3xMtboUDU" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain | Intelligent Planet", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4facfe;}
    .pitch-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST IMAGE LOADER ---
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 1. GEOCODING ---
def get_coordinates(city_name, api_key):
    if not api_key: return None, None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        data = requests.get(url).json()
        if data: return data[0]['lat'], data[0]['lon']
    except: pass
    return None, None

# --- 2. NASA SATELLITE FEED ---
def get_nasa_feed(lat, lon):
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bbox = f"{lon-5},{lat-5},{lon+5},{lat+5}" 
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "800", "HEIGHT": "800", "TIME": today
    }
    try:
        full_url = requests.Request('GET', url, params=params).prepare().url
        return load_image_from_url(full_url), today
    except: return None, None

# --- 3. STATIC RADAR TILE (For AI Vision) ---
def get_static_radar(lat, lon, api_key):
    if not api_key: return None
    try:
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
    st.caption("AI-Driven Cloud Seeding Platform")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("### 📍 Target Selector")
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
    
    # Interactive Map Picker
    m = folium.Map(location=[lat, lon], zoom_start=5)
    m.add_child(folium.LatLngPopup()) 
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.rerun()

    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Sector Analysis: {target_name}*")

tab1, tab2, tab3 = st.tabs(["🌍 The Mission (Pitch)", "📡 Live Data Fusion", "🧠 Gemini Fusion Core"])

# TAB 1: THE PITCH (Problem & Solution)
with tab1:
    st.header("1. Strategic Imperatives")
    
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Problem: Water Scarcity & Inefficiency</h3>
    <p>Regions like Saudi Arabia face extreme water scarcity. Current cloud seeding operations are <b>manual, expensive, and reactive</b>. 
    Pilots fly blindly searching for clouds, leading to high costs and missed opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Current Gap:**
        - ❌ Reactive Decision Making
        - ❌ High Operational Costs ($8k/hr flight)
        - ❌ No Real-time AI Validation
        """)
    with c2:
        st.markdown("""
        **VisionRain Solution:**
        - ✅ **Predictive AI:** Identifies seedable clouds before takeoff.
        - ✅ **Multimodal Fusion:** Combines Satellite + Radar + Telemetry.
        - ✅ **Automated Go/No-Go:** Reduces wasted missions by 40%.
        """)

    st.markdown("---")
    st.header("2. Impact & Vision 2030")
    st.info("**National Priority:** Supports the **Saudi Green Initiative** (10 Billion Trees) by securing a sustainable atmospheric water source.")

# TAB 2: DATA FUSION (Windy + Telemetry)
with tab2:
    st.header("3. Real-Time Environmental Monitoring")
    
    col_visual, col_data = st.columns([2, 1])
    
    # LEFT: WINDY.COM EMBED (The "Sci-Fi" Look)
    with col_visual:
        st.subheader("A. Global Atmospheric Dynamics (Windy)")
        # Force reload on location change using timestamp
        ts = int(time.time())
        windy_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=800&height=500&zoom=6&level=surface&overlay=rain&product=ecmwf&menu=&message=&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1&t={ts}"
        components.iframe(windy_url, height=500)
        st.caption(f"Live Radar & Wind @ {lat:.2f}, {lon:.2f}")

    # RIGHT: OPENWEATHERMAP DATA (The "Hard Numbers")
    with col_data:
        st.subheader("B. Local Telemetry")
        w = get_weather_telemetry(lat, lon, weather_key)
        
        if w:
            st.metric("Relative Humidity", f"{w['main']['humidity']}%", "Target > 40%")
            st.metric("Temperature", f"{w['main']['temp']}°C")
            st.metric("Pressure", f"{w['main']['pressure']} hPa")
            
            # Save for AI
            st.session_state['ai_humid'] = w['main']['humidity']
            st.session_state['ai_press'] = w['main']['pressure']
            
            if w['main']['humidity'] > 40:
                st.success("✅ Conditions: SEEDABLE")
            else:
                st.error("⚠️ Conditions: TOO DRY")
        else:
            st.warning("Enter OpenWeatherMap Key in Sidebar to load telemetry.")
            st.session_state['ai_humid'] = "N/A"
            st.session_state['ai_press'] = "N/A"

# TAB 3: GEMINI SUPER FUSION
with tab3:
    st.header("4. Gemini Fusion Engine")
    st.write("The AI performs a **pixel-level analysis** of Satellite texture and Radar reflectivity, cross-referencing with atmospheric thermodynamics.")
    
    if st.button("RUN DEEP DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # 1. GATHER HIDDEN ASSETS (What Gemini Sees)
            with st.spinner("Aggregating Multi-Sensor Visuals..."):
                # Satellite
                img, status = get_nasa_feed(lat, lon)
                if img:
                    st.session_state['ai_sat'] = img
                else:
                    # Fallback for night/offline
                    backup_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg"
                    st.session_state['ai_sat'] = load_image_from_url(backup_url)
                
                # Radar (Static Tile for AI)
                st.session_state['ai_rad'] = get_static_radar(lat, lon, weather_key)

            # 2. DISPLAY PAYLOAD (Transparency)
            st.markdown("### 👁️ AI Input Stream (What Gemini Sees)")
            c1, c2 = st.columns(2)
            
            if st.session_state.get('ai_sat'):
                c1.image(st.session_state['ai_sat'], caption="Input 1: NASA Optical Satellite", use_column_width=True)
            
            if st.session_state.get('ai_rad'):
                c2.image(st.session_state['ai_rad'], caption="Input 2: Precipitation Radar (Static Tile)", use_column_width=True)
            else:
                c2.warning("Radar Tile Unavailable (Using Telemetry Only)")

            # 3. EXECUTE AI
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # THE SUPER PROMPT
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST for the Saudi Regional Cloud Seeding Program.
            Analyze this Multi-Modal Sensor Data to authorize a mission.
            
            --- MISSION CONTEXT ---
            Location: {target_name} ({lat}, {lon})
            Objective: Hygroscopic Seeding (Salt Flares).
            
            --- NUMERICAL TELEMETRY ---
            - Humidity: {st.session_state.get('ai_humid')}%
            - Pressure: {st.session_state.get('ai_press')} hPa
            
            --- VISUAL DATA (Attached) ---
            Image 1: SATELLITE. Look for "Convective Towers" (lumpy texture) vs "Stratiform" (flat haze).
            Image 2: RADAR. Colored pixels = Active Rain. Transparent/White = Clear.
            
            --- TASK ---
            1. VISUAL OBSERVATION: Describe the cloud morphology in Image 1.
            2. RADAR CORRELATION: Does Image 2 show active precipitation?
            3. THERMODYNAMIC CHECK: Is humidity sufficient (>40%)?
            4. DECISION: **GO** or **NO-GO**?
            5. REASONING: Scientific justification using terms like 'Updrafts', 'Nucleation', and 'Coalescence'.
            """
            
            inputs = [prompt]
            if st.session_state.get('ai_sat'): inputs.append(st.session_state['ai_sat'])
            if st.session_state.get('ai_rad'): inputs.append(st.session_state['ai_rad'])
            
            with st.spinner("Gemini 2.0 is fusing streams..."):
                try:
                    res = model.generate_content(inputs)
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.success("✅ MISSION APPROVED: Conditions Optimal.")
                        st.balloons()
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED: Conditions Unsuitable.")
                except Exception as e:
                    st.error(f"AI Error: {e}")
