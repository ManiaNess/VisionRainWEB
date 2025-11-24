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
    .pitch-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST IMAGE LOADER (With Fail-Safe) ---
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

def create_placeholder_image():
    """Generates a grey square if internet fails completely"""
    return Image.new('RGB', (200, 200), color='#2d2d2d')

# --- 1. GEOCODING ---
def get_coordinates(city_name, api_key):
    if not api_key: return None, None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        data = requests.get(url).json()
        if data: return data[0]['lat'], data[0]['lon']
    except: pass
    return None, None

# --- 2. NASA SCIENTIFIC LAYERS ---
def get_nasa_layer(layer_type, lat, lon):
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bbox = f"{lon-10},{lat-10},{lon+10},{lat+10}" 
    
    # Select Satellite based on region
    if -140 < lon < -30: sat = "GOES-East_ABI_Band02_Red_Visible_1km"
    elif -30 <= lon < 60: sat = "Meteosat_MSG_SEVIRI_Band03_Visible"
    else: sat = "Himawari_AHI_Band3_Red_Visible_1km"

    # Map user selection to NASA Layer IDs
    layer_map = {
        "Visual": sat,
        "Precipitation": "GPM_3IMERGHH_06_Precipitation", 
        "Thermal": "MODIS_Terra_Land_Surface_Temperature_Day",
        "Night": "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance",
    }
    
    selected = layer_map.get(layer_type, sat)
    
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": selected,
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "800", "HEIGHT": "800", "TIME": today
    }
    
    try:
        full_url = requests.Request('GET', url, params=params).prepare().url
        return load_image_from_url(full_url), full_url
    except: return None, None

# --- 3. TELEMETRY ---
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
    
    # Interactive Map
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

# TAB 1: THE PITCH
with tab1:
    st.header("1. Strategic Imperatives")
    st.markdown("""
    <div class="pitch-box">
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

    st.markdown("---")
    st.info("**National Priority:** Supports the **Saudi Green Initiative** (10 Billion Trees).")

# TAB 2: DATA FUSION
with tab2:
    st.header("2. Real-Time Environmental Monitoring")
    
    col_visual, col_data = st.columns([2, 1])
    
    # LEFT: NASA LAYERS
    with col_visual:
        st.subheader("A. Earth Science Data (NASA GIBS)")
        layer_opt = st.selectbox("Select Instrument:", ["Visual (Cloud Texture)", "Precipitation (Rain Radar)", "Thermal (Heat)", "Night Vision"])
        
        # Map selection to function arg
        layer_map = {"Visual": "Visual", "Precipitation": "Precipitation", "Thermal": "Thermal", "Night": "Night"}
        
        with st.spinner("Downlinking..."):
            img, url = get_nasa_layer(layer_map[layer_opt.split()[0]], lat, lon)
            
            if img:
                st.image(img, caption=f"Source: NASA GIBS | {layer_opt}", use_column_width=True)
                # Save MAIN image for AI
                if "Visual" in layer_opt: st.session_state['ai_main'] = img
                if "Precipitation" in layer_opt: st.session_state['ai_radar'] = img
            else:
                st.warning("Layer unavailable. Using Backup.")
                # CRASH PROOF BACKUP
                backup_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg"
                backup_img = load_image_from_url(backup_url)
                
                if backup_img:
                    st.session_state['ai_main'] = backup_img
                    st.image(backup_img, caption="Archive Backup", use_column_width=True)
                else:
                    # ULTIMATE FAILSAFE (Grey Square)
                    st.session_state['ai_main'] = create_placeholder_image()
                    st.image(st.session_state['ai_main'], caption="System Offline (Simulated)", use_column_width=True)

    # RIGHT: OPENWEATHERMAP
    with col_data:
        st.subheader("B. Local Telemetry")
        w = get_weather_telemetry(lat, lon, weather_key)
        
        if w:
            st.metric("Relative Humidity", f"{w['main']['humidity']}%", "Target > 40%")
            st.metric("Temperature", f"{w['main']['temp']}°C")
            st.metric("Pressure", f"{w['main']['pressure']} hPa")
            
            st.session_state['ai_humid'] = w['main']['humidity']
            st.session_state['ai_press'] = w['main']['pressure']
            
            if w['main']['humidity'] > 40:
                st.success("✅ Conditions: SEEDABLE")
            else:
                st.error("⚠️ Conditions: TOO DRY")
        else:
            st.warning("Enter OpenWeatherMap Key in Sidebar.")
            st.session_state['ai_humid'] = "N/A"
            st.session_state['ai_press'] = "N/A"

# TAB 3: GEMINI FUSION
with tab3:
    st.header("3. Gemini Fusion Engine")
    
    if st.button("RUN DEEP DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # ENSURE ASSETS EXIST
            if 'ai_main' not in st.session_state or st.session_state['ai_main'] is None:
                st.warning("Initializing Image Stream...")
                # Force fetch a fallback
                st.session_state['ai_main'] = create_placeholder_image()

            # DISPLAY WHAT AI SEES
            st.markdown("### 👁️ AI Input Stream")
            c1, c2 = st.columns(2)
            
            # Only show if valid
            if st.session_state.get('ai_main'): 
                c1.image(st.session_state['ai_main'], caption="1. Visual Satellite", use_column_width=True)
            
            if st.session_state.get('ai_radar'): 
                c2.image(st.session_state['ai_radar'], caption="2. Precipitation Radar", use_column_width=True)
            else:
                c2.info("No Radar Data Loaded")

            # EXECUTE AI
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST. Analyze this Multi-Modal Sensor Data.
            
            --- MISSION CONTEXT ---
            Location: {target_name} ({lat}, {lon})
            Objective: Hygroscopic Seeding.
            
            --- TELEMETRY ---
            - Humidity: {st.session_state.get('ai_humid')}%
            - Pressure: {st.session_state.get('ai_press')} hPa
            
            --- VISUALS ---
            1. VISUAL SATELLITE: Cloud Texture.
            2. PRECIPITATION RADAR: Rain Intensity (Colors).
            
            --- TASK ---
            1. Describe Cloud Structure.
            2. Check Radar for active rain.
            3. DECISION: **GO** or **NO-GO**?
            4. REASONING: Scientific justification.
            """
            
            # Safe Input List
            inputs = [prompt]
            if st.session_state.get('ai_main'): inputs.append(st.session_state['ai_main'])
            if st.session_state.get('ai_radar'): inputs.append(st.session_state['ai_radar'])
            
            with st.spinner("Gemini 2.0 is fusing streams..."):
                try:
                    res = model.generate_content(inputs)
                    st.markdown("### 🛰️ Mission Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.success("✅ MISSION APPROVED")
                        st.balloons()
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
