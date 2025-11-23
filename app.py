import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import pandas as pd
import math

# --- CONFIGURATION ---
# Paste your keys here if you want them hardcoded, otherwise use Sidebar
DEFAULT_API_KEY = "AIzaSyAZsUnki7M2SJjPYfZ5NHJ8LX3xMtboUDU" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain Omni-Core", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4facfe;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. ROBUST IMAGE LOADER ---
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 2. GEOCODING (Search Function) ---
def get_coordinates(city_name, api_key):
    if not api_key: return None, None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        data = requests.get(url).json()
        if data: return data[0]['lat'], data[0]['lon']
    except: pass
    return None, None

# --- 3. NASA SATELLITE FETCHER (Dynamic) ---
def get_nasa_layer(layer, lat, lon):
    """Fetches Real-Time Satellite Imagery for ANY Location"""
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    
    # Dynamic Bounding Box centered on the search result
    # +/- 5 degrees gives a good regional view
    bbox = f"{lon-5},{lat-5},{lon+5},{lat+5}" 
    
    base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": layer,
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "800", "HEIGHT": "800", "TIME": today
    }
    try:
        full_url = requests.Request('GET', base_url, params=params).prepare().url
        return load_image_from_url(full_url)
    except: return None

# --- 4. OWM RADAR FETCHER ---
def get_radar_tile(lat, lon, api_key):
    if not api_key: return None
    try:
        zoom = 6
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        url = f"https://tile.openweathermap.org/map/precipitation_new/{zoom}/{xtile}/{ytile}.png?appid={api_key}"
        return load_image_from_url(url)
    except: return None

# --- 5. TELEMETRY FETCHER ---
def get_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- SIDEBAR: MISSION CONTROL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Global Data Verification System")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Global Target Selector")
    
    # SEARCH BAR
    target_name = st.text_input("Search City", "Jeddah")
    
    # State Management for Location
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728

    if st.button("Find Location"):
        if weather_key:
            new_lat, new_lon = get_coordinates(target_name, weather_key)
            if new_lat:
                st.session_state['lat'] = new_lat
                st.session_state['lon'] = new_lon
                st.success(f"Locked: {target_name}")
                st.rerun() # Force reload all data for new location
            else:
                st.error("City not found.")
        else:
            st.warning("Need OpenWeather Key to search.")

    # Clickable Map
    m = folium.Map(location=[st.session_state['lat'], st.session_state['lon']], zoom_start=5)
    m.add_child(folium.LatLngPopup()) 
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.rerun()

    lat, lon = st.session_state['lat'], st.session_state['lon']
    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN DASHBOARD ---
st.title("VisionRain Omni-Core")
st.markdown(f"### *Sector Analysis: {target_name}*")

tab1, tab2 = st.tabs(["📡 NASA Sensor Array (Visuals)", "🧠 Gemini Fusion Core"])

# TAB 1: NASA LAYERS & TELEMETRY
with tab1:
    st.header("1. Real-Time Earth Science Data")
    
    col_img, col_data = st.columns([2, 1])
    
    # LEFT: SATELLITE IMAGERY
    with col_img:
        layer_opt = st.selectbox("Select Instrument Layer:", 
            ["1. Visible (Cloud Texture)", "2. Thermal Infrared (Night Mode)", "3. Water Vapor (Moisture)"])
        
        if "1. Visible" in layer_opt:
            # VIIRS / MODIS Visible (Only works in Daytime)
            img = get_nasa_layer("VIIRS_SNPP_CorrectedReflectance_TrueColor", lat, lon)
            if img: 
                st.image(img, caption=f"NASA VIIRS TrueColor ({target_name})", use_column_width=True)
                st.session_state['ai_vis'] = img
            else:
                st.warning("Visual Band Offline (Likely Night). Switching to Thermal.")
                # Auto-fallback to Thermal
                img = get_nasa_layer("VIIRS_SNPP_Brightness_Temperature_BandM15_DayNight", lat, lon)
                st.image(img, caption="Fallback: Thermal Infrared (Night Capable)", use_column_width=True)
                st.session_state['ai_vis'] = img
                
        elif "2. Thermal" in layer_opt:
            img = get_nasa_layer("VIIRS_SNPP_Brightness_Temperature_BandM15_DayNight", lat, lon)
            if img: 
                st.image(img, caption="Thermal Infrared (Cloud Height)", use_column_width=True)
                st.session_state['ai_vis'] = img

        elif "3. Water Vapor" in layer_opt:
            # Using a water vapor proxy or similar global layer available in GIBS
            img = get_nasa_layer("AIRS_L3_Total_Precipitable_Water_Liquid_A_Day", lat, lon)
            if img: 
                st.image(img, caption="Total Precipitable Water", use_column_width=True)
                st.session_state['ai_vis'] = img

    # RIGHT: TELEMETRY & RADAR
    with col_data:
        st.subheader("Telemetry")
        w = get_telemetry(lat, lon, weather_key)
        if w:
            st.metric("Humidity", f"{w['main']['humidity']}%", "Target > 40%")
            st.metric("Temp", f"{w['main']['temp']}°C")
            st.metric("Pressure", f"{w['main']['pressure']} hPa")
            
            # Save for AI
            st.session_state['ai_humid'] = w['main']['humidity']
            st.session_state['ai_press'] = w['main']['pressure']
        else:
            st.error("Enter Weather Key.")
            st.session_state['ai_humid'] = "N/A"
            st.session_state['ai_press'] = "N/A"
        
        st.markdown("---")
        st.subheader("Precipitation Radar")
        radar_img = get_radar_tile(lat, lon, weather_key)
        if radar_img:
            st.image(radar_img, caption=f"Rain Radar ({target_name})", use_column_width=True)
            st.session_state['ai_rad'] = radar_img
        else:
            st.warning("Radar Clear / No Data")

# TAB 2: GEMINI SUPER FUSION
with tab2:
    st.header("2. Gemini Fusion Engine")
    st.write("Gemini analyzes **Satellite Texture** + **Radar Reflectivity** + **Telemetry** simultaneously.")
    
    if st.button("RUN DEEP DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        elif 'ai_vis' not in st.session_state:
            st.error("📡 Data Not Ready. Please load Tab 1 first.")
        else:
            
            # SHOW THE USER WHAT AI IS SEEING
            st.markdown("### 👁️ AI Input Stream")
            c1, c2 = st.columns(2)
            c1.image(st.session_state['ai_vis'], caption="Input 1: Satellite Morphology", use_column_width=True)
            if 'ai_rad' in st.session_state and st.session_state['ai_rad']:
                c2.image(st.session_state['ai_rad'], caption="Input 2: Radar Reflectivity", use_column_width=True)
            
            # CONFIGURE AI
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # --- SUPER PROMPT (Scientific Chain-of-Thought) ---
            prompt = f"""
            ACT AS A LEAD ATMOSPHERIC SCIENTIST.
            Analyze this Multi-Modal Sensor Data for a Cloud Seeding Mission.
            
            --- MISSION CONTEXT ---
            Location: {target_name} ({lat}, {lon})
            Objective: Hygroscopic Seeding (Salt Flares).
            
            --- TELEMETRY ---
            - Humidity: {st.session_state.get('ai_humid')}%
            - Pressure: {st.session_state.get('ai_press')} hPa
            
            --- VISUAL ANALYSIS TASK ---
            1. SATELLITE (Image 1): 
               - Describe the cloud texture. Is it 'Stratus' (flat/hazy) or 'Cumulus' (lumpy/convective)?
               - Are there vertical towers indicating updrafts?
            
            2. RADAR (Image 2): 
               - Do you see colored pixels? (Green/Blue = Light Rain, Yellow/Red = Heavy Storms).
               - If Transparent/White, it means NO rain is currently falling.
            
            --- FINAL DECISION ---
            1. SYNTHESIS: Does the Humidity match the visual clouds?
            2. VERDICT: **GO** or **NO-GO**?
            3. JUSTIFICATION: Explain using meteorological terms (updrafts, coalescence, nucleation).
            """
            
            inputs = [prompt, st.session_state['ai_vis']]
            if st.session_state.get('ai_rad'):
                inputs.append(st.session_state['ai_rad'])
                
            with st.spinner("Gemini is fusing visual and numerical streams..."):
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
