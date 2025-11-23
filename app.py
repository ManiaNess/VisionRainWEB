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
DEFAULT_API_KEY = "AIzaSyAZsUnki7M2SJjPYfZ5NHJ8LX3xMtboUDU" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain Scientific Core", layout="wide", page_icon="⛈️")

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

# --- 2. SMART NASA LAYER FETCHER ---
def get_nasa_layer(layer_type, lat, lon):
    """Fetches specific scientific layers from NASA GIBS"""
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bbox = f"{lon-10},{lat-10},{lon+10},{lat+10}" 
    
    # SELECT SATELLITE BASED ON LONGITUDE
    if -140 < lon < -30: sat_source = "GOES-East_ABI_Band02_Red_Visible_1km"
    elif -30 <= lon < 60: sat_source = "Meteosat_MSG_SEVIRI_Band03_Visible"
    else: sat_source = "Himawari_AHI_Band3_Red_Visible_1km"

    # LAYER DEFINITIONS
    layers = {
        "Visual": sat_source,
        "Thermal": "MODIS_Terra_Land_Surface_Temperature_Day", 
        "Moisture": "AIRS_L3_Total_Precipitable_Water_Liquid_A_Day", 
        "Night": "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance",
    }
    
    selected_layer = layers.get(layer_type, sat_source)
    
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": selected_layer,
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "800", "HEIGHT": "800", "TIME": today
    }
    
    try:
        full_url = requests.Request('GET', url, params=params).prepare().url
        img = load_image_from_url(full_url)
        return img, selected_layer
    except: return None, "Error"

# --- 3. OWM RADAR TILE ---
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

# --- 4. TELEMETRY ---
def get_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 5. GEOCODING ---
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
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Scientific Core")
    
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
st.title("VisionRain Scientific Core")
st.markdown(f"### *Sector Analysis: {target_name}*")

tab1, tab2 = st.tabs(["📡 NASA Sensor Array (Visuals)", "🧠 Gemini Fusion Core"])

# TAB 1: NASA LAYERS
with tab1:
    st.header("1. Earth Science Data (NASA GIBS)")
    
    col_img, col_data = st.columns([2, 1])
    
    with col_img:
        layer_opt = st.selectbox("Select Instrument Layer:", 
            ["1. Visible (Geostationary)", "2. Thermal (Land Surface Temp)", "3. Moisture (Water Vapor)", "4. Night Vision"])
        
        # --- FIX: CHECK IF IMAGE EXISTS BEFORE DISPLAYING ---
        if "1. Visible" in layer_opt:
            img, src = get_nasa_layer("Visual", lat, lon)
            if img: 
                st.image(img, caption=f"Source: {src}", use_column_width=True)
                st.session_state['ai_vis'] = img
            else:
                st.warning("Visible Band Unavailable (Night/Cloud). Using Night Mode recommended.")
                
        elif "2. Thermal" in layer_opt:
            img, src = get_nasa_layer("Thermal", lat, lon)
            if img:
                st.image(img, caption=f"Thermal Map ({src})", use_column_width=True)
                st.session_state['ai_therm'] = img
            else:
                st.warning("Thermal Data Unavailable for this region/time.")

        elif "3. Moisture" in layer_opt:
            img, src = get_nasa_layer("Moisture", lat, lon)
            if img:
                st.image(img, caption=f"Water Vapor ({src})", use_column_width=True)
                st.session_state['ai_moist'] = img
            else:
                st.warning("Moisture Data Processing...")
            
        elif "4. Night" in layer_opt:
            img, src = get_nasa_layer("Night", lat, lon)
            if img:
                st.image(img, caption=f"Night Vision ({src})", use_column_width=True)
            else:
                st.warning("Night Band Unavailable.")

    with col_data:
        st.subheader("Telemetry")
        w = get_telemetry(lat, lon, weather_key)
        if w:
            st.metric("Humidity", f"{w['main']['humidity']}%", "Target > 40%")
            st.metric("Temp", f"{w['main']['temp']}°C")
            st.metric("Pressure", f"{w['main']['pressure']} hPa")
        else:
            st.error("Enter Weather Key.")
        
        st.markdown("---")
        st.subheader("Precipitation Radar")
        radar_img = get_radar_tile(lat, lon, weather_key)
        if radar_img:
            st.image(radar_img, caption="Precipitation Overlay (OWM)", use_column_width=True)
            st.session_state['ai_rad'] = radar_img
        else:
            st.warning("Radar Clear")

# TAB 2: GEMINI AI
with tab2:
    st.header("2. Gemini Fusion Engine")
    
    if st.button("RUN DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # Ensure we have at least one image to analyze
            if 'ai_vis' not in st.session_state:
                st.session_state['ai_vis'], _ = get_nasa_layer("Visual", lat, lon)
            
            # Display Payload
            st.markdown("### 👁️ AI Input Stream")
            c1, c2, c3 = st.columns(3)
            
            # Only display if image exists
            if st.session_state.get('ai_vis'): c1.image(st.session_state['ai_vis'], caption="Visual", use_column_width=True)
            if st.session_state.get('ai_therm'): c2.image(st.session_state['ai_therm'], caption="Thermal", use_column_width=True)
            if st.session_state.get('ai_rad'): c3.image(st.session_state['ai_rad'], caption="Radar", use_column_width=True)
            
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST. Analyze this Multi-Spectral Sensor Data.
            Location: {target_name} ({lat}, {lon})
            
            DATA:
            - Humidity: {w['main']['humidity'] if w else 'N/A'}%
            
            TASK:
            1. Analyze Cloud Morphology.
            2. Evaluate Thermal/Moisture data if present.
            3. DECISION: **GO** or **NO-GO** for Cloud Seeding?
            """
            
            # Safe Input Construction
            inputs = [prompt]
            if st.session_state.get('ai_vis'): inputs.append(st.session_state['ai_vis'])
            if st.session_state.get('ai_therm'): inputs.append(st.session_state['ai_therm'])
            if st.session_state.get('ai_rad'): inputs.append(st.session_state['ai_rad'])
            
            with st.spinner("Gemini is thinking..."):
                try:
                    res = model.generate_content(inputs)
                    st.markdown("### 🛰️ Mission Report")
                    st.write(res.text)
                    if "GO" in res.text.upper(): st.balloons()
                except Exception as e:
                    st.error(f"AI Error: {e}")
