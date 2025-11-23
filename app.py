import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import math

# --- CONFIGURATION ---
DEFAULT_GOOGLE_KEY = "AIzaSyA7Yk4WRdSu976U4EpHZN47m-KA8JbJ5do"
DEFAULT_WEATHER_KEY = "11b260a4212d29eaccbd9754da459059"

st.set_page_config(page_title="VisionRain Sensor Array", layout="wide", page_icon="🛰️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #1f2937; padding: 10px; border-radius: 8px;}
    h1, h2, h3 {color: #60a5fa;}
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST IMAGE LOADER ---
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        return Image.open(BytesIO(r.content))
    except: return None

# --- 1. NASA WORLDVIEW FETCHER ---
def get_nasa_satellite(lat, lon):
    """Fetches True-Color Satellite Image from NASA GIBS"""
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    # Create a 4-degree bounding box for a good view
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
        return load_image_from_url(full_url), "NASA VIIRS (Live)"
    except: return None, "Offline"

# --- 2. OWM MAP TILE FETCHER ---
def get_map_tile(layer, lat, lon, api_key, zoom=6):
    """Fetches specific weather layer tiles (Clouds, Rain, Wind)"""
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    
    url = f"https://tile.openweathermap.org/map/{layer}_new/{zoom}/{xtile}/{ytile}.png?appid={api_key}"
    return load_image_from_url(url)

# --- 3. NUMERICAL DATA FETCHER ---
def get_weather_json(lat, lon, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.title("⛈️ VisionRain")
    st.caption("Multi-Sensor Fusion Core")
    google_key = st.text_input("Google API Key", value=DEFAULT_GOOGLE_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=DEFAULT_WEATHER_KEY, type="password")
    
    st.markdown("### 📍 Target Selection")
    loc_name = st.text_input("Region", "Riyadh, SA")
    lat = st.number_input("Latitude", 24.7136)
    lon = st.number_input("Longitude", 46.6753)
    st.success("Status: **ONLINE**")

# --- MAIN UI ---
st.title(f"📡 Mission Target: {loc_name}")

# 1. NUMERICAL DATA
w_data = get_weather_json(lat, lon, weather_key)
if w_data:
    main = w_data['main']
    wind = w_data['wind']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temp", f"{main['temp']}°C")
    c2.metric("Humidity", f"{main['humidity']}%", "Target > 40%")
    c3.metric("Pressure", f"{main['pressure']} hPa")
    c4.metric("Wind", f"{wind['speed']} m/s")

st.markdown("---")

# 2. VISUAL SENSOR ARRAY
st.header("Sensor Fusion Array")

# TOP ROW: NASA SATELLITE (THE "REAL" VIEW)
st.subheader("1. Optical Satellite Feed (NASA Worldview)")
nasa_img, status = get_nasa_satellite(lat, lon)
if nasa_img:
    st.image(nasa_img, caption=f"Source: VIIRS/Suomi NPP | Status: {status}", width=800)
else:
    st.warning("Satellite Offline (Night Transit). Using OWM Data below.")

# BOTTOM ROW: 4 OWM LAYERS
st.subheader("2. Atmospheric Analysis Layers (OpenWeatherMap)")
col1, col2 = st.columns(2)

with col1:
    st.write("**Layer A: Cloud Density**")
    cloud_img = get_map_tile("clouds", lat, lon, weather_key)
    if cloud_img: st.image(cloud_img, use_column_width=True)
    
    st.write("**Layer C: Wind Speed**")
    wind_img = get_map_tile("wind", lat, lon, weather_key)
    if wind_img: st.image(wind_img, use_column_width=True)

with col2:
    st.write("**Layer B: Precipitation (Radar)**")
    rain_img = get_map_tile("precipitation", lat, lon, weather_key)
    if rain_img: st.image(rain_img, use_column_width=True)
    
    st.write("**Layer D: Pressure Isobars**")
    press_img = get_map_tile("pressure", lat, lon, weather_key)
    if press_img: st.image(press_img, use_column_width=True)

# 3. AI ANALYSIS
st.markdown("---")
st.header("🧠 Gemini 5-Point Fusion")

if st.button("RUN MULTIMODAL DIAGNOSTICS", type="primary"):
    if not google_key:
        st.error("🔑 API Key Missing!")
    else:
        genai.configure(api_key=google_key)
        # Using 1.5 Flash because it handles multiple images extremely well
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Act as a Senior Meteorologist. Analyze this Multi-Sensor Data Array.
        
        NUMERICAL TELEMETRY:
        {w_data}
        
        VISUAL SENSORS:
        1. NASA SATELLITE (Image 1): Real-world optical view. Look for convective towers.
        2. CLOUD DENSITY MAP (Image 2): Grey = Cloud Cover.
        3. PRECIPITATION RADAR (Image 3): Colors = Rain.
        4. WIND SPEED MAP (Image 4).
        5. PRESSURE MAP (Image 5).
        
        TASK:
        1. Correlate the Humidity ({main['humidity']}%) with the visual clouds.
        2. Check if Radar (Image 3) shows active rain.
        3. DECISION: GO or NO-GO for Cloud Seeding?
        4. REASONING: Explain the correlation between the 5 sensors.
        """
        
        # PACKING ALL 5 IMAGES + TEXT
        inputs = [prompt]
        if nasa_img: inputs.append(nasa_img)
        if cloud_img: inputs.append(cloud_img)
        if rain_img: inputs.append(rain_img)
        if wind_img: inputs.append(wind_img)
        if press_img: inputs.append(press_img)
        
        with st.spinner("Fusing 5 Data Streams..."):
            try:
                response = model.generate_content(inputs)
                st.markdown(response.text)
                if "GO" in response.text: st.balloons()
            except Exception as e:
                st.error(f"AI Error: {e}")
