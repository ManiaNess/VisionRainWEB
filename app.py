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
import time

# --- CONFIGURATION ---
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

# --- 2. GEOCODING (FIXED) ---
def get_coordinates(city_name, api_key):
    """Converts City Name -> Lat/Lon"""
    if not api_key: return None, None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        data = requests.get(url).json()
        if data:
            return data[0]['lat'], data[0]['lon']
    except: pass
    return None, None

# --- 3. NASA SATELLITE FEED ---
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

# --- 4. BACKGROUND TILE FETCHER ---
def get_static_map_layer(layer, lat, lon, api_key):
    if not api_key: return None
    try:
        zoom = 6
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        url = f"https://tile.openweathermap.org/map/{layer}_new/{zoom}/{xtile}/{ytile}.png?appid={api_key}"
        return load_image_from_url(url)
    except: return None

# --- 5. NUMERICAL DATA (OWM) ---
def get_owm_data(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 6. NUMERICAL DATA (WINDY/ECMWF) ---
def get_windy_data(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,surface_pressure"
        return requests.get(url).json()['current']
    except: return None

# --- 7. WINDY EMBED BUILDER ---
def render_windy(lat, lon, overlay):
    # Added 'marker=true' and 'location=coordinates' to force it to jump to specific lat/lon
    # Added timestamp to URL to force Streamlit to reload the iframe on change
    ts = int(time.time()) 
    url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=800&height=500&zoom=6&level=surface&overlay={overlay}&product=ecmwf&menu=&message=&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1&t={ts}"
    components.iframe(url, height=500)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Omni-Sensor Fusion Core")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Target Selector")
    
    # --- SEARCH FUNCTIONALITY (RESTORED) ---
    target_input = st.text_input("Region Name", "Jeddah")
    
    # Initialize State
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728
    if 'target_name' not in st.session_state: st.session_state['target_name'] = "Jeddah"

    if st.button("Find Location"):
        if weather_key:
            new_lat, new_lon = get_coordinates(target_input, weather_key)
            if new_lat:
                st.session_state['lat'] = new_lat
                st.session_state['lon'] = new_lon
                st.session_state['target_name'] = target_input
                st.success(f"Locked: {target_input}")
                st.rerun()
            else:
                st.error("City not found.")
        else:
            st.warning("Enter Weather Key to enable Search")

    # Interactive Map
    m = folium.Map(location=[st.session_state['lat'], st.session_state['lon']], zoom_start=5)
    m.add_child(folium.LatLngPopup()) 
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.session_state['target_name'] = "Custom Coordinates"
        st.rerun()

    lat, lon = st.session_state['lat'], st.session_state['lon']
    target_name = st.session_state['target_name']
    
    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN DASHBOARD ---
st.title("VisionRain Omni-Core")
st.markdown(f"### *Sector Analysis: {target_name}*")

tab1, tab2, tab3 = st.tabs(["👁️ The Windy Wall (Visuals)", "📊 Data Truth (Comparison)", "🧠 Gemini Super-Fusion"])

# TAB 1: THE 7 LAYERS (VISUALS)
with tab1:
    st.header("1. Global Atmospheric Dynamics")
    layer_opt = st.selectbox("Select Sensor Layer:", 
        ["1. NASA Satellite (TrueColor)", "2. Windy Satellite (Infrared)", "3. Windy Radar (Precipitation)", 
         "4. Windy Clouds", "5. Windy Wind", "6. Windy Rain & Thunder", "7. Windy Pressure (Weather)"])
    
    if "1. NASA" in layer_opt:
        st.subheader("NASA VIIRS Optical Downlink")
        img, date = get_nasa_feed(lat, lon)
        if img: 
            st.image(img, caption=f"Real-Time Optical Feed | {date}", use_column_width=True)
            st.session_state['ai_sat'] = img
        else:
            st.warning("NASA Feed Offline (Night). Using Archive.")
            backup_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg"
            st.session_state['ai_sat'] = load_image_from_url(backup_url)
            st.image(st.session_state['ai_sat'], caption="Archive Backup")
            
    elif "2. Windy Satellite" in layer_opt: render_windy(lat, lon, "satellite")
    elif "3. Windy Radar" in layer_opt: render_windy(lat, lon, "radar")
    elif "4. Windy Clouds" in layer_opt: render_windy(lat, lon, "clouds")
    elif "5. Windy Wind" in layer_opt: render_windy(lat, lon, "wind")
    elif "6. Windy Rain" in layer_opt: render_windy(lat, lon, "rain")
    elif "7. Windy Pressure" in layer_opt: render_windy(lat, lon, "pressure")

# TAB 2: DATA TRUTH (NUMERICAL)
with tab2:
    st.header("2. Telemetry Validation")
    st.write("Comparing **Live Station Data** (OpenWeather) vs **Predictive Models** (Windy/ECMWF).")
    
    owm = get_owm_data(lat, lon, weather_key)
    windy = get_windy_data(lat, lon)
    
    if owm and windy:
        comp_data = {
            "Metric": ["Temperature", "Humidity", "Wind Speed", "Cloud Cover", "Pressure"],
            "OpenWeather (Live Station)": [f"{owm['main']['temp']} °C", f"{owm['main']['humidity']}%", f"{owm['wind']['speed']} m/s", f"{owm['clouds']['all']}%", f"{owm['main']['pressure']} hPa"],
            "Windy/ECMWF (Model)": [f"{windy['temperature_2m']} °C", f"{windy['relative_humidity_2m']}%", f"{windy['wind_speed_10m']} m/s", f"{windy['cloud_cover']}%", f"{windy['surface_pressure']} hPa"]
        }
        df = pd.DataFrame(comp_data)
        st.table(df)
        consensus_humid = (owm['main']['humidity'] + windy['relative_humidity_2m']) / 2
    else:
        st.error("⚠️ Enter OpenWeatherMap Key in Sidebar to load data.")
        consensus_humid = 65

# TAB 3: GEMINI SUPER FUSION
with tab3:
    st.header("3. Gemini Fusion Core")
    
    if not api_key:
        st.error("🔑 Please enter Google API Key in the Sidebar.")
    elif not weather_key:
        st.error("⚠️ Please enter OpenWeatherMap Key in the Sidebar.")
    else:
        st.success("✅ AUTHENTICATED. Preparing AI Input Payload...")
        
        if st.button("🚀 RUN HYPER-LOCAL DIAGNOSTICS", type="primary"):
            
            # 1. FETCH ALL STATIC MAPS (The "Eyes" for AI)
            with st.spinner("Aggregating Multi-Sensor Visuals..."):
                # Satellite
                if 'ai_sat' not in st.session_state or st.session_state['ai_sat'] is None:
                    img, _ = get_nasa_feed(lat, lon)
                    st.session_state['ai_sat'] = img if img else load_image_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg")
                
                # OWM Layers
                ai_radar = get_static_map_layer("precipitation", lat, lon, weather_key)
                ai_wind = get_static_map_layer("wind", lat, lon, weather_key)
                ai_clouds = get_static_map_layer("clouds", lat, lon, weather_key)

            # 2. DISPLAY PAYLOAD
            st.markdown("### 👁️ AI Visual Inputs")
            c1, c2, c3, c4 = st.columns(4)
            
            # Safe Display
            if st.session_state['ai_sat']: c1.image(st.session_state['ai_sat'], caption="1. Optical Satellite", use_column_width=True)
            if ai_clouds: c2.image(ai_clouds, caption="2. Cloud Density", use_column_width=True)
            if ai_radar: c3.image(ai_radar, caption="3. Radar", use_column_width=True)
            if ai_wind: c4.image(ai_wind, caption="4. Wind", use_column_width=True)

            # 3. EXECUTE AI
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST. Analyze this 4-Layer Sensor Array.
            
            --- MISSION CONTEXT ---
            Location: {target_name} ({lat}, {lon})
            Objective: Hygroscopic Cloud Seeding.
            
            --- NUMERICAL DATA ---
            - Consensus Humidity: {consensus_humid}%
            - Pressure: {owm['main']['pressure'] if owm else 'N/A'} hPa
            
            --- VISUAL DATA (Attached) ---
            Image 1: Satellite (Texture).
            Image 2: Cloud Density (Grey scale).
            Image 3: Radar (Colors = Rain).
            Image 4: Wind (Streamlines).
            
            --- TASK ---
            1. VISUAL ANALYSIS: Synthesize the patterns across all 4 images.
            2. CORRELATION: Does the Radar (Image 3) confirm the Cloud Density (Image 2)?
            3. DECISION: **GO** or **NO-GO**?
            4. REASONING: Scientific justification.
            """
            
            inputs = [prompt, st.session_state['ai_sat']]
            if ai_clouds: inputs.append(ai_clouds)
            if ai_radar: inputs.append(ai_radar)
            if ai_wind: inputs.append(ai_wind)
            
            with st.spinner("Gemini is fusing 4 visual streams + telemetry..."):
                try:
                    res = model.generate_content(inputs)
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.balloons()
                        st.success("✅ MISSION APPROVED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
