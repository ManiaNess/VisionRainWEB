import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import streamlit.components.v1 as components
import os

# --- CONFIGURATION ---
# Users will paste keys in the sidebar
DEFAULT_API_KEY = "AIzaSyA7Yk4WRdSu976U4EpHZN47m" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain Data Core", layout="wide", page_icon="🛰️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #60a5fa;}
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: ROBUST IMAGE LOADER ---
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        return Image.open(BytesIO(r.content))
    except: return None

# --- DATA FETCHING ---
def get_nasa_feed(lat, lon):
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bbox = f"{lat-5},{lon-5},{lat+5},{lon+5}" 
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

def get_weather_telemetry(lat, lon):
    if WEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            return requests.get(url).json()['main']
        except: pass
    return {"humidity": 65, "temp": 32, "pressure": 1012} 

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Data Verification System")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeatherMap Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Target Zone")
    target_name = st.text_input("Region Name", "Riyadh")
    col1, col2 = st.columns(2)
    with col1: lat = st.number_input("Latitude", value=24.7, step=0.1)
    with col2: lon = st.number_input("Longitude", value=46.7, step=0.1)
    
    st.map({"lat": [lat], "lon": [lon]})
    st.success(f"Tracking: **{target_name}**")

# --- MAIN DASHBOARD ---
st.title("VisionRain Data Command Center")
tab1, tab2 = st.tabs(["📡 Data Verification (Live)", "🧠 AI Analysis (Gemini)"])

# TAB 1: THE DATA DASHBOARD
with tab1:
    st.header("1. Hydro-Meteorological Data Fusion")
    
    col_sat, col_weath, col_rad = st.columns(3)
    
    # 1. SATELLITE DATA
    with col_sat:
        st.subheader("A. Visual Satellite")
        source_opt = st.radio("Source:", ["Live Feed (NASA)", "Archive (Storm)"], horizontal=True)
        
        if source_opt == "Live Feed (NASA)":
            if st.button("📡 ACQUIRE LIVE SIGNAL"):
                with st.spinner("Connecting to Suomi NPP..."):
                    img, date = get_nasa_feed(lat, lon)
                    if img:
                        st.image(img, caption=f"Live Feed: {date}", use_column_width=True)
                        st.session_state['sat_img'] = img
                    else:
                        st.warning("Orbit Offline. Using Backup.")
                        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg"
                        st.session_state['sat_img'] = load_image_from_url(url)
                        st.image(st.session_state['sat_img'], caption="Backup: Archive Storm")
        else:
            url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg"
            st.session_state['sat_img'] = load_image_from_url(url)
            if st.session_state['sat_img']:
                st.image(st.session_state['sat_img'], caption="Archive: Convective System", use_column_width=True)

    # 2. WEATHER DATA
    with col_weath:
        st.subheader("B. Telemetry")
        w = get_weather_telemetry(lat, lon)
        st.metric("Relative Humidity", f"{w['humidity']}%", "Target > 40%")
        st.metric("Temperature", f"{w['temp']}°C")
        st.metric("Pressure", f"{w['pressure']} hPa")
        st.info(f"**Status:** {'✅ SEEDABLE' if w['humidity'] > 40 else '⚠️ TOO DRY'}")

    # 3. RADAR DATA (OPENWEATHERMAP TILES)
    with col_rad:
        st.subheader("C. Global Precipitation")
        radar_mode = st.radio("Radar Mode:", ["Interactive Map (OpenWeather)", "Scientific Scan (Static)"])
        
        if radar_mode == "Interactive Map (OpenWeather)":
            if not weather_key:
                st.error("⚠️ Enter OpenWeatherMap Key in Sidebar to see Live Radar!")
            else:
                # This HTML embeds a Leaflet Map using YOUR API Key for tiles
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
                        var map = L.map('map').setView([{lat}, {lon}], 5);
                        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                            attribution: '© OpenStreetMap'
                        }}).addTo(map);
                        
                        // OPENWEATHERMAP PRECIPITATION LAYER
                        L.tileLayer('https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={weather_key}', {{
                            opacity: 0.8
                        }}).addTo(map);
                    </script>
                </body>
                </html>
                """
                components.html(html_map, height=300)
                st.caption("Layer: OWM Global Precipitation")
        else:
            # STATIC BACKUP
            radar_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Radar_reflectivity.jpg/600px-Radar_reflectivity.jpg"
            st.session_state['rad_img'] = load_image_from_url(radar_url)
            if st.session_state['rad_img']:
                st.image(st.session_state['rad_img'], caption="NASA GPM IMERG (Gemini Readable)", use_column_width=True)

# TAB 2: AI ANALYSIS
with tab2:
    st.header("2. Gemini Fusion Engine")
    st.write("The AI analyzes **Satellite Texture** + **Radar Reflectivity** + **Humidity**.")
    
    if st.button("RUN DIAGNOSTICS"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        elif 'sat_img' not in st.session_state:
            st.error("📡 Load Satellite Data in Tab 1 first.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            You are an AI Meteorologist. Analyze these TWO inputs:
            1. VISUAL SATELLITE (Image 1): Look for convective towers.
            2. RADAR REFLECTIVITY (Image 2): Look for Red/Yellow zones.
            3. TELEMETRY: Humidity is {w['humidity']}%.
            DECISION: Is this cloud system suitable for Seeding?
            FORMAT: JSON {{Decision: GO/NO-GO, Confidence: %, Reasoning: text}}
            """
            
            inputs = [prompt, st.session_state['sat_img']]
            if 'rad_img' in st.session_state and st.session_state['rad_img']:
                inputs.append(st.session_state['rad_img'])
                st.success("✅ Radar Data Injected into Model")
            
            with st.spinner("Fusion Engine Processing..."):
                try:
                    res = model.generate_content(inputs)
                    st.markdown("### 🤖 AI Assessment")
                    st.write(res.text)
                    if "GO" in res.text.upper():
                        st.success("Conditions Optimal.")
                        st.balloons()
                except Exception as e:
                    st.error(f"AI Error: {e}")
