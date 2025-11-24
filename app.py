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

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
SCREENSHOT_API_KEY = "f9ededa86ff343819371871884196288" # APIFLASH KEY REQUIRED FOR WINDY SCREENSHOTS

st.set_page_config(page_title="VisionRain Autopilot", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4facfe;}
    .data-box {padding: 10px; background-color: #1e293b; border-radius: 5px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. VISUAL AGENT (ApiFlash) ---
def get_windy_screenshot(lat, lon, layer, api_key):
    if not api_key: return None
    # Windy URL for specific layer
    windy_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=800&height=600&zoom=6&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    
    params = {
        'access_key': api_key,
        'url': windy_url,
        'format': 'jpeg',
        'width': 800,
        'height': 600,
        'delay': 4, # Wait for animation
        'quality': 80,
        'no_cookie_banners': 'true',
        'no_ads': 'true'
    }
    try:
        query = urllib.parse.urlencode(params)
        r = requests.get(f"https://api.apiflash.com/v1/urltoimage?{query}", timeout=20)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 2. NASA SATELLITE ---
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
        # Fetch directly
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 3. TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 4. GEOCODING ---
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
    st.title("VisionRain")
    st.caption("Autopilot Agent")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    screen_key = st.text_input("ApiFlash Key", value=SCREENSHOT_API_KEY, type="password")
    
    st.markdown("### 📍 Target Selector")
    
    # Search Logic
    target_input = st.text_input("Region Search", "Jeddah")
    
    # Session State Init
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728
    if 'target_name' not in st.session_state: st.session_state['target_name'] = "Jeddah"
    if 'last_fetch_coords' not in st.session_state: st.session_state['last_fetch_coords'] = (0,0)

    if st.button("Find City"):
        if weather_key:
            new_lat, new_lon = get_coordinates(target_input, weather_key)
            if new_lat:
                st.session_state['lat'] = new_lat
                st.session_state['lon'] = new_lon
                st.session_state['target_name'] = target_input
                st.rerun()

    # Interactive Map
    m = folium.Map(location=[st.session_state['lat'], st.session_state['lon']], zoom_start=5)
    m.add_child(folium.LatLngPopup()) 
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.session_state['target_name'] = "Map Selection"
        st.rerun()

    lat = st.session_state['lat']
    lon = st.session_state['lon']
    target_name = st.session_state['target_name']
    
    st.success(f"Locked: {lat:.4f}, {lon:.4f}")

# --- AUTOMATIC DATA COLLECTION (Runs when location changes) ---
current_coords = (lat, lon)
if st.session_state['last_fetch_coords'] != current_coords:
    if screen_key:
        with st.spinner("🤖 Agent is deployed! Capturing Global Sensor Data... (Wait ~15s)"):
            # 1. NASA
            st.session_state['img_nasa'] = get_nasa_feed(lat, lon)
            
            # 2. WINDY LAYERS (Batch Capture)
            st.session_state['img_radar'] = get_windy_screenshot(lat, lon, "radar", screen_key)
            st.session_state['img_wind'] = get_windy_screenshot(lat, lon, "wind", screen_key)
            st.session_state['img_clouds'] = get_windy_screenshot(lat, lon, "clouds", screen_key)
            st.session_state['img_rain'] = get_windy_screenshot(lat, lon, "rain", screen_key)
            
            # 3. Telemetry
            st.session_state['telemetry'] = get_weather_telemetry(lat, lon, weather_key)
            
            # Update state to prevent loop
            st.session_state['last_fetch_coords'] = current_coords
            st.rerun()
    else:
        st.warning("⚠️ Enter ApiFlash Key to enable Auto-Capture.")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Target Sector: {target_name}*")

tab1, tab2 = st.tabs(["📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# TAB 1: THE WALL OF SCREENS
with tab1:
    st.header("1. Multi-Spectral Environment")
    
    # Telemetry Row
    w = st.session_state.get('telemetry', None)
    if w:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Humidity", f"{w['main']['humidity']}%")
        c2.metric("Temp", f"{w['main']['temp']}°C")
        c3.metric("Pressure", f"{w['main']['pressure']} hPa")
        c4.metric("Wind", f"{w['wind']['speed']} m/s")
    
    st.divider()
    
    # Visuals Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("A. NASA Optical Satellite")
        if st.session_state.get('img_nasa'): st.image(st.session_state['img_nasa'], use_column_width=True)
        else: st.info("Waiting for NASA...")
        
        st.caption("D. Cloud Density (Windy)")
        if st.session_state.get('img_clouds'): st.image(st.session_state['img_clouds'], use_column_width=True)
        else: st.info("Waiting for Agent...")

    with col2:
        st.caption("B. Doppler Radar (Windy)")
        if st.session_state.get('img_radar'): st.image(st.session_state['img_radar'], use_column_width=True)
        else: st.info("Waiting for Agent...")
        
        st.caption("E. Rain Accumulation (Windy)")
        if st.session_state.get('img_rain'): st.image(st.session_state['img_rain'], use_column_width=True)
        else: st.info("Waiting for Agent...")

    with col3:
        st.caption("C. Wind Flow (Windy)")
        if st.session_state.get('img_wind'): st.image(st.session_state['img_wind'], use_column_width=True)
        else: st.info("Waiting for Agent...")

# TAB 2: GEMINI FUSION
with tab2:
    st.header("2. Gemini Fusion Engine")
    
    # 1. SHOW DATA INPUTS (Transparency)
    st.markdown("### 🔍 Data Ingestion Manifest")
    
    # Comparison Table
    if w:
        df = pd.DataFrame({
            "Parameter": ["Humidity", "Wind Speed", "Cloud Cover", "Temp"],
            "Ideal Seeding Condition": ["> 50%", "< 15 m/s", "Convective", "< 25°C"],
            "Current Actual": [
                f"{w['main']['humidity']}%", 
                f"{w['wind']['speed']} m/s", 
                "Analyzing...", 
                f"{w['main']['temp']}°C"
            ]
        })
        st.table(df)
    
    # Show Thumbnails of what AI reads
    st.write("**Visual Evidence Stream:**")
    cols = st.columns(5)
    imgs = [st.session_state.get(k) for k in ['img_nasa', 'img_radar', 'img_wind', 'img_clouds', 'img_rain']]
    captions = ["NASA", "Radar", "Wind", "Clouds", "Rain"]
    
    valid_imgs = []
    valid_caps = []
    for i, img in enumerate(imgs):
        if img:
            cols[i].image(img, caption=captions[i], use_column_width=True)
            valid_imgs.append(img)
            valid_caps.append(captions[i])

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        elif not valid_imgs:
            st.error("⚠️ No images captured yet.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD ATMOSPHERIC SCIENTIST.
            Analyze this Multi-Source Intelligence for Cloud Seeding.
            
            --- TARGET CONTEXT ---
            Location: {target_name}
            Objective: Hygroscopic Seeding.
            
            --- IDEAL CRITERIA ---
            - Humidity > 50%
            - Wind < 15 m/s
            - Clouds: Convective (Lumpy/Vertical) preferred over Stratiform (Flat).
            - Radar: Active cells (Red/Yellow) indicate storm maturity.
            
            --- ACTUAL TELEMETRY ---
            - Humidity: {w['main']['humidity']}%
            - Wind: {w['wind']['speed']} m/s
            - Pressure: {w['main']['pressure']} hPa
            
            --- VISUAL INPUTS ---
            I have attached {len(valid_imgs)} images: {', '.join(valid_caps)}.
            1. NASA Satellite: Check cloud texture.
            2. Windy Radar: Check precipitation intensity.
            3. Windy Wind: Check airflow patterns.
            
            --- TASK ---
            1. VISUAL ANALYSIS: Describe the structure seen in the Satellite and Radar images.
            2. COMPARISON: Compare Actual conditions vs Ideal Criteria (Table above).
            3. DECISION: **GO** or **NO-GO**?
            4. REASONING: Scientific justification.
            """
            
            with st.spinner("Gemini 2.0 is analyzing Visuals + Telemetry..."):
                try:
                    inputs = [prompt] + valid_imgs
                    res = model.generate_content(inputs)
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.balloons()
                        st.success("✅ MISSION APPROVED")
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
