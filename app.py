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

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "" 
SCREENSHOT_API_KEY = "" # Get from apiflash.com

st.set_page_config(page_title="VisionRain Hybrid Core", layout="wide", page_icon="👁️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4facfe;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. NASA SATELLITE FETCHER (For Clouds) ---
def get_nasa_feed(lat, lon):
    """Fetches Static Satellite Imagery from NASA (Best for Clouds)"""
    try:
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        bbox = f"{lon-5},{lat-5},{lon+5},{lat+5}" 
        url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        params = {
            "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
            "LAYERS": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
            "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
            "BBOX": bbox, "WIDTH": "800", "HEIGHT": "800", "TIME": today
        }
        # Robust Loader
        full_url = requests.Request('GET', url, params=params).prepare().url
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(full_url, headers=headers, timeout=8)
        if r.status_code == 200: return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 2. WINDY VISUAL AGENT (For Radar/Wind) ---
def get_windy_screenshot(lat, lon, layer, api_key):
    """Captures Windy.com for dynamic layers"""
    if not api_key: return None
    
    # Windy URL centered on location
    windy_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=1000&height=600&zoom=6&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    
    params = {
        'access_key': api_key,
        'url': windy_url,
        'format': 'jpeg',
        'width': 1000,
        'height': 600,
        'delay': 5, # Wait for animation
        'quality': 80,
        'no_cookie_banners': 'true',
        'no_ads': 'true'
    }
    
    try:
        query = urllib.parse.urlencode(params)
        r = requests.get(f"https://api.apiflash.com/v1/urltoimage?{query}", timeout=25)
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
    st.caption("Hybrid-Agent Interface")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    screen_key = st.text_input("ApiFlash Key", value=SCREENSHOT_API_KEY, type="password")
    
    st.markdown("### 📍 Target Selector")
    
    # Search Logic
    target_input = st.text_input("Region Search", "Jeddah")
    
    # Init State
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728
    if 'target_name' not in st.session_state: st.session_state['target_name'] = "Jeddah"

    if st.button("Find City"):
        if weather_key:
            new_lat, new_lon = get_coordinates(target_input, weather_key)
            if new_lat:
                st.session_state['lat'] = new_lat
                st.session_state['lon'] = new_lon
                st.session_state['target_name'] = target_input
                st.rerun()
        else:
            st.warning("Need OpenWeather Key")

    # Interactive Map
    m = folium.Map(location=[st.session_state['lat'], st.session_state['lon']], zoom_start=5)
    m.add_child(folium.LatLngPopup()) 
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.session_state['target_name'] = "Selected Coordinates"
        st.rerun()

    lat = st.session_state['lat']
    lon = st.session_state['lon']
    target_name = st.session_state['target_name']
    
    st.success(f"Locked: {target_name}\n({lat:.4f}, {lon:.4f})")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Target Sector: {target_name}*")

tab1, tab2 = st.tabs(["👁️ Hybrid Sensor Array", "🧠 Gemini Fusion Core"])

# TAB 1: HYBRID SENSORS
with tab1:
    st.header("1. Multi-Source Visual Intelligence")
    
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        st.info("Select Instrument Layer:")
        # Distinct logic for Clouds vs Others
        layer_opt = st.selectbox("Layer", ["Satellite (NASA Clouds)", "Radar (Windy)", "Wind (Windy)", "Rain Accumulation (Windy)"])
        
        if st.button("📡 ACQUIRE VISUAL", type="primary"):
            with st.spinner(f"Acquiring {layer_opt} for {target_name}..."):
                
                # LOGIC SPLIT: NASA vs WINDY
                if "Satellite" in layer_opt:
                    # Use NASA for Clouds
                    img = get_nasa_feed(lat, lon)
                    source = "NASA VIIRS (Optical)"
                else:
                    # Use Windy Agent for everything else
                    if not screen_key:
                        st.error("Missing ApiFlash Key!")
                        img = None
                    else:
                        # Map selection to Windy parameter
                        w_layer = "radar" if "Radar" in layer_opt else "wind" if "Wind" in layer_opt else "rain"
                        img = get_windy_screenshot(lat, lon, w_layer, screen_key)
                        source = f"Windy.com Agent ({w_layer.title()})"
                
                # Save result
                if img:
                    st.session_state['current_img'] = img
                    st.session_state['current_source'] = source
                    st.success("Data Acquired")
                else:
                    st.error("Acquisition Failed")

    with col_view:
        # Display Image
        if st.session_state.get('current_img'):
            st.image(st.session_state['current_img'], caption=f"{st.session_state['current_source']} @ {target_name}", use_column_width=True)
        else:
            st.markdown("<div style='height:300px; border:1px dashed #555; display:flex; align-items:center; justify-content:center;'>NO SIGNAL</div>", unsafe_allow_html=True)

# TAB 2: GEMINI FUSION
with tab2:
    st.header("2. Multi-Modal Decision Core")
    
    # Telemetry
    w = get_weather_telemetry(lat, lon, weather_key)
    if w:
        c1, c2, c3 = st.columns(3)
        c1.metric("Humidity", f"{w['main']['humidity']}%")
        c2.metric("Pressure", f"{w['main']['pressure']} hPa")
        c3.metric("Wind", f"{w['wind']['speed']} m/s")
    
    st.divider()
    
    if st.button("RUN MISSION DIAGNOSTICS"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        elif not st.session_state.get('current_img'):
            st.error("⚠️ No visual data! Go to Tab 1 and Acquire Visuals.")
        else:
            # Run AI
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST.
            Analyze this Visual Data + Telemetry for Cloud Seeding suitability.
            
            --- CONTEXT ---
            Target: {target_name} ({lat}, {lon})
            Telemetry: Humidity {w['main']['humidity'] if w else 'N/A'}%
            Source Type: {st.session_state['current_source']}
            
            --- VISUAL ANALYSIS ---
            Look at the attached image.
            1. If SATELLITE: Describe cloud texture (Convective vs Stratiform).
            2. If RADAR/RAIN: Describe precipitation intensity (Colors).
            3. If WIND: Describe airflow patterns.
            
            --- DECISION ---
            1. CORRELATION: Does the image match the Humidity?
            2. VERDICT: **GO** or **NO-GO**?
            3. REASONING: Scientific justification.
            """
            
            with st.spinner("Gemini is analyzing..."):
                try:
                    res = model.generate_content([prompt, st.session_state['current_img']])
                    st.markdown("### 🤖 Mission Report")
                    st.write(res.text)
                    if "GO" in res.text.upper(): st.balloons()
                except Exception as e:
                    st.error(f"AI Error: {e}")
