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

st.set_page_config(page_title="VisionRain | Intelligent Planet", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {
        background-color: #1a1a1a; 
        border: 1px solid #333; 
        border-radius: 12px; 
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .pitch-box {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #00e5ff;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.1);
    }
    .success-box {
        background-color: rgba(0, 255, 128, 0.1); 
        border: 1px solid #00ff80; 
        color: #00ff80; 
        padding: 15px; 
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. MICROLINK AGENT (Captures Windy) ---
def get_windy_capture(lat, lon, layer):
    """
    Uses Microlink to screenshot Windy.com layers.
    """
    # Windy Embed URL
    target_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=600&height=400&zoom=5&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    
    # Microlink API Call (Free Tier)
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(target_url)}&screenshot=true&meta=false&waitFor=4000&viewport.width=600&viewport.height=400"
    
    try:
        r = requests.get(api_url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

# --- 2. NASA SATELLITE (Visual) ---
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
        r = requests.get(url, params=params, timeout=8)
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
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("v2.0 | Autonomous AI Core")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Target Acquisition")
    target_input = st.text_input("Search Region", "Jeddah")
    
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728
    if 'target' not in st.session_state: st.session_state['target'] = "Jeddah"

    if st.button("Locate Target"):
        if weather_key:
            lat, lon = get_coordinates(target_input, weather_key)
            if lat:
                st.session_state['lat'] = lat
                st.session_state['lon'] = lon
                st.session_state['target'] = target_input
                st.session_state['data_fetched'] = False # Force refresh
                st.rerun()

    lat, lon = st.session_state['lat'], st.session_state['lon']
    target = st.session_state['target']

    m = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB dark_matter")
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.session_state['target'] = "Map Pin"
        st.session_state['data_fetched'] = False # Force refresh
        st.rerun()

    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Mission Target: {target}*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision (Pitch)", "📡 Live Sensor Array", "🧠 Gemini Fusion Core"])

# --- TAB 1: THE PITCH ---
with tab1:
    st.header("Strategic Framework")
    
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b>, California, and Australia face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    These issues are intensifying due to climate change and unstable precipitation patterns.</p>
    <p>Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>. Pilots often fly blind, missing critical seeding windows.</p>
    <p>This aligns critically with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b> for water sustainability.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="pitch-box">
        <h3>💡 2. VisionRain Solution</h3>
        <p>An <b>AI-driven Decision Support Platform</b> that automates the entire seeding lifecycle:</p>
        <ul>
        <li><b>Predictive AI:</b> Identifies seedable clouds via Satellite Fusion.</li>
        <li><b>Optimization:</b> Precision timing for intervention.</li>
        <li><b>Cost Reduction:</b> Eliminates chemical flares via Electro-Coalescence logic.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="pitch-box">
        <h3>🚀 3. Implementation Plan</h3>
        <p><b>Phase 1 (Prototype):</b></p>
        <ul>
        <li><b>Data:</b> NASA Worldview + OpenWeatherMap + Radar.</li>
        <li><b>AI:</b> Gemini Pro Multimodal Fusion.</li>
        <li><b>Output:</b> Real-time GO/NO-GO Authorization.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# --- AUTOMATIC DATA COLLECTION (Runs once per location change) ---
if 'data_fetched' not in st.session_state or st.session_state.get('last_coords') != (lat, lon):
    
    progress_text = "📡 Initializing Global Sensor Array..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # 1. NASA
        my_bar.progress(10, text="🛰️ Downlinking NASA VIIRS...")
        st.session_state['img_nasa'] = get_nasa_feed(lat, lon)
        
        # 2. WINDY LAYERS (Batch Capture)
        my_bar.progress(30, text="🌪️ Capturing Windy Layers...")
        st.session_state['img_wind'] = get_windy_capture(lat, lon, "wind")
        st.session_state['img_radar'] = get_windy_capture(lat, lon, "radar")
        st.session_state['img_clouds'] = get_windy_capture(lat, lon, "clouds")
        st.session_state['img_rain'] = get_windy_capture(lat, lon, "rain")
        st.session_state['img_lclouds'] = get_windy_capture(lat, lon, "lclouds")
        st.session_state['img_hclouds'] = get_windy_capture(lat, lon, "hclouds")
        
        # 3. NUMBERS
        my_bar.progress(90, text="📊 Syncing Telemetry...")
        st.session_state['w_data'] = get_weather_telemetry(lat, lon, weather_key)
        
        st.session_state['last_coords'] = (lat, lon)
        st.session_state['data_fetched'] = True
        my_bar.empty()
        st.rerun() # Refresh to show images
        
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")

# --- TAB 2: THE WALL OF SCREENS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    w = st.session_state.get('w_data')
    if w:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Humidity", f"{w['main']['humidity']}%", "Target > 40%")
        c2.metric("Temperature", f"{w['main']['temp']}°C")
        c3.metric("Pressure", f"{w['main']['pressure']} hPa")
        c4.metric("Wind", f"{w['wind']['speed']} m/s")
    
    st.divider()

    # Visuals Grid (3 Rows)
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.caption("A. Optical Satellite (NASA)")
        if st.session_state.get('img_nasa'): st.image(st.session_state['img_nasa'], use_column_width=True)
        else: st.info("Fetching...")
    with r1c2:
        st.caption("B. Doppler Radar (Windy)")
        if st.session_state.get('img_radar'): st.image(st.session_state['img_radar'], use_column_width=True)
        else: st.info("Fetching Agent...")
    with r1c3:
        st.caption("C. Wind Velocity (Windy)")
        if st.session_state.get('img_wind'): st.image(st.session_state['img_wind'], use_column_width=True)
        else: st.info("Fetching Agent...")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.caption("D. Cloud Density (Windy)")
        if st.session_state.get('img_clouds'): st.image(st.session_state['img_clouds'], use_column_width=True)
        else: st.info("Fetching Agent...")
    with r2c2:
        st.caption("E. Rain Accumulation (Windy)")
        if st.session_state.get('img_rain'): st.image(st.session_state['img_rain'], use_column_width=True)
        else: st.info("Fetching Agent...")
    with r2c3:
        st.caption("F. Low Clouds (Seedable)")
        if st.session_state.get('img_lclouds'): st.image(st.session_state['img_lclouds'], use_column_width=True)
        else: st.info("Fetching Agent...")

# --- TAB 3: GEMINI INTELLIGENCE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.markdown("### 🔍 Operational Criteria Check")
    if w:
        comp_df = pd.DataFrame({
            "Parameter": ["Humidity", "Wind Speed", "Cloud Structure", "Precipitation"],
            "Ideal Condition": ["> 45%", "< 15 m/s", "Convective", "Active Cells"],
            "Actual Data": [
                f"{w['main']['humidity']}%", 
                f"{w['wind']['speed']} m/s", 
                "Analyzing Visuals...", 
                "Scanning Radar..."
            ]
        })
        st.table(comp_df)

    # 2. VISUAL EVIDENCE REPLAY (The Fix for Crash)
    st.caption("Visual Evidence Stream (Sent to Gemini):")
    
    # Create dynamic list of valid images and captions
    valid_imgs = []
    valid_captions = []
    
    possible_layers = [
        ('img_nasa', 'NASA Satellite'),
        ('img_radar', 'Radar'),
        ('img_wind', 'Wind'),
        ('img_clouds', 'Clouds'),
        ('img_lclouds', 'Low Clouds')
    ]
    
    for key, label in possible_layers:
        img = st.session_state.get(key)
        if img:
            valid_imgs.append(img)
            valid_captions.append(label)
    
    if valid_imgs:
        # Show the strip of images Gemini will see
        st.image(valid_imgs, width=150, caption=valid_captions)
    else:
        st.warning("⚠️ Waiting for Sensor Data...")

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not valid_imgs:
            st.error("⚠️ No Visual Data Available.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # SUPER PROMPT
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST.
            Analyze this Multi-Modal Sensor Data for the Saudi Cloud Seeding Program.
            
            --- TARGET ---
            Location: {target} ({lat}, {lon})
            
            --- IDEAL CONDITIONS ---
            - Humidity > 45%
            - Wind < 15 m/s
            - Clouds: **Low Clouds** (Water) are SEEDABLE. **High Clouds** (Ice) are NOT.
            
            --- ACTUAL TELEMETRY ---
            - Humidity: {w['main']['humidity']}%
            - Wind: {w['wind']['speed']} m/s
            
            --- VISUALS (Attached) ---
            Layers Provided: {', '.join(valid_captions)}
            
            --- OUTPUT FORMAT (Markdown) ---
            1. **Visual Analysis:** Describe the cloud formation and radar intensity clearly.
            2. **Criteria Match Table:** Generate a Markdown table comparing Ideal vs Actual.
            3. **Decision:** **GO** or **NO-GO**?
            4. **Scientific Reasoning:** Explain why based on Cloud Type (Low vs High) and Humidity.
            """
            
            with st.spinner("Gemini 2.0 is fusing visual streams + telemetry..."):
                try:
                    res = model.generate_content([prompt] + valid_imgs)
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED: Atmospheric Conditions Optimal</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED: Conditions Unsuitable")
                except Exception as e:
                    st.error(f"AI Error: {e}")
