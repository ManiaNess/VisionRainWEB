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

st.set_page_config(page_title="VisionRain | AI Weather Modification", layout="wide", page_icon="⛈️")

# --- ULTRA-MODERN STYLING ---
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
    .pitch-card {
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

# --- 1. MICROLINK AGENT (Zoom Earth) ---
def get_zoom_earth_capture(lat, lon, layer):
    """
    Uses Microlink to capture Zoom Earth HD layers.
    Layers: 'radar', 'wind', 'satellite', 'precipitation'
    """
    # Zoom Earth URL structure
    # We use the 'embed' view or clean map view if possible
    target_url = f"https://zoom.earth/maps/{layer}/#view={lat},{lon},6z/overlays=labels:off,lines:off"
    
    # Microlink API (Free Tier)
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(target_url)}&screenshot=true&meta=false&waitFor=4000&viewport.width=800&viewport.height=600"
    
    try:
        r = requests.get(api_url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
    except: pass
    return None

# --- 2. OPENWEATHER TELEMETRY ---
def get_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 3. GEOCODING ---
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
                st.rerun()

    lat, lon = st.session_state['lat'], st.session_state['lon']
    target = st.session_state['target']

    m = folium.Map(location=[lat, lon], zoom_start=5, tiles="CartoDB dark_matter")
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.session_state['target'] = "Map Pin"
        st.rerun()

    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Mission Target: {target}*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision (Pitch)", "📡 Live Sensor Array", "🧠 Gemini Intelligence"])

# --- TAB 1: THE PITCH ---
with tab1:
    st.header("Strategic Framework")
    
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b>, California, and Australia face escalating environmental crises: water scarcity, prolonged droughts, and wildfires. 
    Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>. Pilots often fly blind, missing critical seeding windows.</p>
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
        <li><b>Cost Reduction:</b> Removes chemical flares via Electro-Coalescence logic.</li>
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

    st.markdown("""
    <div class="pitch-box">
    <h3>📈 4. Impact & Priority</h3>
    <b>Why Now?</b> To support the 10 Billion Trees initiative, we need a sustainable water source.
    <br><b>Economic:</b> Reduces reliance on expensive desalination ($0.50/m³ vs $0.02/m³).
    <br><b>Innovation:</b> Enables future autonomous drone swarms.
    </div>
    """, unsafe_allow_html=True)

# --- AUTOMATIC DATA COLLECTION (Runs on Load) ---
# We fetch everything once and cache it in session state
if 'data_fetched' not in st.session_state or st.session_state['last_coords'] != (lat, lon):
    with st.spinner("🛰️ Auto-Deploying Sensor Agents... (Capturing Global Feeds)"):
        # 1. Capture Visuals (Microlink Agent)
        st.session_state['img_sat'] = get_zoom_earth_capture(lat, lon, "satellite") # HD Clouds
        st.session_state['img_radar'] = get_zoom_earth_capture(lat, lon, "radar")   # Rain
        st.session_state['img_wind'] = get_zoom_earth_capture(lat, lon, "wind")     # Wind
        st.session_state['img_rain'] = get_zoom_earth_capture(lat, lon, "precipitation") # Accumulation
        
        # 2. Fetch Numbers
        st.session_state['w_data'] = get_telemetry(lat, lon, weather_key)
        
        st.session_state['last_coords'] = (lat, lon)
        st.session_state['data_fetched'] = True

# --- TAB 2: THE SENSOR ARRAY ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    # Display Telemetry Row
    w = st.session_state.get('w_data')
    if w:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Humidity", f"{w['main']['humidity']}%", "Target > 40%")
        c2.metric("Temperature", f"{w['main']['temp']}°C")
        c3.metric("Pressure", f"{w['main']['pressure']} hPa")
        c4.metric("Wind", f"{w['wind']['speed']} m/s")
    else:
        st.warning("Telemetry Offline (Check API Key)")

    st.divider()

    # Display "The Wall" of Visuals
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.caption("A. HD Satellite (Zoom Earth)")
        if st.session_state.get('img_sat'): st.image(st.session_state['img_sat'], use_column_width=True)
        else: st.warning("Loading...")
    
    with col2:
        st.caption("B. Precipitation Radar")
        if st.session_state.get('img_radar'): st.image(st.session_state['img_radar'], use_column_width=True)
        else: st.warning("Loading...")

    with col3:
        st.caption("C. Wind Velocity")
        if st.session_state.get('img_wind'): st.image(st.session_state['img_wind'], use_column_width=True)
        else: st.warning("Loading...")

    with col4:
        st.caption("D. Rain Accumulation")
        if st.session_state.get('img_rain'): st.image(st.session_state['img_rain'], use_column_width=True)
        else: st.warning("Loading...")

# --- TAB 3: GEMINI INTELLIGENCE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    # 1. SHOW DATA INPUTS (Transparency)
    st.markdown("### 🔍 AI Input Manifest")
    
    # Comparison Table (Ideal vs Actual)
    if w:
        comp_df = pd.DataFrame({
            "Parameter": ["Humidity", "Wind Speed", "Cloud Type", "Radar"],
            "Ideal Condition": ["> 45%", "< 15 m/s", "Convective (Lumpy)", "Active Cells"],
            "Actual Data": [
                f"{w['main']['humidity']}%", 
                f"{w['wind']['speed']} m/s", 
                "Analyzing...", 
                "Scanning..."
            ]
        })
        st.table(comp_df)

    # Show the images again so user knows what AI is seeing
    st.caption("Visual Evidence Stream (Sent to Gemini):")
    img_list = [st.session_state.get(k) for k in ['img_sat', 'img_radar', 'img_wind', 'img_rain']]
    valid_imgs = [i for i in img_list if i is not None]
    
    if valid_imgs:
        st.image(valid_imgs, width=150, caption=["Satellite", "Radar", "Wind", "Rain"])
    else:
        st.error("⚠️ No visual data captured. Please wait or refresh.")

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not valid_imgs:
            st.error("⚠️ Waiting for Satellite Downlink...")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST.
            Analyze this Multi-Modal Sensor Data for the Saudi Cloud Seeding Program.
            
            --- TARGET ---
            Location: {target} ({lat}, {lon})
            
            --- IDEAL CONDITIONS ---
            - Humidity > 45%
            - Wind < 15 m/s
            - Clouds: Convective (Cumulus) preferred.
            
            --- ACTUAL TELEMETRY ---
            - Humidity: {w['main']['humidity']}%
            - Wind: {w['wind']['speed']} m/s
            
            --- VISUALS (Attached) ---
            1. SATELLITE: Check cloud texture.
            2. RADAR: Check rain intensity (Colors).
            3. WIND: Check flow.
            
            --- OUTPUT FORMAT ---
            1. **Visual Analysis:** Describe the cloud formation and radar intensity clearly.
            2. **Data Sync:** Compare Actual vs Ideal conditions.
            3. **Decision:** **GO** or **NO-GO**?
            4. **Scientific Reasoning:** Explain why.
            """
            
            with st.spinner("Gemini 2.0 is fusing 4 visual streams + telemetry..."):
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
