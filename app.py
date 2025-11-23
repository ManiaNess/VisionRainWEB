import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium

# --- CONFIGURATION ---
# Leave empty for GitHub (User pastes in Sidebar for security)
DEFAULT_API_KEY = "AIzaSyAZsUnki7M2SJjPYfZ5NHJ8LX3xMtboUDU" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain Data Core", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #60a5fa;}
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST IMAGE LOADER ---
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 1. NASA SATELLITE FETCHER (With Fallback) ---
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
        img = load_image_from_url(full_url)
        if img: return img, "Live Feed (NASA VIIRS)"
    except: pass
    
    # FALLBACK IF NASA FAILS
    backup_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg"
    return load_image_from_url(backup_url), "Archive Backup (Night/Offline)"

# --- 2. WEATHER TELEMETRY ---
def get_weather_telemetry(lat, lon):
    if WEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            return requests.get(url).json()['main']
        except: pass
    return {"humidity": 65, "temp": 32, "pressure": 1012} 

# --- SIDEBAR: INTERACTIVE MISSION CONTROL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Data Verification System")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeatherMap Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Select Target")
    st.caption("Click map to update coordinates")

    # --- INTERACTIVE MAP PICKER (FOLIUM) ---
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728

    m = folium.Map(location=[st.session_state['lat'], st.session_state['lon']], zoom_start=5)
    m.add_child(folium.LatLngPopup()) 
    
    map_data = st_folium(m, height=250, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.rerun()

    # Display Coordinates
    c1, c2 = st.columns(2)
    c1.metric("Lat", f"{st.session_state['lat']:.4f}")
    c2.metric("Lon", f"{st.session_state['lon']:.4f}")
    
    lat = st.session_state['lat']
    lon = st.session_state['lon']
    
    st.success("Target Locked")

# --- MAIN DASHBOARD ---
st.title("VisionRain Data Command Center")
st.markdown(f"### *Sector Analysis Coordinates: {lat:.4f}, {lon:.4f}*")

tab1, tab2 = st.tabs(["📡 Live Global Dynamics (Windy)", "🧠 Gemini Fusion Core"])

# TAB 1: DATA FUSION (WINDY + OWM)
with tab1:
    st.header("1. Real-Time Environmental Monitoring")
    
    col_visual, col_data = st.columns([2, 1])
    
    # LEFT: WINDY.COM EMBED
    with col_visual:
        st.subheader("A. Atmospheric Dynamics (Windy.com)")
        windy_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=800&height=500&zoom=6&level=surface&overlay=rain&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
        components.iframe(windy_url, height=500)
        st.caption(f"Live Radar & Wind @ {lat:.2f}, {lon:.2f}")

    # RIGHT: TELEMETRY (THE NUMBERS)
    with col_data:
        st.subheader("B. Local Telemetry")
        w = get_weather_telemetry(lat, lon)
        
        st.metric("Relative Humidity", f"{w['humidity']}%", "Target > 40%")
        st.metric("Temperature", f"{w['temp']}°C")
        st.metric("Pressure", f"{w['pressure']} hPa")
        
        if w['humidity'] > 40:
            st.success("✅ Conditions: SEEDABLE")
        else:
            st.error("⚠️ Conditions: TOO DRY")
            
        st.markdown("---")
        st.info("""
        **Active Data Streams:**
        1. **Visual:** Windy.com (Interactive)
        2. **Numerical:** OpenWeatherMap API
        3. **Analysis:** Gemini 2.0 Flash
        """)

# TAB 2: AI ANALYSIS (SUPER PROMPT)
with tab2:
    st.header("2. Gemini Fusion Engine")
    st.write("The AI captures a **Static Satellite Snapshot** (NASA) + **Telemetry** (OpenWeather) to perform analysis.")
    
    if st.button("RUN DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # 1. FETCH ASSETS FOR AI
            with st.spinner("Acquiring NASA Optical Downlink for Analysis..."):
                sat_img, status = get_nasa_feed(lat, lon)
                st.image(sat_img, caption=f"Input Source: {status}", width=400)

            # 2. RUN GEMINI
            genai.configure(api_key=api_key)
            try:
                # Trying the newest model first
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # --- THE SUPER PROMPT ---
            prompt = f"""
            ACT AS A LEAD ATMOSPHERIC SCIENTIST for the Saudi Rain Enhancement Program.
            Your task is to analyze multi-modal sensor data to authorize a Cloud Seeding Mission.
            
            --- MISSION CONTEXT ---
            Target Coordinates: {lat}, {lon}
            Objective: Enhance precipitation via Hygroscopic Seeding (Salt Flares).
            
            --- INPUT DATA STREAMS ---
            1. VISUAL SATELLITE (Image 1): Visible Spectrum (VIIRS). Look for cloud texture (lumpy = convective).
            2. TELEMETRY SENSORS:
               - Humidity: {w['humidity']}% (Threshold > 40%)
               - Temperature: {w['temp']}°C
               - Pressure: {w['pressure']} hPa
            
            --- ANALYSIS PROTOCOL (Step-by-Step) ---
            
            STEP 1: VISUAL OBSERVATION
            Describe exactly what you see in the Satellite Image. 
            - Are the clouds flat/hazy (Stratiform) or tall/lumpy (Convective)?
            - Distinct features: Anvils, Overshooting tops, or clear skies?
            
            STEP 2: THERMODYNAMIC VALIDATION
            Does the Humidity ({w['humidity']}%) support droplet growth?
            
            STEP 3: OPERATIONAL DECISION
            - STATUS: [GO / NO-GO]
            - CONFIDENCE: [0-100%]
            - ACTION: Recommend specific seeding agent (Hygroscopic vs Glaciogenic).
            
            Output Format: Structured Markdown.
            """
            
            with st.spinner("Gemini is processing..."):
                try:
                    res = model.generate_content([prompt, sat_img])
                    st.markdown("### 🤖 AI Mission Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.success("✅ MISSION APPROVED")
                        st.balloons()
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
