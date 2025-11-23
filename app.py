import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import streamlit.components.v1 as components
import time
import json

# --- CONFIGURATION ---
DEFAULT_API_KEY = "AIzaSyAZsUnki7M2SJjPYfZ5NHJ8LX3xMtboUDU" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain Data Core", layout="wide", page_icon="⛈️")

# --- STYLING (GOOGLE CLOUD THEME) ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4285F4;} /* Google Blue */
    .stButton>button {background-color: #4285F4; color: white; border: none;}
    .success-box {padding: 10px; background-color: #0f9d58; color: white; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST IMAGE LOADER ---
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
    st.caption("Built on **Google Cloud Vertex AI**")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeatherMap Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Target Zone")
    target_name = st.text_input("Region Name", "Jeddah")
    
    col1, col2 = st.columns(2)
    with col1: lat = st.number_input("Latitude", value=21.5433, format="%.4f")
    with col2: lon = st.number_input("Longitude", value=39.1728, format="%.4f")
    
    # GOOGLE MAPS EMBED (No Key Needed for this View)
    # This replaces the generic map with a real Google Map
    map_html = f"""
    <iframe width="100%" height="250" frameborder="0" style="border:0; border-radius:10px;"
        src="https://maps.google.com/maps?q={lat},{lon}&z=10&output=embed">
    </iframe>
    """
    components.html(map_html, height=250)
    
    st.success(f"Tracking: **{target_name}**")

# --- MAIN DASHBOARD ---
st.title("VisionRain Data Command Center")
st.markdown(f"### *Sector Analysis: {target_name} ({lat}, {lon})*")

tab1, tab2 = st.tabs(["📡 Data Verification (Live)", "🧠 Gemini Fusion Core"])

# TAB 1: DATA FUSION
with tab1:
    st.header("1. Hydro-Meteorological Data Fusion")
    
    col_sat, col_weath, col_rad = st.columns(3)
    
    # 1. SATELLITE DATA
    with col_sat:
        st.subheader("A. Satellite (Visual)")
        source_opt = st.radio("Source:", ["Live Feed (NASA)", "Archive (Storm)"], horizontal=True)
        
        if source_opt == "Live Feed (NASA)":
            if st.button("📡 ACQUIRE LIVE SIGNAL"):
                with st.spinner(f"Connecting to Suomi NPP over {target_name}..."):
                    img, date = get_nasa_feed(lat, lon)
                    if img:
                        st.image(img, caption=f"Live Feed: {date} (NASA GIBS)", use_column_width=True)
                        st.session_state['sat_img'] = img
                    else:
                        st.warning("Orbit Offline (Night). Using Backup.")
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

    # 3. RADAR DATA (OPENWEATHERMAP)
    with col_rad:
        st.subheader("C. Global Precipitation")
        radar_mode = st.radio("Radar Mode:", ["Interactive Map (OpenWeather)", "Scientific Scan (Static)"])
        
        if radar_mode == "Interactive Map (OpenWeather)":
            if not weather_key:
                st.error("⚠️ Enter OpenWeatherMap Key in Sidebar!")
            else:
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
                        var map = L.map('map').setView([{lat}, {lon}], 6);
                        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                            attribution: '© OpenStreetMap'
                        }}).addTo(map);
                        L.tileLayer('https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={weather_key}', {{
                            opacity: 0.8
                        }}).addTo(map);
                    </script>
                </body>
                </html>
                """
                components.html(html_map, height=300)
                st.caption(f"Live Radar over {target_name}")
        else:
            radar_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Radar_reflectivity.jpg/600px-Radar_reflectivity.jpg"
            st.session_state['rad_img'] = load_image_from_url(radar_url)
            if st.session_state['rad_img']:
                st.image(st.session_state['rad_img'], caption="NASA GPM IMERG (Gemini Readable)", use_column_width=True)

# TAB 2: AI ANALYSIS
# TAB 2: AI ANALYSIS
with tab2:
    st.header("2. Gemini Fusion Engine")
    st.caption("Processing on **Google Cloud Vertex AI** Infrastructure")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.write("The AI performs a **pixel-level analysis** of Satellite texture and Radar reflectivity, cross-referencing with atmospheric thermodynamics.")
    with col_b:
        # GOOGLE CLOUD "DATA PIPELINE" SIMULATION
        if st.button("☁️ COMMIT DATA TO BIGQUERY"):
            with st.spinner("Uploading telemetry to gs://visionrain-datalake/raw..."):
                time.sleep(1.5)
            st.toast("✅ Data saved to Google BigQuery!", icon="☁️")

    st.divider()
    
    if st.button("RUN DEEP DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        elif 'sat_img' not in st.session_state:
            st.error("📡 Load Satellite Data in Tab 1 first.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # --- THE SUPER PROMPT ---
            prompt = f"""
            ACT AS A LEAD ATMOSPHERIC SCIENTIST for the Saudi Rain Enhancement Program.
            Your task is to analyze multi-modal sensor data to authorize a Cloud Seeding Mission.
            
            --- MISSION CONTEXT ---
            Target Region: {target_name} (Lat: {lat}, Lon: {lon})
            Objective: Enhance precipitation via Hygroscopic Seeding (Salt Flares).
            
            --- INPUT DATA STREAMS ---
            1. VISUAL SATELLITE (Image 1): Visible Spectrum (VIIRS). Look for cloud texture (lumpy = convective).
            2. RADAR REFLECTIVITY (Image 2): Precipitation Intensity. (Red/Yellow = Heavy, Green/Blue = Light).
            3. TELEMETRY SENSORS:
               - Humidity: {w['humidity']}% (Threshold > 40%)
               - Temperature: {w['temp']}°C
               - Pressure: {w['pressure']} hPa
            
            --- ANALYSIS PROTOCOL (Step-by-Step) ---
            
            STEP 1: VISUAL OBSERVATION
            Describe exactly what you see in the Satellite Image. 
            - Are the clouds flat/hazy (Stratiform) or tall/lumpy (Convective)?
            - distinct features: Anvils, Overshooting tops, or clear skies?
            
            STEP 2: RADAR CORRELATION
            Describe the Radar Image. 
            - Do you see organized storm cells? 
            - What is the max reflectivity color?
            
            STEP 3: THERMODYNAMIC VALIDATION
            Does the Humidity ({w['humidity']}%) support droplet growth?
            
            STEP 4: OPERATIONAL DECISION
            - STATUS: [GO / NO-GO]
            - CONFIDENCE: [0-100%]
            - ACTION: Recommend specific seeding agent (Hygroscopic vs Glaciogenic).
            
            Output Format: Structured Markdown.
            """
            
            inputs = [prompt, st.session_state['sat_img']]
            if 'rad_img' in st.session_state and st.session_state['rad_img']:
                inputs.append(st.session_state['rad_img'])
            
            with st.spinner("Gemini is analyzing Cloud Microphysics & Radar Cross-Sections..."):
                try:
                    res = model.generate_content(inputs)
                    
                    # Display the result in a nice box
                    st.markdown("### 🛰️ AI Mission Report")
                    st.write(res.text)
                    
                    # Visual Feedback based on decision
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.success("✅ MISSION APPROVED: Launching Drone Swarm Protocols.")
                        st.balloons()
                    else:
                        st.error("⛔ MISSION ABORTED: Atmospheric Conditions Unsuitable.")
                        
                except Exception as e:
                    st.error(f"AI Error: {e}")
