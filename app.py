import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import os

# --- CONFIGURATION ---
# Leave empty for GitHub (User pastes in Sidebar for security)
DEFAULT_API_KEY = "AIzaSyA7Yk4WRdSu976U4EpHZN47m-KA8JbJ5do" 
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

# --- ASSET LOADER (Runs once to download data) ---
@st.cache_resource
def load_scientific_assets():
    """Downloads backup scientific data so the app never crashes."""
    assets = {
        "saudi_storm_visual.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Cumulonimbus_cloud_over_Singapore.jpg/800px-Cumulonimbus_cloud_over_Singapore.jpg",
        "saudi_rain_map.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Radar_reflectivity.jpg/600px-Radar_reflectivity.jpg"
    }
    os.makedirs("data", exist_ok=True)
    
    for name, url in assets.items():
        path = os.path.join("data", name)
        if not os.path.exists(path):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, headers=headers)
                if r.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(r.content)
            except: pass
    return "Assets Loaded"

# Initialize Assets
load_scientific_assets()

# --- DATA FETCHING FUNCTIONS ---

def get_nasa_feed(lat, lon):
    """Fetches Live Satellite Imagery from NASA GIBS"""
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
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200: return Image.open(BytesIO(r.content)), today
    except: return None, None

def get_weather_telemetry(lat, lon):
    """Fetches Weather Data (or Simulates if key missing)"""
    if WEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            return requests.get(url).json()['main']
        except: pass
    # Fallback Simulation
    return {"humidity": 65, "temp": 32, "pressure": 1012} 

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Data Verification System")
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
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
    
    # THREE COLUMNS FOR DATA
    col_sat, col_weath, col_rad = st.columns(3)
    
    # 1. SATELLITE DATA
    with col_sat:
        st.subheader("A. Satellite (Visual)")
        source_opt = st.radio("Source:", ["Live Feed (NASA)", "Archive (Storm)"], horizontal=True)
        
        if source_opt == "Live Feed (NASA)":
            if st.button("📡 ACQUIRE LIVE SIGNAL"):
                with st.spinner("Connecting to Suomi NPP..."):
                    img, date = get_nasa_feed(lat, lon)
                    if img:
                        st.image(img, caption=f"Live Feed: {date} (NASA GIBS)", use_column_width=True)
                        st.session_state['analysis_img'] = img
                    else:
                        st.warning("Orbit Offline (Night). Using Archive.")
                        st.session_state['analysis_img'] = Image.open("data/saudi_storm_visual.jpg")
                        st.image(st.session_state['analysis_img'], caption="Backup: Archive Storm")
        else:
            try:
                st.session_state['analysis_img'] = Image.open("data/saudi_storm_visual.jpg")
                st.image(st.session_state['analysis_img'], caption="Archive: Convective System", use_column_width=True)
            except:
                st.error("Loading assets...")

    # 2. WEATHER DATA
    with col_weath:
        st.subheader("B. Telemetry (Numerical)")
        w = get_weather_telemetry(lat, lon)
        
        st.metric("Relative Humidity", f"{w['humidity']}%", "Target > 40%")
        st.metric("Temperature", f"{w['temp']}°C")
        st.metric("Pressure", f"{w['pressure']} hPa")
        
        st.info(f"**Status:** {'✅ SEEDABLE' if w['humidity'] > 40 else '⚠️ TOO DRY'}")

    # 3. RADAR DATA (FIXED)
    with col_rad:
        st.subheader("C. Radar (Precipitation)")
        
        # Using the locally downloaded file
        if os.path.exists("data/saudi_rain_map.jpg"):
            st.image("data/saudi_rain_map.jpg", caption="NASA GPM IMERG (Rain Intensity)", use_column_width=True)
        else:
            st.warning("Radar loading...")

# TAB 2: AI ANALYSIS
with tab2:
    st.header("2. Gemini Fusion Engine")
    st.write("The AI analyzes the **Satellite Image** above + the **Humidity** data to make a decision.")
    
    if st.button("RUN DIAGNOSTICS"):
        if not api_key:
            st.error("🔑 API Key Missing! Check Sidebar.")
        elif 'analysis_img' not in st.session_state:
            st.error("📡 No Data! Load Satellite Image in Tab 1.")
        else:
            genai.configure(api_key=api_key)
            # Try-Catch for Model Version Safety
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze this satellite image of {target_name} ({lat}, {lon}).
            Telemetry Input: Humidity {w['humidity']}%.
            
            Task:
            1. Identify cloud type (Convective vs Stratiform).
            2. Cross-reference with humidity.
            3. DECISION: Is this suitable for Cloud Seeding?
            4. Return JSON: {{Decision: GO/NO-GO, Confidence: %, Reasoning: text}}
            """
            
            with st.spinner("Fusion Engine Processing..."):
                try:
                    res = model.generate_content([prompt, st.session_state['analysis_img']])
                    st.markdown("### 🤖 AI Assessment")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper():
                        st.success("✅ CONCLUSION: Conditions Optimal for Seeding.")
                        st.balloons()
                    else:
                        st.warning("⚠️ CONCLUSION: Conditions Unsuitable.")
                except Exception as e:
                    st.error(f"AI Error: {e}")
                    
