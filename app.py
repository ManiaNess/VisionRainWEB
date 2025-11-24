import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import urllib.parse
import time

# --- CONFIGURATION ---
# Paste your keys here or keep them blank to use the sidebar
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "" 
SCREENSHOT_API_KEY = "" # Get this from apiflash.com

st.set_page_config(page_title="VisionRain Visual Agent", layout="wide", page_icon="👁️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px;}
    h1, h2, h3 {color: #4facfe;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. THE VISUAL AGENT (ApiFlash) ---
def get_windy_screenshot(lat, lon, layer, api_key):
    """
    Spins up a Headless Chrome browser in the cloud to photograph Windy.com.
    """
    if not api_key: return None
    
    # We use the 'Embed' version of Windy because it has less UI clutter (buttons/menus)
    # This makes it easier for Gemini to read the actual weather data.
    windy_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=1000&height=600&zoom=6&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    
    params = {
        'access_key': api_key,
        'url': windy_url,
        'format': 'jpeg',
        'width': 1000,
        'height': 600,
        'delay': 4,   # CRITICAL: Wait 4 seconds for the animation to load/spin up
        'quality': 90,
        'no_cookie_banners': 'true',
        'no_ads': 'true'
    }
    
    try:
        query = urllib.parse.urlencode(params)
        # The actual API call that takes the photo
        r = requests.get(f"https://api.apiflash.com/v1/urltoimage?{query}", timeout=20)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 2. TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
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
    st.title("VisionRain")
    st.caption("Autonomous Visual Agent")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    screen_key = st.text_input("Screenshot Key (ApiFlash)", value=SCREENSHOT_API_KEY, type="password")
    
    st.markdown("### 📍 Target Selector")
    target_name = st.text_input("Region Name", "Jeddah")
    
    if 'lat' not in st.session_state: st.session_state['lat'] = 21.5433
    if 'lon' not in st.session_state: st.session_state['lon'] = 39.1728

    if st.button("Find Location"):
        if weather_key:
            new_lat, new_lon = get_coordinates(target_name, weather_key)
            if new_lat:
                st.session_state['lat'] = new_lat
                st.session_state['lon'] = new_lon
                st.success(f"Locked: {target_name}")
                st.rerun()
            else:
                st.error("City not found.")
        else:
            st.warning("Need Weather Key to search!")

    lat, lon = st.session_state['lat'], st.session_state['lon']
    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN DASHBOARD ---
st.title("VisionRain Agent Interface")
st.markdown(f"### *Targeting Sector: {target_name}*")

tab1, tab2 = st.tabs(["👁️ Visual Reconnaissance", "🧠 Gemini Fusion Core"])

# TAB 1: THE VISUAL AGENT
with tab1:
    st.header("1. Autonomous Visual Capture")
    st.write("The system deploys a **Cloud Agent** to browse Windy.com and capture live intelligence.")
    
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        st.info("Select layer to capture:")
        # These are the exact layer names Windy uses
        target_layer = st.selectbox("Instrument Layer", ["radar", "satellite", "rain", "wind", "clouds"])
        
        if st.button("📸 CAPTURE LAYER", type="primary"):
            if not screen_key:
                st.error("Missing ApiFlash Key!")
            else:
                with st.spinner(f"Agent browsing Windy for {target_layer}... (Wait 5s)"):
                    # 1. Take the Screenshot
                    img = get_windy_screenshot(lat, lon, target_layer, screen_key)
                    
                    if img:
                        # 2. Save it to Session State so Gemini can see it later
                        st.session_state[f'img_{target_layer}'] = img
                        st.success(f"Successfully captured {target_layer}!")
                    else:
                        st.error("Agent timed out. Try again.")

    with col_view:
        # Display the captured image if it exists
        if st.session_state.get(f'img_{target_layer}'):
            st.image(st.session_state[f'img_{target_layer}'], caption=f"Agent Capture: {target_layer.upper()}", use_column_width=True)
        else:
            st.markdown("""
                <div style="height:300px; border:2px dashed #555; display:flex; align-items:center; justify-content:center; color:#888;">
                    NO VISUAL DATA CAPTURED
                </div>
            """, unsafe_allow_html=True)

# TAB 2: GEMINI FUSION
with tab2:
    st.header("2. Multi-Modal Decision Core")
    
    # Get Telemetry for Context
    w = get_weather_telemetry(lat, lon, weather_key)
    if w:
        c1, c2, c3 = st.columns(3)
        c1.metric("Humidity", f"{w['main']['humidity']}%")
        c2.metric("Pressure", f"{w['main']['pressure']} hPa")
        c3.metric("Wind", f"{w['wind']['speed']} m/s")
    
    st.divider()
    
    if st.button("RUN FULL MISSION DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # 1. COLLECT ALL CAPTURED IMAGES
            evidence = []
            layer_names = []
            
            for l in ["radar", "satellite", "rain", "wind", "clouds"]:
                if st.session_state.get(f'img_{l}'):
                    evidence.append(st.session_state[f'img_{l}'])
                    layer_names.append(l.upper())
            
            if not evidence:
                st.error("⚠️ No visual evidence found! Please go to Tab 1 and capture at least one layer.")
            else:
                # 2. SHOW WHAT AI IS SEEING
                st.write(f"Gemini is analyzing {len(evidence)} visual inputs: {', '.join(layer_names)}")
                st.image(evidence, width=200, caption=layer_names)
                
                # 3. RUN GEMINI
                genai.configure(api_key=api_key)
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                except:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST.
                Analyze these WEBSITE SCREENSHOTS captured from Windy.com.
                
                --- MISSION CONTEXT ---
                Location: {target_name} ({lat}, {lon})
                Telemetry: Humidity {w['main']['humidity'] if w else 'N/A'}%
                
                --- VISUALS PROVIDED ---
                The attached images are real-time captures of: {', '.join(layer_names)}.
                - RADAR: Shows Rain Intensity (Red/Yellow = Heavy).
                - SATELLITE: Shows Cloud Texture.
                - WIND: Shows Airflow Particles.
                
                --- TASK ---
                1. VISUAL ANALYSIS: Describe what you see in the screenshots.
                   (e.g., "I see a swirling low-pressure system" or "The radar is completely clear").
                2. CORRELATION: Does the visual intensity match the Humidity?
                3. DECISION: **GO** or **NO-GO** for Cloud Seeding?
                4. REASONING: Scientific justification.
                """
                
                inputs = [prompt] + evidence
                
                with st.spinner("Gemini is processing visual intelligence..."):
                    try:
                        res = model.generate_content(inputs)
                        st.markdown("### 🤖 Mission Report")
                        st.write(res.text)
                        if "GO" in res.text.upper(): st.balloons()
                    except Exception as e:
                        st.error(f"AI Error: {e}")
