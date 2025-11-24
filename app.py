import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
import datetime
from io import BytesIO
import urllib.parse

# --- CONFIGURATION ---
# Paste your keys here or use the Sidebar
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
SCREENSHOT_API_KEY = "f9ededa86ff343819371871884196288" # Get this from https://apiflash.com (Free)

st.set_page_config(page_title="VisionRain AI Agent", layout="wide", page_icon="👁️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    h1, h2, h3 {color: #4facfe;}
    .agent-badge {background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. THE VISUAL AGENT (Captures Real Websites) ---
def get_website_screenshot(url, api_key):
    """
    Uses a Headless Browser Agent to visit a URL, wait for render, and take a photo.
    """
    if not api_key: return None
    
    # ApiFlash parameters to ensure Windy loads correctly
    params = {
        'access_key': api_key,
        'url': url,
        'format': 'jpeg',
        'width': 1024,
        'height': 768,
        'delay': 3, # Wait 3 seconds for Windy animation to load
        'quality': 80
    }
    
    try:
        query = urllib.parse.urlencode(params)
        api_url = f"https://api.apiflash.com/v1/urltoimage?{query}"
        r = requests.get(api_url, timeout=15)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    except: pass
    return None

# --- 2. WINDY URL GENERATOR ---
def get_windy_url(lat, lon, layer):
    """Generates the specific Windy.com URL for a location and layer"""
    # Layer map: 'radar' -> 'radar', 'satellite' -> 'satellite', etc.
    # Windy URL format: https://www.windy.com/LAT/LON?LAYER,LAT,LON,ZOOM
    return f"https://www.windy.com/{lat}/{lon}?{layer},{lat},{lon},6"

# --- 3. TELEMETRY (Numerical Truth) ---
def get_telemetry(lat, lon, key):
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
    st.caption("Autonomous Visual Agent")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    screen_key = st.text_input("Screenshot API Key", value=SCREENSHOT_API_KEY, type="password", help="Get free key from apiflash.com")
    
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
                st.rerun()

    lat, lon = st.session_state['lat'], st.session_state['lon']
    st.info(f"Locked: {lat:.4f}, {lon:.4f}")

# --- MAIN UI ---
st.title("VisionRain Agent Interface")
st.markdown(f"### *Targeting Sector: {target_name}*")

tab1, tab2 = st.tabs(["👁️ Live Visual Intelligence", "🧠 Gemini Fusion Core"])

# TAB 1: VISUAL AGENT
with tab1:
    st.header("1. Autonomous Visual Reconnaissance")
    
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        st.markdown("**Select Target Layer:**")
        layer = st.selectbox("Instrument", ["radar", "satellite", "rain", "wind", "clouds", "temp"])
        
        st.info("Click below to dispatch the Visual Agent to Windy.com and capture live intelligence.")
        
        if st.button("📸 CAPTURE LIVE SITE"):
            if not screen_key:
                st.error("Need Screenshot API Key!")
            else:
                with st.spinner(f"Agent browsing Windy.com for {layer}..."):
                    target_url = get_windy_url(lat, lon, layer)
                    img = get_website_screenshot(target_url, screen_key)
                    
                    if img:
                        st.session_state[f'img_{layer}'] = img
                        st.success("Capture Successful")
                    else:
                        st.error("Agent Timeout. Check API Key.")

    with col_view:
        # Display whatever we captured
        if st.session_state.get(f'img_{layer}'):
            st.image(st.session_state[f'img_{layer}'], caption=f"Live Agent Capture: {layer.upper()}", use_column_width=True)
        else:
            st.markdown(f"""
            <div style="height:400px; border:2px dashed #333; display:flex; align-items:center; justify-content:center; color:#555;">
                NO VISUAL DATA FOR {layer.upper()}
            </div>
            """, unsafe_allow_html=True)

# TAB 2: GEMINI FUSION
with tab2:
    st.header("2. Multi-Modal Decision Core")
    
    # Telemetry
    w = get_telemetry(lat, lon, weather_key)
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
            # Check what images we have
            available_images = []
            for l in ["radar", "satellite", "rain", "wind", "clouds"]:
                if st.session_state.get(f'img_{l}'):
                    available_images.append(st.session_state[f'img_{l}'])
            
            if not available_images:
                st.warning("⚠️ No visual evidence captured yet! Go to Tab 1 and capture at least one layer.")
            else:
                # Display what AI sees
                st.write(f"Analyzing {len(available_images)} visual inputs + Telemetry...")
                st.image(available_images, width=200, caption=["Input"] * len(available_images))
                
                genai.configure(api_key=api_key)
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                except:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST.
                Analyze these WEBSITE SCREENSHOTS from Windy.com and the Telemetry.
                
                --- CONTEXT ---
                Location: {target_name} ({lat}, {lon})
                Telemetry: Humidity {w['main']['humidity'] if w else 'N/A'}%
                
                --- TASK ---
                1. Look at the screenshots. Describe the weather patterns (colors, swirls, density).
                2. Note: Windy.com uses RED/YELLOW for intense rain/wind, WHITE for clouds.
                3. Correlate visual intensity with the humidity number.
                4. DECISION: **GO** or **NO-GO** for Cloud Seeding?
                """
                
                inputs = [prompt] + available_images
                
                with st.spinner("Gemini is analyzing website visuals..."):
                    try:
                        res = model.generate_content(inputs)
                        st.markdown("### 🤖 Mission Report")
                        st.write(res.text)
                        if "GO" in res.text.upper(): st.balloons()
                    except Exception as e:
                        st.error(f"AI Error: {e}")
