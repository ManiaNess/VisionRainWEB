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
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 

st.set_page_config(page_title="VisionRain | Intelligent Planet", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {
        background-color: #1f2937; 
        border: 1px solid #374151; 
        border-radius: 8px; 
        padding: 15px;
    }
    h1, h2, h3 {color: #4facfe;}
    .pitch-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
        margin-bottom: 10px;
    }
    iframe {border-radius: 10px; border: 1px solid #334155;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. MICROLINK AGENT (The "Eyes") ---
def get_windy_screenshot(lat, lon, layer):
    """
    Uses Microlink (Free) to capture a snapshot of Windy.com for Gemini.
    """
    # Windy Embed URL
    target_url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=800&height=600&zoom=6&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1"
    
    # Microlink API (Free Tier, no key needed for low volume)
    api_url = f"https://api.microlink.io?url={urllib.parse.quote(target_url)}&screenshot=true&meta=false&waitFor=5000&viewport.width=800&viewport.height=600"
    
    try:
        r = requests.get(api_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            img_url = data['data']['screenshot']['url']
            return Image.open(BytesIO(requests.get(img_url).content))
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
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("AI-Driven Cloud Seeding")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
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
    
    # Map Picker
    m = folium.Map(location=[lat, lon], zoom_start=5)
    m.add_child(folium.LatLngPopup()) 
    map_data = st_folium(m, height=200, width=280)

    if map_data['last_clicked']:
        st.session_state['lat'] = map_data['last_clicked']['lat']
        st.session_state['lon'] = map_data['last_clicked']['lng']
        st.rerun()

    st.info(f"Coords: {lat:.4f}, {lon:.4f}")

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown(f"### *Sector Analysis: {target_name}*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Real-Time Monitor", "🧠 Gemini Fusion Core"])

# TAB 1: THE PITCH (Strategic Vision)
with tab1:
    st.header("1. Strategic Imperatives")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Problem: Water Scarcity & Climate Resilience</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b>, California, and Australia are facing escalating environmental crises including water scarcity, prolonged droughts, and wildfire escalation. 
    These issues are intensifying due to climate change and unstable precipitation patterns.</p>
    <p>Current cloud seeding operations are <b>manual, expensive, and reactive</b>. Existing systems lack real-time AI analysis, limiting their effectiveness.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="pitch-box">
        <h4>💡 The Solution: VisionRain</h4>
        <p>An <b>AI-driven Decision Support Platform</b> that automates the seeding lifecycle:</p>
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
        <h4>🚀 Impact & Vision 2030</h4>
        <p>This challenge is critically aligned with the <b>Saudi Green Initiative</b> (10 Billion Trees) and Water Sustainability goals.</p>
        <ul>
        <li><b>Economic:</b> Reduces reliance on desalination ($0.50/m³ vs $0.02/m³).</li>
        <li><b>Innovation:</b> Enables scalable, autonomous drone deployment.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# TAB 2: REAL-TIME MONITOR (Windy + Telemetry)
with tab2:
    st.header("2. Live Environmental Dynamics")
    
    # Layer Selector for Windy
    layer = st.selectbox("Select Instrument Layer:", ["radar", "satellite", "rain", "wind", "clouds"])
    
    col_vis, col_dat = st.columns([2, 1])
    
    # LEFT: WINDY EMBED
    with col_vis:
        st.subheader(f"Global Dynamics ({layer.title()})")
        # Windy Embed Code
        ts = int(time.time())
        url = f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}&detailLat={lat}&detailLon={lon}&width=800&height=450&zoom=6&level=surface&overlay={layer}&product=ecmwf&menu=&message=&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=&metricWind=default&metricTemp=default&radarRange=-1&t={ts}"
        components.iframe(url, height=450)
        st.caption("Interactive Visualization powered by Windy.com")

    # RIGHT: TELEMETRY
    with col_dat:
        st.subheader("Local Telemetry")
        w = get_weather_telemetry(lat, lon, weather_key)
        
        if w:
            st.metric("Humidity", f"{w['main']['humidity']}%", "Target > 40%")
            st.metric("Temperature", f"{w['main']['temp']}°C")
            st.metric("Pressure", f"{w['main']['pressure']} hPa")
            st.metric("Wind", f"{w['wind']['speed']} m/s")
            
            st.session_state['ai_humid'] = w['main']['humidity']
            st.session_state['ai_press'] = w['main']['pressure']
        else:
            st.warning("Enter OpenWeatherMap Key in Sidebar.")
            st.session_state['ai_humid'] = "N/A"

# TAB 3: GEMINI FUSION (The AI Agent)
with tab3:
    st.header("3. Gemini Fusion Engine")
    
    st.write("The AI deploys a **Visual Agent** to capture the Windy map and analyze it alongside telemetry.")
    
    if st.button("RUN DEEP DIAGNOSTICS", type="primary"):
        if not api_key:
            st.error("🔑 API Key Missing!")
        else:
            # 1. CAPTURE THE SCREENSHOT (Agent)
            with st.spinner("Deploying Visual Agent to capture Windy Radar..."):
                # We capture the 'Radar' layer specifically for analysis
                screenshot = get_windy_screenshot(lat, lon, "radar")
                
                if screenshot:
                    st.session_state['ai_evidence'] = screenshot
                    st.success("Visual Evidence Acquired")
                else:
                    st.warning("Agent Timeout. Visuals might be unavailable.")

            # 2. SHOW THE EVIDENCE
            st.markdown("### 👁️ AI Visual Input")
            if st.session_state.get('ai_evidence'):
                st.image(st.session_state['ai_evidence'], caption="Real-Time Agent Capture (Radar Layer)", use_column_width=True)
            
            # 3. EXECUTE AI
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            ACT AS A LEAD METEOROLOGIST. Analyze this Multi-Modal Sensor Data.
            
            --- MISSION CONTEXT ---
            Location: {target_name} ({lat}, {lon})
            Objective: Hygroscopic Cloud Seeding.
            
            --- TELEMETRY ---
            - Humidity: {st.session_state.get('ai_humid')}%
            - Pressure: {st.session_state.get('ai_press')} hPa
            
            --- VISUALS (Attached) ---
            Image: Real-time Radar Capture from Windy.com.
            - Colored Blobs = Rain/Storms.
            - Grey/Black = Clear.
            
            --- TASK ---
            1. VISUAL ANALYSIS: Describe the radar intensity seen in the image.
            2. CORRELATION: Does the visual rain match the humidity level?
            3. DECISION: **GO** or **NO-GO**?
            4. REASONING: Scientific justification.
            """
            
            inputs = [prompt]
            if st.session_state.get('ai_evidence'): inputs.append(st.session_state['ai_evidence'])
            
            with st.spinner("Gemini 2.0 is fusing streams..."):
                try:
                    res = model.generate_content(inputs)
                    st.markdown("### 🛰️ Mission Report")
                    st.write(res.text)
                    
                    if "GO" in res.text.upper() and "NO-GO" not in res.text.upper():
                        st.success("✅ MISSION APPROVED")
                        st.balloons()
                    elif "NO-GO" in res.text.upper():
                        st.error("⛔ MISSION ABORTED")
                except Exception as e:
                    st.error(f"AI Error: {e}")
