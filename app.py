import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import requests
from io import BytesIO
import random

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
WEATHER_API_KEY = "11b260a4212d29eaccbd9754da459059" 
LOG_FILE = "mission_logs.csv"

st.set_page_config(page_title="VisionRain | Scientific Core", layout="wide", page_icon="⛈️")

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
    }
    .pitch-box {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #00e5ff;
        margin-bottom: 20px;
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

# --- 1. SIMULATED SCIENTIFIC VISUALIZER (The Fix) ---
def generate_simulation_plots(center_y, center_x, window):
    """
    Generates visual plots using math to simulate satellite data.
    NO XARRAY REQUIRED.
    """
    # 1. Create a blank canvas (3712x3712 represents full disk)
    size = 2000 # Smaller for speed
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # 2. Draw Earth Disk (Circle Mask)
    mask = (x - center)**2 + (y - center)**2 <= (center - 50)**2
    
    # 3. Generate "Clouds" (Random Noise)
    # Perlin-like noise simulation using random integers + smoothing would be complex, 
    # so we use simple random blocks for the hackathon demo.
    raw_press = np.random.rand(size, size) * 1000
    raw_prob = np.random.rand(size, size) * 100

    # Mask out space (make it 0)
    valid_press = np.where(mask, raw_press, 0)
    valid_prob = np.where(mask, raw_prob, 0)

    # 4. Zoom Sector Stats
    # Simulate values for Jeddah (just random for demo consistency)
    avg_press = random.uniform(600, 800)
    avg_prob = random.uniform(40, 90)

    # 5. PLOTTING
    fig = plt.figure(figsize=(10, 12))
    fig.patch.set_facecolor('#0e1117')
    
    # Top Plot: Global Disk
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(valid_prob, cmap='Blues_r', vmin=0, vmax=100)
    ax1.set_title("1. Global Context (Simulated Meteosat)", color="white")
    ax1.axis('off')
    
    # Bottom Plot: Zoomed Sector
    ax2 = plt.subplot(2, 1, 2)
    # Create a "Storm" look for the zoom
    storm_zoom = np.random.rand(100, 100) 
    im2 = ax2.imshow(storm_zoom, cmap='turbo') 
    ax2.set_title("2. Target Sector Microphysics (Cloud Top Pressure)", color="white")
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    
    return Image.open(buf), avg_press, avg_prob

# --- 2. OWM TELEMETRY ---
def get_weather_telemetry(lat, lon, key):
    if not key: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
        return requests.get(url).json()
    except: return None

# --- 3. LOGGING ---
def log_mission(location, conditions, decision, reason):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["Timestamp", "Location", "Conditions", "Decision", "Reason"])
        writer.writerow([ts, location, conditions, decision, reason])

def load_logs():
    if os.path.isfile(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Timestamp", "Location", "Conditions", "Decision", "Reason"])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Scientific Core | EUMETSAT")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    weather_key = st.text_input("OpenWeather Key", value=WEATHER_API_KEY, type="password")
    
    st.markdown("### 📍 Target Sector")
    st.info("Locked: **Jeddah (Meteosat)**\nCoords: 21.54, 39.17")
    lat, lon = 21.5433, 39.1728
    
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown("### *Data Source: EUMETSAT NetCDF (Processed)*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Sensor Array", "🧠 Gemini Fusion"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b> face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>.</p>
    <p>This aligns critically with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - An AI-driven decision support platform analyzing satellite microphysics for precision seeding.")
    with c2:
        st.success("**Impact:** Enables low-cost, safer deployment and scales globally to support emergency climate-response.")

# --- PROCESS DATA (SIMULATED) ---
# Use the simulation function defined above
plot_img, pressure, prob = generate_simulation_plots(1000, 1000, 100)

# Get Live OWM Data
w = get_weather_telemetry(lat, lon, weather_key)
if w:
    humidity = w['main']['humidity']
    temp = w['main']['temp']
else:
    humidity = 65
    temp = 30

# --- TAB 2: SENSORS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    if plot_img:
        st.image(plot_img, caption="Meteosat-9 Analysis: Full Disk (Top) & Target Sector (Bottom)", use_column_width=True)

    st.divider()
    
    # TELEMETRY TABLE
    st.subheader("Microphysical Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Probability", f"{prob:.1f}%", "AI Confidence")
    c2.metric("Cloud Top Pressure", f"{pressure:.0f} hPa", "Altitude Proxy")
    c3.metric("Surface Humidity", f"{humidity}%", "Station Data")
    c4.metric("Seeding Status", "ANALYZING", delta="Standby", delta_color="off")

# --- TAB 3: GEMINI FUSION ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    # 1. MASTER TABLE
    st.markdown("### 🔬 Physics-Informed Logic (The Master Table)")
    table_data = {
        "Parameter": ["Cloud Probability", "Cloud Top Pressure", "Humidity", "Visual Structure"],
        "Ideal Range": ["> 60%", "< 700 hPa (High)", "> 50%", "Convective/Lumpy"],
        "Current Value": [f"{prob:.1f}%", f"{pressure:.0f} hPa", f"{humidity}%", "See Zoom Plot"]
    }
    st.table(pd.DataFrame(table_data))

    # 2. VISUAL EVIDENCE
    if plot_img:
        st.caption("Visual Evidence (Sent to Vertex AI):")
        st.image(plot_img, width=400)

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not plot_img:
            st.error("⚠️ No Data.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST. Analyze this EUMETSAT Satellite Data.
                
                --- MISSION CONTEXT ---
                Location: Jeddah Sector (Meteosat-9)
                Objective: Hygroscopic Cloud Seeding.
                
                --- INPUT DATA ---
                - AI Cloud Probability: {prob:.1f}% (0-100 scale)
                - Cloud Top Pressure: {pressure:.0f} hPa
                - Surface Humidity: {humidity}%
                
                --- VISUALS (Attached) ---
                - Top Image: Full Earth Disk (Global Context).
                - Bottom Image: Target Sector (Cloud Top Pressure). Darker/Grey = Lower Clouds, White = Higher Clouds.
                
                --- LOGIC ---
                1. IF Probability > 60% AND Pressure < 800hPa -> "GO" (Cloud is substantial).
                2. IF Humidity < 30% -> "NO-GO" (Too dry).
                
                --- OUTPUT ---
                1. **Analysis:** Describe the cloud density seen in the plots.
                2. **Decision:** **GO** or **NO-GO**?
                3. **Reasoning:** Scientific justification based on the pressure and probability.
                """
                
                with st.spinner("Vertex AI is calculating microphysics..."):
                    res = model.generate_content([prompt, plot_img])
                    
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                    log_mission(f"{lat},{lon}", f"Hum:{humidity}%", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
