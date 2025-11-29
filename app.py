import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import random
from io import BytesIO
import folium
from streamlit_folium import st_folium

# --- SAUDI SECTOR CONFIGURATION (NEW) ---
# Defines coordinates and "Climate Bias" (higher bias = more likely to find rain)
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {
        "coords": [21.5433, 39.1728], 
        "bias_prob": 0,    # Neutral baseline
        "bias_temp": 0, 
        "humidity_base": 60
    },
    "Riyadh (Central Arid)": {
        "coords": [24.7136, 46.6753], 
        "bias_prob": -30,  # Harder to find rain
        "bias_temp": 5,    # Hotter
        "humidity_base": 20
    },
    "Abha (Asir Mountains)": {
        "coords": [18.2164, 42.5053], 
        "bias_prob": 40,   # Very high chance of rain (Monsoon influence)
        "bias_temp": -10,  # Cooler
        "humidity_base": 70
    },
    "Dammam (Gulf Coast)": {
        "coords": [26.4207, 50.0888], 
        "bias_prob": -10, 
        "bias_temp": 2, 
        "humidity_base": 65
    },
    "Tabuk (Northern Region)": {
        "coords": [28.3835, 36.5662], 
        "bias_prob": 10,   # Moderate chance (Winter rains)
        "bias_temp": -5, 
        "humidity_base": 35
    }
}

# --- FIREBASE / FIRESTORE IMPORTS ---
# WRAPPED IN TRY/EXCEPT TO PREVENT CRASHES IF LIB IS MISSING
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Autonomous Core", layout="wide", page_icon="⛈️")

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 15px;}
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    .success-box {background-color: rgba(0, 255, 128, 0.1); border: 1px solid #00ff80; color: #00ff80; padding: 15px; border-radius: 10px;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- FIRESTORE SETUP (Safe Fallback Mode) ---
# If Firebase is missing or no keys, we use a SessionState "Database"

if "firestore_db" not in st.session_state:
    st.session_state.firestore_db = []

def init_firebase():
    """Attempts to initialize Firebase, returns DB or None."""
    if not FIREBASE_AVAILABLE:
        return None
    try:
        # Check if already initialized to prevent 'App already exists' error
        if not firebase_admin._apps:
            # Look for secrets in .streamlit/secrets.toml
            if "firebase" in st.secrets:
                # specific handling for streamlit secrets dictionary
                key_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(key_dict)
                initialize_app(cred)
            else:
                return None # Run in offline mode
        return firestore.client()
    except Exception as e:
        return None

# Initialize DB safely
db = init_firebase()

def save_mission_log(region, stats, decision, reasoning):
    """Saves mission data to Firestore (if avail) AND Session State (Fallback)."""
    # 1. Save to Session State (Always works immediately for the UI)
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "region": region,
        "stats": stats,
        "decision": decision,
        "reasoning": reasoning
    }
    st.session_state.firestore_db.append(entry)
    
    # 2. Save to Firestore (If configured and online)
    if db:
        try:
            db.collection("mission_logs").add(entry)
        except Exception:
            pass # Fail silently to keep app running

def get_mission_logs():
    """Retrieves logs from Session State (primary for demo speed)."""
    return pd.DataFrame(st.session_state.firestore_db)

# --- SCIENTIFIC DATA ENGINE (Simulates .nc/.grib files) ---
# Since we don't have the 500MB physical files, we generate realistic meteorological fields
# based on the user's physics constraints.

def generate_weather_field(shape=(100, 100), seed=42, mode="random"):
    """Generates 2D arrays resembling cloud structures."""
    np.random.seed(seed)
    if mode == "blobs":
        # Gaussian blobs for clouds
        x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
        d = np.sqrt(x*x + y*y)
        sigma, mu = 0.5, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        noise = np.random.normal(0, 0.1, shape)
        return g + noise
    else:
        return np.random.rand(*shape)

def scan_saudi_sector():
    """
    Simulates scanning the Jeddah sector and extracting specific metrics.
    Returns a dictionary of 'Real' values based on the file variables described.
    """
    # Simulate a variety of weather conditions
    conditions = [
        # Case 1: Perfect Seeding Candidate
        {"prob": 85, "press": 650, "rad": 12.5, "opt": 15, "lwc": 0.005, "rh": 70, "temp": -8},
        # Case 2: Dry/No Cloud
        {"prob": 10, "press": 900, "rad": 0.0, "opt": 1, "lwc": 0.000, "rh": 20, "temp": 25},
        # Case 3: Already Raining / Ice Phase
        {"prob": 90, "press": 400, "rad": 22.0, "opt": 30, "lwc": 0.008, "rh": 90, "temp": -25},
    ]
    
    # Randomly select a condition for this "Live Scan"
    data = random.choice(conditions)
    
# ... (Keep Firestore Logic as is) ...

# --- SCIENTIFIC DATA ENGINE (Simulates .nc/.grib files) ---
# ... (Keep generate_weather_field as is) ...

def scan_saudi_sector(sector_name):
    """
    Simulates scanning a specific region with climate biases.
    """
    # Get sector profile
    profile = SAUDI_SECTORS[sector_name]
    
    # Simulate a variety of weather conditions
    conditions = [
        # Case 1: Perfect Seeding Candidate
        {"prob": 85, "press": 650, "rad": 12.5, "opt": 15, "lwc": 0.005, "rh": profile['humidity_base'] + 10, "temp": -8},
        # Case 2: Dry/No Cloud
        {"prob": 10, "press": 900, "rad": 0.0, "opt": 1, "lwc": 0.000, "rh": profile['humidity_base'] - 10, "temp": 25},
        # Case 3: Already Raining / Ice Phase
        {"prob": 90, "press": 400, "rad": 22.0, "opt": 30, "lwc": 0.008, "rh": 90, "temp": -25},
    ]
    
    # Randomly select a condition
    data = random.choice(conditions)
    
    # APPLY GEOGRAPHIC BIAS
    data['prob'] += profile['bias_prob']
    data['temp'] += profile['bias_temp']
    
    # Clamp values to realistic ranges
    data['prob'] = max(0, min(100, data['prob'])) # 0-100%
    data['rh'] = max(5, min(100, data['rh']))
    
    # Add sensor noise
    data['prob'] += random.uniform(-2, 2)
    data['press'] += random.uniform(-10, 10)
    data['rad'] += random.uniform(-0.5, 0.5)
    
    return data

# ... (Keep plot_scientific_matrix as is) ...

# --- SIDEBAR & SETUP ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v20.1")
    
    # NEW: REGION SELECTOR
    st.markdown("### 🎯 Target Sector")
    selected_sector = st.selectbox("Active Region", list(SAUDI_SECTORS.keys()))
    
    api_key = st.text_input("Gemini API Key", type="password", help="Enter Google AI Studio Key")
    
    st.markdown("---")
    st.markdown("### 📡 Telemetry Status")
    col1, col2 = st.columns(2)
    col1.metric("Meteosat-11", "ONLINE", delta_color="normal")
    col2.metric("ERA5 Reanalysis", "SYNCED", delta_color="normal")
    
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        pwd = st.text_input("Access Code", type="password")
        if pwd == "123456":
            st.success("Access Granted")
            df_logs = get_mission_logs()
            if not df_logs.empty:
                st.dataframe(df_logs)
                st.download_button("Export CSV", df_logs.to_csv(), "mission_logs.csv")
            else:
                st.info("No missions logged yet.")
        elif pwd:
            st.error("Invalid Code")

# --- MAIN APP ---
st.title("VisionRain Command Center")

# INITIALIZE SESSION STATE DATA
# Check if sector changed, if so, trigger re-scan
if 'current_sector' not in st.session_state:
    st.session_state.current_sector = selected_sector
    st.session_state.scan_data = scan_saudi_sector(selected_sector)

# If user changed dropdown, auto-refresh data
if st.session_state.current_sector != selected_sector:
    st.session_state.current_sector = selected_sector
    st.session_state.scan_data = scan_saudi_sector(selected_sector)
    st.rerun()

if st.button("🔄 Rescan Sector (Refresh Data)"):
    st.session_state.scan_data = scan_saudi_sector(selected_sector)
    st.rerun()

current_data = st.session_state.scan_data

# TABS
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Pitch", "🗺️ Operations Dashboard", "🧠 Gemini Autopilot"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Vision 2030: The Rain Enhancement Strategy")
    
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 Problem: The Water Scarcity Crisis</h3>
    <p>Saudi Arabia faces extreme heat, drought, and wildfire risks. Current cloud seeding is <b>manual, reactive, and dangerous</b> for pilots.
    Valuable "seedable" cloud formations are missed because analysis takes hours.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**The Solution: VisionRain**\n\nAn AI-driven, pilotless ecosystem that detects seedable clouds in real-time.")
    with col_b:
        st.warning("**The Tech**\n\nFused Meteosat Optical Data with ERA5 Atmospheric Physics for 3D storm reconstruction.")
    with col_c:
        st.success("**The Impact**\n\nSupports Saudi Green Initiative. Reduces cost by 80% (No Manned Flights).")

# --- TAB 2: OPERATIONS ---
with tab2:
    st.header(f"Real-Time Sector Analysis: {selected_sector}")
    
    # Top Section: Map and Key Metrics
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.subheader("Target Identification Map")
        
        # DYNAMIC COORDINATES BASED ON SELECTION
        sector_coords = SAUDI_SECTORS[selected_sector]['coords']
        
        m = folium.Map(location=sector_coords, zoom_start=9, tiles="CartoDB dark_matter")
        
        # Determine Color based on probability
        icon_color = "green" if current_data['prob'] > 60 else "gray"
        
        folium.CircleMarker(
            location=sector_coords,
            radius=50,
            popup=f"Scan Sector: {selected_sector}",
            color="#00e5ff",
            fill=True,
            fill_opacity=0.1
        ).add_to(m)
        
        folium.Marker(
            sector_coords,
            popup=f"Target Candidate\nProb: {current_data['prob']:.1f}%",
            icon=folium.Icon(color=icon_color, icon="cloud", prefix="fa")
        ).add_to(m)
        
        st_folium(m, height=400, use_container_width=True)

    with c2:
        st.subheader("Live Telemetry")
        c2a, c2b = st.columns(2)
        c2a.metric("Probability", f"{current_data['prob']:.1f}%", delta="High Conf" if current_data['prob']>50 else "Low Conf")
        c2b.metric("Pressure", f"{current_data['press']:.0f} hPa")
        c2a.metric("Eff. Radius", f"{current_data['rad']:.1f} µm", help="Ideal < 14µm")
        c2b.metric("Optical Depth", f"{current_data['opt']:.1f}")
        
        st.metric("Liquid Water Content", f"{current_data['lwc']:.4f} kg/m³")
        st.progress(min(current_data['prob']/100, 1.0), text="Seeding Suitability Index")

    st.divider()
    st.subheader("Multi-Spectral Physics Matrix (Meteosat + ERA5)")
    matrix_img = plot_scientific_matrix(current_data)
    st.image(matrix_img, use_column_width=True)

# --- TAB 3: AUTOPILOT ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.markdown("The AI Brain analyzes the visual matrix and the numerical data to make a **GO / NO-GO** decision.")
    
    col_input, col_action = st.columns([1, 1])
    
    with col_input:
        st.image(matrix_img, caption="Visual Input Tensor", use_column_width=True)
        
        # Prepare Data Table for AI
        df_display = pd.DataFrame([
            {"Metric": "Cloud Probability", "Value": f"{current_data['prob']:.1f}", "Unit": "%", "Threshold": "> 50%"},
            {"Metric": "Cloud Top Pressure", "Value": f"{current_data['press']:.1f}", "Unit": "hPa", "Threshold": "400-900"},
            {"Metric": "Effective Radius", "Value": f"{current_data['rad']:.1f}", "Unit": "µm", "Threshold": "< 14"},
            {"Metric": "Temperature", "Value": f"{current_data['temp']:.1f}", "Unit": "°C", "Threshold": "-5 to -15"},
        ])
        st.dataframe(df_display, hide_index=True)

    with col_action:
        st.subheader("Mission Control")
        
        if st.button("🚀 AUTHORIZE DRONE SWARM", type="primary", use_container_width=True):
            
            with st.status("Initializing AI Command Sequence...", expanded=True) as status:
                st.write("📡 Uploading Telemetry to Vertex AI...")
                
                # 1. BUILD PROMPT
                prompt = f"""
                ACT AS A METEOROLOGICAL MISSION COMMANDER. 
                Analyze this data for a Cloud Seeding Mission in Saudi Arabia.

                DATA TELEMETRY:
                - Probability: {current_data['prob']}% (Chance of rain)
                - Pressure: {current_data['press']} hPa
                - Effective Radius: {current_data['rad']} microns (µm)
                - Optical Depth: {current_data['opt']}
                - Temperature: {current_data['temp']} C
                
                LOGIC RULES (Strict Physics):
                1. IF Radius < 14 AND Radius > 5 (Small droplets need growth) -> POSITIVE INDICATOR.
                2. IF Temp is between -5C and -20C (Supercooled liquid water) -> POSITIVE INDICATOR.
                3. IF Probability > 50 -> POSITIVE INDICATOR.
                4. IF Radius > 20 (Already raining) -> ABORT/NO-GO.
                
                OUTPUT FORMAT:
                Decision: [GO or NO-GO]
                Reasoning: [Scientific explanation in 1 sentence]
                Protocol: [Action to take]
                """
                
                # 2. CALL GEMINI (Or Simulate if no key)
                response_text = ""
                decision = "PENDING"
                
                if not api_key:
                    st.warning("⚠️ No API Key Detected - SIMULATING AI RESPONSE")
                    # Simple rule-based simulation for demo purposes
                    if current_data['rad'] < 14 and current_data['prob'] > 50 and current_data['temp'] < 0:
                        decision = "GO"
                        response_text = f"**Decision:** GO\n\n**Reasoning:** Radius {current_data['rad']:.1f}µm indicates harvestable droplets and Temperature {current_data['temp']}°C suggests supercooled water present.\n\n**Protocol:** Deploy Hygroscopic Flares."
                    else:
                        decision = "NO-GO"
                        response_text = f"**Decision:** NO-GO\n\n**Reasoning:** Conditions unfavorable. Radius {current_data['rad']:.1f}µm or Probability {current_data['prob']}% out of bounds.\n\n**Protocol:** Stand down and continue scanning."
                else:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        # Pass both text and image
                        response = model.generate_content([prompt, matrix_img])
                        response_text = response.text
                        decision = "GO" if "GO" in response_text.upper() else "NO-GO"
                    except Exception as e:
                        st.error(f"AI Connection Failed: {e}")
                        decision = "ERROR"

                st.write("🧠 Analyzing Microphysics...")
                status.update(label="Analysis Complete", state="complete", expanded=False)
            
            # 3. DISPLAY RESULT
            if "GO" in decision and "NO-GO" not in decision:
                st.markdown(f"""
                <div class="success-box">
                <h1>✅ MISSION APPROVED</h1>
                {response_text}
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.error("⛔ MISSION ABORTED")
                st.write(response_text)
                
            # 4. SAVE TO DATABASE
            save_mission_log(selected_sector, str(current_data), decision, response_text)
            st.toast("Mission Logged to Database", icon="💾")
