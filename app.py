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

# --- FIREBASE / FIRESTORE IMPORTS (SAFE MODE) ---
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
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {border: 1px solid #333; border-radius: 10px; overflow: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Abha (Asir Mountains)": {
        "coords": [18.2164, 42.5053], 
        "bias_prob": 40, "bias_temp": -10, "humidity_base": 70
    },
    "Jeddah (Red Sea Coast)": {
        "coords": [21.5433, 39.1728], 
        "bias_prob": 0, "bias_temp": 0, "humidity_base": 60
    },
    "Tabuk (Northern Region)": {
        "coords": [28.3835, 36.5662], 
        "bias_prob": 10, "bias_temp": -5, "humidity_base": 35
    },
    "Dammam (Gulf Coast)": {
        "coords": [26.4207, 50.0888], 
        "bias_prob": -10, "bias_temp": 2, "humidity_base": 65
    },
    "Riyadh (Central Arid)": {
        "coords": [24.7136, 46.6753], 
        "bias_prob": -30, "bias_temp": 5, "humidity_base": 20
    }
}

# --- FIRESTORE SETUP ---
if "firestore_db" not in st.session_state:
    st.session_state.firestore_db = []

def init_firebase():
    """Attempts to initialize Firebase, returns DB or None."""
    if not FIREBASE_AVAILABLE: return None
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                initialize_app(cred)
            else:
                return None
        return firestore.client()
    except Exception: return None

db = init_firebase()

def save_mission_log(region, stats, decision, reasoning):
    """Saves mission data to Firestore (if avail) AND Session State."""
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "region": region,
        "stats": str(stats), # Save as string summary
        "decision": decision,
        "reasoning": reasoning
    }
    st.session_state.firestore_db.append(entry)
    if db:
        try: db.collection("mission_logs").add(entry)
        except: pass

def get_mission_logs():
    return pd.DataFrame(st.session_state.firestore_db)

# --- SCIENTIFIC DATA ENGINE (Simulates .nc/.grib files) ---
def generate_weather_field(shape=(100, 100), seed=42, mode="random"):
    np.random.seed(seed)
    if mode == "blobs":
        x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
        d = np.sqrt(x*x + y*y)
        sigma, mu = 0.5, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        noise = np.random.normal(0, 0.1, shape)
        return g + noise
    else:
        return np.random.rand(*shape)

def scan_single_sector(sector_name):
    """Generates data for ONE sector based on its climate profile."""
    profile = SAUDI_SECTORS[sector_name]
    
    # Base Conditions
    conditions = [
        {"prob": 85, "press": 650, "rad": 12.5, "opt": 15, "lwc": 0.005, "rh": profile['humidity_base'] + 10, "temp": -8, "w": 2.5, "phase": 1}, # Liquid
        {"prob": 10, "press": 900, "rad": 0.0, "opt": 1, "lwc": 0.000, "rh": profile['humidity_base'] - 10, "temp": 25, "w": 0.1, "phase": 0}, # Clear
        {"prob": 90, "press": 400, "rad": 22.0, "opt": 30, "lwc": 0.008, "rh": 90, "temp": -25, "w": 5.0, "phase": 2}, # Ice
    ]
    data = random.choice(conditions)
    
    # Apply Bias & Noise
    data['prob'] += profile['bias_prob']
    data['temp'] += profile['bias_temp']
    
    # Clamp
    data['prob'] = max(0, min(100, data['prob']))
    data['rh'] = max(5, min(100, data['rh']))
    
    # Noise
    data['prob'] += random.uniform(-2, 2)
    data['press'] += random.uniform(-10, 10)
    data['rad'] += random.uniform(-0.5, 0.5)
    
    # Determine Status
    if data['prob'] > 60 and data['rad'] < 14 and data['phase'] == 1:
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 50:
        data['status'] = "MONITORING"
    else:
        data['status'] = "UNSUITABLE"
        
    return data

def run_kingdom_wide_scan():
    """Scans ALL sectors and returns a DataFrame of results."""
    results = {}
    for sector in SAUDI_SECTORS:
        results[sector] = scan_single_sector(sector)
    return results

# --- VISUALIZATION ENGINE (2x5 Matrix) ---
def plot_scientific_matrix(data_points):
    """Generates the 2x5 Matrix of Scientific Plots."""
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    fig.patch.set_facecolor('#0e1117')
    
    # Row 1: Satellite (Meteosat)
    # Row 2: ERA5 (Atmospheric)
    
    seed = int(data_points['prob']) # consistent seed per data
    
    plots = [
        # Row 1
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": generate_weather_field(seed=seed, mode="blobs") * data_points['prob']},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": generate_weather_field(seed=seed+1, mode="blobs") * data_points['press']},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": generate_weather_field(seed=seed+2, mode="blobs") * data_points['rad']},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": generate_weather_field(seed=seed+3, mode="blobs") * data_points['opt']},
        {"ax": axes[0,4], "title": "Cloud Phase (0=Clr,1=Liq,2=Ice)", "cmap": "cool", "data": generate_weather_field(seed=seed+4) * data_points['phase']},
        
        # Row 2
        {"ax": axes[1,0], "title": "Liquid Water (kg/m³)", "cmap": "Blues", "data": generate_weather_field(seed=seed+5) * data_points['lwc']},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu", "data": generate_weather_field(seed=seed+6) * (data_points['lwc']/2)},
        {"ax": axes[1,2], "title": "Relative Humidity (%)", "cmap": "Greens", "data": generate_weather_field(seed=seed+7, mode="blobs") * data_points['rh']},
        {"ax": axes[1,3], "title": "Vertical Velocity (m/s)", "cmap": "RdBu", "data": generate_weather_field(seed=seed+8) * data_points['w']},
        {"ax": axes[1,4], "title": "Temperature (°C)", "cmap": "inferno", "data": generate_weather_field(seed=seed+9, mode="blobs") * data_points['temp']},
    ]

    for p in plots:
        ax = p['ax']
        ax.set_facecolor('#0e1117')
        im = ax.imshow(p['data'], cmap=p['cmap'], aspect='auto')
        ax.set_title(p['title'], color="white", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    return Image.open(buf)

# --- APP INIT ---
if 'all_sector_data' not in st.session_state:
    st.session_state.all_sector_data = run_kingdom_wide_scan()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v21.0")
    
    if st.button("🔄 RE-SCAN KINGDOM"):
        st.session_state.all_sector_data = run_kingdom_wide_scan()
        st.rerun()

    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.success("Access Granted")
            df_logs = get_mission_logs()
            if not df_logs.empty:
                st.dataframe(df_logs)
                st.download_button("Export CSV", df_logs.to_csv(), "mission_logs.csv")
            else:
                st.info("No missions logged.")

# --- MAIN UI ---
st.title("VisionRain Command Center")
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Pitch", "🛰️ Surveillance & Ops", "🧠 Gemini Autopilot"])

# TAB 1: PITCH
with tab1:
    st.header("Vision 2030: The Rain Enhancement Strategy")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 Problem: The Water Scarcity Crisis</h3>
    <p>Saudi Arabia faces extreme heat, drought, and wildfire risks. Current cloud seeding is <b>manual, reactive, and dangerous</b> for pilots.
    Valuable "seedable" cloud formations are missed because analysis takes hours.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**Solution: VisionRain**\n\nAI-driven, pilotless ecosystem."); c2.warning("**Tech**\n\nFused Meteosat + ERA5 Data."); c3.success("**Impact**\n\nSupports Saudi Green Initiative.")

# TAB 2: OPS
with tab2:
    st.header("Kingdom-Wide Surveillance Wall")
    
    # 1. PREPARE TABLE DATA
    raw_data = st.session_state.all_sector_data
    table_rows = []
    for region, d in raw_data.items():
        table_rows.append({
            "Region": region,
            "Status": d['status'],
            "Probability": f"{d['prob']:.1f}%",
            "Radius": f"{d['rad']:.1f} µm",
            "Pressure": f"{d['press']:.0f} hPa",
            "Temp": f"{d['temp']:.1f} °C",
            "_raw_prob": d['prob'] # Hidden for sorting
        })
    
    df_overview = pd.DataFrame(table_rows).sort_values("_raw_prob", ascending=False)
    
    # 2. DISPLAY INTERACTIVE TABLE
    st.dataframe(
        df_overview.drop(columns=["_raw_prob"]), 
        use_container_width=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Mission Status",
                help="AI Derived Status",
                validate="^(SEEDABLE TARGET|MONITORING|UNSUITABLE)$"
            ),
            "Probability": st.column_config.ProgressColumn(
                "Cloud Prob",
                format="%s",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True
    )
    
    st.divider()
    
    # 3. SELECTOR FOR DEEP DIVE
    # Default to the highest probability region
    top_region = df_overview.iloc[0]["Region"]
    st.subheader("Deep Analysis Protocol")
    
    c_sel, c_map = st.columns([1, 2])
    with c_sel:
        target_region = st.selectbox("Select Target for Analysis:", df_overview["Region"], index=0)
        target_data = raw_data[target_region]
        
        # Quick Stats for selection
        st.metric("Target Probability", f"{target_data['prob']:.1f}%")
        st.metric("Cloud Phase", "Liquid" if target_data['phase']==1 else "Ice" if target_data['phase']==2 else "Clear")

    with c_map:
        coords = SAUDI_SECTORS[target_region]['coords']
        m = folium.Map(location=coords, zoom_start=8, tiles="CartoDB dark_matter")
        color = "green" if target_data['prob'] > 60 else "red"
        folium.Marker(coords, popup=f"{target_region}", icon=folium.Icon(color=color, icon="cloud")).add_to(m)
        st_folium(m, height=250, use_container_width=True)

    # 4. 2x5 MATRIX
    st.subheader(f"Multi-Spectral Physics Matrix: {target_region}")
    matrix_img = plot_scientific_matrix(target_data)
    st.image(matrix_img, use_column_width=True)

# TAB 3: GEMINI
with tab3:
    st.header(f"Gemini Fusion Engine: {target_region}")
    
    # Show Inputs
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(matrix_img, caption="Visual Input Tensor")
    with c2:
        st.dataframe(pd.DataFrame([target_data]).T, use_container_width=True)

    if st.button("🚀 AUTHORIZE DRONE SWARM", type="primary"):
        with st.status("Initializing AI Command Sequence...") as status:
            st.write("📡 Uploading Telemetry to Vertex AI...")
            
            prompt = f"""
            ACT AS A MISSION COMMANDER. Analyze data for {target_region}.
            
            DATA:
            - Probability: {target_data['prob']:.1f}%
            - Radius: {target_data['rad']:.1f} microns
            - Phase: {target_data['phase']} (1=Liquid, 2=Ice)
            - Temp: {target_data['temp']:.1f} C
            
            RULES:
            1. GO IF: Radius < 14 AND Radius > 5 AND Phase=1 (Liquid).
            2. NO-GO IF: Phase=2 (Ice) OR Probability < 50.
            
            OUTPUT:
            Decision: [GO/NO-GO]
            Reasoning: [1 sentence]
            Protocol: [Action]
            """
            
            decision, response_text = "PENDING", ""
            
            if not api_key:
                st.warning("⚠️ Using Onboard Logic (No API Key)")
                if target_data['prob'] > 60 and target_data['rad'] < 14 and target_data['phase'] == 1:
                    decision = "GO"
                    response_text = f"**Decision:** GO\n\n**Reasoning:** High probability ({target_data['prob']:.1f}%) and optimal liquid radius ({target_data['rad']:.1f}µm).\n\n**Protocol:** Engage Electro-Coalescence."
                else:
                    decision = "NO-GO"
                    response_text = f"**Decision:** NO-GO\n\n**Reasoning:** Conditions unsuitable. Radius {target_data['rad']:.1f}µm or Phase {target_data['phase']} out of bounds.\n\n**Protocol:** Continue Monitoring."
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    res = model.generate_content([prompt, matrix_img])
                    response_text = res.text
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                except Exception as e:
                    decision = "ERROR"; response_text = str(e)

            status.update(label="Analysis Complete", state="complete")
        
        if "GO" in decision and "NO-GO" not in decision:
            st.balloons()
            st.success("✅ MISSION APPROVED")
            st.write(response_text)
        else:
            st.error("⛔ MISSION ABORTED")
            st.write(response_text)
            
        save_mission_log(target_region, str(target_data), decision, response_text)
