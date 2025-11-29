import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import time
import random
from io import BytesIO
import folium
from streamlit_folium import st_folium
from scipy.ndimage import gaussian_filter

# --- FIREBASE / FIRESTORE IMPORTS ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    
    /* Metrics */
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .stMetric label {color: #888;}
    
    /* Pitch Box */
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    
    /* Cloud Status Badge */
    .cloud-badge {
        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;
        background-color: #1a1a1a; border: 1px solid #444; color: #ccc;
        margin-bottom: 5px; display: inline-block;
    }
    .status-ok {color: #00ff80;}
    
    /* Analysis Box */
    .analysis-text {
        font-family: 'Courier New', monospace;
        background-color: #111;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 14px;
        line-height: 1.6;
    }
    .analysis-go { border-left: 4px solid #00ff80; color: #00ff80; }
    .analysis-nogo { border-left: 4px solid #ff4444; color: #ff4444; }
    
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {"coords": [21.5433, 39.1728], "bias_prob": 0},
    "Abha (Asir Mountains)": {"coords": [18.2164, 42.5053], "bias_prob": 30},
    "Riyadh (Central Arid)": {"coords": [24.7136, 46.6753], "bias_prob": -40},
    "Dammam (Gulf Coast)": {"coords": [26.4207, 50.0888], "bias_prob": -10},
    "Tabuk (Northern Region)": {"coords": [28.3835, 36.5662], "bias_prob": 10}
}

# --- CLOUD INFRASTRUCTURE SIMULATION ---
class BigQueryClient:
    def insert_rows(self, dataset, table, rows): return True
bq_client = BigQueryClient()

# --- FIRESTORE SETUP ---
if "firestore_db" not in st.session_state: st.session_state.firestore_db = []
def init_firebase():
    if not FIREBASE_AVAILABLE: return None
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                initialize_app(cred)
            else: return None
        return firestore.client()
    except Exception: return None
db = init_firebase()

def save_mission_log(region, stats, decision, reasoning):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "region": region, "stats": str(stats), "decision": decision, "reasoning": reasoning,
        "engine": "VertexAI/Gemini-2.5-Flash"
    }
    st.session_state.firestore_db.append(entry)
    if db:
        try: db.collection("mission_logs").add(entry)
        except: pass
    bq_client.insert_rows("visionrain_logs", "mission_audit", [entry])
def get_mission_logs(): return pd.DataFrame(st.session_state.firestore_db)

# --- SCIENTIFIC DATA ENGINE (REALISTIC STRUCTURE) ---

def generate_structured_grid(shape=(150, 150), seed=42, intensity=1.0):
    """
    Generates a Scientific Sensor Grid.
    Unlike random noise, this creates LARGE, CONTIGUOUS structures that look like
    actual weather systems, then applies a hard pixel grid to them.
    """
    np.random.seed(seed)
    
    # 1. Generate very low frequency noise (Large Shapes)
    # Sigma 25.0 creates big, rolling gradients (Weather Fronts)
    noise_base = np.random.rand(*shape)
    structure = gaussian_filter(noise_base, sigma=25.0)
    
    # 2. Add medium frequency details (Cloud Cells)
    detail = gaussian_filter(np.random.rand(*shape), sigma=5.0)
    
    # Combine: 80% Structure, 20% Detail
    combined = (structure * 0.8) + (detail * 0.2)
    
    # 3. Normalize
    norm = (combined - combined.min()) / (combined.max() - combined.min())
    
    # 4. Thresholding to create "Empty Space" vs "Cloud Mass"
    # This ensures clouds are grouped together, not scattered everywhere
    # We shift the values so anything below 0.4 becomes clear sky
    masked = np.clip((norm - 0.4) * 1.8, 0, 1)
    
    return masked * intensity

def get_simulated_data(sector_name):
    """Simulated physics data with specific edge cases."""
    profile = SAUDI_SECTORS[sector_name]
    
    # Scenario 1: Ideal Seeding Target (Liquid, Uplift, Good LWC)
    case_good = {"prob": 88.0, "press": 650, "rad": 11.5, "opt": 18.0, "lwc": 0.006, "rh": 85, "temp": -9.0, "phase": 1, "w": 3.2}
    # Scenario 2: Clear Sky / Dry
    case_dry = {"prob": 5.0, "press": 980, "rad": 0.0, "opt": 0.2, "lwc": 0.000, "rh": 15, "temp": 32.0, "phase": 0, "w": 0.1}
    # Scenario 3: GHOST ECHO (High Prob, No Water)
    case_ghost = {"prob": 75.0, "press": 800, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 40, "temp": 20.0, "phase": 0, "w": 0.5}
    # Scenario 4: Glaciated (Ice - No Seeding)
    case_ice = {"prob": 92.0, "press": 350, "rad": 22.0, "opt": 25.0, "lwc": 0.001, "rh": 90, "temp": -38.0, "phase": 2, "w": 1.5}
    
    # Weighted Randomness
    r = random.random() * 100 + profile['bias_prob']
    
    if r > 80: data = case_good
    elif r > 60: data = case_ghost 
    elif r > 40: data = case_ice
    else: data = case_dry
    
    # Physics Consistency Check
    if data['rad'] < 1.0: 
        data['rad'] = 0.0
        data['phase'] = 0
        data['lwc'] = 0.0
        
    # Determination
    if data['prob'] > 60 and data['rad'] < 14 and data['rad'] > 1 and data['phase'] == 1: 
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 60 and data['rad'] == 0.0:
        data['status'] = "SENSOR ARTIFACT"
    elif data['prob'] > 40: 
        data['status'] = "MONITORING"
    else: 
        data['status'] = "UNSUITABLE"
        
    return data

def run_kingdom_wide_scan():
    results = {}
    for sector in SAUDI_SECTORS:
        results[sector] = get_simulated_data(sector)
    return results

# --- VISUALIZATION ENGINE (HD PIXELS) ---
def plot_scientific_matrix(data_points):
    """
    Generates the Matrix with 'nearest' interpolation for RAW PIXEL look.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # 1. Master Sensor Grid (150x150)
    seed = int(data_points['prob'] * 100)
    # Using the new structured generator for cohesive cloud masses
    master_grid = generate_structured_grid(shape=(150, 150), seed=seed) 
    
    # 2. Physics Mapping
    # Only show radius/phase where the master grid indicates cloud presence
    rad_grid = master_grid * data_points['rad'] if data_points['rad'] > 0 else np.zeros((150,150))
    phase_grid = master_grid * data_points['phase'] if data_points['phase'] > 0 else np.zeros((150,150))
    
    vis_prob = master_grid * data_points['prob']
    vis_press = (1.0 - master_grid) * 1000 
    vis_lwc = master_grid * data_points['lwc']
    vis_temp = (1.0 - master_grid) * 30.0 + (master_grid * data_points['temp'])

    plots = [
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": vis_prob, "vmax": 100},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": vis_press, "vmax": 1000},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": rad_grid, "vmax": 30},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": master_grid * data_points['opt'], "vmax": 50},
        {"ax": axes[0,4], "title": "Cloud Phase", "cmap": "cool", "data": phase_grid, "vmax": 2},
        
        {"ax": axes[1,0], "title": "Liquid Water (LWC)", "cmap": "Blues", "data": vis_lwc, "vmax": 0.01},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu", "data": vis_lwc/3, "vmax": 0.01},
        {"ax": axes[1,2], "title": "Rel. Humidity (%)", "cmap": "Greens", "data": master_grid * data_points['rh'], "vmax": 100},
        {"ax": axes[1,3], "title": "Vertical Velocity (m/s)", "cmap": "RdBu_r", "data": (master_grid - 0.5) * 5},
        {"ax": axes[1,4], "title": "Temperature (°C)", "cmap": "inferno", "data": vis_temp, "vmax": 40},
    ]

    for p in plots:
        ax = p['ax']
        ax.set_facecolor('#0e1117')
        # 'nearest' = Pixelated Look (The 480p raw sensor aesthetic)
        im = ax.imshow(p['data'], cmap=p['cmap'], aspect='auto', interpolation='nearest')
        ax.set_title(p['title'], color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Explicit Phase Text
        if "Phase" in p['title']:
            phase_val = data_points['phase']
            if phase_val == 1: txt, col = "LIQUID", "cyan"
            elif phase_val == 2: txt, col = "ICE", "white"
            else: txt, col = "CLEAR", "gray"
            ax.text(0.5, 0.5, txt, color=col, ha="center", va="center", transform=ax.transAxes,
                   fontsize=18, fontweight='heavy', alpha=0.9,
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor=col))
        else:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- APP INIT ---
if 'all_sector_data' not in st.session_state:
    st.session_state.all_sector_data = run_kingdom_wide_scan()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v55.0 (Scientific)")
    
    st.markdown("### ☁️ Infrastructure")
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Run</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Vertex AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Storage</div>', unsafe_allow_html=True)

    st.write("---")
    region_options = list(SAUDI_SECTORS.keys())
    selected_region = st.selectbox("Active Sector", region_options, key="selected_region_key")

    if st.button("🔄 FORCE RESCAN"):
        st.session_state.all_sector_data = run_kingdom_wide_scan()
        st.rerun()

    api_key = st.text_input("Gemini API Key", type="password")
    with st.expander("🔒 Admin Logs"):
        if st.text_input("Key", type="password") == "123456": st.dataframe(get_mission_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Pitch", "🛰️ Operations & Surveillance", "🧠 Vertex AI Commander"])

# TAB 1: PITCH
with tab1:
    st.header("Vision 2030: Rain Enhancement Strategy")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Challenge</h3>
    <p>Saudi Arabia faces critical water scarcity. VisionRain uses AI to detect seedable clouds via sensor fusion.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**Solution**\n\nAI-driven, pilotless ecosystem."); c2.warning("**Tech**\n\nVertex AI + BigQuery."); c3.success("**Impact**\n\nAutomated Water Security.")

# TAB 2: OPS
with tab2:
    current_region = st.session_state.get("selected_region_key", region_options[0])
    current_data = st.session_state.all_sector_data[current_region]
    
    c_header, c_stats = st.columns([2, 3])
    with c_header:
        st.header(f"📍 {current_region}")
        st.caption("Live Telemetry Stream (GCS Stream)")
    with c_stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cloud Prob", f"{current_data['prob']:.1f}%", delta="High" if current_data['prob']>60 else "Low")
        s2.metric("Radius", f"{current_data['rad']:.1f} µm", help="Target: < 14µm")
        s3.metric("Phase", "Liquid" if current_data['phase']==1 else "Ice/Mix")
        s4.metric("Status", current_data['status'])

    col_map, col_matrix = st.columns([1, 2])
    with col_map:
        st.markdown("**Live Sector Map**")
        m = folium.Map(location=SAUDI_SECTORS[current_region]['coords'], zoom_start=6, tiles="CartoDB dark_matter")
        for reg_name, info in SAUDI_SECTORS.items():
            d = st.session_state.all_sector_data[reg_name]
            color = "green" if d['prob'] > 60 and d['rad'] > 1 else "orange" if d['prob'] > 30 else "gray"
            folium.Marker(info['coords'], popup=f"{reg_name}", icon=folium.Icon(color=color, icon="cloud", prefix="fa")).add_to(m)
            if reg_name == current_region:
                folium.CircleMarker(info['coords'], radius=20, color="#00e5ff", fill=True, fill_opacity=0.2).add_to(m)
        st_folium(m, height=300, use_container_width=True)

    with col_matrix:
        st.markdown("**Real-Time Microphysics Matrix**")
        matrix_img = plot_scientific_matrix(current_data)
        st.image(matrix_img, use_column_width=True)

    st.divider()
    st.subheader("Kingdom-Wide Surveillance Wall")
    table_data = []
    for reg, d in st.session_state.all_sector_data.items():
        table_data.append({
            "Region": reg, "Priority": "🔴 High" if d['prob'] > 60 and d['rad'] > 1 else "🟡 Medium" if d['prob'] > 30 else "⚪ Low",
            "Probability": d['prob'], "Effective Radius": f"{d['rad']:.1f} µm", "Condition": d['status']
        })
    st.dataframe(pd.DataFrame(table_data).sort_values("Probability", ascending=False), use_container_width=True, hide_index=True)

# TAB 3: GEMINI
with tab3:
    st.header(f"Vertex AI Commander: {current_region}")
    c1, c2 = st.columns([1, 1])
    with c1: st.image(matrix_img, caption="Visual Input Tensor")
    with c2: st.write("### Telemetry Packet"); st.json(current_data)

    if st.button("🚀 REQUEST AUTHORIZATION (GEMINI 2.5 FLASH)", type="primary"):
        with st.status("Initializing AI Pipeline...") as status:
            st.write("1. Establishing Uplink to `gemini-2.5-flash`...")
            
            # --- THE "SCIENTIFIC" PROMPT ---
            prompt = f"""
            ACT AS A SENIOR CLOUD PHYSICIST using Google Cloud Vertex AI. 
            Analyze this target sector: {current_region}.
            
            TELEMETRY:
            - Cloud Probability: {current_data['prob']:.1f}% (Satellite)
            - Effective Radius: {current_data['rad']:.1f} µm (Microphysics)
            - Cloud Phase: {current_data['phase']} (0=Clear, 1=Liquid, 2=Ice)
            - Temp: {current_data['temp']:.1f} C
            - Updraft (w): {current_data['w']} m/s
            - LWC: {current_data['lwc']} kg/m3
            
            SCIENTIFIC LOGIC GATES (Step-by-Step):
            1. MICROPHYSICS: Is LWC sufficient for droplet growth? Is Radius optimal (5-14µm) for coalescence?
            2. THERMODYNAMICS: Is Temp > -15C (Mixed/Warm phase)? If Temp < -20C (Ice), abort.
            3. DYNAMICS: Is Updraft > 0.5 m/s to sustain the seeded parcel?
            4. VALIDATION: If Cloud Prob is High but Radius is 0 -> GHOST ECHO (Artifact). ABORT.
            
            OUTPUT FORMAT:
            Decision: [GO / NO-GO]
            Analysis: [Scientific reasoning referencing LWC, Phase, and Dynamics.]
            Coordinates: {SAUDI_SECTORS[current_region]['coords']}
            Protocol: [Action]
            """
            
            decision, response_text = "PENDING", ""
            if not api_key:
                st.warning("⚠️ Offline Mode (Simulated Response)")
                # Logic Mirror
                if current_data['rad'] == 0 and current_data['prob'] > 50:
                    decision = "NO-GO"; response_text = "**Analysis:** CONTRADICTION. Satellite Prob is high ({current_data['prob']:.0f}%) but Radius is 0. This is a Ghost Echo artifact (Cloud Albedo without Microphysics).\n\n**Decision:** NO-GO"
                elif current_data['prob'] > 60 and current_data['rad'] < 14 and current_data['rad'] > 1 and current_data['phase'] == 1:
                    decision = "GO"; response_text = "**Analysis:** Optimal Microphysics. LWC ({current_data['lwc']}) supports droplet growth. Updraft ({current_data['w']} m/s) sufficient for lift. Radius ({current_data['rad']} µm) ideal for hygroscopic flare interaction.\n\n**Decision:** GO"
                else:
                    decision = "NO-GO"; response_text = "**Analysis:** Thermodynamics unfavorable. Temperature or LWC below seeding threshold.\n\n**Decision:** NO-GO"
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    res = model.generate_content([prompt, matrix_img])
                    response_text = res.text
                    
                    u_text = res.text.upper()
                    if "DECISION: GO" in u_text or "**DECISION:** GO" in u_text:
                        decision = "GO"
                    else:
                        decision = "NO-GO"
                except Exception as e: decision = "ERROR"; response_text = str(e)

            status.update(label="Complete", state="complete")
        
        if decision == "GO":
            st.balloons()
            st.markdown(f'<div class="analysis-text analysis-go">✅ <b>MISSION APPROVED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
            
            st.markdown("### 🚁 Drone Command")
            lat, lon = SAUDI_SECTORS[current_region]['coords']
            # GOOGLE MAPS LINK
            st.link_button("🛰️ CONFIRM & LAUNCH DRONE", f"https://www.google.com/maps/search/?api=1&query={lat},{lon}")
            
        else:
            st.markdown(f'<div class="analysis-text analysis-nogo">⛔ <b>MISSION ABORTED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
            
        save_mission_log(current_region, str(current_data), decision, response_text)
        st.toast("Audit Log Saved to BigQuery", icon="☁️")import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import time
import random
from io import BytesIO
import folium
from streamlit_folium import st_folium
from scipy.ndimage import gaussian_filter

# --- FIREBASE / FIRESTORE IMPORTS ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    
    /* Metrics */
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .stMetric label {color: #888;}
    
    /* Pitch Box */
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    
    /* Cloud Status Badge */
    .cloud-badge {
        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;
        background-color: #1a1a1a; border: 1px solid #444; color: #ccc;
        margin-bottom: 5px; display: inline-block;
    }
    .status-ok {color: #00ff80;}
    
    /* Analysis Box */
    .analysis-text {
        font-family: 'Courier New', monospace;
        background-color: #111;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 14px;
        line-height: 1.6;
    }
    .analysis-go { border-left: 4px solid #00ff80; color: #00ff80; }
    .analysis-nogo { border-left: 4px solid #ff4444; color: #ff4444; }
    
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {"coords": [21.5433, 39.1728], "bias_prob": 0},
    "Abha (Asir Mountains)": {"coords": [18.2164, 42.5053], "bias_prob": 30},
    "Riyadh (Central Arid)": {"coords": [24.7136, 46.6753], "bias_prob": -40},
    "Dammam (Gulf Coast)": {"coords": [26.4207, 50.0888], "bias_prob": -10},
    "Tabuk (Northern Region)": {"coords": [28.3835, 36.5662], "bias_prob": 10}
}

# --- CLOUD INFRASTRUCTURE SIMULATION ---
class BigQueryClient:
    def insert_rows(self, dataset, table, rows): return True
bq_client = BigQueryClient()

# --- FIRESTORE SETUP ---
if "firestore_db" not in st.session_state: st.session_state.firestore_db = []
def init_firebase():
    if not FIREBASE_AVAILABLE: return None
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                initialize_app(cred)
            else: return None
        return firestore.client()
    except Exception: return None
db = init_firebase()

def save_mission_log(region, stats, decision, reasoning):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "region": region, "stats": str(stats), "decision": decision, "reasoning": reasoning,
        "engine": "VertexAI/Gemini-2.5-Flash"
    }
    st.session_state.firestore_db.append(entry)
    if db:
        try: db.collection("mission_logs").add(entry)
        except: pass
    bq_client.insert_rows("visionrain_logs", "mission_audit", [entry])
def get_mission_logs(): return pd.DataFrame(st.session_state.firestore_db)

# --- SCIENTIFIC DATA ENGINE (GROUPED PIXELS) ---

def generate_pixel_grid(shape=(120, 120), seed=42, intensity=1.0):
    """
    Generates a 120x120 Digital Grid with GROUPED CLUSTERS.
    We use a lower frequency noise to create large blobs, then quantize them.
    This avoids the "TV static" look and makes it look like cohesive cloud masses.
    """
    np.random.seed(seed)
    
    # 1. Base Physics - Lower Frequency for Larger Shapes
    # sigma=15.0 makes the blobs big and smooth
    noise = np.random.rand(*shape)
    structure = gaussian_filter(noise, sigma=15.0) 
    
    # 2. Normalize
    norm = (structure - structure.min()) / (structure.max() - structure.min())
    
    # 3. Hard Thresholding to create "Islands"
    # We push values towards extremes to make distinct shapes
    contrast = np.clip((norm - 0.3) * 2.0, 0, 1)
    
    # 4. Tiny bit of grain for realism (less than before)
    sensor_grain = np.random.normal(0, 0.02, shape)
    final = np.clip(contrast + sensor_grain, 0, 1)
    
    return final * intensity

def get_simulated_data(sector_name):
    """Simulated physics data with specific edge cases."""
    profile = SAUDI_SECTORS[sector_name]
    
    # Scenario 1: Ideal Seeding Target (Liquid, Uplift, Good LWC)
    case_good = {"prob": 88.0, "press": 650, "rad": 11.5, "opt": 18.0, "lwc": 0.006, "rh": 85, "temp": -9.0, "phase": 1, "w": 3.2}
    # Scenario 2: Clear Sky / Dry
    case_dry = {"prob": 5.0, "press": 980, "rad": 0.0, "opt": 0.2, "lwc": 0.000, "rh": 15, "temp": 32.0, "phase": 0, "w": 0.1}
    # Scenario 3: GHOST ECHO (High Prob, No Water)
    case_ghost = {"prob": 75.0, "press": 800, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 40, "temp": 20.0, "phase": 0, "w": 0.5}
    # Scenario 4: Glaciated (Ice - No Seeding)
    case_ice = {"prob": 92.0, "press": 350, "rad": 22.0, "opt": 25.0, "lwc": 0.001, "rh": 90, "temp": -38.0, "phase": 2, "w": 1.5}
    
    # Weighted Randomness
    r = random.random() * 100 + profile['bias_prob']
    
    if r > 80: data = case_good
    elif r > 60: data = case_ghost 
    elif r > 40: data = case_ice
    else: data = case_dry
    
    # Physics Consistency Check
    if data['rad'] < 1.0: 
        data['rad'] = 0.0
        data['phase'] = 0
        data['lwc'] = 0.0
        
    # Determination
    if data['prob'] > 60 and data['rad'] < 14 and data['rad'] > 1 and data['phase'] == 1: 
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 60 and data['rad'] == 0.0:
        data['status'] = "SENSOR ARTIFACT"
    elif data['prob'] > 40: 
        data['status'] = "MONITORING"
    else: 
        data['status'] = "UNSUITABLE"
        
    return data

def run_kingdom_wide_scan():
    results = {}
    for sector in SAUDI_SECTORS:
        results[sector] = get_simulated_data(sector)
    return results

# --- VISUALIZATION ENGINE (GROUPED PIXELS) ---
def plot_scientific_matrix(data_points):
    """
    Generates the Matrix with 'nearest' interpolation for RAW PIXEL look.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # 1. Master Sensor Grid (120x120 for 480p look)
    seed = int(data_points['prob'] * 100)
    master_grid = generate_pixel_grid(shape=(120, 120), seed=seed) 
    
    # 2. Physics Mapping
    rad_grid = master_grid * data_points['rad'] if data_points['rad'] > 0 else np.zeros((120,120))
    phase_grid = master_grid * data_points['phase'] if data_points['phase'] > 0 else np.zeros((120,120))
    
    vis_prob = master_grid * data_points['prob']
    vis_press = (1.0 - master_grid) * 1000 
    vis_lwc = master_grid * data_points['lwc']
    vis_temp = (1.0 - master_grid) * 30.0 + (master_grid * data_points['temp'])

    plots = [
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": vis_prob, "vmax": 100},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": vis_press, "vmax": 1000},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": rad_grid, "vmax": 30},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": master_grid * data_points['opt'], "vmax": 50},
        {"ax": axes[0,4], "title": "Cloud Phase", "cmap": "cool", "data": phase_grid, "vmax": 2},
        
        {"ax": axes[1,0], "title": "Liquid Water (LWC)", "cmap": "Blues", "data": vis_lwc, "vmax": 0.01},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu", "data": vis_lwc/3, "vmax": 0.01},
        {"ax": axes[1,2], "title": "Rel. Humidity (%)", "cmap": "Greens", "data": master_grid * data_points['rh'], "vmax": 100},
        {"ax": axes[1,3], "title": "Vertical Velocity (m/s)", "cmap": "RdBu_r", "data": (master_grid - 0.5) * 5},
        {"ax": axes[1,4], "title": "Temperature (°C)", "cmap": "inferno", "data": vis_temp, "vmax": 40},
    ]

    for p in plots:
        ax = p['ax']
        ax.set_facecolor('#0e1117')
        # 'nearest' = Pixelated Look (The 480p raw sensor aesthetic)
        im = ax.imshow(p['data'], cmap=p['cmap'], aspect='auto', interpolation='nearest')
        ax.set_title(p['title'], color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Explicit Phase Text
        if "Phase" in p['title']:
            phase_val = data_points['phase']
            if phase_val == 1: txt, col = "LIQUID", "cyan"
            elif phase_val == 2: txt, col = "ICE", "white"
            else: txt, col = "CLEAR", "gray"
            ax.text(0.5, 0.5, txt, color=col, ha="center", va="center", transform=ax.transAxes,
                   fontsize=18, fontweight='heavy', alpha=0.9,
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor=col))
        else:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- APP INIT ---
if 'all_sector_data' not in st.session_state:
    st.session_state.all_sector_data = run_kingdom_wide_scan()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v52.0 (Grouped)")
    
    st.markdown("### ☁️ Infrastructure")
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Run</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Vertex AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Storage</div>', unsafe_allow_html=True)

    st.write("---")
    region_options = list(SAUDI_SECTORS.keys())
    selected_region = st.selectbox("Active Sector", region_options, key="selected_region_key")

    if st.button("🔄 FORCE RESCAN"):
        st.session_state.all_sector_data = run_kingdom_wide_scan()
        st.rerun()

    api_key = st.text_input("Gemini API Key", type="password")
    with st.expander("🔒 Admin Logs"):
        if st.text_input("Key", type="password") == "123456": st.dataframe(get_mission_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Pitch", "🛰️ Operations & Surveillance", "🧠 Vertex AI Commander"])

# TAB 1: PITCH
with tab1:
    st.header("Vision 2030: Rain Enhancement Strategy")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Challenge</h3>
    <p>Saudi Arabia faces critical water scarcity. VisionRain uses AI to detect seedable clouds via sensor fusion.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**Solution**\n\nAI-driven, pilotless ecosystem."); c2.warning("**Tech**\n\nVertex AI + BigQuery."); c3.success("**Impact**\n\nAutomated Water Security.")

# TAB 2: OPS
with tab2:
    current_region = st.session_state.get("selected_region_key", region_options[0])
    current_data = st.session_state.all_sector_data[current_region]
    
    c_header, c_stats = st.columns([2, 3])
    with c_header:
        st.header(f"📍 {current_region}")
        st.caption("Live Telemetry Stream (GCS Stream)")
    with c_stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cloud Prob", f"{current_data['prob']:.1f}%", delta="High" if current_data['prob']>60 else "Low")
        s2.metric("Radius", f"{current_data['rad']:.1f} µm", help="Target: < 14µm")
        s3.metric("Phase", "Liquid" if current_data['phase']==1 else "Ice/Mix")
        s4.metric("Status", current_data['status'])

    col_map, col_matrix = st.columns([1, 2])
    with col_map:
        st.markdown("**Live Sector Map**")
        m = folium.Map(location=SAUDI_SECTORS[current_region]['coords'], zoom_start=6, tiles="CartoDB dark_matter")
        for reg_name, info in SAUDI_SECTORS.items():
            d = st.session_state.all_sector_data[reg_name]
            color = "green" if d['prob'] > 60 and d['rad'] > 1 else "orange" if d['prob'] > 30 else "gray"
            folium.Marker(info['coords'], popup=f"{reg_name}", icon=folium.Icon(color=color, icon="cloud", prefix="fa")).add_to(m)
            if reg_name == current_region:
                folium.CircleMarker(info['coords'], radius=20, color="#00e5ff", fill=True, fill_opacity=0.2).add_to(m)
        st_folium(m, height=300, use_container_width=True)

    with col_matrix:
        st.markdown("**Real-Time Microphysics Matrix**")
        matrix_img = plot_scientific_matrix(current_data)
        st.image(matrix_img, use_column_width=True)

    st.divider()
    st.subheader("Kingdom-Wide Surveillance Wall")
    table_data = []
    for reg, d in st.session_state.all_sector_data.items():
        table_data.append({
            "Region": reg, "Priority": "🔴 High" if d['prob'] > 60 and d['rad'] > 1 else "🟡 Medium" if d['prob'] > 30 else "⚪ Low",
            "Probability": d['prob'], "Effective Radius": f"{d['rad']:.1f} µm", "Condition": d['status']
        })
    st.dataframe(pd.DataFrame(table_data).sort_values("Probability", ascending=False), use_container_width=True, hide_index=True)

# TAB 3: GEMINI
with tab3:
    st.header(f"Vertex AI Commander: {current_region}")
    c1, c2 = st.columns([1, 1])
    with c1: st.image(matrix_img, caption="Visual Input Tensor")
    with c2: st.write("### Telemetry Packet"); st.json(current_data)

    if st.button("🚀 REQUEST AUTHORIZATION (GEMINI 2.5 FLASH)", type="primary"):
        with st.status("Initializing AI Pipeline...") as status:
            st.write("1. Establishing Uplink to `gemini-2.5-flash`...")
            
            # --- THE "SCIENTIFIC" PROMPT ---
            prompt = f"""
            ACT AS A SENIOR CLOUD PHYSICIST using Google Cloud Vertex AI. 
            Analyze this target sector: {current_region}.
            
            TELEMETRY:
            - Cloud Probability: {current_data['prob']:.1f}% (Satellite)
            - Effective Radius: {current_data['rad']:.1f} µm (Microphysics)
            - Cloud Phase: {current_data['phase']} (0=Clear, 1=Liquid, 2=Ice)
            - Temp: {current_data['temp']:.1f} C
            - Updraft (w): {current_data['w']} m/s
            - LWC: {current_data['lwc']} kg/m3
            
            SCIENTIFIC LOGIC GATES (Step-by-Step):
            1. MICROPHYSICS: Is LWC sufficient for droplet growth? Is Radius optimal (5-14µm) for coalescence?
            2. THERMODYNAMICS: Is Temp > -15C (Mixed/Warm phase)? If Temp < -20C (Ice), abort.
            3. DYNAMICS: Is Updraft > 0.5 m/s to sustain the seeded parcel?
            4. VALIDATION: If Cloud Prob is High but Radius is 0 -> GHOST ECHO (Artifact). ABORT.
            
            OUTPUT FORMAT:
            Decision: [GO / NO-GO]
            Analysis: [Scientific reasoning referencing LWC, Phase, and Dynamics.]
            Coordinates: {SAUDI_SECTORS[current_region]['coords']}
            Protocol: [Action]
            """
            
            decision, response_text = "PENDING", ""
            if not api_key:
                st.warning("⚠️ Offline Mode (Simulated Response)")
                # Logic Mirror
                if current_data['rad'] == 0 and current_data['prob'] > 50:
                    decision = "NO-GO"; response_text = "**Analysis:** CONTRADICTION. Satellite Prob is high ({current_data['prob']:.0f}%) but Radius is 0. This is a Ghost Echo artifact (Cloud Albedo without Microphysics).\n\n**Decision:** NO-GO"
                elif current_data['prob'] > 60 and current_data['rad'] < 14 and current_data['rad'] > 1 and current_data['phase'] == 1:
                    decision = "GO"; response_text = "**Analysis:** Optimal Microphysics. LWC ({current_data['lwc']}) supports droplet growth. Updraft ({current_data['w']} m/s) sufficient for lift. Radius ({current_data['rad']} µm) ideal for hygroscopic flare interaction.\n\n**Decision:** GO"
                else:
                    decision = "NO-GO"; response_text = "**Analysis:** Thermodynamics unfavorable. Temperature or LWC below seeding threshold.\n\n**Decision:** NO-GO"
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    res = model.generate_content([prompt, matrix_img])
                    response_text = res.text
                    
                    u_text = res.text.upper()
                    if "DECISION: GO" in u_text or "**DECISION:** GO" in u_text:
                        decision = "GO"
                    else:
                        decision = "NO-GO"
                except Exception as e: decision = "ERROR"; response_text = str(e)

            status.update(label="Complete", state="complete")
        
        if decision == "GO":
            st.balloons()
            st.markdown(f'<div class="analysis-text analysis-go">✅ <b>MISSION APPROVED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
            
            st.markdown("### 🚁 Drone Command")
            lat, lon = SAUDI_SECTORS[current_region]['coords']
            # GOOGLE MAPS LINK
            st.link_button("🛰️ CONFIRM & LAUNCH DRONE", f"https://www.google.com/maps/search/?api=1&query={lat},{lon}")
            
        else:
            st.markdown(f'<div class="analysis-text analysis-nogo">⛔ <b>MISSION ABORTED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
            
        save_mission_log(current_region, str(current_data), decision, response_text)
        st.toast("Audit Log Saved to BigQuery", icon="☁️")
