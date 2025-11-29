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
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .stMetric label {color: #888;}
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    .analysis-text {
        font-family: 'Courier New', monospace;
        color: #00ff80;
        background-color: #111;
        padding: 15px;
        border-left: 3px solid #00ff80;
        border-radius: 5px;
        margin-top: 10px;
    }
    .analysis-fail {
        color: #ff4444;
        border-left: 3px solid #ff4444;
    }
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {"coords": [21.5433, 39.1728], "bias_prob": 0, "humidity_base": 60},
    "Abha (Asir Mountains)": {"coords": [18.2164, 42.5053], "bias_prob": 30, "humidity_base": 70},
    "Riyadh (Central Arid)": {"coords": [24.7136, 46.6753], "bias_prob": -40, "humidity_base": 20},
    "Dammam (Gulf Coast)": {"coords": [26.4207, 50.0888], "bias_prob": -10, "humidity_base": 65},
    "Tabuk (Northern Region)": {"coords": [28.3835, 36.5662], "bias_prob": 10, "humidity_base": 35}
}

# --- CLOUD INFRASTRUCTURE (MOCKED) ---
class BigQueryClient:
    def insert_rows(self, dataset, table, rows): return True
bq_client = BigQueryClient()

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

# --- SCIENTIFIC DATA ENGINE (PIXELATED SIMULATION) ---

def generate_pixel_grid(shape=(100, 100), seed=42, intensity=1.0):
    """
    Generates a realistic 'Digital Sensor' look.
    It creates a coherent cloud shape, but then pixelates it into 10x10 blocks
    so it looks like raw satellite grid data, not soft blobs.
    """
    np.random.seed(seed)
    
    # 1. Generate Base Coherent Noise (The Physics)
    noise = np.random.rand(*shape)
    smooth = gaussian_filter(noise, sigma=15.0) # Very smooth base
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    
    # 2. Pixelate It (The Sensor View)
    # Downsample to 10x10 grid
    block_size = 10
    small_view = smooth[::block_size, ::block_size]
    
    # Upsample back to 100x100 with Nearest Neighbor (Hard edges)
    pixelated = small_view.repeat(block_size, axis=0).repeat(block_size, axis=1)
    
    return pixelated * intensity

def get_simulated_data(sector_name):
    profile = SAUDI_SECTORS[sector_name]
    conditions = [
        {"prob": 85.0, "press": 650, "rad": 12.5, "opt": 15.0, "lwc": 0.005, "rh": 80, "temp": -8.0, "phase": 1}, 
        {"prob": 5.0, "press": 950, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 20, "temp": 28.0, "phase": 0},
        {"prob": 70.0, "press": 350, "rad": 25.0, "opt": 5.0, "lwc": 0.001, "rh": 60, "temp": -35.0, "phase": 2},
    ]
    data = random.choice(conditions).copy()
    data['prob'] += profile['bias_prob'] + random.uniform(-5, 5)
    data['prob'] = max(0.0, min(100.0, data['prob']))
    
    if data['rad'] < 1.0: 
        data['rad'] = 0.0
        data['phase'] = 0
        
    if data['prob'] > 60 and data['rad'] < 14 and data['rad'] > 1 and data['phase'] == 1: 
        data['status'] = "SEEDABLE TARGET"
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

# --- VISUALIZATION ENGINE (GRID SQUARES) ---
def plot_scientific_matrix(data_points):
    """
    Generates the Matrix with HARD PIXEL EDGES and TEXT OVERLAYS.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # Unified Seed
    seed = int(data_points['prob'] * 100)
    master_grid = generate_pixel_grid(seed=seed) 
    
    # Apply Master Grid to Metrics
    vis_prob = master_grid * data_points['prob']
    vis_press = (1.0 - master_grid) * 1000 
    vis_rad = master_grid * data_points['rad']
    vis_phase = master_grid * data_points['phase']
    vis_lwc = master_grid * data_points['lwc']
    vis_ice = master_grid * (data_points['lwc'] * 0.1)
    vis_temp = (1.0 - master_grid) * 30.0 + (master_grid * data_points['temp'])

    plots = [
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": vis_prob, "vmax": 100},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": vis_press, "vmax": 1000},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": vis_rad, "vmax": 30},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": master_grid * data_points['opt'], "vmax": 50},
        {"ax": axes[0,4], "title": "Cloud Phase", "cmap": "cool", "data": vis_phase, "vmax": 2},
        
        {"ax": axes[1,0], "title": "Liquid Water (kg/m³)", "cmap": "Blues", "data": vis_lwc, "vmax": 0.01},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu", "data": vis_ice, "vmax": 0.01},
        {"ax": axes[1,2], "title": "Rel. Humidity (%)", "cmap": "Greens", "data": master_grid * data_points['rh'], "vmax": 100},
        {"ax": axes[1,3], "title": "Vertical Velocity (m/s)", "cmap": "RdBu_r", "data": (master_grid - 0.5) * 5},
        {"ax": axes[1,4], "title": "Temperature (°C)", "cmap": "inferno", "data": vis_temp, "vmax": 40},
    ]

    for p in plots:
        ax = p['ax']
        ax.set_facecolor('#0e1117')
        
        # KEY CHANGE: 'nearest' interpolation creates hard square pixels
        im = ax.imshow(p['data'], cmap=p['cmap'], aspect='auto', interpolation='nearest')
        
        ax.set_title(p['title'], color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # --- EXPLICIT PHASE TEXT ---
        if "Phase" in p['title']:
            phase_val = data_points['phase']
            if phase_val == 1:
                txt = "LIQUID"
                col = "cyan"
            elif phase_val == 2:
                txt = "ICE"
                col = "white"
            else:
                txt = "CLEAR"
                col = "gray"
            
            # Draw heavy text in center
            ax.text(50, 50, txt, color=col, ha="center", va="center", 
                   fontsize=20, fontweight='heavy',
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor=col, boxstyle='round,pad=0.5'))
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
    st.caption("Kingdom Commander | v35.0 (Grid)")
    
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
            "Region": reg, "Priority": "🔴 High" if d['prob'] > 60 else "🟡 Medium" if d['prob'] > 30 else "⚪ Low",
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
            
            prompt = f"""
            ACT AS A SENIOR CLOUD PHYSICIST. Analyze this target.
            DATA: Prob: {current_data['prob']:.1f}%, Radius: {current_data['rad']:.1f}um, Phase: {current_data['phase']}, Temp: {current_data['temp']:.1f}C.
            LOGIC: 
            1. IF Prob > 60% AND Radius 0 -> NO-GO (Ghost Echo).
            2. IF Phase Ice -> NO-GO.
            3. IF Prob > 60% AND Phase Liquid AND Radius 5-14 -> GO.
            OUTPUT: Decision (GO/NO-GO), Analysis, Protocol.
            """
            
            decision, response_text = "PENDING", ""
            if not api_key:
                st.warning("⚠️ Offline Mode (Simulated Response)")
                if current_data['rad'] == 0 and current_data['prob'] > 50:
                    decision = "NO-GO"; response_text = "SENSOR CONTRADICTION DETECTED. Probability high, Radius 0. Ghost Echo. Aborting."
                elif current_data['prob'] > 60 and current_data['rad'] < 14 and current_data['phase'] == 1:
                    decision = "GO"; response_text = "Cross-validation successful. Conditions ideal for collision-coalescence."
                else:
                    decision = "NO-GO"; response_text = "Metrics below threshold."
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    res = model.generate_content([prompt, matrix_img])
                    response_text = res.text
                    decision = "GO" if "GO" in res.text.upper() and "NO-GO" not in res.text.upper() else "NO-GO"
                except Exception as e: decision = "ERROR"; response_text = str(e)

            status.update(label="Complete", state="complete")
        
        if decision == "GO":
            st.balloons()
            st.markdown(f'<div class="analysis-text" style="border-left: 3px solid #00ff80;">✅ <b>MISSION APPROVED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="analysis-text analysis-fail">⛔ <b>MISSION ABORTED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
            
        save_mission_log(current_region, str(current_data), decision, response_text)
        st.toast("Audit Log Saved to BigQuery", icon="☁️")
