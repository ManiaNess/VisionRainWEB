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
    .status-warn {color: #ffaa00;}
    
    /* Map Container */
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {
        "coords": [21.5433, 39.1728], 
        "bias_prob": 0, "bias_temp": 0, "humidity_base": 60
    },
    "Abha (Asir Mountains)": {
        "coords": [18.2164, 42.5053], 
        "bias_prob": 30, "bias_temp": -10, "humidity_base": 70
    },
    "Riyadh (Central Arid)": {
        "coords": [24.7136, 46.6753], 
        "bias_prob": -40, "bias_temp": 5, "humidity_base": 20
    },
    "Dammam (Gulf Coast)": {
        "coords": [26.4207, 50.0888], 
        "bias_prob": -10, "bias_temp": 2, "humidity_base": 65
    },
    "Tabuk (Northern Region)": {
        "coords": [28.3835, 36.5662], 
        "bias_prob": 10, "bias_temp": -5, "humidity_base": 35
    }
}

# --- GOOGLE CLOUD ARCHITECTURE SIMULATION ---
class BigQueryClient:
    def insert_rows(self, dataset, table, rows):
        time.sleep(0.1) 
        return True

class CloudStorageClient:
    def fetch_satellite_data(self, region):
        time.sleep(0.2)
        return True

bq_client = BigQueryClient()
gcs_client = CloudStorageClient()

# --- FIRESTORE SETUP (Using session state as proxy) ---
if "firestore_db" not in st.session_state:
    st.session_state.firestore_db = []

def save_mission_log(region, stats, decision, reasoning):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "region": region,
        "stats": str(stats),
        "decision": decision,
        "reasoning": reasoning,
        "engine": "VertexAI/Gemini-2.5-Flash"
    }
    st.session_state.firestore_db.append(entry)
    bq_client.insert_rows("visionrain_logs", "mission_audit", [entry])

def get_mission_logs():
    return pd.DataFrame(st.session_state.firestore_db)

# --- SCIENTIFIC DATA ENGINE (Simulation) ---
def scan_single_sector(sector_name):
    profile = SAUDI_SECTORS[sector_name]
    
    conditions = [
        # Ideal Storm
        {"prob": 85.0, "press": 650, "rad": 12.5, "opt": 15.0, "lwc": 0.005, "rh": 80, "temp": -8.0, "w": 2.5, "phase": 1}, 
        # Clear Sky
        {"prob": 5.0, "press": 950, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 20, "temp": 28.0, "w": 0.1, "phase": 0},
        # High Ice Cloud
        {"prob": 70.0, "press": 350, "rad": 25.0, "opt": 5.0, "lwc": 0.001, "rh": 60, "temp": -35.0, "w": 0.5, "phase": 2},
    ]
    
    data = random.choice(conditions).copy()
    data['prob'] += profile['bias_prob']
    data['rh'] = profile['humidity_base'] + random.uniform(-10, 10)
    data['temp'] += profile['bias_temp']
    data['prob'] += random.uniform(-5, 5)
    
    # --- CRITICAL RANGE CHECKS ---
    data['prob'] = max(0.0, min(100.0, data['prob'])) 
    data['rh'] = max(5.0, min(100.0, data['rh']))
    
    if data['prob'] > 60 and data['rad'] < 14 and data['phase'] == 1:
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 40:
        data['status'] = "MONITORING"
    else:
        data['status'] = "UNSUITABLE"
        
    return data

def run_kingdom_wide_scan():
    """Scans ALL sectors and returns a DataFrame of results."""
    with st.spinner("Fetching NetCDF Packets from Cloud Storage..."):
        gcs_client.fetch_satellite_data("all")
        
    results = {}
    for sector in SAUDI_SECTORS:
        results[sector] = scan_single_sector(sector)
    return results

# --- APP STATE INIT ---
if 'all_sector_data' not in st.session_state:
    st.session_state.all_sector_data = run_kingdom_wide_scan()
    sorted_regions = sorted(st.session_state.all_sector_data.items(), key=lambda x: x[1]['prob'], reverse=True)
    st.session_state.selected_region = sorted_regions[0][0]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v25.0 (Simulation Stable)")
    
    st.markdown("### ☁️ Cloud Architecture")
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Run (App)</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Vertex AI (Gemini)</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> BigQuery (Logs)</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Storage (Data)</div>', unsafe_allow_html=True)

    st.write("---")
    
    # REGION SELECTOR
    st.markdown("### 📡 Active Sector")
    region_options = list(SAUDI_SECTORS.keys())
    if st.session_state.selected_region not in region_options:
        st.session_state.selected_region = region_options[0]
        
    selected = st.selectbox("Select Region to Monitor", region_options, 
                           index=region_options.index(st.session_state.selected_region))
    
    if selected != st.session_state.selected_region:
        st.session_state.selected_region = selected
        st.rerun()

    if st.button("🔄 FORCE RESCAN"):
        st.session_state.all_sector_data = run_kingdom_wide_scan()
        st.rerun()

    st.write("---")
    api_key = st.text_input("Gemini API Key", type="password")
    
    with st.expander("🔒 Admin Logs (BigQuery)"):
        if st.text_input("Admin Key", type="password") == "123456":
            st.dataframe(get_mission_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Pitch", "🛰️ Operations & Surveillance", "🧠 Vertex AI Commander"])

# TAB 1: PITCH
with tab1:
    st.header("Vision 2030: The Rain Enhancement Strategy")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Challenge</h3>
    <p>Saudi Arabia faces critical water scarcity. Current seeding operations are <b>manual and reactive</b>. 
    VisionRain uses AI to detect short-lived seedable clouds across the Kingdom in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.info("**Solution**\n\nAI-driven, pilotless ecosystem on Cloud Run.")
    c2.warning("**Tech**\n\nVertex AI + BigQuery + Cloud Storage.")
    c3.success("**Impact**\n\nAutomated Water Security.")

    st.subheader("Google Cloud Platform Architecture")
    st.markdown("""
    | Component | GCP Service | Function |
    | :--- | :--- | :--- |
    | **The Brain** | **Vertex AI (Gemini)** | Analyzes microphysics for GO/NO-GO decisions. |
    | **The App** | **Cloud Run** | Auto-scaling dashboard hosting. |
    | **Data Lake** | **Cloud Storage** | Stores raw Meteosat NetCDF files. |
    | **Audit Logs** | **BigQuery** | Immutable flight recorder for every mission. |
    """)

# TAB 2: OPS
with tab2:
    # --- SECTION A: SELECTED REGION VISUALS (TOP) ---
    current_region = st.session_state.selected_region
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

    # MAP & METRICS
    col_map, col_matrix = st.columns([1, 2])
    
    with col_map:
        st.markdown("**Live Sector Map**")
        # Zoomed-in Map
        m = folium.Map(location=SAUDI_SECTORS[current_region]['coords'], zoom_start=8, tiles="CartoDB dark_matter")
        
        for region_name, info in SAUDI_SECTORS.items():
            r_data = st.session_state.all_sector_data[region_name]
            color = "green" if r_data['prob'] > 60 else "orange" if r_data['prob'] > 30 else "gray"
            tooltip_html = f"<b>{region_name}</b><br>Prob: {r_data['prob']:.1f}%"
            
            folium.Marker(
                info['coords'], popup=tooltip_html, tooltip=f"{region_name}",
                icon=folium.Icon(color=color, icon="cloud", prefix="fa")
            ).add_to(m)
            
            if region_name == current_region:
                folium.CircleMarker(info['coords'], radius=20, color="#00e5ff", fill=True, fill_opacity=0.2).add_to(m)
                
        st_folium(m, height=450, use_container_width=True)

    with col_matrix:
        st.markdown("**Full Microphysics Data**")
        # Numerical data display instead of the 2x5 visual
        st.dataframe(
            pd.DataFrame([current_data]).T.rename(columns={0: "Value"}),
            use_container_width=True
        )

    st.divider()

    # --- SECTION B: KINGDOM WIDE SURVEILLANCE TABLE (BOTTOM) ---
    st.subheader("Kingdom-Wide Surveillance Wall")
    
    table_data = []
    for reg, d in st.session_state.all_sector_data.items():
        table_data.append({
            "Region": reg,
            "Priority": "🔴 High" if d['prob'] > 60 else "🟡 Medium" if d['prob'] > 30 else "⚪ Low",
            "Probability": d['prob'],
            "Effective Radius": f"{d['rad']:.1f} µm",
            "Cloud Pressure": f"{d['press']:.0f} hPa",
            "Temp": f"{d['temp']:.1f} °C",
            "Condition": "Seedable" if d['status'] == "SEEDABLE TARGET" else "Wait"
        })
    
    df_table = pd.DataFrame(table_data).sort_values("Probability", ascending=False)
    
    st.dataframe(
        df_table, use_container_width=True,
        column_config={
            "Probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0, max_value=100),
        }, hide_index=True
    )

# TAB 3: GEMINI
with tab3:
    st.header(f"Vertex AI Commander: {current_region}")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        # st.image(matrix_img, caption="Visual Input Tensor (NetCDF Visualized)") # Removed
        st.write("### AI Input Metrics")
        st.json(current_data)
        
    with c2:
        st.write("### Decision Rules")
        st.markdown("""
        * **Radius:** < 14 µm (Needed for seeding).
        * **Temp:** -5°C to -15°C (Supercooled water required).
        * **Phase:** Liquid (Ice is a No-Go).
        """)

    if st.button("🚀 REQUEST AUTHORIZATION (VERTEX AI)", type="primary"):
        with st.status("Initializing Vertex AI Pipeline...") as status:
            status.update(label="1. Fetching Model: `gemini-2.5-flash`...")
            
            prompt = f"""
            ACT AS A METEOROLOGIST. Analyze {current_region}.
            DATA: {json.dumps(current_data)}.
            RULES: GO IF Radius < 14 AND Radius > 5 AND Phase=1 (Liquid). NO-GO IF Phase=2 (Ice) OR Prob < 50.
            OUTPUT: Decision (GO/NO-GO), Reasoning, Protocol.
            """
            
            decision = "PENDING"
            response_text = ""
            
            if not api_key:
                st.warning("⚠️ Offline Mode (Simulated Response)")
                time.sleep(1)
                if current_data['prob'] > 60 and current_data['rad'] < 14 and current_data['phase'] == 1:
                    decision = "GO"
                    response_text = "Vertex AI Confidence: 98%. Conditions Optimal for Hygroscopic Seeding."
                else:
                    decision = "NO-GO"
                    response_text = f"Vertex AI Confidence: 95%. Conditions Unfavorable (Radius {current_data['rad']:.1f}µm)."
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    status.update(label="2. Awaiting Gemini Decision...")
                    res = model.generate_content(prompt)
                    response_text = res.text
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                except Exception as e:
                    decision = "ERROR"; response_text = str(e)

            status.update(label="3. Logging Decision to BigQuery...", state="running")
            save_mission_log(current_region, current_data, decision, response_text)
            status.update(label="Complete", state="complete")
            
            if "GO" in decision:
                st.success(f"✅ MISSION APPROVED: {response_text}")
            else:
                st.error(f"⛔ MISSION ABORTED: {response_text}")
