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
import math # Added for distance calculation

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

# --- MISSION PLANNER FUNCTIONS (MODIFIED) ---

def find_best_target(all_data):
    """
    Identifies the best seedable cloud based on physics rules.
    Returns the target dict OR a dictionary with the failure reason.
    """
    best_candidate = None
    best_prob = 0
    
    # Track overall state for failure reporting
    reasons = {'TOO_COLD': 0, 'TOO_LARGE': 0, 'LOW_PROB': 0}
    total_regions = len(all_data)
    
    for region_name, data in all_data.items():
        is_seedable = (data['rad'] < 14) and (data['rad'] > 5) and (data['phase'] == 1) and (data['prob'] > 50)
        
        if is_seedable and data['prob'] > best_prob:
            best_prob = data['prob']
            best_candidate = {
                "region": region_name, 
                "coords": SAUDI_SECTORS[region_name]['coords'],
                "prob": data['prob']
            }
        
        # Analyze why it failed (if not seedable)
        if not is_seedable:
            if data['phase'] == 2 or data['temp'] < -15:
                reasons['TOO_COLD'] += 1
            elif data['rad'] >= 14:
                reasons['TOO_LARGE'] += 1
            elif data['prob'] <= 50:
                reasons['LOW_PROB'] += 1
                
    if best_candidate:
        return best_candidate
    else:
        # Determine the dominant reason for failure across the Kingdom
        dominant_reason = max(reasons, key=reasons.get)
        
        if reasons[dominant_reason] > 0.6 * total_regions:
            if dominant_reason == 'TOO_COLD':
                return {"failure_reason": "Cloud conditions are predominantly below the ideal -15°C threshold (Ice Phase dominant). Seeding is not recommended."}
            elif dominant_reason == 'TOO_LARGE':
                return {"failure_reason": "Cloud droplet radii are too large (>14µm) across the Kingdom. Natural precipitation is already likely."}
            else:
                return {"failure_reason": "Insufficient cloud coverage or low probability (<50%) detected across all sectors."}
        else:
            return {"failure_reason": "No single region meets all seeding criteria simultaneously."}


def find_launch_base(target_coords):
    """Finds the closest launch base to the target cloud."""
    target_lat, target_lon = target_coords
    closest_base = None
    min_distance = float('inf')
    
    for base_name, base_info in SAUDI_SECTORS.items():
        base_lat, base_lon = base_info['coords']
        
        distance = math.sqrt((base_lat - target_lat)**2 + (base_lon - target_lon)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_base = {"name": base_name, "coords": base_info['coords']}
            
    return closest_base

def generate_mission_map(base, target):
    """Creates a Folium map showing the flight path."""
    center_lat = (base['coords'][0] + target['coords'][0]) / 2
    center_lon = (base['coords'][1] + target['coords'][1]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB dark_matter")
    
    folium.PolyLine(
        locations=[base['coords'], target['coords']],
        color="red",
        weight=4,
        opacity=0.8,
        tooltip=f"Mission Path: {base['name']} to {target['region']}"
    ).add_to(m)
    
    folium.Marker(
        base['coords'],
        popup=f"LAUNCH BASE: {base['name']}",
        icon=folium.Icon(color="green", icon="plane", prefix="fa")
    ).add_to(m)
    
    folium.Marker(
        target['coords'],
        popup=f"TARGET CLOUD: {target['region']} ({target['prob']:.1f}% Prob)",
        icon=folium.Icon(color="blue", icon="bolt", prefix="fa")
    ).add_to(m)
    
    return m

# --- SCIENTIFIC DATA ENGINE (Simulation) ---
def scan_single_sector(sector_name):
    # (Original simulation logic remains the same for stability)
    profile = SAUDI_SECTORS[sector_name]
    
    conditions = [
        {"prob": 85.0, "press": 650, "rad": 12.5, "opt": 15.0, "lwc": 0.005, "rh": 80, "temp": -8.0, "w": 2.5, "phase": 1}, 
        {"prob": 5.0, "press": 950, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 20, "temp": 28.0, "w": 0.1, "phase": 0},
        {"prob": 70.0, "press": 350, "rad": 25.0, "opt": 5.0, "lwc": 0.001, "rh": 60, "temp": -35.0, "w": 0.5, "phase": 2},
    ]
    
    data = random.choice(conditions).copy()
    data['prob'] += profile['bias_prob']
    data['rh'] = profile['humidity_base'] + random.uniform(-10, 10)
    data['temp'] += profile['bias_temp']
    data['prob'] += random.uniform(-5, 5)
    
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

# --- SIDEBAR & MAIN UI (REST OF CODE) ---
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
        st.markdown("Logs are simulated for security.")


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
    
    st.subheader("Google Cloud Platform Architecture")
    st.markdown("""
    | Component | GCP Service | Function |
    | :--- | :--- | :--- |
    | **The Brain** | **Vertex AI (Gemini)** | Analyzes microphysics for GO/NO-GO decisions. |
    | **Data Lake** | **Cloud Storage** | Stores raw Meteosat NetCDF files. |
    """)

# TAB 2: OPS
with tab2:
    current_region = st.session_state.selected_region
    current_data = st.session_state.all_sector_data[current_region]
    
    c_header, c_stats = st.columns([2, 3])
    with c_header:
        st.header(f"📍 {current_region}")
    with c_stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cloud Prob", f"{current_data['prob']:.1f}%", delta="High" if current_data['prob']>60 else "Low")
        s2.metric("Radius", f"{current_data['rad']:.1f} µm", help="Target: < 14µm")
        s3.metric("Phase", "Liquid" if current_data['phase']==1 else "Ice/Mix")
        s4.metric("Status", current_data['status'])

    col_map, col_matrix = st.columns([1, 2])
    
    with col_map:
        st.markdown("**Live Sector Map**")
        lat, lon = SAUDI_SECTORS[current_region]['coords']
        m = folium.Map(location=[lat, lon], zoom_start=8, tiles="CartoDB dark_matter")
        
        for region_name, info in SAUDI_SECTORS.items():
            r_data = st.session_state.all_sector_data[region_name]
            color = "green" if r_data['prob'] > 60 else "orange" if r_data['prob'] > 30 else "gray"
            folium.Marker(info['coords'], icon=folium.Icon(color=color, icon="cloud", prefix="fa")).add_to(m)
            
        st_folium(m, height=450, use_container_width=True)

    with col_matrix:
        st.markdown("**Full Microphysics Data**")
        st.dataframe(
            pd.DataFrame([current_data]).T.rename(columns={0: "Value"}),
            use_container_width=True
        )

    st.divider()

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
            "Condition": d['status']
        })
    
    df_table = pd.DataFrame(table_data).sort_values("Probability", ascending=False)
    
    st.dataframe(
        df_table, use_container_width=True,
        column_config={
            "Probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0, max_value=100),
        }, hide_index=True
    )

# TAB 3: GEMINI (Mission Planning Integration)
with tab3:
    st.header(f"🧠 Mission Control: {current_region}")
    st.markdown("---")
    
    # Run the logic to find the best target OR the reason for failure
    mission_status = find_best_target(st.session_state.all_sector_data)
    
    if st.button("🚀 REQUEST AUTHORIZATION & PLOT MISSION PATH", type="primary"):
        
        if "failure_reason" in mission_status:
            # Display Abort Reason
            st.error("⛔ MISSION ABORTED: Kingdom-Wide Scan Failed.")
            st.warning(f"**Justification:** {mission_status['failure_reason']} and a table why")
            
        else:
            best_target = mission_status
            
            # Step 1: Find Launch Base and Plot Path
            launch_base = find_launch_base(best_target['coords'])
            mission_map = generate_mission_map(launch_base, best_target)
            
            # Step 2: Request Gemini Decision (Simplified Simulation)
            with st.status("Initializing Vertex AI Pipeline...") as status:
                status.update(label="1. Awaiting Gemini Decision...", state="running")
                
                prompt = f"""
                ACT AS A METEOROLOGIST. Analyze {best_target['region']} for deployment.
                DATA: {json.dumps(st.session_state.all_sector_data[best_target['region']])}.
                RULES: GO IF Radius < 14 AND Radius > 5 AND Phase=1 (Liquid). NO-GO IF Phase=2 (Ice) OR Prob < 50.
                OUTPUT: Decision (GO/NO-GO), Reasoning, Protocol.
                """
                
                time.sleep(2) # Simulate processing time
                
                # Simplified GO/NO-GO for display stability
                if best_target['prob'] > 70:
                    response_text = f"DECISION: GO | REASONING: Optimal microphysics (Radius {current_data['rad']:.1f}µm, Liquid Phase). | PROTOCOL: Launch drone from {launch_base['name']}."
                    st.success(f"✅ MISSION APPROVED: LAUNCH FROM **{launch_base['name']}**")
                    
                    st.subheader("🗺️ Autonomous Drone Flight Path")
                    st_folium(mission_map, height=500, use_container_width=True)
                else:
                    st.error("⛔ AI Decision Override: Probability too low for immediate deployment.")
                    response_text = "Decision Override: Probability too low."

                status.update(label="3. Mission Log Complete.", state="complete")

            # Step 4: Log the Outcome
            save_mission_log(best_target['region'], st.session_state.all_sector_data[best_target['region']], "GO", response_text)
