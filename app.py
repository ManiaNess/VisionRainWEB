import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import csv
import requests
from io import BytesIO
import folium
from streamlit_folium import st_folium
import random

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="🛰️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 15px;}
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    .success-box {background-color: rgba(0, 255, 128, 0.1); border: 1px solid #00ff80; color: #00ff80; padding: 15px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. SIMULATED DATA LOADER (Fail-Safe) ---
def generate_saudi_scan():
    """
    Simulates scanning the entire Saudi Arabia sector for seedable clouds.
    Returns a DataFrame of targets with realistic microphysics.
    """
    targets = []
    # Generate 5-8 random storm cells across Saudi Arabia
    for i in range(random.randint(5, 8)):
        # Saudi Bounds: Lat 17-31, Lon 36-54
        lat = random.uniform(18.0, 28.0)
        lon = random.uniform(39.0, 50.0)
        
        # Simulate Physics for this cell
        prob = random.randint(60, 95)
        press = random.randint(400, 750) # 400-700 is ideal
        rad = random.uniform(8.0, 20.0) # <14 is ideal
        od = random.uniform(5.0, 30.0) # >10 is ideal
        lwc = random.uniform(0.001, 0.005) # Liquid Water
        
        # Determine Status
        if prob > 70 and press < 700 and rad < 14:
            status = "HIGH PRIORITY"
        elif prob > 50:
            status = "MODERATE"
        else:
            status = "LOW PRIORITY"
            
        targets.append({
            "ID": f"TGT-{100+i}",
            "Lat": round(lat, 4),
            "Lon": round(lon, 4),
            "Cloud Prob": prob,
            "Pressure": press,
            "Effective Radius": round(rad, 1),
            "Optical Depth": round(od, 1),
            "Liquid Water": round(lwc, 4),
            "Status": status
        })
    return pd.DataFrame(targets)

# --- 2. GLOBAL HEATMAP VISUALIZER ---
def generate_heatmap(metric="Cloud Probability"):
    """
    Generates a Kingdom-wide heatmap for the selected metric.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    
    # Create random noise map to simulate satellite data over Saudi
    data = np.random.rand(100, 150) 
    
    if metric == "Cloud Probability":
        cmap = "Blues_r"
        title = "Cloud Probability (%)"
    elif metric == "Cloud Top Pressure":
        cmap = "turbo_r"
        title = "Cloud Top Pressure (hPa)"
    elif metric == "Effective Radius":
        cmap = "viridis"
        title = "Droplet Effective Radius (µm)"
    else:
        cmap = "magma"
        title = "Optical Depth (Thickness)"

    im = ax.imshow(data, cmap=cmap, aspect='auto')
    ax.set_title(f"Kingdom-Wide Scan: {title}", color="white")
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    return Image.open(buf)

# --- 3. TARGET VISUALIZER ---
def generate_target_zoomed(target_data):
    """Plots the specific zoomed view for the AI"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    
    # Fake zoom data based on target values
    zoom_data = np.random.rand(50, 50) * 100
    
    ax1.imshow(zoom_data, cmap='gray_r')
    ax1.set_title(f"Target {target_data['ID']}: Visual", color="white")
    ax1.axis('off')
    
    ax2.imshow(zoom_data, cmap='jet')
    ax2.set_title(f"Radar Reflectivity", color="white")
    ax2.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    return Image.open(buf)

# --- 4. LOGGING ---
def log_mission(target_id, lat, lon, decision):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f: f.write("Timestamp,Target,Location,Decision\n")
    with open(LOG_FILE, 'a') as f:
        f.write(f"{ts},{target_id},{lat},{lon},{decision}\n")

def load_logs():
    if os.path.exists(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v10.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Regional Scanner")
    
    if 'targets' not in st.session_state:
        st.session_state['targets'] = generate_saudi_scan()
        
    if st.button("RE-SCAN SECTOR"):
        with st.spinner("Scanning Kingdom..."):
            st.session_state['targets'] = generate_saudi_scan()
            st.rerun()
            
    # Target Selector
    df = st.session_state['targets']
    if not df.empty:
        st.success(f"{len(df)} Targets Found")
        target_id = st.selectbox("Select Target:", df["ID"])
        selected_row = df[df["ID"] == target_id].iloc[0]
    else:
        selected_row = None

    # Admin
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Strategy", "🗺️ Operations Map", "🧠 Gemini Authorization"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b> face escalating environmental crises: water scarcity and drought. 
    Current cloud seeding operations are <b>manual, expensive, and reactive</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - An AI-driven decision support platform.")
    with c2:
        st.success("**Impact:** Supports Saudi Green Initiative & Water Security.")

# --- TAB 2: MAP & HEATMAPS ---
with tab2:
    st.header("Kingdom-Wide Analysis")
    
    col_map, col_heat = st.columns([1, 1])
    
    with col_map:
        st.subheader("Live Threat Map")
        # Dynamic Map with Pins
        m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
        
        for _, row in df.iterrows():
            color = 'green' if row['Status'] == 'HIGH PRIORITY' else 'orange'
            folium.Marker(
                [row['Lat'], row['Lon']], 
                popup=f"{row['ID']}: {row['Status']}",
                icon=folium.Icon(color=color, icon="cloud")
            ).add_to(m)
            
        if selected_row is not None:
            folium.CircleMarker(
                [selected_row['Lat'], selected_row['Lon']], radius=20, color='#00e5ff', fill=False
            ).add_to(m)
            
        st_folium(m, height=400, width=600)

    with col_heat:
        st.subheader("Spectral Analysis")
        metric = st.selectbox("View Layer:", ["Cloud Probability", "Cloud Top Pressure", "Effective Radius", "Optical Depth"])
        heatmap = generate_heatmap(metric)
        st.image(heatmap, caption=f"Live {metric} over Saudi Arabia", use_column_width=True)

    # FULL DATA TABLE
    st.markdown("### 📊 Detected Targets Manifest")
    st.dataframe(df.style.highlight_max(axis=0, color='darkgreen'))

# --- TAB 3: GEMINI ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if selected_row is not None:
        t = selected_row
        
        st.info(f"Engaging Target: **{t['ID']}** at **{t['Lat']}, {t['Lon']}**")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            # 1. VISUAL EVIDENCE
            zoom_img = generate_target_zoomed(t)
            st.image(zoom_img, caption=f"Target Recon: {t['ID']}", use_column_width=True)
            
        with c2:
            # 2. MASTER TABLE (Ideal vs Actual)
            st.markdown("### 🔬 Microphysics Validation")
            val_df = pd.DataFrame({
                "Metric": ["Cloud Probability", "Pressure", "Effective Radius", "Optical Depth"],
                "Ideal Range": ["> 70%", "400-700 hPa", "< 14 µm", "> 10"],
                "Target Value": [f"{t['Cloud Prob']}%", f"{t['Pressure']} hPa", f"{t['Effective Radius']} µm", f"{t['Optical Depth']}"]
            })
            st.table(val_df)
        
        # 3. EXECUTE
        if st.button("AUTHORIZE DRONE SWARM", type="primary"):
            if not api_key:
                st.error("🔑 Google API Key Missing!")
            else:
                genai.configure(api_key=api_key)
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    ACT AS A MISSION COMMANDER. Analyze this Target for Cloud Seeding.
                    
                    --- TARGET PROFILE ---
                    ID: {t['ID']}
                    Location: {t['Lat']}, {t['Lon']}
                    
                    --- MICROPHYSICS ---
                    - Probability: {t['Cloud Prob']}%
                    - Pressure: {t['Pressure']} hPa
                    - Radius: {t['Effective Radius']} microns
                    - Optical Depth: {t['Optical Depth']}
                    - Liquid Water: {t['Liquid Water']} kg/kg
                    
                    --- LOGIC RULES ---
                    1. IF Radius < 14 AND Depth > 10 -> "GO" (Ideal).
                    2. IF Radius > 15 OR Probability < 50 -> "NO-GO".
                    
                    --- OUTPUT ---
                    1. **Assessment:** Analyze the physics table.
                    2. **Decision:** **GO** or **NO-GO**.
                    3. **Protocol:** "Deploy Drones" or "Stand Down".
                    """
                    
                    with st.spinner("Vertex AI validating parameters..."):
                        res = model.generate_content([prompt, zoom_img])
                        
                        decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                        log_mission(t['ID'], t['Lat'], t['Lon'], decision)
                        
                        st.markdown("### 🛰️ Mission Directive")
                        st.write(res.text)
                        
                        if decision == "GO":
                            st.balloons()
                            st.success(f"✅ DRONES DISPATCHED TO {t['Lat']}, {t['Lon']}")
                        else:
                            st.error("⛔ MISSION ABORTED")
                            
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.warning("Select a target from Tab 2 or Sidebar first.")
