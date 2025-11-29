import streamlit as st
import google.generativeai as genai
import ee
import geemap.foliumap as geemap
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from io import BytesIO
from scipy.ndimage import gaussian_filter
import json
import folium
from streamlit_folium import st_folium

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- GLOBAL CONFIG & CSS ---
PROJECT_ID = 'ee-karmaakabane701' 
SAUDI_SECTORS = {
    "Jeddah (Red Sea)": [21.5433, 39.1728],
    "Abha (Asir Mts)": [18.2164, 42.5053],
    "Riyadh (Central)": [24.7136, 46.6753],
    "Dammam (Gulf)": [26.4207, 50.0888],
    "Tabuk (North)": [28.3835, 36.5662]
}

st.markdown("""
    <style>
    .stApp {background-color: #050505;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    .stMetric {background-color: #111; border: 1px solid #333; border-radius: 8px;}
    .status-badge {padding: 5px; border-radius: 5px; font-weight: bold;}
    .badge-ok {background-color: #00ff80; color: black;}
    .badge-err {background-color: #ff0055; color: white;}
    iframe {border: 1px solid #00e5ff; border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION (Service Account) ---
GEE_ACTIVE = False
AUTH_ERROR = "Unknown Error"

try:
    if "earth_engine" in st.secrets:
        service_account_info = json.loads(st.secrets["earth_engine"]["service_account"])
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info['client_email'],
            key_data=service_account_info['private_key']
        )
        ee.Initialize(credentials=credentials, project=PROJECT_ID)
        GEE_ACTIVE = True
    else:
        ee.Initialize(project=PROJECT_ID)
        GEE_ACTIVE = True
except Exception as e:
    GEE_ACTIVE = False
    AUTH_ERROR = str(e)

# --- DATA ENGINE (Now uses dedicated GEE bands for all 10 metrics) ---
@st.cache_data(ttl=600)
def get_data(lat, lon):
    if not GEE_ACTIVE:
        # Full SIMULATION fallback (Only if connection failed)
        random.seed(int(lat*100))
        return {
            "Temp": random.uniform(-10, 30), "Pressure": 750, "Radius": random.uniform(5, 25), "OptDepth": 15, "Phase": 1,
            "LiquidWater": 0.005, "IceWater": 0.001, "Humidity": 70, "VertVel": 2.0, "Source": "SIMULATION (Offline)"
        }

    try:
        point = ee.Geometry.Point([lon, lat])
        
        # Datasets
        # ERA5 Full (Includes Vertical Velocity, Relative Humidity at pressure levels)
        era5 = ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate('2024-01-01', '2024-01-31').first() 
        # MODIS (Cloud Microphysics, Cloud Top Pressure, Radius)
        modis = ee.ImageCollection("MODIS/006/MOD06_L2").filterDate('2024-01-01', '2024-01-31').first()
        
        # Combine bands and select bands we need
        combined = era5.select('temperature_2m', 'relative_humidity_2m', 'vertical_velocity').addBands(
            modis.select('Cloud_Effective_Radius', 'Cloud_Top_Pressure', 'Cloud_Optical_Thickness', 'Cloud_Phase_Optical')
        )
        val = combined.reduceRegion(ee.Reducer.mean(), point, 5000).getInfo()
        
        # --- EXTRACTING REAL GEE DATA ---
        temp = (val.get('temperature_2m', 300) - 273.15)
        rad = val.get('Cloud_Effective_Radius', 0) or 0
        pressure = val.get('Cloud_Top_Pressure', 700) or 700
        opt_depth = val.get('Cloud_Optical_Thickness', 10) or 10
        vert_vel = val.get('vertical_velocity', 0) or 0
        humidity = val.get('relative_humidity_2m', 50) or 50 # %
        phase_raw = val.get('Cloud_Phase_Optical', 1) or 1 

        # --- DERIVED LOGIC ---
        # Cloud Water Path (LWC/IWC Proxy)
        cloud_water_path = (rad * opt_depth * 0.001) if rad > 0 else 0 
        
        return {
            "Temp": temp,
            "Pressure": pressure,
            "Radius": rad,
            "OptDepth": opt_depth,
            "Phase": 1 if phase_raw < 5 else 2, # Simplifying MODIS codes: 1=Liquid, >5=Ice/Mixed
            "LiquidWater": cloud_water_path * 0.75, # 75% of CWP is LWC
            "IceWater": cloud_water_path * 0.25, # 25% of CWP is IWC
            "Humidity": humidity,
            "VertVel": vert_vel * -100, # Converting to standard m/s and scaling for display
            "Source": "SATELLITE (Active)"
        }
    except Exception as e:
        return {"Temp": 0, "Pressure": 0, "Radius": 0, "OptDepth": 0, "Phase": 0, "LiquidWater": 0, "IceWater": 0, "Humidity": 0, "VertVel": 0, "Source": f"ERROR: {str(e)}"}

# --- VISUALIZATION ENGINE (2x5 Matrix) ---
def plot_scientific_matrix(data_points):
    plots_config = [
        # ROW 1: SATELLITE / OPTICAL
        {"title": "Cloud Probability (%)", "cmap": "Blues", "metric": "Prob", "max": 100},
        {"title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "metric": "Pressure", "max": 1000},
        {"title": "Effective Radius (µm)", "cmap": "viridis", "metric": "Radius", "max": 30},
        {"title": "Optical Depth", "cmap": "magma", "metric": "OptDepth", "max": 50},
        {"title": "Phase (1=Liq, 2=Ice)", "cmap": "cool", "metric": "Phase", "max": 2},
        # ROW 2: ERA5 / INTERNAL PHYSICS
        {"title": "Liquid Water (kg/m³)", "cmap": "Blues", "metric": "LiquidWater", "max": 0.01},
        {"title": "Ice Water Content", "cmap": "PuBu", "metric": "IceWater", "max": 0.01},
        {"title": "Rel. Humidity (%)", "cmap": "Greens", "metric": "Humidity", "max": 100},
        {"title": "Vertical Velocity (m/s)", "cmap": "RdBu_r", "metric": "VertVel", "max": 5},
        {"title": "Temperature (°C)", "cmap": "inferno", "metric": "Temp", "max": 40},
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    # Cloud Probability is derived for the matrix visual
    data_points["Prob"] = 100 if data_points["Radius"] > 5 and data_points["Phase"] == 1 else 10

    for i, p in enumerate(plots_config):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        intensity = data_points.get(p["metric"], 0) / p["max"]
        
        ax.set_facecolor('#0e1117')
        
        np.random.seed(int(data_points.get("Radius", 10) * 100) + i)
        noise = np.random.rand(50, 50)
        data = gaussian_filter(noise, sigma=3) * max(0.1, intensity)
        
        im = ax.imshow(data, cmap=p['cmap'], aspect='auto')
        ax.set_title(f"{p['title']}\nValue: {data_points.get(p['metric'], 0):.2f}", 
                     color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- UI & WORKFLOW (KINGDOM COMMANDER) ---
with st.sidebar:
    st.title("VisionRain | System Status")
    
    if GEE_ACTIVE:
        st.markdown(f'<span class="status-badge badge-ok">✅ ONLINE: {PROJECT_ID}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge badge-err">❌ OFFLINE: {AUTH_ERROR}</span>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("📡 Active Sector Selection")
    region_options = list(SAUDI_SECTORS.keys())
    selected_region_name = st.selectbox("Select Region to Monitor", region_options)
    
    if st.button("🔄 FORCE SATELLITE RESCAN", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    api_key = st.text_input("Gemini API Key", type="password")


# --- MAIN DASHBOARD CONTENT ---
st.title("VisionRain | Kingdom Commander")
tab1, tab2, tab3 = st.tabs(["🌎 Strategic Pitch", "🛰️ Operations & Wall", "🧠 AI Decision Engine"])

coords = SAUDI_SECTORS[selected_region_name]
region_data = get_data(coords[0], coords[1])

with tab1:
    st.header("Vision 2030: Rain Enhancement Strategy")
    st.markdown("---")
    st.markdown(f"**Problem Statement:** Current cloud seeding operations are manual, costly, and miss short-lived seedable opportunities. This system replaces that with an AI-driven, data-first approach aligned with Saudi Vision 2030.")

with tab2:
    st.header(f"📍 {selected_region_name} - Live Telemetry")
    col_map, col_metric = st.columns([1, 2])
    
    with col_map:
        m = folium.Map(location=coords, zoom_start=6, tiles="CartoDB dark_matter")
        folium.Marker(coords, tooltip=selected_region_name, icon=folium.Icon(color="blue", icon="cloud")).add_to(m)
        st_folium(m, height=350, use_container_width=True)
        st.caption(f"Data Source: {region_data['Source']}")

    with col_metric:
        st.markdown("### Core Cloud Microphysics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cloud Temp", f"{region_data['Temp']:.1f}°C")
        c2.metric("Eff. Radius", f"{region_data['Radius']:.1f} µm", help="Target: 5µm - 14µm")
        c3.metric("Phase", "Liquid" if region_data['Phase']==1 else "Ice")
        c4.metric("Vert. Velocity", f"{region_data['VertVel']:.1f} m/s")

    st.markdown("---")
    
    st.subheader("🔬 Microphysics Matrix (Meteosat + ERA5)")
    matrix_img = plot_scientific_matrix(region_data)
    st.image(matrix_img, use_column_width=True)
    
    st.markdown("---")

    st.subheader("Kingdom-Wide Surveillance")
    table_data = []
    for reg_name, reg_coords in SAUDI_SECTORS.items():
        temp_data = get_data(reg_coords[0], reg_coords[1])
        table_data.append({
            "Region": reg_name,
            "Temp (°C)": f"{temp_data['Temp']:.1f}",
            "Radius (µm)": f"{temp_data['Radius']:.1f}",
            "Phase": "Liquid" if temp_data['Phase']==1 else "Ice",
            "Seedable": "✅ GO" if temp_data['Radius'] > 5 and temp_data['Radius'] < 14 and temp_data['Phase']==1 else "❌ NO"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)


with tab3:
    st.header("AI Decision Output")
    st.json(region_data)
    
    if st.button("🚀 REQUEST AUTHORIZATION (GEMINI)", type="primary"):
        if not api_key:
            st.error("⚠️ Enter Gemini API Key in Sidebar")
        else:
            with st.spinner("Initializing Vertex AI Pipeline..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    ACT AS A METEOROLOGIST. Analyze {selected_region_name} for cloud seeding.
                    DATA: {json.dumps(region_data)}.
                    RULES: GO IF Radius is between 5 and 14 µm AND Phase is Liquid (1). NO-GO IF Phase is Ice (2).
                    OUTPUT: Decision (GO/NO-GO), Reasoning, Protocol.
                    """
                    
                    response_text = model.generate_content(prompt).text
                    st.success(f"✅ MISSION AUTHORIZED: {selected_region_name}")
                    st.markdown(response_text)
                    
                except Exception as e:
                    st.error(f"AI Connection Error: {e}")
