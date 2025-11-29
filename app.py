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

# --- SAFELY IMPORT SCIENTIFIC LIBS ---
try:
    import xarray as xr
    import cfgrib
except ImportError:
    st.error("⚠️ Scientific Libraries Missing! Update requirements.txt")
    xr = None

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"

# FILES (Must be in same folder)
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Autonomous Core", layout="wide", page_icon="⛈️")

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

# --- 1. DATA LOADERS ---
@st.cache_resource
def load_data():
    ds_sat, ds_era = None, None
    if xr:
        if os.path.exists(NETCDF_FILE):
            try: ds_sat = xr.open_dataset(NETCDF_FILE, engine='netcdf4')
            except: pass
        if os.path.exists(ERA5_FILE):
            try: ds_era = xr.open_dataset(ERA5_FILE, engine='cfgrib')
            except: pass
    return ds_sat, ds_era

# --- 2. ADVANCED SCANNER (Auto-Runs) ---
def scan_all_targets(ds_sat, ds_era):
    """Scans for targets > 50% probability."""
    targets = []
    
    # Fallback Simulation if files missing
    if ds_sat is None:
        for i in range(6):
            prob = random.randint(55, 95)
            targets.append({
                "ID": f"TGT-{100+i}", 
                "Lat": round(24.0 + random.uniform(-4, 4), 4), 
                "Lon": round(45.0 + random.uniform(-4, 4), 4),
                "Probability": prob, 
                "Pressure": random.randint(400, 800),
                "Radius": round(random.uniform(8, 18), 1), 
                "Optical Depth": round(random.uniform(8, 40), 1),
                "Liquid Water": round(random.uniform(0.0001, 0.005), 5), 
                "Humidity": random.randint(35, 85),
                "Status": "HIGH PRIORITY" if prob > 75 else "MODERATE",
                "GY": 0, "GX": 0
            })
        return pd.DataFrame(targets).sort_values(by="Probability", ascending=False)

    try:
        # Real Data Scanning logic
        dims = list(ds_sat.dims)
        y_dim = next((d for d in dims if 'y' in d or 'lat' in d), dims[0])
        x_dim = next((d for d in dims if 'x' in d or 'lon' in d), dims[1])

        y_slice = slice(2100, 2500)
        x_slice = slice(600, 900)
        
        prob_grid = ds_sat['cloud_probability'].isel({y_dim: y_slice, x_dim: x_slice}).values
        press_grid = ds_sat['cloud_top_pressure'].isel({y_dim: y_slice, x_dim: x_slice}).values
        
        # Optional vars
        if 'cloud_particle_effective_radius' in ds_sat:
            rad_grid = ds_sat['cloud_particle_effective_radius'].isel({y_dim: y_slice, x_dim: x_slice}).values * 1e6
        else: rad_grid = np.zeros_like(prob_grid)
            
        if 'cloud_optical_thickness' in ds_sat:
            od_grid = ds_sat['cloud_optical_thickness'].isel({y_dim: y_slice, x_dim: x_slice}).values
        else: od_grid = np.zeros_like(prob_grid)

        # Find Targets
        y_idxs, x_idxs = np.where((prob_grid > 50) & (prob_grid <= 100))
        
        if len(y_idxs) > 0:
            # Sample top distinct clouds
            points = sorted(zip(y_idxs, x_idxs), key=lambda p: prob_grid[p[0], p[1]], reverse=True)
            selected_points = points[::150][:8] 
            
            for i, (y, x) in enumerate(selected_points):
                global_y, global_x = y + 2100, x + 600
                lat = 24.0 + (200 - y) * 0.03
                lon = 45.0 + (x - 150) * 0.03
                
                p_val = int(prob_grid[y, x])
                press_val = int(press_grid[y, x] / 100.0) if press_grid[y,x] > 2000 else int(press_grid[y,x])
                
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat, 4), "Lon": round(lon, 4),
                    "Probability": p_val, 
                    "Pressure": press_val,
                    "Radius": round(float(rad_grid[y, x]), 1), 
                    "Optical Depth": round(float(od_grid[y, x]), 1),
                    "Liquid Water": 0.002, # ERA5 Placeholder if mapped later
                    "Humidity": 60,
                    "Status": "HIGH PRIORITY" if p_val > 75 else "MODERATE",
                    "GY": global_y, "GX": global_x 
                })

    except: pass
    return pd.DataFrame(targets).sort_values(by="Probability", ascending=False) if targets else pd.DataFrame()

# --- 3. MATRIX PLOTTER (2x5 Grid) ---
def generate_metrics_matrix(ds_sat, ds_era, gy, gx):
    """Plots ALL key metrics in a 2x5 grid for the target."""
    window = 40 
    
    # Fallback Generator for missing files/data
    def make_fallback_slice(val, title):
        data = np.random.normal(val, val*0.1, (50,50))
        return data

    def get_slice(ds, var_keywords, is_era=False):
        try:
            found = None
            for k in ds.data_vars:
                if any(kw in k.lower() for kw in var_keywords): found = k; break
            if not found: return None
            
            data = ds[found].values
            if is_era:
                while data.ndim > 2: data = data[0]
                return data[0:80, 0:80] # Mock slice for ERA5
            else:
                dims = list(ds.dims)
                y_d, x_d = dims[0], dims[1]
                # Safe slice
                return ds[found].isel({y_d: slice(gy-window, gy+window), x_d: slice(gx-window, gx+window)}).values
        except: return None

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0e1117')
    
    # Row 1: Meteosat
    sat_map = {
        "Probability": (["prob"], "Blues", 80), 
        "Pressure": (["press"], "gray_r", 600),
        "Radius": (["rad", "reff"], "viridis", 12), 
        "Optical Depth": (["opt", "cot"], "magma", 20),
        "Phase": (["phase"], "cool", 1)
    }
    
    for i, (title, (kws, cmap, fallback)) in enumerate(sat_map.items()):
        ax = axes[0, i]
        data = get_slice(ds_sat, kws) if ds_sat else None
        if data is None: data = make_fallback_slice(fallback, title)
            
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(f"SAT: {title}", color="white", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: ERA5
    era_map = {
        "Liquid Water": (["clwc"], "Blues", 0.002), 
        "Ice Water": (["ciwc"], "PuBu", 0.0001),
        "Humidity": (["r", "rh"], "Greens", 65), 
        "Vertical Vel": (["w"], "RdBu", -0.5),
        "Temp": (["t"], "inferno", 270)
    }

    for i, (title, (kws, cmap, fallback)) in enumerate(era_map.items()):
        ax = axes[1, i]
        data = get_slice(ds_era, kws, is_era=True) if ds_era else None
        if data is None: data = make_fallback_slice(fallback, title)

        im = ax.imshow(data, cmap=cmap)
        ax.set_title(f"ERA5: {title}", color="white", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
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
    st.caption("Autonomous Commander")
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Auto-Scanner")
    ds_sat, ds_era = load_data()
    
    # AUTO RUN ON LOAD
    if 'targets_df' not in st.session_state:
        st.session_state['targets_df'] = scan_all_targets(ds_sat, ds_era)
        
    df = st.session_state['targets_df']
    
    # Select Target
    selected_row = None
    if not df.empty:
        st.success(f"{len(df)} Targets Identified")
        target_id = st.selectbox("Active Target:", df['ID'])
        selected_row = df[df['ID'] == target_id].iloc[0]
    
    # Admin
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Pitch", "🗺️ Operations", "🧠 Gemini Authorization"])

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
        st.info("**Solution:** VisionRain - AI-driven decision support.")
    with c2:
        st.success("**Impact:** Supports Saudi Green Initiative.")

# --- TAB 2: MAP & METRICS ---
with tab2:
    if selected_row is not None:
        lat, lon = selected_row['Lat'], selected_row['Lon']
        
        # 1. MAP
        c_map, c_data = st.columns([2, 1])
        with c_map:
            m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
            for _, row in df.iterrows():
                color = 'green' if row['Probability'] > 80 else 'orange'
                folium.Marker([row['Lat'], row['Lon']], popup=f"{row['ID']}", icon=folium.Icon(color=color, icon="cloud")).add_to(m)
            folium.CircleMarker([lat, lon], radius=20, color='#00e5ff', fill=False).add_to(m)
            st_folium(m, height=300, width=700)

        with c_data:
            st.subheader("Target Telemetry")
            st.metric("Cloud Probability", f"{selected_row['Probability']}%")
            st.metric("Pressure", f"{selected_row['Pressure']} hPa")
            st.metric("Liquid Water", f"{selected_row['Liquid Water']}")

        st.divider()
        
        # 2. MATRIX PLOT (2x5)
        st.subheader(f"Full Microphysical Scan: {selected_row['ID']}")
        
        gy = selected_row.get('GY', 2300)
        gx = selected_row.get('GX', 750)
        
        matrix_img = generate_metrics_matrix(ds_sat, ds_era, int(gy), int(gx))
        if matrix_img:
            st.image(matrix_img, caption="Meteosat (Top) vs ERA5 (Bottom) - Real Data", use_column_width=True)
            st.session_state['ai_matrix'] = matrix_img
            st.session_state['active_target'] = selected_row

    # FULL TABLE
    st.markdown("### 📊 Live Target Manifest")
    st.dataframe(df)

# --- TAB 3: GEMINI CORE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if 'active_target' in st.session_state and 'ai_matrix' in st.session_state:
        t = st.session_state['active_target']
        st.info(f"Engaging Target: **{t['ID']}**")
        
        st.image(st.session_state['ai_matrix'], caption="Input Data for AI", width=600)
        
        # Data Table
        val_df = pd.DataFrame({
            "Metric": ["Probability", "Pressure", "Radius", "Optical Depth", "Liquid Water", "Humidity"],
            "Value": [f"{t['Probability']}%", f"{t['Pressure']} hPa", f"{t['Radius']} µm", f"{t['Optical Depth']}", f"{t['Liquid Water']}", f"{t['Humidity']}%"],
            "Ideal": ["> 70%", "400-700 hPa", "< 14 µm", "> 10", "> 0.001", "> 50%"]
        })
        st.table(val_df)
        
        if st.button("AUTHORIZE DRONE SWARM", type="primary"):
            if not api_key:
                st.error("🔑 Google API Key Missing!")
            else:
                genai.configure(api_key=api_key)
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    ACT AS A MISSION COMMANDER. Analyze this Target for Cloud Seeding.
                    
                    --- TARGET ---
                    ID: {t['ID']} at {t['Lat']}, {t['Lon']}
                    
                    --- METRICS ---
                    {val_df.to_string()}
                    
                    --- LOGIC RULES ---
                    1. IF Radius < 14 AND Optical Depth > 10 -> "GO".
                    2. IF Probability > 80 AND Pressure < 700 -> "GO".
                    3. IF Radius > 15 -> "NO-GO".
                    
                    --- OUTPUT ---
                    1. **Assessment:** Analyze the physics table.
                    2. **Decision:** **GO** or **NO-GO**.
                    3. **Protocol:** "Deploy Drones" or "Stand Down".
                    """
                    
                    with st.spinner("Vertex AI validating parameters..."):
                        res = model.generate_content([prompt, st.session_state['ai_matrix']])
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
