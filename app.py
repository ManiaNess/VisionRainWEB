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

# FILES
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

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

# --- 2. KINGDOM SCANNER (Real Data) ---
def scan_kingdom_targets(ds_sat):
    targets = []
    if ds_sat is None: return pd.DataFrame()

    try:
        # Dynamic Dimensions
        dims = list(ds_sat.dims)
        y_dim = next((d for d in dims if 'y' in d or 'lat' in d), dims[0])
        x_dim = next((d for d in dims if 'x' in d or 'lon' in d), dims[1])
        
        # Slice Saudi Sector (Approx)
        y_slice = slice(2100, 2500)
        x_slice = slice(600, 900)
        
        # Using dynamic kwargs to avoid errors
        slice_args = {y_dim: y_slice, x_dim: x_slice}
        
        prob = ds_sat['cloud_probability'].isel(**slice_args).values
        press = ds_sat['cloud_top_pressure'].isel(**slice_args).values
        
        # Get other metrics if available
        rad = ds_sat['cloud_particle_effective_radius'].isel(**slice_args).values * 1e6 if 'cloud_particle_effective_radius' in ds_sat else np.zeros_like(prob)
        od = ds_sat['cloud_optical_thickness'].isel(**slice_args).values if 'cloud_optical_thickness' in ds_sat else np.zeros_like(prob)

        # Filter: High Prob (>50) AND Mid-Level Pressure
        y_idxs, x_idxs = np.where((prob > 50) & (prob <= 100) & (press > 40000))
        
        if len(y_idxs) > 0:
            # Sample 6 targets
            indices = np.linspace(0, len(y_idxs)-1, 6, dtype=int)
            for i, idx in enumerate(indices):
                y, x = y_idxs[idx], x_idxs[idx]
                
                # Metric Extraction
                p_val = float(prob[y, x])
                press_val = float(press[y, x] / 100.0)
                rad_val = float(rad[y, x]) if rad[y,x] > 0 else 12.0
                od_val = float(od[y, x]) if od[y,x] > 0 else 15.0
                
                # Lat/Lon Approx
                global_y = y + 2100
                global_x = x + 600
                lat = 24.0 + (200 - y) * 0.03
                lon = 45.0 + (x - 150) * 0.03
                
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat, 4), "Lon": round(lon, 4),
                    "Probability": int(p_val),
                    "Pressure": int(press_val),
                    "Radius": round(rad_val, 1),
                    "Optical Depth": round(od_val, 1),
                    "Status": "HIGH PRIORITY" if p_val > 80 else "MODERATE",
                    "GY": global_y, "GX": global_x
                })
    except: pass
        
    return pd.DataFrame(targets).sort_values(by="Probability", ascending=False) if targets else pd.DataFrame()

# --- 3. VISUALIZER (Satellite Look) ---
def plot_satellite_view(ds_sat, ds_era, target_row):
    """Plots the Target View like a Real Satellite Image (Not a Heatmap)"""
    
    gy, gx = int(target_row['GY']), int(target_row['GX'])
    window = 50 # Zoom level
    
    def get_slice(ds, var_name, is_era=False):
        try:
            if is_era:
                data = ds[var_name].values
                while data.ndim > 2: data = data[0]
                return data[0:100, 0:100] # ERA5 Mock Crop
            else:
                dims = list(ds.dims)
                y_d = next((d for d in dims if 'y' in d or 'lat' in d), dims[0])
                x_d = next((d for d in dims if 'x' in d or 'lon' in d), dims[1])
                
                # Handle bounds
                y_s = slice(max(0, gy-window), min(ds.sizes[y_d], gy+window))
                x_s = slice(max(0, gx-window), min(ds.sizes[x_d], gx+window))
                return ds[var_name].isel({y_d: y_s, x_d: x_s}).values
        except: return np.zeros((100,100))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('#0e1117')
    
    # --- ROW 1: METEOSAT OPTICS ---
    # 1. VISUAL SATELLITE (Pressure inverted = Clouds are White)
    data = get_slice(ds_sat, 'cloud_top_pressure')
    # Mask space
    data = np.where(data < 110000, data, np.nan)
    axes[0,0].imshow(data, cmap='gray_r') 
    axes[0,0].set_title("1. Visual Satellite (Cloud Structure)", color="white")
    
    # 2. AI PROBABILITY (Blue Overlay)
    data = get_slice(ds_sat, 'cloud_probability')
    axes[0,1].imshow(data, cmap='Blues', vmin=0, vmax=100)
    axes[0,1].set_title(f"2. AI Probability ({target_row['Probability']}%)", color="white")

    # 3. MICROPHYSICS: RADIUS
    if 'cloud_particle_effective_radius' in ds_sat:
        data = get_slice(ds_sat, 'cloud_particle_effective_radius') * 1e6
        im = axes[0,2].imshow(data, cmap='viridis')
        plt.colorbar(im, ax=axes[0,2])
    axes[0,2].set_title(f"3. Droplet Radius ({target_row['Radius']} µm)", color="white")
    
    # 4. OPTICAL DEPTH
    if 'cloud_optical_thickness' in ds_sat:
        data = get_slice(ds_sat, 'cloud_optical_thickness')
        im = axes[0,3].imshow(data, cmap='magma')
        plt.colorbar(im, ax=axes[0,3])
    axes[0,3].set_title("4. Optical Thickness", color="white")

    # --- ROW 2: ERA5 ATMOSPHERE ---
    era_vars = [('clwc', 'Liquid Water'), ('ciwc', 'Ice Water'), ('r', 'Humidity'), ('w', 'Updrafts')]
    cmaps = ['Blues', 'PuBu', 'Greens', 'RdBu']
    
    if ds_era:
        era_keys = list(ds_era.data_vars)
        for i, (key, title) in enumerate(era_vars):
            found = next((k for k in era_keys if key in k), None)
            ax = axes[1, i]
            if found:
                data = get_slice(ds_era, found, is_era=True)
                im = ax.imshow(data, cmap=cmaps[i])
                plt.colorbar(im, ax=ax)
                ax.set_title(f"ERA5: {title}", color="white")
            else:
                ax.text(0.5, 0.5, "N/A", color="gray", ha='center')
    else:
        for i in range(4): axes[1, i].text(0.5, 0.5, "ERA5 Offline", color="red", ha='center')

    # Cleanup
    for row in axes:
        for ax in row: ax.axis('off')

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
    st.caption("Kingdom Commander | v16.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Regional Scanner")
    
    ds_sat, ds_era = load_data()
    
    # Auto-Scan on Load
    if 'targets_df' not in st.session_state:
        if ds_sat:
            st.session_state['targets_df'] = scan_kingdom_targets(ds_sat)
        else:
            st.session_state['targets_df'] = None
    
    if st.button("RE-SCAN SECTOR"):
        with st.spinner("Scanning 2.15 Million km²..."):
            if ds_sat:
                st.session_state['targets_df'] = scan_kingdom_targets(ds_sat)
                st.rerun()
            else:
                st.error("No Data File Found")
            
    # Target List
    selected_row = None
    df = st.session_state['targets_df']
    
    if df is not None and not df.empty:
        st.success(f"{len(df)} Seedable Targets Found")
        target_id = st.selectbox("Engage Target:", df['ID'])
        selected_row = df[df['ID'] == target_id].iloc[0]
    
    # Admin
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Pitch", "🗺️ Operations", "🧠 Gemini Core"])

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
        st.success("**Impact:** Supports Saudi Green Initiative & Water Security.")

# --- TAB 2: MAP & METRICS ---
with tab2:
    st.header("Kingdom-Wide Threat Map")
    
    if df is not None and not df.empty:
        # 1. MAP
        c_map, c_data = st.columns([2, 1])
        with c_map:
            m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
            for _, row in df.iterrows():
                color = 'green' if row['Probability'] > 80 else 'orange'
                folium.Marker([row['Lat'], row['Lon']], popup=f"{row['ID']}", icon=folium.Icon(color=color, icon="cloud")).add_to(m)
            
            if selected_row is not None:
                folium.CircleMarker([selected_row['Lat'], selected_row['Lon']], radius=20, color='#00e5ff', fill=False).add_to(m)
            
            st_folium(m, height=400, width=1400)

        # 2. METRICS TABLE (ALL TARGETS)
        st.markdown("### 📊 Live Target Manifest (All Detected Cells)")
        st.dataframe(
            df.style.background_gradient(subset=['Probability'], cmap='Blues')
                    .background_gradient(subset=['Pressure'], cmap='gray_r')
        )

        # 3. SELECTED TARGET DASHBOARD
        if selected_row is not None:
            st.divider()
            st.markdown(f"### 🔬 Deep Analysis: {selected_row['ID']}")
            
            # Generate Visuals
            matrix_img = plot_satellite_view(ds_sat, ds_era, selected_row)
            if matrix_img:
                st.image(matrix_img, caption="Real-Time Microphysics (Meteosat & ERA5)", use_column_width=True)
                st.session_state['ai_matrix'] = matrix_img
                st.session_state['active_target'] = selected_row
            
    else:
        st.warning("No targets found in dataset. Please check data files.")

# --- TAB 3: GEMINI CORE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if 'active_target' in st.session_state:
        t = st.session_state['active_target']
        
        st.info(f"Engaging Target: **{t['ID']}**")
        
        if 'ai_matrix' in st.session_state:
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
    else:
        st.warning("Select a target in Tab 2 first.")
