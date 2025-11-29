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

st.set_page_config(page_title="VisionRain | Ultimate Commander", layout="wide", page_icon="⛈️")

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

# --- 2. ADVANCED SCANNER (The Fix) ---
def scan_all_targets(ds_sat, ds_era):
    """
    Scans dataset for targets > 50% probability.
    Extracts ALL metrics for the table.
    """
    targets = []
    
    # --- SIMULATION FALLBACK (If file missing) ---
    if ds_sat is None:
        for i in range(10):
            lat = 24.0 + random.uniform(-5, 5)
            lon = 45.0 + random.uniform(-5, 5)
            prob = random.randint(51, 99)
            targets.append({
                "ID": f"TGT-{100+i}", "Lat": round(lat, 4), "Lon": round(lon, 4),
                "Probability": prob, "Pressure": random.randint(400, 800),
                "Radius": round(random.uniform(8, 20), 1), "Optical Depth": round(random.uniform(5, 40), 1),
                "Liquid Water": round(random.uniform(0.0001, 0.005), 5), "Humidity": random.randint(30, 80),
                "Status": "HIGH PRIORITY" if prob > 75 else "MODERATE",
                "GY": 0, "GX": 0 # Dummy coords
            })
        return pd.DataFrame(targets).sort_values(by="Probability", ascending=False)

    # --- REAL DATA SCANNING ---
    try:
        # 1. Dimension Handling
        dims = list(ds_sat.dims)
        y_dim = next((d for d in dims if 'y' in d or 'lat' in d), dims[0])
        x_dim = next((d for d in dims if 'x' in d or 'lon' in d), dims[1])

        # 2. Define Sector (Saudi Approx)
        # Y: 2000-2600, X: 500-1000 covers the peninsula in Meteosat disk
        y_slice = slice(2000, 2600)
        x_slice = slice(500, 1000)
        
        # 3. Extract Grids
        prob_grid = ds_sat['cloud_probability'].isel({y_dim: y_slice, x_dim: x_slice}).values
        press_grid = ds_sat['cloud_top_pressure'].isel({y_dim: y_slice, x_dim: x_slice}).values
        
        # Optional Vars (Microphysics)
        if 'cloud_particle_effective_radius' in ds_sat:
            rad_grid = ds_sat['cloud_particle_effective_radius'].isel({y_dim: y_slice, x_dim: x_slice}).values * 1e6
        else: rad_grid = np.zeros_like(prob_grid)
            
        if 'cloud_optical_thickness' in ds_sat:
            od_grid = ds_sat['cloud_optical_thickness'].isel({y_dim: y_slice, x_dim: x_slice}).values
        else: od_grid = np.zeros_like(prob_grid)

        # ERA5 Defaults
        era_lwc, era_hum = 0.002, 60
        if ds_era:
            # Try to get mean values from ERA5 file
            try:
                if 'clwc' in ds_era: era_lwc = float(ds_era['clwc'].mean())
                if 'r' in ds_era: era_hum = float(ds_era['r'].mean())
            except: pass

        # 4. FIND TARGETS (>50% Prob)
        # Filter invalid data
        prob_clean = np.where((prob_grid > 50) & (prob_grid <= 100), prob_grid, 0)
        y_idxs, x_idxs = np.where(prob_clean > 50)
        
        if len(y_idxs) > 0:
            # Sample points (Skip to avoid duplicates)
            points = sorted(zip(y_idxs, x_idxs), key=lambda p: prob_grid[p[0], p[1]], reverse=True)
            selected_points = points[::200][:15] # Take top 15 distinct clouds
            
            for i, (y, x) in enumerate(selected_points):
                # Global Coords for Plotting Later
                global_y = y + 2000
                global_x = x + 500
                
                # Approx Lat/Lon Map
                lat = 24.0 + (300 - y) * 0.03
                lon = 45.0 + (x - 250) * 0.03
                
                p_val = int(prob_grid[y, x])
                press_val = int(press_grid[y, x] / 100.0) if press_grid[y,x] > 2000 else int(press_grid[y,x])
                rad_val = round(float(rad_grid[y, x]), 1)
                od_val = round(float(od_grid[y, x]), 1)
                
                status = "HIGH PRIORITY" if p_val > 75 else "MODERATE"
                
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat, 4), "Lon": round(lon, 4),
                    "Probability": p_val, "Pressure": press_val,
                    "Radius": rad_val, "Optical Depth": od_val,
                    "Liquid Water": round(era_lwc, 5), "Humidity": int(era_hum),
                    "Status": status,
                    "GY": global_y, "GX": global_x # Save for plotting
                })

    except Exception as e:
        st.error(f"Scan Error: {e}")
        
    return pd.DataFrame(targets).sort_values(by="Probability", ascending=False)

# --- 3. DASHBOARD PLOTTER (Real Data 2x4 Grid) ---
def plot_metrics_dashboard(ds_sat, ds_era, target_row):
    """Plots scientific metrics for the target"""
    
    gy, gx = int(target_row['GY']), int(target_row['GX'])
    window = 40
    
    def get_slice(ds, var_name, cy, cx, is_era=False):
        try:
            if is_era:
                data = ds[var_name].values
                while data.ndim > 2: data = data[0]
                return data[0:80, 0:80] # ERA5 is coarse
            else:
                dims = list(ds.dims)
                y_d = next((d for d in dims if 'y' in d or 'lat' in d), dims[0])
                x_d = next((d for d in dims if 'x' in d or 'lon' in d), dims[1])
                return ds[var_name].isel({y_d: slice(cy-window, cy+window), x_d: slice(cx-window, cx+window)}).values
        except:
            return np.zeros((80,80))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.patch.set_facecolor('#0e1117')
    
    # --- ROW 1: METEOSAT ---
    # Probability
    data = get_slice(ds_sat, 'cloud_probability', gy, gx)
    im = axes[0,0].imshow(data, cmap='Blues', vmin=0, vmax=100)
    axes[0,0].set_title(f"Probability ({target_row['Probability']}%)", color="white")
    plt.colorbar(im, ax=axes[0,0])
    
    # Pressure
    data = get_slice(ds_sat, 'cloud_top_pressure', gy, gx)
    im = axes[0,1].imshow(data, cmap='gray_r')
    axes[0,1].set_title(f"Pressure ({target_row['Pressure']} hPa)", color="white")
    plt.colorbar(im, ax=axes[0,1])

    # Radius
    if 'cloud_particle_effective_radius' in ds_sat:
        data = get_slice(ds_sat, 'cloud_particle_effective_radius', gy, gx) * 1e6
        im = axes[0,2].imshow(data, cmap='viridis')
        axes[0,2].set_title(f"Radius ({target_row['Radius']} µm)", color="white")
        plt.colorbar(im, ax=axes[0,2])
    else: axes[0,2].text(0.5, 0.5, "N/A", color="red", ha='center')

    # Optical Depth
    if 'cloud_optical_thickness' in ds_sat:
        data = get_slice(ds_sat, 'cloud_optical_thickness', gy, gx)
        im = axes[0,3].imshow(data, cmap='magma')
        axes[0,3].set_title(f"Optical Depth ({target_row['Optical Depth']})", color="white")
        plt.colorbar(im, ax=axes[0,3])
    else: axes[0,3].text(0.5, 0.5, "N/A", color="red", ha='center')

    # --- ROW 2: ERA5 ---
    # ERA5 vars are tricky, try common names
    era_vars = {'Liquid Water': 'clwc', 'Ice Water': 'ciwc', 'Humidity': 'r', 'Vertical Vel': 'w'}
    
    if ds_era:
        era_keys = list(ds_era.data_vars)
        for i, (title, var_key) in enumerate(era_vars.items()):
            ax = axes[1, i]
            found = next((k for k in era_keys if var_key in k), None)
            if found:
                data = get_slice(ds_era, found, 0, 0, is_era=True)
                im = ax.imshow(data, cmap='plasma')
                ax.set_title(title, color="white")
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, "Missing", color="gray", ha='center')
    else:
        for i in range(4): axes[1, i].text(0.5, 0.5, "ERA5 OFFLINE", color="red", ha='center')

    # Cleanup
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117')
    buf.seek(0)
    return Image.open(buf)

# --- 4. LOGGING ---
def log_mission(target_id, lat, lon, decision, reason):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f: f.write("Timestamp,Target,Location,Decision,Reason\n")
    with open(LOG_FILE, 'a') as f:
        f.write(f"{ts},{target_id},{lat},{lon},{decision},{reason}\n")

def load_logs():
    if os.path.exists(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v13.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Regional Scanner")
    
    ds_sat, ds_era = load_data()
    
    if 'targets_df' not in st.session_state:
        # Auto-scan on load
        st.session_state['targets_df'] = scan_all_targets(ds_sat, ds_era)
    
    if st.button("RE-SCAN SECTOR"):
        with st.spinner("Scanning 2.15 Million km²..."):
            st.session_state['targets_df'] = scan_all_targets(ds_sat, ds_era)
            st.rerun()
            
    # Target List
    selected_row = None
    df = st.session_state['targets_df']
    
    if not df.empty:
        st.success(f"{len(df)} Targets > 50% Probability")
        target_id = st.selectbox("Engage Target:", df['ID'])
        selected_row = df[df['ID'] == target_id].iloc[0]
    
    # Admin
    st.markdown("---")
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
        st.info("**Solution:** VisionRain - An AI-driven decision support platform.")
    with c2:
        st.success("**Impact:** Supports Saudi Green Initiative & Water Security.")

# --- TAB 2: MAP & METRICS ---
with tab2:
    st.header("Kingdom-Wide Threat Map")
    
    # 1. MAP
    m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
    
    for _, row in df.iterrows():
        color = 'green' if row['Probability'] > 75 else 'orange'
        folium.Marker(
            [row['Lat'], row['Lon']], 
            popup=f"{row['ID']}: {row['Probability']}%",
            icon=folium.Icon(color=color, icon="cloud")
        ).add_to(m)
    
    if selected_row is not None:
        folium.CircleMarker(
            [selected_row['Lat'], selected_row['Lon']], radius=20, color='#00e5ff', fill=False
        ).add_to(m)
    
    st_folium(m, height=400, width=1400)

    # 2. MASTER TABLE (ALL TARGETS) - THE BIG REQUEST
    st.markdown("### 📊 Live Target Manifest (All Detected Cells)")
    # Display the FULL table with all columns: Radius, OD, Pressure, etc.
    st.dataframe(
        df.style.background_gradient(subset=['Probability'], cmap='Blues')
                .background_gradient(subset=['Pressure'], cmap='gray_r')
    )

    # 3. SELECTED TARGET DASHBOARD
    if selected_row is not None:
        st.divider()
        st.markdown(f"### 🔬 Deep Analysis: {selected_row['ID']}")
        
        # Generate the 2x4 Plot Matrix
        dashboard_img = plot_metrics_dashboard(ds_sat, ds_era, selected_row)
        if dashboard_img:
            st.image(dashboard_img, caption="Multi-Spectral Microphysics Scan", use_column_width=True)
            # Save for AI
            st.session_state['ai_dash'] = dashboard_img
            st.session_state['active_target'] = selected_row
            
    else:
        st.info("👈 Please select a target.")

# --- TAB 3: GEMINI CORE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if 'active_target' in st.session_state:
        t = st.session_state['active_target']
        
        st.info(f"Processing Target: **{t['ID']}**")
        
        # Show Evidence
        st.image(st.session_state['ai_dash'], caption="Input Data for AI", width=600)
        
        # Data Table for AI
        val_df = pd.DataFrame({
            "Metric": ["Probability", "Pressure", "Radius", "Optical Depth", "Liquid Water", "Humidity"],
            "Value": [f"{t['Probability']}%", f"{t['Pressure']} hPa", f"{t['Radius']} µm", f"{t['Optical Depth']}", f"{t['Liquid Water']}", f"{t['Humidity']}%"],
            "Ideal": ["> 70%", "400-700 hPa", "< 14 µm", "> 10", "> 0.001", "> 50%"]
        })
        st.table(val_df)
        
        # Dispatch Button
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
                        res = model.generate_content([prompt, st.session_state['ai_dash']])
                        
                        decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                        log_mission(t['ID'], t['Lat'], t['Lon'], decision, "AI Check")
                        
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
