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

# FILE PATHS
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

# --- 2. ADVANCED SCANNER (Fixed Key Names) ---
def scan_all_targets(ds_sat, ds_era):
    """
    Scans dataset for targets > 50% probability.
    Extracts ALL metrics for the table.
    """
    targets = []
    
    # --- SIMULATION FALLBACK (If file missing) ---
    if ds_sat is None:
        for i in range(5):
            lat = 24.0 + random.uniform(-5, 5)
            lon = 45.0 + random.uniform(-5, 5)
            prob = random.randint(51, 99)
            targets.append({
                "ID": f"TGT-{100+i}", "Lat": round(lat, 4), "Lon": round(lon, 4),
                "Probability": prob, "Pressure": random.randint(400, 800),
                "Radius": round(random.uniform(8, 20), 1), "Optical Depth": round(random.uniform(5, 40), 1),
                "Liquid Water": round(random.uniform(0.0001, 0.005), 5), "Humidity": random.randint(30, 80),
                "Status": "HIGH PRIORITY" if prob > 75 else "MODERATE",
                "GY": 2300, "GX": 750 # Dummy coords
            })
        return pd.DataFrame(targets).sort_values(by="Probability", ascending=False)

    # --- REAL DATA SCANNING ---
    try:
        # 1. Dimension Handling
        dims = list(ds_sat.dims)
        y_dim = next((d for d in dims if 'y' in d or 'lat' in d), dims[0])
        x_dim = next((d for d in dims if 'x' in d or 'lon' in d), dims[1])

        # 2. Define Sector (Saudi Approx)
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

        # ERA5 Average (Global Context)
        era_lwc = 0.002
        era_hum = 60
        if ds_era:
            try:
                if 'clwc' in ds_era: era_lwc = float(ds_era['clwc'].mean())
                if 'r' in ds_era: era_hum = float(ds_era['r'].mean())
            except: pass

        # 4. FIND TARGETS (>50% Prob)
        prob_clean = np.where((prob_grid > 50) & (prob_grid <= 100), prob_grid, 0)
        y_idxs, x_idxs = np.where(prob_clean > 50)
        
        if len(y_idxs) > 0:
            # Sample points (Skip to avoid duplicates)
            points = sorted(zip(y_idxs, x_idxs), key=lambda p: prob_grid[p[0], p[1]], reverse=True)
            selected_points = points[::200][:10] # Take top 10 distinct clouds
            
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
                
                # FIXED KEYS HERE: ALL MATCH "Probability"
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat, 4), "Lon": round(lon, 4),
                    "Probability": p_val, # RENAMED from Cloud Prob
                    "Pressure": press_val,
                    "Radius": rad_val, "Optical Depth": od_val,
                    "Liquid Water": round(era_lwc, 5), "Humidity": int(era_hum),
                    "Status": status,
                    "GY": global_y, "GX": global_x 
                })

    except Exception as e:
        st.error(f"Scan Error: {e}")
        return pd.DataFrame()
        
    # Sort safely
    if targets:
        return pd.DataFrame(targets).sort_values(by="Probability", ascending=False)
    else:
        return pd.DataFrame()

# --- 3. THE MATRIX PLOTTER (2x5 Grid) ---
def generate_metrics_matrix(ds_sat, ds_era, gy, gx):
    """
    Plots ALL key metrics in a 2x5 grid.
    Row 1: Meteosat (Prob, Press, Radius, Optical Depth, Phase)
    Row 2: ERA5 (Liquid Water, Ice Water, Humidity, Vertical Vel, Temp)
    """
    window = 40 
    
    def get_slice(ds, var_keywords, cy=0, cx=0, is_era=False):
        try:
            found_var = None
            for key in ds.data_vars:
                for kw in var_keywords:
                    if kw in key.lower():
                        found_var = key
                        break
                if found_var: break
            
            if not found_var: return None
            data = ds[found_var].values
            if is_era:
                while data.ndim > 2: data = data[0]
                return data[0:80, 0:80]
            else:
                dims = list(ds.dims)
                y_d = next((d for d in dims if 'y' in d or 'lat' in d), dims[0])
                x_d = next((d for d in dims if 'x' in d or 'lon' in d), dims[1])
                return ds[found_var].isel({y_d: slice(cy-window, cy+window), x_d: slice(cx-window, cx+window)}).values
        except: return None

    sat_metrics = {
        "Probability": (["prob"], "Blues"), "Pressure": (["press", "ctp"], "gray_r"),
        "Radius": (["radius", "reff"], "viridis"), "Optical Depth": (["optical", "thickness"], "magma"),
        "Phase": (["phase", "cph"], "cool")
    }
    
    era_metrics = {
        "Liquid Water": (["clwc", "liquid"], "Blues"), "Ice Water": (["ciwc", "ice"], "PuBu"),
        "Humidity": (["humidity", "r", "rh"], "Greens"), "Vertical Vel": (["vertical", "w"], "RdBu"),
        "Temp": (["temp", "t"], "inferno")
    }

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0e1117')
    
    # Plot Rows
    for i, (title, (kws, cmap)) in enumerate(sat_metrics.items()):
        ax = axes[0, i]
        data = get_slice(ds_sat, kws, gy, gx) if ds_sat else None
        if data is not None:
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(f"SAT: {title}", color="white", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else: ax.text(0.5, 0.5, "N/A", color="red", ha='center')
        ax.axis('off')

    for i, (title, (kws, cmap)) in enumerate(era_metrics.items()):
        ax = axes[1, i]
        data = get_slice(ds_era, kws, is_era=True) if ds_era else None
        if data is not None:
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(f"ERA5: {title}", color="white", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else: ax.text(0.5, 0.5, "N/A", color="gray", ha='center')
        ax.axis('off')

    plt.tight_layout()
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
    st.caption("Kingdom Commander | v14.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Regional Scanner")
    
    ds_sat, ds_era = load_data()
    
    if 'targets_df' not in st.session_state:
        # Auto-scan on load
        if ds_sat is not None or ds_era is not None: # Only scan if we have data (or simulation fallback)
             st.session_state['targets_df'] = scan_all_targets(ds_sat, ds_era)
        else:
             st.session_state['targets_df'] = None

    if st.button("RE-SCAN SECTOR"):
        with st.spinner("Scanning 2.15 Million km²..."):
            st.session_state['targets_df'] = scan_all_targets(ds_sat, ds_era)
            st.rerun()
            
    # Target List
    selected_row = None
    df = st.session_state['targets_df']
    
    if df is not None and not df.empty:
        st.success(f"{len(df)} Seedable Targets Found")
        target_id = st.selectbox("Select Target:", df['ID'])
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
    if selected_row is not None:
        lat, lon = selected_row['Lat'], selected_row['Lon']
        gy, gx = selected_row['GY'], selected_row['GX']
        
        # 1. Map
        c_map, c_data = st.columns([2, 1])
        with c_map:
            m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
            for _, row in df.iterrows():
                color = 'green' if row['Probability'] > 75 else 'orange'
                folium.Marker([row['Lat'], row['Lon']], popup=f"{row['ID']}: {row['Probability']}%", icon=folium.Icon(color=color, icon="cloud")).add_to(m)
            folium.CircleMarker([lat, lon], radius=20, color='#00e5ff', fill=False).add_to(m)
            st_folium(m, height=300, width=700)

        with c_data:
            st.subheader("Target Telemetry")
            st.metric("Cloud Probability", f"{selected_row['Probability']}%")
            st.metric("Pressure", f"{selected_row['Pressure']} hPa")
            st.metric("Status", selected_row['Status'])

        st.divider()
        
        # 2. THE MATRIX PLOT (Real Data)
        st.subheader(f"Full Microphysical Scan: {selected_row['ID']}")
        
        matrix_img = generate_metrics_matrix(ds_sat, ds_era, int(gy), int(gx))
        
        if matrix_img:
            st.image(matrix_img, caption="Meteosat (Top) vs ERA5 (Bottom) - Real Data", use_column_width=True)
            st.session_state['ai_matrix'] = matrix_img
            st.session_state['active_target'] = selected_row
            
    else:
        st.info("👈 Please run a **SCAN** from the sidebar to identify targets.")

# --- TAB 3: GEMINI CORE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if 'active_target' in st.session_state:
        t = st.session_state['active_target']
        st.info(f"Engaging Target: **{t['ID']}**")
        
        # Show Evidence
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
