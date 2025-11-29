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

# --- SAFELY IMPORT SCIENTIFIC LIBS ---
try:
    import xarray as xr
    import cfgrib
except ImportError:
    xr = None

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {
        background-color: #1a1a1a; 
        border: 1px solid #333; 
        border-radius: 12px; 
        padding: 15px;
    }
    .pitch-box {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #00e5ff;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: rgba(0, 255, 128, 0.1); 
        border: 1px solid #00ff80; 
        color: #00ff80; 
        padding: 15px; 
        border-radius: 10px;
    }
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

# --- 2. REAL DATA SCANNER (Saudi Sector) ---
def scan_kingdom_targets(ds_sat, ds_era):
    """
    Scans the actual data files for targets > 50% probability.
    Extracts metrics for the table.
    """
    targets = []
    
    if ds_sat is None: return pd.DataFrame()

    try:
        # 1. Slice Saudi Sector (Approx Pixels for Meteosat)
        # Y: 2000-2600, X: 500-1000 covers the peninsula
        y_slice = slice(2000, 2600)
        x_slice = slice(500, 1000)
        
        # 2. Extract Arrays (Handle missing vars gracefully)
        # Probability
        prob_grid = ds_sat['cloud_probability'].isel(y=y_slice, x=x_slice).values
        # Pressure
        press_grid = ds_sat['cloud_top_pressure'].isel(y=y_slice, x=x_slice).values
        
        # Radius (Optional)
        if 'cloud_particle_effective_radius' in ds_sat:
            rad_grid = ds_sat['cloud_particle_effective_radius'].isel(y=y_slice, x=x_slice).values * 1e6
        else: rad_grid = np.zeros_like(prob_grid)

        # Optical Depth (Optional)
        if 'cloud_optical_thickness' in ds_sat:
            od_grid = ds_sat['cloud_optical_thickness'].isel(y=y_slice, x=x_slice).values
        else: od_grid = np.zeros_like(prob_grid)

        # ERA5 Global Averages (For table context if matching is hard)
        era_lwc = 0.002
        era_hum = 60
        if ds_era:
            try:
                # Try to find liquid water variable
                for v in ['clwc', 'liquid']:
                     if v in ds_era: era_lwc = float(ds_era[v].mean()); break
                # Try to find humidity
                for v in ['r', 'rh', 'humidity']:
                     if v in ds_era: era_hum = float(ds_era[v].mean()); break
            except: pass

        # 3. SCAN Logic: Probability > 50%
        # We filter out invalid data (>100) and low prob (<50)
        valid_mask = (prob_grid > 50) & (prob_grid <= 100)
        y_idxs, x_idxs = np.where(valid_mask)
        
        if len(y_idxs) > 0:
            # Sort by probability to get best targets
            points = sorted(zip(y_idxs, x_idxs), key=lambda p: prob_grid[p[0], p[1]], reverse=True)
            # Sample distinct points (skip 200 pixels)
            selected_points = points[::200][:10]
            
            for i, (y, x) in enumerate(selected_points):
                # Restore Global Coords for Plotting
                global_y = y + 2000
                global_x = x + 500
                
                # Approx Lat/Lon Mapping
                lat = 24.0 + (300 - y) * 0.03
                lon = 45.0 + (x - 250) * 0.03
                
                # Get Values
                p_val = int(prob_grid[y, x])
                press_val = int(press_grid[y, x] / 100.0) # Pa -> hPa
                rad_val = round(float(rad_grid[y, x]), 1)
                od_val = round(float(od_grid[y, x]), 1)
                
                status = "HIGH PRIORITY" if p_val > 75 else "MODERATE"
                
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat, 4), "Lon": round(lon, 4),
                    "Probability": p_val,
                    "Pressure": press_val,
                    "Radius": rad_val,
                    "Optical Depth": od_val,
                    "Liquid Water": f"{era_lwc:.2e}",
                    "Humidity": int(era_hum),
                    "Status": status,
                    "GY": global_y, "GX": global_x # Hidden coords for plotter
                })

    except Exception as e:
        st.error(f"Scan Error: {e}")
        
    return pd.DataFrame(targets)

# --- 3. MATRIX PLOTTER (2x5 Grid - Real Data) ---
def generate_metrics_matrix(ds_sat, ds_era, gy, gx):
    """
    Plots the 2x5 Grid of REAL metrics. No noise.
    """
    window = 40 
    
    def get_slice(ds, keywords, cy=0, cx=0, is_era=False):
        try:
            # Find Variable
            found = None
            for k in ds.data_vars:
                if any(kw in k.lower() for kw in keywords): found = k; break
            if not found: return None
            
            data = ds[found].values
            
            if is_era:
                # ERA5 Handling
                while data.ndim > 2: data = data[0]
                return data[0:80, 0:80] # Crop for visibility
            else:
                # Meteosat Handling
                y_slice = slice(max(0, cy-window), cy+window)
                x_slice = slice(max(0, cx-window), cx+window)
                return ds[found].isel(y=y_slice, x=x_slice).values
        except: return None

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0e1117')
    
    # METEOSAT ROW
    sat_cfg = [
        ("Probability", ["prob"], "Blues"),
        ("Pressure", ["press", "ctp"], "gray_r"),
        ("Radius", ["rad", "reff"], "viridis"),
        ("Optical Depth", ["opt", "cot"], "magma"),
        ("Phase", ["phase"], "cool")
    ]
    
    for i, (title, kws, cmap) in enumerate(sat_cfg):
        ax = axes[0, i]
        data = get_slice(ds_sat, kws, gy, gx)
        if data is not None:
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else: ax.text(0.5, 0.5, "N/A", color="red", ha='center')
        ax.set_title(f"SAT: {title}", color="white", fontsize=10)
        ax.axis('off')

    # ERA5 ROW
    era_cfg = [
        ("Liquid Water", ["clwc", "liquid"], "Blues"),
        ("Ice Water", ["ciwc", "ice"], "PuBu"),
        ("Humidity", ["r", "rh", "humid"], "Greens"),
        ("Vertical Vel", ["w", "omega"], "RdBu"),
        ("Temp", ["t", "temp"], "inferno")
    ]

    for i, (title, kws, cmap) in enumerate(era_cfg):
        ax = axes[1, i]
        data = get_slice(ds_era, kws, is_era=True)
        if data is not None:
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else: ax.text(0.5, 0.5, "Missing", color="gray", ha='center')
        ax.set_title(f"ERA5: {title}", color="white", fontsize=10)
        ax.axis('off')

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
    st.caption("Kingdom Commander | v18.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Regional Scanner")
    
    ds_sat, ds_era = load_data()
    
    # Auto-Scan
    if 'targets_df' not in st.session_state:
        if ds_sat:
            with st.spinner("Scanning Kingdom Sector..."):
                st.session_state['targets_df'] = scan_kingdom_targets(ds_sat, ds_era)
        else:
            st.session_state['targets_df'] = pd.DataFrame()

    if st.button("RE-SCAN SECTOR"):
        if ds_sat:
            st.session_state['targets_df'] = scan_kingdom_targets(ds_sat, ds_era)
            st.rerun()
            
    df = st.session_state['targets_df']
    selected_row = None
    if not df.empty:
        st.success(f"{len(df)} Targets Identified")
        target_id = st.selectbox("Select Target:", df['ID'])
        selected_row = df[df['ID'] == target_id].iloc[0]
    
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "🗺️ Operations & Data", "🧠 Gemini Core"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Regions such as <b>Saudi Arabia</b> face increasing environmental challenges, including water scarcity, prolonged droughts, and heightened wildfire risk.
    Although cloud seeding is an established method, current operations remain manual, costly, and reactive.</p>
    <p>This challenge is strongly aligned with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - AI-driven decision support platform using satellite fusion.")
    with c2:
        st.success("**Impact:** Enables safer, cost-efficient unmanned deployment and scales globally.")

# --- TAB 2: OPERATIONS ---
with tab2:
    st.header("Kingdom-Wide Operations")
    
    # 1. MAP
    col_map, col_data = st.columns([2, 1])
    
    with col_map:
        m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
        if not df.empty:
            for _, row in df.iterrows():
                color = 'green' if row['Probability'] > 80 else 'orange'
                folium.Marker([row['Lat'], row['Lon']], popup=row['ID'], icon=folium.Icon(color=color, icon="cloud")).add_to(m)
            if selected_row is not None:
                folium.CircleMarker([selected_row['Lat'], selected_row['Lon']], radius=20, color='#00e5ff', fill=False).add_to(m)
        st_folium(m, height=400, width=700)

    with col_data:
        st.subheader("Live Targets")
        if not df.empty:
            st.dataframe(df[['ID', 'Probability', 'Status']].style.highlight_max(axis=0))
        else:
            st.warning("No targets found. Check files.")

    # 2. TARGET ANALYSIS (The Wall)
    if selected_row is not None:
        st.divider()
        st.markdown(f"### 🔬 Deep Analysis: {selected_row['ID']}")
        
        gy, gx = selected_row.get('GY', 2300), selected_row.get('GX', 750)
        matrix_img = generate_metrics_matrix(ds_sat, ds_era, int(gy), int(gx))
        
        if matrix_img:
            st.image(matrix_img, caption="Scientific Data Matrix (Meteosat + ERA5)", use_column_width=True)
            st.session_state['ai_matrix'] = matrix_img
            st.session_state['active_target'] = selected_row

# --- TAB 3: GEMINI ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if 'active_target' in st.session_state and 'ai_matrix' in st.session_state:
        t = st.session_state['active_target']
        st.info(f"Engaging: **{t['ID']}**")
        
        st.image(st.session_state['ai_matrix'], width=600, caption="Visual Evidence")
        
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
                    1. IF Radius < 14 AND Optical Depth > 10 -> "GO" (Ideal).
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
                except Exception as e: st.error(f"AI Error: {e}")
    else:
        st.warning("Select a target in Tab 2 first.")
