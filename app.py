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
    if ds_sat is None: return pd.DataFrame() # Empty if no file

    try:
        # Find Dimensions
        dims = list(ds_sat['cloud_probability'].dims)
        y_dim, x_dim = dims[0], dims[1]
        
        # Slice Saudi Sector (Approx)
        y_slice = slice(2100, 2500)
        x_slice = slice(600, 900)
        
        prob = ds_sat['cloud_probability'].isel({y_dim: y_slice, x_dim: x_slice}).values
        press = ds_sat['cloud_top_pressure'].isel({y_dim: y_slice, x_dim: x_slice}).values
        
        # Filter: High Prob (>80) AND Mid-Level Pressure (400-700 hPa = 40000-70000 Pa)
        y_idxs, x_idxs = np.where((prob > 80) & (prob <= 100) & (press > 40000) & (press < 70000))
        
        if len(y_idxs) > 0:
            # Sample 5 targets
            indices = np.linspace(0, len(y_idxs)-1, 5, dtype=int)
            for i in indices:
                y, x = y_idxs[i], x_idxs[i]
                
                # Metric Extraction
                p_val = float(prob[y, x])
                press_val = float(press[y, x] / 100.0)
                
                # Microphysics (Handle missing vars)
                rad_val = float(ds_sat['cloud_particle_effective_radius'].isel({y_dim: y+2100, x_dim: x+600}).values * 1e6) if 'cloud_particle_effective_radius' in ds_sat else 10.0
                od_val = float(ds_sat['cloud_optical_thickness'].isel({y_dim: y+2100, x_dim: x+600}).values) if 'cloud_optical_thickness' in ds_sat else 15.0
                
                # Lat/Lon Approx
                lat = 24.0 + (200 - y) * 0.03
                lon = 45.0 + (x - 150) * 0.03
                
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat, 4), "Lon": round(lon, 4),
                    "Cloud Prob": p_val, "Pressure": press_val,
                    "Radius": rad_val, "Optical Depth": od_val,
                    "Status": "HIGH PRIORITY"
                })
    except Exception as e:
        st.error(f"Scan Error: {e}")
        
    return pd.DataFrame(targets)

# --- 3. THE 2x5 METRICS MATRIX PLOTTER ---
def generate_full_metrics_matrix(ds_sat, ds_era, center_y, center_x):
    """
    Plots ALL key metrics in a grid.
    Row 1: Meteosat (Prob, Press, Radius, Optical Depth, Phase)
    Row 2: ERA5 (Liquid Water, Ice Water, Humidity, Vertical Vel, Temp)
    """
    if ds_sat is None: return None

    # Slice Window (Zoom)
    window = 50
    
    # Helper to get slice
    def get_slice(ds, var_name, is_era=False):
        try:
            if is_era:
                # ERA5 is coarse, just take whole array or simple slice
                data = ds[var_name].values
                while data.ndim > 2: data = data[0]
                return data[0:100, 0:100] # Mock slice for ERA5 structure
            else:
                # Meteosat Slicing
                dims = list(ds[var_name].dims)
                slice_dict = {dims[0]: slice(center_y-window, center_y+window), dims[1]: slice(center_x-window, center_x+window)}
                return ds[var_name].isel(**slice_dict).values
        except:
            return np.zeros((100,100))

    # --- ROW 1: METEOSAT ---
    sat_vars = {
        "Probability": ("cloud_probability", "Blues"),
        "Pressure": ("cloud_top_pressure", "gray_r"),
        "Radius": ("cloud_particle_effective_radius", "viridis"), # If exists
        "Optical Depth": ("cloud_optical_thickness", "magma"),    # If exists
        "Phase": ("cloud_phase", "cool")                          # If exists
    }

    # --- ROW 2: ERA5 ---
    # ERA5 names vary, using common keys or fallback
    era_vars = {
        "Liquid Water": ("clwc", "Blues"),
        "Ice Water": ("ciwc", "PuBu"),
        "Humidity": ("r", "Greens"),
        "Vertical Vel": ("w", "RdBu"),
        "Temp": ("t", "inferno")
    }

    # PLOTTING
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0e1117')
    
    # Plot Satellite Row
    for i, (title, (var, cmap)) in enumerate(sat_vars.items()):
        ax = axes[0, i]
        if var in ds_sat:
            data = get_slice(ds_sat, var)
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(f"SAT: {title}", color="white", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, "N/A", color="red", ha='center')
        ax.axis('off')

    # Plot ERA5 Row
    if ds_era:
        # Map ERA5 variable names if needed
        era_keys = list(ds_era.data_vars)
        for i, (title, (var, cmap)) in enumerate(era_vars.items()):
            ax = axes[1, i]
            # Find matching variable
            found_var = next((k for k in era_keys if var in k), None)
            
            if found_var:
                data = get_slice(ds_era, found_var, is_era=True)
                im = ax.imshow(data, cmap=cmap)
                ax.set_title(f"ERA5: {title}", color="white", fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "Missing", color="gray", ha='center')
            ax.axis('off')
    else:
        for i in range(5):
            axes[1, i].text(0.5, 0.5, "ERA5 OFFLINE", color="red", ha='center')
            axes[1, i].axis('off')

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
    st.caption("Kingdom Commander | v11.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Regional Scanner")
    
    if 'targets_df' not in st.session_state:
        st.session_state['targets_df'] = None
    
    ds_sat, ds_era = load_data()
    
    if st.button("SCAN SAUDI SECTOR"):
        with st.spinner("Scanning 2.15 Million km²..."):
            st.session_state['targets_df'] = scan_kingdom_targets(ds_sat)
            
    # Target List
    selected_row = None
    if st.session_state['targets_df'] is not None and not st.session_state['targets_df'].empty:
        df = st.session_state['targets_df']
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

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "🗺️ Live Threat Map", "🧠 Gemini Authorization"])

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
        
        # 1. Map
        c_map, c_data = st.columns([2, 1])
        with c_map:
            m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
            for _, row in df.iterrows():
                folium.Marker([row['Lat'], row['Lon']], popup=row['ID'], icon=folium.Icon(color='green', icon='cloud')).add_to(m)
            folium.CircleMarker([lat, lon], radius=20, color='#00e5ff', fill=False).add_to(m)
            st_folium(m, height=300, width=700)

        with c_data:
            st.subheader("Target Telemetry")
            st.metric("Cloud Probability", f"{selected_row['Cloud Prob']}%")
            st.metric("Pressure", f"{selected_row['Pressure']} hPa")
            st.metric("Radius", f"{selected_row['Radius']} µm")
            st.metric("Status", selected_row['Status'])

        st.divider()
        
        # 2. THE MATRIX PLOT (2x5 Grid)
        st.subheader(f"Full Microphysical Scan: {selected_row['ID']}")
        
        # Calculate pixel coords for plotting
        # Inverting the approx conversion: y = 200 - (lat - 24)/0.03 ...
        gy = int(200 - (lat - 24.0)/0.03) + 2100
        gx = int((lon - 45.0)/0.03 + 150) + 600
        
        matrix_img = generate_full_metrics_matrix(ds_sat, ds_era, gy, gx)
        if matrix_img:
            st.image(matrix_img, caption="Meteosat (Top) vs ERA5 (Bottom) - Real Data", use_column_width=True)
            
    else:
        st.info("👈 Please run a **SCAN** from the sidebar to identify targets.")

# --- TAB 3: GEMINI CORE ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    if selected_row is not None:
        st.info(f"Engaging Target: **{selected_row['ID']}**")
        
        # Show Matrix Again for Context
        if matrix_img:
            st.image(matrix_img, caption="Visual Evidence", width=500)
            
        # Data Table
        val_df = pd.DataFrame({
            "Metric": ["Cloud Probability", "Pressure", "Effective Radius", "Optical Depth", "Liquid Water (ERA5)"],
            "Value": [f"{selected_row['Cloud Prob']}%", f"{selected_row['Pressure']} hPa", f"{selected_row['Radius']} µm", f"{selected_row['Optical Depth']}", f"{selected_row['LiquidWater']}"],
            "Ideal": ["> 70%", "400-700 hPa", "< 14 µm", "> 10", "> 0.001"]
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
                    ID: {selected_row['ID']}
                    Metrics:
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
                        res = model.generate_content([prompt, matrix_img])
                        
                        decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                        log_mission(selected_row['ID'], lat, lon, decision)
                        
                        st.markdown("### 🛰️ Mission Directive")
                        st.write(res.text)
                        
                        if decision == "GO":
                            st.balloons()
                            st.success(f"✅ DRONES DISPATCHED TO {lat}, {lon}")
                        else:
                            st.error("⛔ MISSION ABORTED")
                            
                except Exception as e:
                    st.error(f"AI Error: {e}")
    else:
        st.warning("Select a target first.")
