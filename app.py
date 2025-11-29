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
    st.error("⚠️ Scientific Libraries Missing! Please check requirements.txt")
    xr = None

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"

# FILES (Must be in same folder)
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Heatmap Commander", layout="wide", page_icon="⛈️")

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

# --- 1. DATA LOADERS (Real Data Only) ---
@st.cache_resource
def load_data():
    ds_sat, ds_era = None, None
    if xr:
        if os.path.exists(NETCDF_FILE):
            try: ds_sat = xr.open_dataset(NETCDF_FILE, engine='netcdf4')
            except Exception as e: st.error(f"Satellite Load Error: {e}")
        
        if os.path.exists(ERA5_FILE):
            try: ds_era = xr.open_dataset(ERA5_FILE, engine='cfgrib')
            except Exception as e: st.warning(f"ERA5 Load Error (Check packages.txt): {e}")
            
    return ds_sat, ds_era

# --- 2. HEATMAP GENERATOR (The "Colorful Map") ---
def generate_heatmap(ds, var_name, title, cmap, is_era=False):
    """Generates a heatmap for a specific variable from the dataset."""
    if ds is None: return None
    
    try:
        # Find variable
        target_var = None
        for v in ds.data_vars:
            if var_name in v.lower():
                target_var = v
                break
        
        if not target_var:
            # Fallback: Try to find anything that matches keywords
            keywords = var_name.split()
            for v in ds.data_vars:
                if any(k in v.lower() for k in keywords):
                    target_var = v
                    break
        
        if not target_var: return None

        # Extract Data
        data = ds[target_var].values
        
        # Flatten dimensions (Time/Level)
        if is_era:
            while data.ndim > 2: data = data[0]
        else:
            # For Meteosat, we might need to slice if the file is Full Disk
            # Slicing Saudi Sector (approximate indices)
            if data.shape[0] > 3000:
                data = data[2100:2500, 600:900] 
        
        # Mask invalid values (Space/Errors)
        data = np.where((data > -999) & (data < 99999), data, np.nan)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0e1117')
        
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        ax.set_title(f"{title}", color="white", fontsize=14)
        ax.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor='#0e1117', bbox_inches='tight')
        buf.seek(0)
        return Image.open(buf)

    except Exception as e:
        st.error(f"Heatmap Error ({var_name}): {e}")
        return None

# --- 3. TARGET SCANNER (Real Data) ---
def scan_targets(ds_sat, ds_era):
    targets = []
    if ds_sat is None: return pd.DataFrame()

    try:
        # Extract Key Metrics
        # Probability
        prob_var = next((v for v in ds_sat if 'probability' in v), None)
        if not prob_var: return pd.DataFrame()
        
        # Slicing (Saudi)
        y_slice = slice(2100, 2500)
        x_slice = slice(600, 900)
        
        prob_data = ds_sat[prob_var].isel(y=y_slice, x=x_slice).values
        
        # Find targets > 50%
        y_idxs, x_idxs = np.where((prob_data > 50) & (prob_data <= 100))
        
        # Sample points
        if len(y_idxs) > 0:
            # Pick top 10 distinct
            indices = np.linspace(0, len(y_idxs)-1, 10, dtype=int)
            
            for i in indices:
                y, x = y_idxs[i], x_idxs[i]
                
                # Coords (Approx)
                lat = 24.0 + (200 - y) * 0.03
                lon = 45.0 + (x - 150) * 0.03
                
                # Get Value
                p_val = int(prob_data[y, x])
                
                # Determine Status
                status = "HIGH PRIORITY" if p_val > 75 else "MODERATE"
                
                # Extract other metrics if available
                press_val = 0
                if 'cloud_top_pressure' in ds_sat:
                    press_val = int(ds_sat['cloud_top_pressure'].isel(y=y+2100, x=x+600).values / 100)
                
                targets.append({
                    "ID": f"TGT-{100+i}",
                    "Lat": round(lat, 4), "Lon": round(lon, 4),
                    "Probability": p_val, "Pressure": press_val,
                    "Status": status,
                    "GY": y + 2100, "GX": x + 600
                })
                
    except Exception as e:
        st.error(f"Scan Error: {e}")
        
    return pd.DataFrame(targets).sort_values(by="Probability", ascending=False)

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
    st.caption("Kingdom Commander | v17.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📡 Auto-Scanner")
    ds_sat, ds_era = load_data()
    
    if 'targets_df' not in st.session_state:
        if ds_sat:
            with st.spinner("Scanning Sector..."):
                st.session_state['targets_df'] = scan_targets(ds_sat, ds_era)
        else:
            st.session_state['targets_df'] = pd.DataFrame()
            
    df = st.session_state['targets_df']
    
    selected_row = None
    if not df.empty:
        st.success(f"{len(df)} Targets Found")
        target_id = st.selectbox("Select Target:", df['ID'])
        selected_row = df[df['ID'] == target_id].iloc[0]
    
    st.markdown("---")
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")

tab1, tab2, tab3 = st.tabs(["🌍 Pitch", "🗺️ Heatmaps & Targets", "🧠 Gemini Authorization"])

# TAB 1
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
    c1.info("**Solution:** VisionRain - AI-driven decision support.")
    c2.success("**Impact:** Supports Saudi Green Initiative.")

# TAB 2
with tab2:
    st.header("Kingdom-Wide Analysis")
    
    col_map, col_heat = st.columns([1, 1])
    
    with col_map:
        st.subheader("Live Threat Map")
        m = folium.Map(location=[24.0, 45.0], zoom_start=5, tiles="CartoDB dark_matter")
        
        if not df.empty:
            for _, row in df.iterrows():
                color = 'green' if row['Probability'] > 80 else 'orange'
                folium.Marker([row['Lat'], row['Lon']], popup=f"{row['ID']}", icon=folium.Icon(color=color, icon="cloud")).add_to(m)
            
            if selected_row is not None:
                folium.CircleMarker([selected_row['Lat'], selected_row['Lon']], radius=20, color='#00e5ff', fill=False).add_to(m)
                
        st_folium(m, height=400, width=600)

    with col_heat:
        st.subheader("Spectral Layer Viewer")
        
        layer_option = st.selectbox("Select Metric Layer:", [
            "Meteosat: Cloud Probability",
            "Meteosat: Cloud Top Pressure",
            "Meteosat: Effective Radius",
            "Meteosat: Optical Depth",
            "ERA5: Liquid Water Content",
            "ERA5: Humidity",
            "ERA5: Vertical Velocity"
        ])
        
        # Map selection to variable names
        layer_map = {
            "Meteosat: Cloud Probability": ("probability", "Blues", ds_sat, False),
            "Meteosat: Cloud Top Pressure": ("pressure", "gray_r", ds_sat, False),
            "Meteosat: Effective Radius": ("radius", "viridis", ds_sat, False),
            "Meteosat: Optical Depth": ("optical", "magma", ds_sat, False),
            "ERA5: Liquid Water Content": ("clwc", "Blues", ds_era, True),
            "ERA5: Humidity": ("r", "Greens", ds_era, True),
            "ERA5: Vertical Velocity": ("w", "RdBu", ds_era, True)
        }
        
        var_key, cmap, ds_source, is_era = layer_map[layer_option]
        
        if ds_source:
            heatmap_img = generate_heatmap(ds_source, var_key, layer_option, cmap, is_era)
            if heatmap_img:
                st.image(heatmap_img, caption=f"Live Layer: {layer_option}", use_column_width=True)
                if selected_row is not None:
                    st.session_state['ai_heatmap'] = heatmap_img # Save for AI
            else:
                st.warning(f"Variable '{var_key}' not found in dataset.")
        else:
            st.error("Dataset not loaded.")

    # FULL DATA TABLE
    st.markdown("### 📊 Live Target Manifest")
    if not df.empty:
        st.dataframe(df.style.background_gradient(subset=['Probability'], cmap='Blues'))

# TAB 3
with tab3:
    st.header("Gemini Fusion Engine")
    
    if selected_row is not None:
        t = selected_row
        st.info(f"Engaging: **{t['ID']}**")
        
        # Show Context
        if 'ai_heatmap' in st.session_state:
            st.image(st.session_state['ai_heatmap'], caption="Contextual Heatmap", width=500)
            
        # Metrics Table
        val_df = pd.DataFrame({
            "Metric": ["Probability", "Pressure", "Status"],
            "Value": [f"{t['Probability']}%", f"{t['Pressure']} hPa", t['Status']],
            "Ideal": ["> 70%", "400-700 hPa", "HIGH"]
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
                    ACT AS A MISSION COMMANDER. Analyze this Target.
                    ID: {t['ID']} | Location: {t['Lat']}, {t['Lon']}
                    
                    METRICS:
                    - Probability: {t['Probability']}%
                    - Pressure: {t['Pressure']} hPa
                    
                    LOGIC:
                    1. IF Probability > 80 AND Pressure < 700 -> GO.
                    2. ELSE -> NO-GO.
                    
                    OUTPUT:
                    1. Decision: GO/NO-GO.
                    2. Reasoning.
                    """
                    
                    with st.spinner("Vertex AI Validating..."):
                        res = model.generate_content([prompt])
                        decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                        log_mission(t['ID'], t['Lat'], t['Lon'], decision)
                        
                        st.markdown("### 🛰️ Mission Directive")
                        st.write(res.text)
                        if decision == "GO": 
                            st.balloons()
                            st.success("✅ DRONES DISPATCHED")
                        else: 
                            st.error("⛔ ABORTED")
                            
                except Exception as e: st.error(f"AI Error: {e}")
    else:
        st.warning("Select a target in Tab 2 first.")
