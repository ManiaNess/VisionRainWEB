import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import time
from io import BytesIO
import folium
from streamlit_folium import st_folium

# --- SCIENTIFIC LIBRARIES ---
try:
    import xarray as xr
    import cfgrib
except ImportError:
    st.error("Scientific libraries not installed. Please check requirements.txt")

# --- FIREBASE ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- FILE PATHS (Your Exact Files) ---
DEFAULT_NC = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
DEFAULT_GRIB = "ce636265319242f2fef4a83020b30ecf.grib"

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .analysis-text {font-family: 'Courier New', monospace; color: #00ff80; background-color: #111; padding: 15px; border-left: 3px solid #00ff80; border-radius: 5px; margin-top: 10px;}
    .analysis-fail {color: #ff4444; border-left: 3px solid #ff4444;}
    </style>
    """, unsafe_allow_html=True)

# --- SECTOR MAPPING (Pixel Coordinates for your NC file) ---
SAUDI_SECTORS = {
    "Jeddah": {"coords": [21.5433, 39.1728], "y": 2300, "x": 750},
    "Abha":   {"coords": [18.2164, 42.5053], "y": 2500, "x": 800},
    "Riyadh": {"coords": [24.7136, 46.6753], "y": 2100, "x": 900},
    "Dammam": {"coords": [26.4207, 50.0888], "y": 2000, "x": 950},
    "Tabuk":  {"coords": [28.3835, 36.5662], "y": 1900, "x": 700}
}

# --- DATA ENGINE: REAL FILE READER ONLY ---
@st.cache_resource
def load_datasets(nc_path, grib_path):
    """Loads the heavy scientific files once."""
    ds_sat, ds_era = None, None
    
    if os.path.exists(nc_path):
        try: ds_sat = xr.open_dataset(nc_path, engine='netcdf4')
        except: pass
    
    if os.path.exists(grib_path):
        try: ds_era = xr.open_dataset(grib_path, engine='cfgrib')
        except: pass
            
    return ds_sat, ds_era

def find_var(ds, keys):
    """Smart variable hunter."""
    if ds is None: return None
    for v in ds.data_vars:
        if any(k in v.lower() for k in keys): return ds[v]
    return None

def extract_real_data(ds_sat, ds_era, sector_name, window=50):
    """
    Extracts RAW numpy arrays from the files. 
    Strictly no simulation. Returns empty zeros if files missing.
    """
    coords = SAUDI_SECTORS[sector_name]
    cy, cx = coords['y'], coords['x']
    
    metrics = {}
    arrays = {}
    
    # 1. Satellite Extraction (Pixel Logic)
    if ds_sat:
        y_slice = slice(max(0, cy - window), cy + window)
        x_slice = slice(max(0, cx - window), cx + window)
        
        def get_s(keys, name, scale=1.0):
            var = find_var(ds_sat, keys)
            if var is not None:
                # Assuming (y, x) dims are last two
                d = var.isel(y=y_slice, x=x_slice).values * scale
                # Mask fill values (usually extremely high or low numbers in NetCDF)
                d = np.ma.masked_where((d < -100) | (d > 5000), d)
                arrays[name] = d
                metrics[name] = float(np.ma.mean(d)) if np.ma.count(d) > 0 else 0.0
            else:
                arrays[name] = np.zeros((window*2, window*2))
                metrics[name] = 0.0

        get_s(['prob', 'cloud_probability'], 'prob')
        get_s(['press', 'pressure'], 'press', 0.01) # Pa -> hPa
        get_s(['rad', 'effective', 'cre'], 'rad')
        get_s(['opt', 'thickness'], 'opt')
        get_s(['phase'], 'phase')
        
    else:
        # FAIL SAFE: Zeros if file missing
        for k in ['prob','press','rad','opt','phase']: 
            arrays[k] = np.zeros((100,100))
            metrics[k] = 0.0

    # 2. ERA5 Extraction (Approximate Crop)
    if ds_era:
        def get_e(keys, name, scale=1.0):
            var = find_var(ds_era, keys)
            if var is not None:
                # Take raw values, handle time/step dims
                d = var.values
                while len(d.shape) > 2: d = d[0] # Flatten time
                
                # Simple center crop for visual consistency
                my, mx = d.shape[0]//2, d.shape[1]//2
                crop = d[my-window:my+window, mx-window:mx+window] * scale
                arrays[name] = crop
                metrics[name] = float(np.mean(crop))
            else:
                arrays[name] = np.zeros((window*2, window*2))
                metrics[name] = 0.0

        get_e(['clwc', 'liquid'], 'lwc')
        get_e(['ciwc', 'ice'], 'ice')
        get_e(['r', 'humidity'], 'rh')
        get_e(['w', 'vertical'], 'w')
        get_e(['t', 'temp'], 'temp', 1.0 - 273.15) # K -> C
    else:
        for k in ['lwc','ice','rh','w','temp']:
            arrays[k] = np.zeros((100,100))
            metrics[k] = 0.0

    return metrics, arrays

# --- VISUALIZER ---
def plot_real_matrix(arrays):
    """
    Plots the ACTUAL extracted arrays.
    Uses 'bicubic' interpolation to look like a smooth weather map (REALISTIC), not blobs.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # Configuration
    plots = [
        # Row 1: Satellite
        (0,0, "Cloud Probability", "Blues", arrays['prob'], 0, 100),
        (0,1, "Cloud Top Pressure", "gray_r", arrays['press'], 200, 1000),
        (0,2, "Effective Radius", "viridis", arrays['rad'], 0, 30),
        (0,3, "Optical Depth", "magma", arrays['opt'], 0, 50),
        (0,4, "Cloud Phase", "cool", arrays['phase'], 0, 3), # 0=Clear, 1=Liquid, 2=Ice
        
        # Row 2: ERA5
        (1,0, "Liquid Water", "Blues", arrays['lwc'], None, None),
        (1,1, "Ice Water", "PuBu", arrays['ice'], None, None),
        (1,2, "Rel. Humidity", "Greens", arrays['rh'], 0, 100),
        (1,3, "Vertical Velocity", "RdBu", arrays['w'], -2, 2),
        (1,4, "Temperature (°C)", "inferno", arrays['temp'], -40, 40),
    ]
    
    for r, c, title, cmap, data, vmin, vmax in plots:
        ax = axes[r, c]
        ax.set_facecolor('#0e1117')
        
        # INTERPOLATION IS KEY FOR REALISM
        # 'bicubic' makes grid data look like a smooth photo/heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='bicubic', vmin=vmin, vmax=vmax)
        
        ax.set_title(title, color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Handle Phase Text Overlay
        if "Phase" in title:
            # Calculate dominant phase in center
            center_val = data[data.shape[0]//2, data.shape[1]//2]
            phase_txt = "LIQUID" if 0.5 < center_val < 1.5 else "ICE" if center_val > 1.5 else "CLEAR"
            color = "cyan" if phase_txt == "LIQUID" else "white"
            ax.text(0.5, 0.5, phase_txt, color=color, ha="center", va="center", 
                   transform=ax.transAxes, fontsize=14, fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        else:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- INIT ---
ds_sat, ds_era = load_datasets(DEFAULT_NC, DEFAULT_GRIB)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("v31.0 | Real Data Core")
    
    st.markdown("### 📡 Data Link")
    if ds_sat: st.markdown('<span style="color:#00ff80">● Meteosat Stream Active</span>', unsafe_allow_html=True)
    else: st.error("Meteosat .nc File Not Found")
    
    if ds_era: st.markdown('<span style="color:#00ff80">● ERA5 Stream Active</span>', unsafe_allow_html=True)
    else: st.warning("ERA5 .grib File Not Found")
    
    st.divider()
    
    # Region Selector
    region_options = list(SAUDI_SECTORS.keys())
    if 'selected_region' not in st.session_state: st.session_state.selected_region = region_options[0]
    
    selected = st.selectbox("Active Sector", region_options, key="region_box")
    if selected != st.session_state.selected_region:
        st.session_state.selected_region = selected
        
    api_key = st.text_input("Gemini API Key", type="password")

# --- MAIN UI ---
st.title("VisionRain Command Center")

# 1. EXTRACT REAL DATA
metrics, arrays = extract_real_data(ds_sat, ds_era, st.session_state.selected_region)

# 2. TELEMETRY HEADER
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cloud Probability", f"{metrics['prob']:.1f}%")
c2.metric("Effective Radius", f"{metrics['rad']:.1f} µm")
# Phase Text Logic
p_val = metrics['phase']
p_str = "Liquid" if 0.5 < p_val < 1.5 else "Ice" if p_val > 1.5 else "Clear"
c3.metric("Cloud Phase", p_str)
c4.metric("Temperature", f"{metrics['temp']:.1f} °C")

# 3. VISUALIZATION
st.subheader(f"Multispectral Analysis: {st.session_state.selected_region}")
if ds_sat:
    viz_img = plot_real_matrix(arrays)
    st.image(viz_img, use_column_width=True, caption="Real-Time Data Feed (Bicubic Interpolation)")
else:
    st.error("⚠️ DATA STREAM OFFLINE. PLEASE UPLOAD .NC FILES.")

# 4. AI COMMANDER
st.subheader("Vertex AI Commander")
if st.button("🚀 REQUEST AUTHORIZATION (GEMINI 2.5)", type="primary"):
    if not api_key:
        st.warning("⚠️ Enter API Key to engage AI.")
    else:
        with st.status("Analyzing Physics Vectors...") as status:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""
            ACT AS A METEOROLOGIST. Analyze this real weather data.
            
            METRICS:
            - Cloud Prob: {metrics['prob']:.1f}%
            - Radius: {metrics['rad']:.1f} microns
            - Phase: {p_str}
            
            LOGIC:
            1. FALSE POSITIVE CHECK: If Prob > 50% but Radius is < 1.0, it is a GHOST ECHO. DECISION: NO-GO.
            2. GLACIATION CHECK: If Phase is "Ice", seeding is useless. DECISION: NO-GO.
            3. VALID TARGET: Prob > 60%, Radius 5-14, Phase Liquid. DECISION: GO.
            
            Output strictly: Decision, Analysis, Protocol.
            """
            
            try:
                response = model.generate_content([prompt, viz_img])
                text = response.text
                
                # Strict Parser
                if "NO-GO" in text.upper():
                    decision = "NO-GO"
                elif "GO" in text.upper():
                    decision = "GO"
                else:
                    decision = "NO-GO"
                    
                status.update(label="Complete", state="complete")
                
                if decision == "GO":
                    st.balloons()
                    st.markdown(f'<div class="analysis-text" style="border-left: 3px solid #00ff80;">✅ <b>MISSION APPROVED</b><br><br>{text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="analysis-text analysis-fail">⛔ <b>MISSION ABORTED</b><br><br>{text}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"AI Error: {e}")

# 5. SURVEILLANCE TABLE
st.subheader("Kingdom-Wide Surveillance")
if ds_sat:
    rows = []
    for reg in SAUDI_SECTORS:
        m, _ = extract_real_data(ds_sat, ds_era, reg)
        rows.append({
            "Region": reg,
            "Prob": f"{m['prob']:.1f}%",
            "Radius": f"{m['rad']:.1f} µm",
            "Phase": "Liquid" if 0.5 < m['phase'] < 1.5 else "Ice" if m['phase'] > 1.5 else "Clear",
            "Temp": f"{m['temp']:.1f} C"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
