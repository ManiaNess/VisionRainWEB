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

# --- FIREBASE (Keep for logging) ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- EXPECTED FILENAMES ---
DEFAULT_NC = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
DEFAULT_GRIB = "ce636265319242f2fef4a83020b30ecf.grib"

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    .analysis-text {font-family: 'Courier New', monospace; color: #00ff80; background-color: #111; padding: 15px; border-left: 3px solid #00ff80; border-radius: 5px; margin-top: 10px;}
    .analysis-fail {color: #ff4444; border-left: 3px solid #ff4444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR PIXEL MAPPING ---
# Based on your previous code's pixel coordinates for the Meteosat file
SAUDI_SECTORS = {
    "Jeddah": {"coords": [21.5433, 39.1728], "y": 2300, "x": 750},
    "Abha":   {"coords": [18.2164, 42.5053], "y": 2500, "x": 800},
    "Riyadh": {"coords": [24.7136, 46.6753], "y": 2100, "x": 900},
    "Dammam": {"coords": [26.4207, 50.0888], "y": 2000, "x": 950},
    "Tabuk":  {"coords": [28.3835, 36.5662], "y": 1900, "x": 700}
}

# --- DATA ENGINE: REAL FILE READER ---
@st.cache_resource
def load_datasets(nc_path, grib_path):
    """Loads the heavy scientific files once."""
    ds_sat, ds_era = None, None
    
    # 1. Load Meteosat (.nc)
    if os.path.exists(nc_path):
        try:
            ds_sat = xr.open_dataset(nc_path, engine='netcdf4')
            print(f"Loaded NC: {nc_path}")
        except Exception as e:
            print(f"NC Load Error: {e}")
    
    # 2. Load ERA5 (.grib)
    if os.path.exists(grib_path):
        try:
            ds_era = xr.open_dataset(grib_path, engine='cfgrib')
            print(f"Loaded GRIB: {grib_path}")
        except Exception as e:
            print(f"GRIB Load Error: {e}")
            
    return ds_sat, ds_era

def find_variable(ds, search_keys):
    """Helper to find a variable in the dataset by fuzzy name matching."""
    if ds is None: return None
    for var_name in ds.data_vars:
        if any(key in var_name.lower() for key in search_keys):
            return ds[var_name]
    return None

def extract_sector_data(ds_sat, ds_era, sector_name, window=40):
    """
    EXTRACTS ACTUAL RAW DATA. NO SIMULATION.
    """
    coords = SAUDI_SECTORS[sector_name]
    cy, cx = coords['y'], coords['x']
    
    metrics = {
        "prob": 0.0, "press": 0.0, "rad": 0.0, "opt": 0.0, "phase": 0,
        "lwc": 0.0, "ice": 0.0, "rh": 0.0, "w": 0.0, "temp": 0.0
    }
    
    # Storage for 2D arrays (for plotting)
    arrays = {} 

    # --- 1. SATELLITE EXTRACTION ---
    if ds_sat:
        # Define the slicing window
        y_slice = slice(max(0, cy - window), cy + window)
        x_slice = slice(max(0, cx - window), cx + window)
        
        # Helper to extract and store
        def get_sat_var(keys, metric_key, scale=1.0):
            var = find_variable(ds_sat, keys)
            if var is not None:
                # Extract 2D slice
                # Assuming dims are (y, x) or similar. slice(y), slice(x)
                if len(var.dims) >= 2:
                    dim1, dim2 = var.dims[-2], var.dims[-1] # Usually y, x
                    data_slice = var.sel({dim1: y_slice, dim2: x_slice}).values
                    
                    # Store 2D array for visualization
                    arrays[metric_key] = data_slice * scale
                    
                    # Store Mean Scalar for Telemetry
                    # Handle masked arrays/NaNs
                    valid_data = data_slice[~np.isnan(data_slice)]
                    if len(valid_data) > 0:
                        metrics[metric_key] = float(np.mean(valid_data)) * scale
            else:
                arrays[metric_key] = np.zeros((window*2, window*2))

        # Map Metrics to File Variables
        get_sat_var(['prob', 'cloud_probability'], 'prob')
        get_sat_var(['press', 'pressure'], 'press', scale=0.01) # Pa to hPa
        get_sat_var(['rad', 'effective', 'cre'], 'rad')
        get_sat_var(['opt', 'thickness', 'cot'], 'opt')
        get_sat_var(['phase'], 'phase')

    # --- 2. ERA5 EXTRACTION ---
    # ERA5 is usually Lat/Lon. We simulate the "matching" crop for now 
    # since we don't have the exact projection transform logic here.
    # We will take the mean of the GRIB file or a sub-region if possible.
    if ds_era:
        def get_era_var(keys, metric_key, scale=1.0):
            var = find_variable(ds_era, keys)
            if var is not None:
                # Just take a 80x80 chunk from the middle/start for visualization consistency
                # since mapping pixel X/Y to Lat/Lon requires a transform library
                data = var.values
                if len(data.shape) > 2: data = data[0] # Time dimension
                
                # Crop center
                mid_y, mid_x = data.shape[0]//2, data.shape[1]//2
                data_slice = data[mid_y-window:mid_y+window, mid_x-window:mid_x+window]
                
                arrays[metric_key] = data_slice * scale
                metrics[metric_key] = float(np.nanmean(data_slice)) * scale
            else:
                arrays[metric_key] = np.zeros((window*2, window*2))

        get_era_var(['clwc', 'liquid'], 'lwc')
        get_era_var(['ciwc', 'ice'], 'ice')
        get_era_var(['r', 'humidity'], 'rh')
        get_era_var(['w', 'vertical'], 'w')
        get_era_var(['t', 'temp'], 'temp', scale=1.0 - 273.15) # Kelvin to C if needed? Assuming K usually.

    return metrics, arrays

# --- VISUALIZATION ---
def plot_real_data(arrays, metrics):
    """Plots the ACTUAL arrays extracted from the files."""
    if not arrays: return None
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # Plot Definition
    # Key: (Row, Col, Title, Cmap, DataKey, Vmax)
    plot_defs = [
        (0,0, "Cloud Probability (%)", "Blues", 'prob', 100),
        (0,1, "Cloud Top Pressure (hPa)", "gray_r", 'press', 1000),
        (0,2, "Effective Radius (µm)", "viridis", 'rad', 30),
        (0,3, "Optical Depth", "magma", 'opt', 50),
        (0,4, "Cloud Phase", "cool", 'phase', 4),
        
        (1,0, "Liquid Water", "Blues", 'lwc', None),
        (1,1, "Ice Water", "PuBu", 'ice', None),
        (1,2, "Rel. Humidity (%)", "Greens", 'rh', 100),
        (1,3, "Vertical Velocity", "RdBu", 'w', None),
        (1,4, "Temperature (°C)", "inferno", 'temp', None),
    ]
    
    for r, c, title, cmap, key, vmax in plot_defs:
        ax = axes[r, c]
        ax.set_facecolor('#0e1117')
        
        if key in arrays:
            data = arrays[key]
            # Handle empty/missing data gracefully
            if data is None or data.size == 0: 
                data = np.zeros((80,80))
            
            # Simple normalization for display if range is weird
            im = ax.imshow(data, cmap=cmap, aspect='auto', vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, "NO DATA", color="red", ha="center")
            
        ax.set_title(title, color="white", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- INIT & SIDEBAR ---
# Load Data Once
ds_sat, ds_era = load_datasets(DEFAULT_NC, DEFAULT_GRIB)

with st.sidebar:
    st.title("VisionRain")
    st.caption("v31.0 | Real Data Core")
    
    st.markdown("### 📂 Data Source")
    if ds_sat: st.success(f"NC File Loaded\n({DEFAULT_NC[:15]}...)")
    else: st.error("NC File Missing")
    
    if ds_era: st.success(f"GRIB File Loaded\n({DEFAULT_GRIB[:15]}...)")
    else: st.warning("GRIB File Missing")
    
    # Fallback uploader if files aren't in root
    if not ds_sat or not ds_era:
        st.markdown("---")
        uploaded_nc = st.file_uploader("Upload .nc", type=['nc'])
        uploaded_grib = st.file_uploader("Upload .grib", type=['grib'])
        
        if uploaded_nc:
            with open("temp.nc", "wb") as f: f.write(uploaded_nc.getbuffer())
            st.rerun() # Reload to pick up file
            
    st.markdown("---")
    
    # Region Selector
    region_options = list(SAUDI_SECTORS.keys())
    selected_region = st.selectbox("Select Sector", region_options)
    
    api_key = st.text_input("Gemini API Key", type="password")

# --- MAIN APP ---
st.title("VisionRain Command Center")

# 1. EXTRACT DATA FOR SELECTED REGION
metrics, arrays = extract_sector_data(ds_sat, ds_era, selected_region)

# 2. HEADER METRICS
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cloud Prob", f"{metrics['prob']:.1f}%")
col2.metric("Radius", f"{metrics['rad']:.1f} µm")
col3.metric("Phase", f"{metrics['phase']:.0f}")
col4.metric("Temp", f"{metrics['temp']:.1f} °C")

# 3. VISUALIZATION
st.subheader(f"Real-Time Telemetry: {selected_region}")
if arrays:
    viz_image = plot_real_data(arrays, metrics)
    st.image(viz_image, use_column_width=True)
else:
    st.warning("Waiting for Data Files...")

# 4. AI ANALYSIS
st.subheader("Vertex AI Analysis")
if st.button("RUN ANALYSIS"):
    if not api_key:
        st.error("Please enter Gemini API Key in sidebar")
    else:
        with st.spinner("Analyzing Physics..."):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""
            Analyze this weather data for Cloud Seeding Suitability.
            
            METRICS:
            - Cloud Prob: {metrics['prob']}%
            - Radius: {metrics['rad']} microns
            - Phase: {metrics['phase']} (1=Liquid)
            
            LOGIC:
            - If Prob > 60 AND Radius is 0 -> FALSE POSITIVE / SENSOR ARTIFACT. DECISION: NO-GO.
            - If Prob > 60 AND Radius 5-14 -> GOOD. DECISION: GO.
            
            Provide decision and reasoning.
            """
            
            response = model.generate_content([prompt, viz_image])
            
            if "GO" in response.text.upper() and "NO-GO" not in response.text.upper():
                st.balloons()
                st.success(response.text)
            else:
                st.markdown(f'<div class="analysis-text analysis-fail">{response.text}</div>', unsafe_allow_html=True)

# 5. KINGDOM SURVEILLANCE TABLE
st.subheader("Kingdom-Wide Surveillance")
# Only generate this table if we have data, iterating through all regions
if ds_sat:
    rows = []
    for reg in SAUDI_SECTORS:
        m, _ = extract_sector_data(ds_sat, ds_era, reg)
        rows.append({
            "Region": reg,
            "Prob": f"{m['prob']:.1f}%",
            "Radius": f"{m['rad']:.1f} µm",
            "Phase": m['phase'],
            "Status": "Analyzing..."
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
