import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from io import BytesIO
import folium
from streamlit_folium import st_folium

# --- SAFE IMPORTS FOR SCIENTIFIC LIBS ---
try:
    import xarray as xr
    import cfgrib
except ImportError:
    st.error("⚠️ Scientific Libraries Missing. Please install xarray, cfgrib, netCDF4, eccodes.")

# --- FIREBASE ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- FILE CONSTANTS (Your Uploaded Files) ---
FILE_NC = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
FILE_GRIB = "ce636265319242f2fef4a83020b30ecf.grib"

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .analysis-text {
        font-family: 'Courier New', monospace;
        background-color: #111;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 14px;
        line-height: 1.6;
    }
    .analysis-go { border-left: 4px solid #00ff80; color: #00ff80; }
    .analysis-nogo { border-left: 4px solid #ff4444; color: #ff4444; }
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SECTOR CONFIGURATION (Spatial Mapping) ---
# We map Cities to both Lat/Lon (for GRIB) and Pixel Y/X (for NC)
SAUDI_SECTORS = {
    "Jeddah": {"lat": 21.54, "lon": 39.17, "py": 2300, "px": 750},
    "Abha":   {"lat": 18.22, "lon": 42.51, "py": 2500, "px": 800},
    "Riyadh": {"lat": 24.71, "lon": 46.68, "py": 2100, "px": 900},
    "Dammam": {"lat": 26.42, "lon": 50.09, "py": 2000, "px": 950},
    "Tabuk":  {"lat": 28.38, "lon": 36.57, "py": 1900, "px": 700}
}

# --- DATA LOADER ---
@st.cache_resource
def load_data_files():
    ds_sat, ds_era = None, None
    
    # 1. Load Meteosat (NC)
    if os.path.exists(FILE_NC):
        try: 
            ds_sat = xr.open_dataset(FILE_NC, engine='netcdf4')
            print("✅ Meteosat NC Loaded")
        except Exception as e: print(f"❌ NC Error: {e}")
            
    # 2. Load ERA5 (GRIB)
    if os.path.exists(FILE_GRIB):
        try:
            # GRIB often requires filtering by typeOfLevel
            ds_era = xr.open_dataset(FILE_GRIB, engine='cfgrib')
            print("✅ ERA5 GRIB Loaded")
        except Exception as e: print(f"❌ GRIB Error: {e}")
            
    return ds_sat, ds_era

# --- SMART DATA EXTRACTOR (REAL DATA ONLY) ---
def get_real_data(sector_name, ds_sat, ds_era):
    coords = SAUDI_SECTORS[sector_name]
    
    # Container for the 10 data grids (Visuals) and Scalar Metrics (AI)
    data = {
        "grids": {},
        "metrics": {}
    }
    
    # Window Size for Visualization (Pixels)
    w = 60 # 120x120 grid
    
    # Helper to init empty if file missing
    def empty_grid(): return np.zeros((w*2, w*2))

    # --- 1. SATELLITE EXTRACTION (NC) ---
    if ds_sat:
        # Search for variables in the NC file
        # We try to match common names from EUMETSAT products
        var_map = {
            "prob": ["probability", "prob", "c_prob"],
            "rad": ["radius", "effective", "cre"],
            "phase": ["phase", "cph"],
            "opt": ["optical", "thickness", "cot"],
            "press": ["pressure", "ctp"]
        }
        
        y_sl = slice(coords['py']-w, coords['py']+w)
        x_sl = slice(coords['px']-w, coords['px']+w)
        
        for key, patterns in var_map.items():
            found_var = None
            for v in ds_sat.data_vars:
                if any(p in v.lower() for p in patterns):
                    found_var = v
                    break
            
            if found_var:
                # Extract Grid
                # Assuming dims are (y, x)
                try:
                    raw = ds_sat[found_var].isel(y=y_sl, x=x_sl).values
                    # Handle Fill Values/Masks
                    grid = np.nan_to_num(raw, nan=0.0)
                    
                    # Store Grid
                    data['grids'][key] = grid
                    # Store Metric (Center Pixel)
                    data['metrics'][key] = float(grid[w, w])
                except:
                    data['grids'][key] = empty_grid()
                    data['metrics'][key] = 0.0
            else:
                data['grids'][key] = empty_grid()
                data['metrics'][key] = 0.0
                
    else:
        # Fill empty if no NC
        for k in ["prob", "rad", "phase", "opt", "press"]:
            data['grids'][k] = empty_grid()
            data['metrics'][k] = 0.0

    # --- 2. ERA5 EXTRACTION (GRIB) ---
    if ds_era:
        # GRIB uses Lat/Lon. We use xarray's .sel(method='nearest')
        
        # ERA5 Variables usually: 
        # t (temp), r (humidity), w (vertical velocity), clwc (liquid water), ciwc (ice water)
        era_map = {
            "temp": ["t", "temp"],
            "rh": ["r", "humid"],
            "w": ["w", "vertical"],
            "lwc": ["clwc", "liquid"],
            "ice": ["ciwc", "ice"]
        }
        
        for key, patterns in era_map.items():
            found_var = None
            for v in ds_era.data_vars:
                if any(p in v.lower() for p in patterns):
                    found_var = v
                    break
            
            if found_var:
                try:
                    # Select nearest to city center
                    # We grab a small window around the lat/lon to create a "Grid"
                    # NOTE: GRIB lat/lon spacing varies. We take +/- 0.5 degrees
                    lat_sl = slice(coords['lat']+0.5, coords['lat']-0.5) 
                    lon_sl = slice(coords['lon']-0.5, coords['lon']+0.5)
                    
                    raw = ds_era[found_var].sel(latitude=lat_sl, longitude=lon_sl).values
                    
                    # Handle Time dimension if present (take last)
                    if raw.ndim >= 3: raw = raw[-1]
                    
                    # Resize to match visualization window (120x120) using simple zoom
                    # This creates the "Grid" look from the coarser GRIB data
                    from scipy.ndimage import zoom
                    zoom_factor = (w*2) / max(raw.shape)
                    grid = zoom(raw, zoom_factor, order=0) # Order 0 = Nearest Neighbor (Pixelated)
                    
                    # Ensure exact shape match
                    grid = grid[:w*2, :w*2]
                    
                    data['grids'][key] = grid
                    data['metrics'][key] = float(np.mean(raw)) # Mean of the sector
                    
                except Exception as e:
                    data['grids'][key] = empty_grid()
                    data['metrics'][key] = 0.0
            else:
                data['grids'][key] = empty_grid()
                data['metrics'][key] = 0.0
    else:
        for k in ["temp", "rh", "w", "lwc", "ice"]:
            data['grids'][k] = empty_grid()
            data['metrics'][k] = 0.0
            
    # --- 3. MISSING DATA HANDLER ---
    # If something like Phase is missing in NC, we infer from ERA5 Temp/Ice
    if data['metrics']['phase'] == 0 and data['metrics']['ice'] > 0:
        # If we have Ice Water Content but Phase is 0, infer Ice Phase
        data['metrics']['phase'] = 2 
        data['grids']['phase'] = np.full((w*2, w*2), 2)
        
    # Scale Temp from Kelvin to C
    if data['metrics']['temp'] > 100:
        data['metrics']['temp'] -= 273.15
        data['grids']['temp'] -= 273.15

    return data

# --- VISUALIZATION (RAW PIXELS) ---
def plot_real_grids(data):
    grids = data['grids']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # Plot Config
    plots = [
        # Row 1 (Sat)
        (0,0, "Cloud Probability", "Blues", grids['prob'], 0, 100),
        (0,1, "Cloud Top Pressure", "gray_r", grids['press'], 200, 1000),
        (0,2, "Effective Radius", "viridis", grids['rad'], 0, 30),
        (0,3, "Optical Depth", "magma", grids['opt'], 0, 50),
        (0,4, "Cloud Phase", "cool", grids['phase'], 0, 3),
        # Row 2 (ERA5)
        (1,0, "Liquid Water", "Blues", grids['lwc'], None, None),
        (1,1, "Ice Water", "PuBu", grids['ice'], None, None),
        (1,2, "Rel. Humidity", "Greens", grids['rh'], 0, 100),
        (1,3, "Vertical Velocity", "RdBu", grids['w'], -2, 2),
        (1,4, "Temperature", "inferno", grids['temp'], -40, 40),
    ]
    
    for r, c, title, cmap, arr, vmin, vmax in plots:
        ax = axes[r,c]
        ax.set_facecolor('#0e1117')
        
        # interpolation='nearest' -> THIS GIVES THE PIXELATED/GRID LOOK
        im = ax.imshow(arr, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        
        ax.set_title(title, color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Phase Text Overlay
        if "Phase" in title:
            val = data['metrics']['phase']
            txt = "LIQUID" if 0.5 < val < 1.5 else "ICE" if val > 1.5 else "CLEAR"
            col = "cyan" if "LIQUID" in txt else "white"
            ax.text(0.5, 0.5, txt, color=col, ha="center", va="center", transform=ax.transAxes,
                   fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
        else:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- INIT ---
ds_sat, ds_era = load_data_files()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("v60.0 | REAL DATA CORE")
    
    st.markdown("### 📡 Data Link")
    if ds_sat: st.success("Meteosat (NC) Linked")
    else: st.error("Meteosat (NC) Missing")
    if ds_era: st.success("ERA5 (GRIB) Linked")
    else: st.warning("ERA5 (GRIB) Missing")
    
    st.markdown("---")
    
    region_options = list(SAUDI_SECTORS.keys())
    # State-bound Selector to prevent refresh loops
    selected_region = st.selectbox("Active Sector", region_options, key="region_select")
    
    api_key = st.text_input("Gemini API Key", type="password")

# --- MAIN UI ---
st.title("VisionRain Command Center")

# 1. EXTRACT DATA
current_data = get_real_data(selected_region, ds_sat, ds_era)
m = current_data['metrics']

# 2. TELEMETRY
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cloud Prob", f"{m['prob']:.1f}%")
c2.metric("Radius", f"{m['rad']:.1f} µm")
phase_str = "Liquid" if 0.5 < m['phase'] < 1.5 else "Ice" if m['phase'] > 1.5 else "Clear"
c3.metric("Phase", phase_str)
c4.metric("Temp", f"{m['temp']:.1f} °C")

# 3. VISUALIZATION
st.subheader(f"Multispectral Grid: {selected_region}")
if ds_sat or ds_era:
    viz_img = plot_real_grids(current_data)
    st.image(viz_img, use_column_width=True)
else:
    st.error("NO DATA FILES LOADED. UPLOAD .NC OR .GRIB FILES.")

# 4. AI DECISION
st.subheader("Vertex AI Commander")
if st.button("🚀 REQUEST AUTHORIZATION"):
    if not api_key:
        st.warning("⚠️ Enter API Key")
    else:
        with st.status("Analyzing Physics Vectors...") as status:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""
            ACT AS A SENIOR CLOUD PHYSICIST. Analyze this REAL satellite data for {selected_region}.
            
            METRICS:
            - Cloud Prob: {m['prob']:.1f}%
            - Radius: {m['rad']:.1f} µm
            - Phase: {phase_str} (Value: {m['phase']})
            - Temp: {m['temp']:.1f} C
            - LWC: {m['lwc']:.4f}
            
            LOGIC GATES:
            1. FALSE POSITIVE: If Prob > 50% but Radius is 0 -> GHOST ECHO. ABORT.
            2. GLACIATION: If Phase is Ice -> ABORT.
            3. VALID: Prob > 60%, Radius 5-14µm, Phase Liquid -> GO.
            
            OUTPUT: Decision (GO/NO-GO), Analysis, Protocol.
            """
            
            try:
                res = model.generate_content([prompt, viz_img])
                text = res.text
                
                decision = "GO" if "DECISION: GO" in text.upper() or "**DECISION:** GO" in text.upper() else "NO-GO"
                
                status.update(label="Complete", state="complete")
                
                if decision == "GO":
                    st.balloons()
                    st.markdown(f'<div class="analysis-text analysis-go">✅ <b>MISSION APPROVED</b><br><br>{text}</div>', unsafe_allow_html=True)
                    lat, lon = SAUDI_SECTORS[selected_region]['lat'], SAUDI_SECTORS[selected_region]['lon']
                    st.link_button("🛰️ CONFIRM & LAUNCH DRONE", f"https://www.google.com/maps/search/?api=1&query={lat},{lon}")
                else:
                    st.markdown(f'<div class="analysis-text analysis-nogo">⛔ <b>MISSION ABORTED</b><br><br>{text}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"AI Error: {e}")
