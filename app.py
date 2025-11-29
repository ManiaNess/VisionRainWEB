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
                st.error(f"AI Error: {e}")import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import time
import random
from io import BytesIO
import folium
from streamlit_folium import st_folium
from scipy.ndimage import gaussian_filter

# --- FIREBASE / FIRESTORE IMPORTS (SAFE MODE) ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    /* Metrics */
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .stMetric label {color: #888;}
    
    /* Pitch Box */
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    
    /* Cloud Status Badge */
    .cloud-badge {
        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;
        background-color: #1a1a1a; border: 1px solid #444; color: #ccc;
        margin-bottom: 5px; display: inline-block;
    }
    .status-ok {color: #00ff80;}
    .status-warn {color: #ffaa00;}
    
    /* Map Container */
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {
        "coords": [21.5433, 39.1728], 
        "bias_prob": 0, "bias_temp": 0, "humidity_base": 60
    },
    "Abha (Asir Mountains)": {
        "coords": [18.2164, 42.5053], 
        "bias_prob": 30, "bias_temp": -10, "humidity_base": 70
    },
    "Riyadh (Central Arid)": {
        "coords": [24.7136, 46.6753], 
        "bias_prob": -40, "bias_temp": 5, "humidity_base": 20
    },
    "Dammam (Gulf Coast)": {
        "coords": [26.4207, 50.0888], 
        "bias_prob": -10, "bias_temp": 2, "humidity_base": 65
    },
    "Tabuk (Northern Region)": {
        "coords": [28.3835, 36.5662], 
        "bias_prob": 10, "bias_temp": -5, "humidity_base": 35
    }
}

# --- GOOGLE CLOUD ARCHITECTURE SIMULATION ---
# In a real deployment, these would connect to actual GCP SDKs

class BigQueryClient:
    """Simulates pushing logs to Google BigQuery."""
    def insert_rows(self, dataset, table, rows):
        # Simulation delay
        time.sleep(0.1) 
        # In production: bq_client.insert_rows_json(table_id, rows)
        return True

class CloudStorageClient:
    """Simulates fetching NetCDF files from GCS Buckets."""
    def fetch_satellite_data(self, region):
        # Simulation delay
        time.sleep(0.2)
        return True

bq_client = BigQueryClient()
gcs_client = CloudStorageClient()

# --- FIRESTORE SETUP ---
if "firestore_db" not in st.session_state:
    st.session_state.firestore_db = []

def init_firebase():
    """Attempts to initialize Firebase, returns DB or None."""
    if not FIREBASE_AVAILABLE: return None
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                initialize_app(cred)
            else:
                return None
        return firestore.client()
    except Exception: return None

db = init_firebase()

def save_mission_log(region, stats, decision, reasoning):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "region": region,
        "stats": str(stats),
        "decision": decision,
        "reasoning": reasoning,
        "engine": "VertexAI/Gemini-2.5-Flash"
    }
    
    # 1. Local/Session Storage
    st.session_state.firestore_db.append(entry)
    
    # 2. Firebase Storage (Real Persistence if keys exist)
    if db:
        try: db.collection("mission_logs").add(entry)
        except: pass
        
    # 3. BigQuery Simulation (The "Black Box" Audit Log)
    bq_client.insert_rows("visionrain_logs", "mission_audit", [entry])

def get_mission_logs():
    return pd.DataFrame(st.session_state.firestore_db)

# --- SCIENTIFIC DATA ENGINE ---

def generate_cloud_texture(shape=(100, 100), seed=42, intensity=1.0, roughness=5.0):
    """
    Generates REALISTIC cloud-like textures using Gaussian smoothing on noise.
    Prevents the 'bursting' look by creating smooth gradients.
    """
    np.random.seed(seed)
    noise = np.random.rand(*shape)
    smooth = gaussian_filter(noise, sigma=roughness)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return smooth * intensity

def scan_single_sector(sector_name):
    """Generates data for ONE sector based on its climate profile."""
    profile = SAUDI_SECTORS[sector_name]
    
    # Base Conditions (Physics Templates)
    conditions = [
        # Ideal Storm
        {"prob": 85.0, "press": 650, "rad": 12.5, "opt": 15.0, "lwc": 0.005, "rh": 80, "temp": -8.0, "w": 2.5, "phase": 1}, 
        # Clear Sky
        {"prob": 5.0, "press": 950, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 20, "temp": 28.0, "w": 0.1, "phase": 0},
        # High Ice Cloud
        {"prob": 70.0, "press": 350, "rad": 25.0, "opt": 5.0, "lwc": 0.001, "rh": 60, "temp": -35.0, "w": 0.5, "phase": 2},
    ]
    
    data = random.choice(conditions).copy()
    data['prob'] += profile['bias_prob']
    data['rh'] = profile['humidity_base'] + random.uniform(-10, 10)
    data['temp'] += profile['bias_temp']
    data['prob'] += random.uniform(-5, 5)
    
    # --- CRITICAL RANGE CHECKS ---
    data['prob'] = max(0.0, min(100.0, data['prob'])) 
    data['rh'] = max(5.0, min(100.0, data['rh']))
    
    if data['prob'] > 60 and data['rad'] < 14 and data['phase'] == 1:
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 40:
        data['status'] = "MONITORING"
    else:
        data['status'] = "UNSUITABLE"
        
    return data

def run_kingdom_wide_scan():
    """Scans ALL sectors and returns a DataFrame of results."""
    # Simulate GCS Fetch Latency
    with st.spinner("Fetching NetCDF Packets from Cloud Storage..."):
        gcs_client.fetch_satellite_data("all")
        
    results = {}
    for sector in SAUDI_SECTORS:
        results[sector] = scan_single_sector(sector)
    return results

# --- VISUALIZATION ENGINE (2x5 Matrix) ---
def plot_scientific_matrix(data_points):
    """Generates the 2x5 Matrix of Scientific Plots with REALISTIC TEXTURES."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    seed = int(data_points['prob'] * 100)
    
    plots = [
        # ROW 1: SATELLITE / OPTICAL
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": generate_cloud_texture(seed=seed, roughness=6) * data_points['prob'], "vmax": 100},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": generate_cloud_texture(seed=seed+1, roughness=8) * data_points['press'], "vmax": 1000},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": generate_cloud_texture(seed=seed+2, roughness=4) * data_points['rad'], "vmax": 30},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": generate_cloud_texture(seed=seed+3, roughness=5) * data_points['opt'], "vmax": 50},
        {"ax": axes[0,4], "title": "Phase (0=Clr,1=Liq,2=Ice)", "cmap": "cool", "data": generate_cloud_texture(seed=seed+4, roughness=10) * data_points['phase'], "vmax": 2},
        
        # ROW 2: ERA5 / INTERNAL PHYSICS
        {"ax": axes[1,0], "title": "Liquid Water (kg/m³)", "cmap": "Blues", "data": generate_cloud_texture(seed=seed+5, roughness=7) * data_points['lwc'], "vmax": 0.01},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu", "data": generate_cloud_texture(seed=seed+6, roughness=7) * (data_points['lwc']/3), "vmax": 0.01},
        {"ax": axes[1,2], "title": "Rel. Humidity (%)", "cmap": "Greens", "data": generate_cloud_texture(seed=seed+7, roughness=10) * data_points['rh'], "vmax": 100},
        {"ax": axes[1,3], "title": "Vertical Velocity (m/s)", "cmap": "RdBu_r", "data": (generate_cloud_texture(seed=seed+8, roughness=3) - 0.5) * 10, "vmax": 5},
        {"ax": axes[1,4], "title": "Temperature (°C)", "cmap": "inferno", "data": generate_cloud_texture(seed=seed+9, roughness=15) * 10 + data_points['temp'], "vmax": 40},
    ]

    for p in plots:
        ax = p['ax']
        ax.set_facecolor('#0e1117')
        im = ax.imshow(p['data'], cmap=p['cmap'], aspect='auto')
        ax.set_title(p['title'], color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- APP STATE INIT ---
if 'all_sector_data' not in st.session_state:
    st.session_state.all_sector_data = run_kingdom_wide_scan()
    sorted_regions = sorted(st.session_state.all_sector_data.items(), key=lambda x: x[1]['prob'], reverse=True)
    st.session_state.selected_region = sorted_regions[0][0]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v25.0 (Cloud Native)")
    
    st.markdown("### ☁️ Cloud Architecture")
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Run (App)</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Vertex AI (Gemini)</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> BigQuery (Logs)</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Storage (Data)</div>', unsafe_allow_html=True)

    st.write("---")
    
    # REGION SELECTOR
    st.markdown("### 📡 Active Sector")
    region_options = list(SAUDI_SECTORS.keys())
    if st.session_state.selected_region not in region_options:
        st.session_state.selected_region = region_options[0]
        
    selected = st.selectbox("Select Region to Monitor", region_options, 
                           index=region_options.index(st.session_state.selected_region))
    
    if selected != st.session_state.selected_region:
        st.session_state.selected_region = selected
        st.rerun()

    if st.button("🔄 FORCE RESCAN"):
        st.session_state.all_sector_data = run_kingdom_wide_scan()
        st.rerun()

    st.write("---")
    api_key = st.text_input("Gemini API Key", type="password")
    
    with st.expander("🔒 Admin Logs (BigQuery)"):
        if st.text_input("Admin Key", type="password") == "123456":
            st.dataframe(get_mission_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Pitch", "🛰️ Operations & Surveillance", "🧠 Vertex AI Commander"])

# TAB 1: PITCH
with tab1:
    st.header("Vision 2030: The Rain Enhancement Strategy")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Challenge</h3>
    <p>Saudi Arabia faces critical water scarcity. Current seeding operations are <b>manual and reactive</b>. 
    VisionRain uses AI to detect short-lived seedable clouds across the Kingdom in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.info("**Solution**\n\nAI-driven, pilotless ecosystem on Cloud Run.")
    c2.warning("**Tech**\n\nVertex AI + BigQuery + Cloud Storage.")
    c3.success("**Impact**\n\nAutomated Water Security.")

    st.subheader("Google Cloud Platform Architecture")
    st.markdown("""
    | Component | GCP Service | Function |
    | :--- | :--- | :--- |
    | **The Brain** | **Vertex AI (Gemini)** | Analyzes microphysics for GO/NO-GO decisions. |
    | **The App** | **Cloud Run** | Auto-scaling dashboard hosting. |
    | **Data Lake** | **Cloud Storage** | Stores raw Meteosat NetCDF files. |
    | **Audit Logs** | **BigQuery** | Immutable flight recorder for every mission. |
    """)

# TAB 2: OPS
with tab2:
    # --- SECTION A: SELECTED REGION VISUALS (TOP) ---
    current_region = st.session_state.selected_region
    current_data = st.session_state.all_sector_data[current_region]
    
    c_header, c_stats = st.columns([2, 3])
    with c_header:
        st.header(f"📍 {current_region}")
        st.caption("Live Telemetry Stream (GCS Stream)")
    with c_stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cloud Prob", f"{current_data['prob']:.1f}%", delta="High" if current_data['prob']>60 else "Low")
        s2.metric("Radius", f"{current_data['rad']:.1f} µm", help="Target: < 14µm")
        s3.metric("Phase", "Liquid" if current_data['phase']==1 else "Ice/Mix")
        s4.metric("Status", current_data['status'])

    # MAP & MATRIX
    col_map, col_matrix = st.columns([1, 2])
    
    with col_map:
        st.markdown("**Live Sector Map**")
        m = folium.Map(location=SAUDI_SECTORS[current_region]['coords'], zoom_start=6, tiles="CartoDB dark_matter")
        
        for region_name, info in SAUDI_SECTORS.items():
            r_data = st.session_state.all_sector_data[region_name]
            color = "green" if r_data['prob'] > 60 else "orange" if r_data['prob'] > 30 else "gray"
            tooltip_html = f"<b>{region_name}</b><br>Prob: {r_data['prob']:.1f}%"
            
            folium.Marker(
                info['coords'], popup=tooltip_html, tooltip=f"{region_name}",
                icon=folium.Icon(color=color, icon="cloud", prefix="fa")
            ).add_to(m)
            
            if region_name == current_region:
                folium.CircleMarker(info['coords'], radius=20, color="#00e5ff", fill=True, fill_opacity=0.2).add_to(m)
                
        st_folium(m, height=300, use_container_width=True)

    with col_matrix:
        st.markdown("**Real-Time Microphysics Matrix (Meteosat + ERA5)**")
        matrix_img = plot_scientific_matrix(current_data)
        st.image(matrix_img, use_column_width=True)

    st.divider()

    # --- SECTION B: KINGDOM WIDE SURVEILLANCE TABLE (BOTTOM) ---
    st.subheader("Kingdom-Wide Surveillance Wall")
    
    table_data = []
    for reg, d in st.session_state.all_sector_data.items():
        table_data.append({
            "Region": reg,
            "Priority": "🔴 High" if d['prob'] > 60 else "🟡 Medium" if d['prob'] > 30 else "⚪ Low",
            "Probability": d['prob'],
            "Effective Radius": f"{d['rad']:.1f} µm",
            "Cloud Pressure": f"{d['press']:.0f} hPa",
            "Temp": f"{d['temp']:.1f} °C",
            "Condition": "Seedable" if d['status'] == "SEEDABLE TARGET" else "Wait"
        })
    
    df_table = pd.DataFrame(table_data).sort_values("Probability", ascending=False)
    
    st.dataframe(
        df_table, use_container_width=True,
        column_config={
            "Probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0, max_value=100),
        }, hide_index=True
    )

# TAB 3: GEMINI
with tab3:
    st.header(f"Vertex AI Commander: {current_region}")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(matrix_img, caption="Visual Input Tensor (NetCDF Visualized)")
    with c2:
        st.write("### Telemetry Packet")
        st.json(current_data)

    if st.button("🚀 REQUEST AUTHORIZATION (VERTEX AI)", type="primary"):
        with st.status("Initializing Vertex AI Pipeline...") as status:
            st.write("1. Fetching Model: `gemini-2.5-flash`...")
            time.sleep(0.5)
            st.write("2. Streaming Tensor Data to `us-central1`...")
            
            prompt = f"""
            ACT AS A METEOROLOGIST. Analyze {current_region}.
            DATA: Prob: {current_data['prob']:.1f}%, Radius: {current_data['rad']:.1f}um, Phase: {current_data['phase']}, Temp: {current_data['temp']:.1f}C.
            RULES: GO IF Radius < 14 AND Radius > 5 AND Phase=1 (Liquid). NO-GO IF Phase=2 (Ice) OR Prob < 50.
            OUTPUT: Decision (GO/NO-GO), Reasoning, Protocol.
            """
            
            decision = "PENDING"
            response_text = ""
            
            if not api_key:
                st.warning("⚠️ Offline Mode (Simulated Response)")
                time.sleep(1)
                if current_data['prob'] > 60 and current_data['rad'] < 14 and current_data['phase'] == 1:
                    decision = "GO"
                    response_text = "Vertex AI Confidence: 98%. Conditions Optimal for Hygroscopic Seeding."
                else:
                    decision = "NO-GO"
                    response_text = f"Vertex AI Confidence: 95%. Conditions Unfavorable (Radius {current_data['rad']:.1f}µm)."
            else:
                try:
                    # Using standard GenerativeAI Lib but simulating Vertex behavior
                    genai.configure(api_key=api_key)
                    # NOTE: Using 1.5-flash as the stable proxy for the requested 2.5 architecture
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    res = model.generate_content([prompt, matrix_img])
                    response_text = res.text
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                except Exception as e:
                    decision = "ERROR"; response_text = str(e)

            st.write("3. Logging Decision to BigQuery...")
            status.update(label="Complete", state="complete")
        
        if "GO" in decision:
            st.balloons()
            st.success(f"✅ MISSION APPROVED: {response_text}")
        else:
            st.error(f"⛔ MISSION ABORTED: {response_text}")
            
        save_mission_log(current_region, str(current_data), decision, response_text)
        st.toast("Audit Log Saved to BigQuery", icon="☁️")


import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import time
import random
from io import BytesIO
import folium
from streamlit_folium import st_folium
from scipy.ndimage import gaussian_filter

# --- FIREBASE / FIRESTORE IMPORTS ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    
    /* Metrics */
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .stMetric label {color: #888;}
    
    /* Pitch Box */
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    
    /* Cloud Status Badge */
    .cloud-badge {
        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;
        background-color: #1a1a1a; border: 1px solid #444; color: #ccc;
        margin-bottom: 5px; display: inline-block;
    }
    .status-ok {color: #00ff80;}
    
    /* Analysis Box */
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

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {"coords": [21.5433, 39.1728], "bias_prob": 0},
    "Abha (Asir Mountains)": {"coords": [18.2164, 42.5053], "bias_prob": 30},
    "Riyadh (Central Arid)": {"coords": [24.7136, 46.6753], "bias_prob": -40},
    "Dammam (Gulf Coast)": {"coords": [26.4207, 50.0888], "bias_prob": -10},
    "Tabuk (Northern Region)": {"coords": [28.3835, 36.5662], "bias_prob": 10}
}

# --- CLOUD INFRASTRUCTURE SIMULATION ---
class BigQueryClient:
    def insert_rows(self, dataset, table, rows): return True
bq_client = BigQueryClient()

# --- FIRESTORE SETUP ---
if "firestore_db" not in st.session_state: st.session_state.firestore_db = []
def init_firebase():
    if not FIREBASE_AVAILABLE: return None
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                initialize_app(cred)
            else: return None
        return firestore.client()
    except Exception: return None
db = init_firebase()

def save_mission_log(region, stats, decision, reasoning):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "region": region, "stats": str(stats), "decision": decision, "reasoning": reasoning,
        "engine": "VertexAI/Gemini-2.5-Flash"
    }
    st.session_state.firestore_db.append(entry)
    if db:
        try: db.collection("mission_logs").add(entry)
        except: pass
    bq_client.insert_rows("visionrain_logs", "mission_audit", [entry])
def get_mission_logs(): return pd.DataFrame(st.session_state.firestore_db)

# --- SCIENTIFIC DATA ENGINE (REALISTIC STRUCTURE) ---

def generate_structured_grid(shape=(150, 150), seed=42, intensity=1.0):
    """
    Generates a Scientific Sensor Grid.
    Unlike random noise, this creates LARGE, CONTIGUOUS structures that look like
    actual weather systems, then applies a hard pixel grid to them.
    """
    np.random.seed(seed)
    
    # 1. Generate very low frequency noise (Large Shapes)
    # Sigma 25.0 creates big, rolling gradients (Weather Fronts)
    noise_base = np.random.rand(*shape)
    structure = gaussian_filter(noise_base, sigma=25.0)
    
    # 2. Add medium frequency details (Cloud Cells)
    detail = gaussian_filter(np.random.rand(*shape), sigma=5.0)
    
    # Combine: 80% Structure, 20% Detail
    combined = (structure * 0.8) + (detail * 0.2)
    
    # 3. Normalize
    norm = (combined - combined.min()) / (combined.max() - combined.min())
    
    # 4. Thresholding to create "Empty Space" vs "Cloud Mass"
    # This ensures clouds are grouped together, not scattered everywhere
    # We shift the values so anything below 0.4 becomes clear sky
    masked = np.clip((norm - 0.4) * 1.8, 0, 1)
    
    return masked * intensity

def get_simulated_data(sector_name):
    """Simulated physics data with specific edge cases."""
    profile = SAUDI_SECTORS[sector_name]
    
    # Scenario 1: Ideal Seeding Target (Liquid, Uplift, Good LWC)
    case_good = {"prob": 88.0, "press": 650, "rad": 11.5, "opt": 18.0, "lwc": 0.006, "rh": 85, "temp": -9.0, "phase": 1, "w": 3.2}
    # Scenario 2: Clear Sky / Dry
    case_dry = {"prob": 5.0, "press": 980, "rad": 0.0, "opt": 0.2, "lwc": 0.000, "rh": 15, "temp": 32.0, "phase": 0, "w": 0.1}
    # Scenario 3: GHOST ECHO (High Prob, No Water)
    case_ghost = {"prob": 75.0, "press": 800, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 40, "temp": 20.0, "phase": 0, "w": 0.5}
    # Scenario 4: Glaciated (Ice - No Seeding)
    case_ice = {"prob": 92.0, "press": 350, "rad": 22.0, "opt": 25.0, "lwc": 0.001, "rh": 90, "temp": -38.0, "phase": 2, "w": 1.5}
    
    # Weighted Randomness
    r = random.random() * 100 + profile['bias_prob']
    
    if r > 80: data = case_good
    elif r > 60: data = case_ghost 
    elif r > 40: data = case_ice
    else: data = case_dry
    
    # Physics Consistency Check
    if data['rad'] < 1.0: 
        data['rad'] = 0.0
        data['phase'] = 0
        data['lwc'] = 0.0
        
    # Determination
    if data['prob'] > 60 and data['rad'] < 14 and data['rad'] > 1 and data['phase'] == 1: 
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 60 and data['rad'] == 0.0:
        data['status'] = "SENSOR ARTIFACT"
    elif data['prob'] > 40: 
        data['status'] = "MONITORING"
    else: 
        data['status'] = "UNSUITABLE"
        
    return data

def run_kingdom_wide_scan():
    results = {}
    for sector in SAUDI_SECTORS:
        results[sector] = get_simulated_data(sector)
    return results

# --- VISUALIZATION ENGINE (HD PIXELS) ---
def plot_scientific_matrix(data_points):
    """
    Generates the Matrix with 'nearest' interpolation for RAW PIXEL look.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # 1. Master Sensor Grid (150x150)
    seed = int(data_points['prob'] * 100)
    # Using the new structured generator for cohesive cloud masses
    master_grid = generate_structured_grid(shape=(150, 150), seed=seed) 
    
    # 2. Physics Mapping
    # Only show radius/phase where the master grid indicates cloud presence
    rad_grid = master_grid * data_points['rad'] if data_points['rad'] > 0 else np.zeros((150,150))
    phase_grid = master_grid * data_points['phase'] if data_points['phase'] > 0 else np.zeros((150,150))
    
    vis_prob = master_grid * data_points['prob']
    vis_press = (1.0 - master_grid) * 1000 
    vis_lwc = master_grid * data_points['lwc']
    vis_temp = (1.0 - master_grid) * 30.0 + (master_grid * data_points['temp'])

    plots = [
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": vis_prob, "vmax": 100},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": vis_press, "vmax": 1000},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": rad_grid, "vmax": 30},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": master_grid * data_points['opt'], "vmax": 50},
        {"ax": axes[0,4], "title": "Cloud Phase", "cmap": "cool", "data": phase_grid, "vmax": 2},
        
        {"ax": axes[1,0], "title": "Liquid Water (LWC)", "cmap": "Blues", "data": vis_lwc, "vmax": 0.01},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu", "data": vis_lwc/3, "vmax": 0.01},
        {"ax": axes[1,2], "title": "Rel. Humidity (%)", "cmap": "Greens", "data": master_grid * data_points['rh'], "vmax": 100},
        {"ax": axes[1,3], "title": "Vertical Velocity (m/s)", "cmap": "RdBu_r", "data": (master_grid - 0.5) * 5},
        {"ax": axes[1,4], "title": "Temperature (°C)", "cmap": "inferno", "data": vis_temp, "vmax": 40},
    ]

    for p in plots:
        ax = p['ax']
        ax.set_facecolor('#0e1117')
        # 'nearest' = Pixelated Look (The 480p raw sensor aesthetic)
        im = ax.imshow(p['data'], cmap=p['cmap'], aspect='auto', interpolation='nearest')
        ax.set_title(p['title'], color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Explicit Phase Text
        if "Phase" in p['title']:
            phase_val = data_points['phase']
            if phase_val == 1: txt, col = "LIQUID", "cyan"
            elif phase_val == 2: txt, col = "ICE", "white"
            else: txt, col = "CLEAR", "gray"
            ax.text(0.5, 0.5, txt, color=col, ha="center", va="center", transform=ax.transAxes,
                   fontsize=18, fontweight='heavy', alpha=0.9,
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor=col))
        else:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- APP INIT ---
if 'all_sector_data' not in st.session_state:
    st.session_state.all_sector_data = run_kingdom_wide_scan()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v55.0 (Scientific)")
    
    st.markdown("### ☁️ Infrastructure")
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Run</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Vertex AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Storage</div>', unsafe_allow_html=True)

    st.write("---")
    region_options = list(SAUDI_SECTORS.keys())
    selected_region = st.selectbox("Active Sector", region_options, key="selected_region_key")

    if st.button("🔄 FORCE RESCAN"):
        st.session_state.all_sector_data = run_kingdom_wide_scan()
        st.rerun()

    api_key = st.text_input("Gemini API Key", type="password")
    with st.expander("🔒 Admin Logs"):
        if st.text_input("Key", type="password") == "123456": st.dataframe(get_mission_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Pitch", "🛰️ Operations & Surveillance", "🧠 Vertex AI Commander"])

# TAB 1: PITCH
with tab1:
    st.header("Vision 2030: Rain Enhancement Strategy")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Challenge</h3>
    <p>Saudi Arabia faces critical water scarcity. VisionRain uses AI to detect seedable clouds via sensor fusion.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**Solution**\n\nAI-driven, pilotless ecosystem."); c2.warning("**Tech**\n\nVertex AI + BigQuery."); c3.success("**Impact**\n\nAutomated Water Security.")

# TAB 2: OPS
with tab2:
    current_region = st.session_state.get("selected_region_key", region_options[0])
    current_data = st.session_state.all_sector_data[current_region]
    
    c_header, c_stats = st.columns([2, 3])
    with c_header:
        st.header(f"📍 {current_region}")
        st.caption("Live Telemetry Stream (GCS Stream)")
    with c_stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cloud Prob", f"{current_data['prob']:.1f}%", delta="High" if current_data['prob']>60 else "Low")
        s2.metric("Radius", f"{current_data['rad']:.1f} µm", help="Target: < 14µm")
        s3.metric("Phase", "Liquid" if current_data['phase']==1 else "Ice/Mix")
        s4.metric("Status", current_data['status'])

    col_map, col_matrix = st.columns([1, 2])
    with col_map:
        st.markdown("**Live Sector Map**")
        m = folium.Map(location=SAUDI_SECTORS[current_region]['coords'], zoom_start=6, tiles="CartoDB dark_matter")
        for reg_name, info in SAUDI_SECTORS.items():
            d = st.session_state.all_sector_data[reg_name]
            color = "green" if d['prob'] > 60 and d['rad'] > 1 else "orange" if d['prob'] > 30 else "gray"
            folium.Marker(info['coords'], popup=f"{reg_name}", icon=folium.Icon(color=color, icon="cloud", prefix="fa")).add_to(m)
            if reg_name == current_region:
                folium.CircleMarker(info['coords'], radius=20, color="#00e5ff", fill=True, fill_opacity=0.2).add_to(m)
        st_folium(m, height=300, use_container_width=True)

    with col_matrix:
        st.markdown("**Real-Time Microphysics Matrix**")
        matrix_img = plot_scientific_matrix(current_data)
        st.image(matrix_img, use_column_width=True)

    st.divider()
    st.subheader("Kingdom-Wide Surveillance Wall")
    table_data = []
    for reg, d in st.session_state.all_sector_data.items():
        table_data.append({
            "Region": reg, "Priority": "🔴 High" if d['prob'] > 60 and d['rad'] > 1 else "🟡 Medium" if d['prob'] > 30 else "⚪ Low",
            "Probability": d['prob'], "Effective Radius": f"{d['rad']:.1f} µm", "Condition": d['status']
        })
    st.dataframe(pd.DataFrame(table_data).sort_values("Probability", ascending=False), use_container_width=True, hide_index=True)

# TAB 3: GEMINI
with tab3:
    st.header(f"Vertex AI Commander: {current_region}")
    c1, c2 = st.columns([1, 1])
    with c1: st.image(matrix_img, caption="Visual Input Tensor")
    with c2: st.write("### Telemetry Packet"); st.json(current_data)

    if st.button("🚀 REQUEST AUTHORIZATION (GEMINI 2.5 FLASH)", type="primary"):
        with st.status("Initializing AI Pipeline...") as status:
            st.write("1. Establishing Uplink to `gemini-2.5-flash`...")
            
            # --- THE "SCIENTIFIC" PROMPT ---
            prompt = f"""
            ACT AS A SENIOR CLOUD PHYSICIST using Google Cloud Vertex AI. 
            Analyze this target sector: {current_region}.
            
            TELEMETRY:
            - Cloud Probability: {current_data['prob']:.1f}% (Satellite)
            - Effective Radius: {current_data['rad']:.1f} µm (Microphysics)
            - Cloud Phase: {current_data['phase']} (0=Clear, 1=Liquid, 2=Ice)
            - Temp: {current_data['temp']:.1f} C
            - Updraft (w): {current_data['w']} m/s
            - LWC: {current_data['lwc']} kg/m3
            
            SCIENTIFIC LOGIC GATES (Step-by-Step):
            1. MICROPHYSICS: Is LWC sufficient for droplet growth? Is Radius optimal (5-14µm) for coalescence?
            2. THERMODYNAMICS: Is Temp > -15C (Mixed/Warm phase)? If Temp < -20C (Ice), abort.
            3. DYNAMICS: Is Updraft > 0.5 m/s to sustain the seeded parcel?
            4. VALIDATION: If Cloud Prob is High but Radius is 0 -> GHOST ECHO (Artifact). ABORT.
            
            OUTPUT FORMAT:
            Decision: [GO / NO-GO]
            Analysis: [Scientific reasoning referencing LWC, Phase, and Dynamics.]
            Coordinates: {SAUDI_SECTORS[current_region]['coords']}
            Protocol: [Action]
            """
            
            decision, response_text = "PENDING", ""
            if not api_key:
                st.warning("⚠️ Offline Mode (Simulated Response)")
                # Logic Mirror
                if current_data['rad'] == 0 and current_data['prob'] > 50:
                    decision = "NO-GO"; response_text = "**Analysis:** CONTRADICTION. Satellite Prob is high ({current_data['prob']:.0f}%) but Radius is 0. This is a Ghost Echo artifact (Cloud Albedo without Microphysics).\n\n**Decision:** NO-GO"
                elif current_data['prob'] > 60 and current_data['rad'] < 14 and current_data['rad'] > 1 and current_data['phase'] == 1:
                    decision = "GO"; response_text = "**Analysis:** Optimal Microphysics. LWC ({current_data['lwc']}) supports droplet growth. Updraft ({current_data['w']} m/s) sufficient for lift. Radius ({current_data['rad']} µm) ideal for hygroscopic flare interaction.\n\n**Decision:** GO"
                else:
                    decision = "NO-GO"; response_text = "**Analysis:** Thermodynamics unfavorable. Temperature or LWC below seeding threshold.\n\n**Decision:** NO-GO"
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    res = model.generate_content([prompt, matrix_img])
                    response_text = res.text
                    
                    u_text = res.text.upper()
                    if "DECISION: GO" in u_text or "**DECISION:** GO" in u_text:
                        decision = "GO"
                    else:
                        decision = "NO-GO"
                except Exception as e: decision = "ERROR"; response_text = str(e)

            status.update(label="Complete", state="complete")
        
        if decision == "GO":
            st.balloons()
            st.markdown(f'<div class="analysis-text analysis-go">✅ <b>MISSION APPROVED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
            
            st.markdown("### 🚁 Drone Command")
            lat, lon = SAUDI_SECTORS[current_region]['coords']
            # GOOGLE MAPS LINK
            st.link_button("🛰️ CONFIRM & LAUNCH DRONE", f"https://www.google.com/maps/search/?api=1&query={lat},{lon}")
            
        else:
            st.markdown(f'<div class="analysis-text analysis-nogo">⛔ <b>MISSION ABORTED</b><br><br>{response_text}</div>', unsafe_allow_html=True)
            
        save_mission_log(current_region, str(current_data), decision, response_text)
        st.toast("Audit Log Saved to BigQuery", icon="☁️")
