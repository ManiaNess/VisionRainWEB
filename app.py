import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from io import BytesIO
from scipy.ndimage import zoom

# --- SCIENTIFIC LIBRARIES ---
try:
    import xarray as xr
    import cfgrib
except ImportError:
    st.error("⚠️ Scientific Libraries Missing. Please install xarray, cfgrib, netCDF4, eccodes, scipy.")

# --- FIREBASE ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- FILE CONSTANTS ---
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
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_resource
def load_data_files():
    ds_sat, ds_era = None, None
    if os.path.exists(FILE_NC):
        try: ds_sat = xr.open_dataset(FILE_NC, engine='netcdf4')
        except: pass
    if os.path.exists(FILE_GRIB):
        try: ds_era = xr.open_dataset(FILE_GRIB, engine='cfgrib')
        except: pass
    return ds_sat, ds_era

# --- SMART VARIABLE FINDER ---
def get_var(ds, keys, scale=1.0, fill_value=0.0):
    if ds is None: return None
    for v in ds.data_vars:
        if any(k in v.lower() for k in keys):
            data = ds[v].values
            # Handle Time/Level dimensions (flatten to 2D)
            while data.ndim > 2: data = data[0]
            # Handle Fill Values
            data = np.nan_to_num(data, nan=fill_value)
            return data * scale
    return None

# --- THE "ALL OF SAUDI" SCANNER (FIXED RESIZING) ---
def scan_whole_kingdom(ds_sat, ds_era):
    """
    Scans the ENTIRE file content to find the best seedable pixel.
    CRITICAL FIX: Resizes ALL arrays to match the 'prob' array shape to prevent index errors.
    """
    # 1. Extract Master Grid (Probability)
    prob = get_var(ds_sat, ['prob', 'cloud_probability'], 1.0)
    
    # Fallback if probability is missing
    if prob is None: 
        prob = np.zeros((500, 500))
    
    target_shape = prob.shape
    
    # --- SYNCHRONIZATION FUNCTION ---
    def sync_grid(arr):
        """Forces any array to match the Probability Grid's shape."""
        if arr is None: 
            return np.zeros(target_shape)
        if arr.shape == target_shape:
            return arr
        # Calculate zoom factors
        zf = np.array(target_shape) / np.array(arr.shape)
        return zoom(arr, zf, order=0) # Nearest neighbor

    # 2. Extract & Sync Satellite Metrics
    rad = sync_grid(get_var(ds_sat, ['rad', 'effective', 'cre'], 1.0))
    phase = sync_grid(get_var(ds_sat, ['phase', 'cph'], 1.0))
    press = sync_grid(get_var(ds_sat, ['press', 'pressure'], 0.01))
    opt = sync_grid(get_var(ds_sat, ['opt', 'thickness'], 1.0))

    # 3. Extract & Sync ERA5 Metrics
    lwc = sync_grid(get_var(ds_era, ['clwc', 'liquid']))
    ice = sync_grid(get_var(ds_era, ['ciwc', 'ice']))
    rh = sync_grid(get_var(ds_era, ['r', 'humid']))
    w = sync_grid(get_var(ds_era, ['w', 'vertical']))
    
    # Temp special handling
    raw_temp = get_var(ds_era, ['t', 'temp'], 1.0)
    if raw_temp is not None:
        if np.max(raw_temp) > 100: raw_temp -= 273.15 # K to C
    temp = sync_grid(raw_temp)

    # 4. CALCULATE SEEDABILITY SCORE (Vectorized Physics)
    score_map = np.zeros_like(prob)
    
    mask_prob = (prob > 50).astype(float)
    mask_phase = ((phase > 0.5) & (phase < 1.5)).astype(float) # Liquid = 1
    mask_rad = ((rad > 5) & (rad < 15)).astype(float)
    
    # Weighted Score: High Prob (40%) + Liquid (30%) + Good Radius (30%)
    score_map = (prob * 0.4) + (mask_phase * 100 * 0.3) + (mask_rad * 100 * 0.3)
    
    # 5. FIND HOTSPOT
    if np.max(score_map) > 0:
        y_hot, x_hot = np.unravel_index(np.argmax(score_map), score_map.shape)
    else:
        y_hot, x_hot = prob.shape[0]//2, prob.shape[1]//2

    # 6. EXTRACT VISUALIZATION WINDOW (Safe Slicing)
    win = 60 # 120x120 total
    y_min, y_max = max(0, y_hot-win), min(prob.shape[0], y_hot+win)
    x_min, x_max = max(0, x_hot-win), min(prob.shape[1], x_hot+win)
    
    data = {
        "grids": {
            "prob": prob[y_min:y_max, x_min:x_max],
            "rad": rad[y_min:y_max, x_min:x_max],
            "phase": phase[y_min:y_max, x_min:x_max],
            "temp": temp[y_min:y_max, x_min:x_max],
            "lwc": lwc[y_min:y_max, x_min:x_max],
            "press": press[y_min:y_max, x_min:x_max],
            "opt": opt[y_min:y_max, x_min:x_max],
            "ice": ice[y_min:y_max, x_min:x_max],
            "rh": rh[y_min:y_max, x_min:x_max],
            "w": w[y_min:y_max, x_min:x_max],
        },
        "target_metrics": {
            "prob": float(prob[y_hot, x_hot]),
            "rad": float(rad[y_hot, x_hot]),
            "phase": float(phase[y_hot, x_hot]),
            "temp": float(temp[y_hot, x_hot]),
            "lwc": float(lwc[y_hot, x_hot]),
            "w": float(w[y_hot, x_hot]),
            "score": float(score_map[y_hot, x_hot])
        },
        "coords": (y_hot, x_hot)
    }
    return data

# --- VISUALIZATION (RAW PIXELS) ---
def plot_real_grids(grids):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    plots = [
        (0,0, "Cloud Probability", "Blues", grids['prob'], 0, 100),
        (0,1, "Cloud Top Pressure", "gray_r", grids['press'], 200, 1000),
        (0,2, "Effective Radius", "viridis", grids['rad'], 0, 30),
        (0,3, "Optical Depth", "magma", grids['opt'], 0, 50),
        (0,4, "Cloud Phase", "cool", grids['phase'], 0, 3),
        (1,0, "Liquid Water", "Blues", grids['lwc'], None, None),
        (1,1, "Ice Water", "PuBu", grids['ice'], None, None),
        (1,2, "Rel. Humidity", "Greens", grids['rh'], 0, 100),
        (1,3, "Vertical Velocity", "RdBu", grids['w'], -2, 2),
        (1,4, "Temperature", "inferno", grids['temp'], -40, 40),
    ]
    
    for r, c, title, cmap, arr, vmin, vmax in plots:
        ax = axes[r,c]
        ax.set_facecolor('#0e1117')
        im = ax.imshow(arr, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(title, color="white", fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Center Crosshair
        cy, cx = arr.shape[0]//2, arr.shape[1]//2
        ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
        
        if "Phase" in title:
            val = arr[cy, cx] if arr.size > 0 else 0
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
    st.caption("v71.0 | KINGDOM SCAN")
    
    if ds_sat: st.success("Meteosat Online")
    if ds_era: st.success("ERA5 Online")
    
    st.write("---")
    api_key = st.text_input("Gemini API Key", type="password")

# --- MAIN UI ---
st.title("VisionRain Command Center")

if ds_sat:
    # 1. AUTO-SCAN
    with st.spinner("SCANNING FULL DATASET FOR OPTIMAL TARGET..."):
        scan_results = scan_whole_kingdom(ds_sat, ds_era)
        m = scan_results['target_metrics']
        
    # 2. SHOW RESULT
    st.markdown(f"### 🎯 Optimal Target Found (Pixel: {scan_results['coords']})")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Prob", f"{m['prob']:.1f}%")
    c2.metric("Radius", f"{m['rad']:.1f} µm")
    p_str = "Liquid" if 0.5 < m['phase'] < 1.5 else "Ice" if m['phase'] > 1.5 else "Clear"
    c3.metric("Phase", p_str)
    c4.metric("Score", f"{m['score']:.1f}")

    # 3. VISUALIZE
    viz_img = plot_real_grids(scan_results['grids'])
    st.image(viz_img, use_column_width=True, caption="Target Sector Analysis (Red Cross = AI Target)")

    # 4. AI CONFIRMATION
    st.subheader("Vertex AI Validation")
    if st.button("RUN GEMINI 2.5 DIAGNOSTIC"):
        if not api_key: st.warning("Enter API Key")
        else:
            with st.status("Gemini 2.5 Flash Analyzing Physics...") as status:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt = f"""
                ACT AS A SENIOR CLOUD PHYSICIST. 
                This is the BEST target found in the Saudi dataset scan.
                
                METRICS:
                - Cloud Prob: {m['prob']:.1f}%
                - Radius: {m['rad']:.1f} µm
                - Phase: {p_str} (Value: {m['phase']:.1f})
                - Temp: {m['temp']:.1f} C
                - LWC: {m['lwc']:.4f}
                - Vertical Velocity: {m['w']:.2f} m/s
                
                LOGIC GATES:
                1. GHOST ECHO CHECK: If Prob > 50% but Radius is near 0 -> FALSE POSITIVE. ABORT.
                2. PHASE CHECK: If Ice, ABORT.
                3. GROWTH CHECK: Is LWC > 0 and Updraft (w) > 0?
                
                Provide GO/NO-GO Decision and Scientific Reasoning.
                """
                
                try:
                    res = model.generate_content([prompt, viz_img])
                    text = res.text
                    decision = "GO" if "GO" in text.upper() and "NO-GO" not in text.upper() else "NO-GO"
                    
                    status.update(label="Complete", state="complete")
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown(f'<div class="analysis-text analysis-go">✅ <b>MISSION APPROVED</b><br>{text}</div>', unsafe_allow_html=True)
                        # Approximating Lat/Lon from Pixel (Mock transform for demo)
                        lat_est = 24.0 + (scan_results['coords'][0] * 0.01)
                        lon_est = 45.0 + (scan_results['coords'][1] * 0.01)
                        st.link_button("🛰️ LAUNCH DRONES (Google Maps)", f"https://www.google.com/maps/search/?api=1&query={lat_est},{lon_est}")
                    else:
                        st.markdown(f'<div class="analysis-text analysis-nogo">⛔ <b>MISSION ABORTED</b><br>{text}</div>', unsafe_allow_html=True)
                        
                except Exception as e: st.error(f"AI Error: {e}")

else:
    st.error("DATA FILES NOT FOUND. Please upload .nc and .grib files to root.")
