import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import datetime
import os
from io import BytesIO
from scipy.ndimage import zoom
import folium
from streamlit_folium import st_folium

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

# --- COLORMAP GENERATOR FOR MAP ---
def array_to_colormap(array, cmap_name='jet'):
    """Converts a 2D numpy array to a color-mapped image for Leaflet overlay."""
    # Normalize 0-1
    norm = plt.Normalize(vmin=np.min(array), vmax=np.max(array))
    # Get Colormap
    cmap = cm.get_cmap(cmap_name)
    # Apply
    rgba_img = cmap(norm(array))
    # Make 0 values transparent
    rgba_img[..., 3] = np.where(array < 10, 0.0, 0.6) # Alpha channel logic
    return rgba_img

# --- THE "ALL OF SAUDI" SCANNER ---
@st.cache_data
def scan_whole_kingdom_cached(_ds_sat_dummy, _ds_era_dummy):
    # This wrapper exists just to use Streamlit cache on the heavy math
    # We reload raw data inside to avoid Pickling errors with Xarray objects in cache key
    ds_sat, ds_era = load_data_files()
    
    # 1. EXTRACT MASTER GRID (PROBABILITY)
    prob = get_var(ds_sat, ['prob', 'cloud_probability'], 1.0)
    if prob is None: prob = np.zeros((1000, 1000))
    
    target_shape = prob.shape
    
    # --- CRITICAL FIX: SYNCHRONIZATION ---
    def sync(arr):
        if arr is None: return np.zeros(target_shape)
        if arr.shape == target_shape: return arr
        zf = np.array(target_shape) / np.array(arr.shape)
        # Order 0 = Nearest Neighbor (Preserves Raw Pixels, prevents blurring)
        return zoom(arr, zf, order=0)

    # 2. SYNC ALL METRICS TO MASTER GRID
    rad = sync(get_var(ds_sat, ['rad', 'effective', 'cre'], 1.0))
    phase = sync(get_var(ds_sat, ['phase', 'cph'], 1.0))
    press = sync(get_var(ds_sat, ['press', 'pressure'], 0.01))
    
    # ERA5 is usually much smaller, so we upscale it
    lwc = sync(get_var(ds_era, ['clwc', 'liquid']))
    temp = sync(get_var(ds_era, ['t', 'temp'], 1.0))
    if np.max(temp) > 100: temp -= 273.15 # K to C
    w_vel = sync(get_var(ds_era, ['w', 'vertical']))

    # 3. GLOBAL SCORING (THE AI LOGIC)
    # Score = Prob(40%) + LiquidPhase(30%) + GoodRadius(30%)
    mask_prob = (prob > 40).astype(float)
    mask_phase = ((phase > 0.5) & (phase < 1.5)).astype(float) # Liquid = 1
    mask_rad = ((rad > 5) & (rad < 20)).astype(float) # Seedable size
    
    score_map = (prob * 0.4) + (mask_phase * 100 * 0.3) + (mask_rad * 100 * 0.3)
    
    # 4. FIND TARGET
    if np.max(score_map) > 0:
        y_hot, x_hot = np.unravel_index(np.argmax(score_map), score_map.shape)
    else:
        y_hot, x_hot = prob.shape[0]//2, prob.shape[1]//2

    # 5. CROP TARGET DATA (100x100 Window around hotspot)
    win = 50
    y1, y2 = max(0, y_hot-win), min(target_shape[0], y_hot+win)
    x1, x2 = max(0, x_hot-win), min(target_shape[1], x_hot+win)
    
    return {
        "grids": {
            "prob": prob[y1:y2, x1:x2],
            "rad": rad[y1:y2, x1:x2],
            "phase": phase[y1:y2, x1:x2],
            "temp": temp[y1:y2, x1:x2],
            "lwc": lwc[y1:y2, x1:x2],
            "press": press[y1:y2, x1:x2],
            "w": w_vel[y1:y2, x1:x2]
        },
        "metrics": {
            "prob": float(prob[y_hot, x_hot]),
            "rad": float(rad[y_hot, x_hot]),
            "phase": float(phase[y_hot, x_hot]),
            "temp": float(temp[y_hot, x_hot]),
            "lwc": float(lwc[y_hot, x_hot]),
            "w": float(w_vel[y_hot, x_hot]),
            "score": float(score_map[y_hot, x_hot])
        },
        "map_layer": score_map, # Full map for Folium Overlay
        "coords_px": (y_hot, x_hot),
        "full_shape": target_shape
    }

# --- VISUALIZATION (RAW PIXELS 480p) ---
def plot_real_grids(grids):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8)) # 8 main plots
    fig.patch.set_facecolor('#0e1117')
    
    plots = [
        (0,0, "Cloud Probability (%)", "Blues", grids['prob'], 0, 100),
        (0,1, "Cloud Top Pressure (hPa)", "gray_r", grids['press'], 200, 1000),
        (0,2, "Effective Radius (µm)", "viridis", grids['rad'], 0, 30),
        (0,3, "Cloud Phase", "cool", grids['phase'], 0, 3),
        (1,0, "Liquid Water (LWC)", "Blues", grids['lwc'], None, None),
        (1,1, "Vertical Velocity (m/s)", "RdBu", grids['w'], -2, 2),
        (1,2, "Temperature (°C)", "inferno", grids['temp'], -40, 40),
        # Histogram Analysis
        (1,3, "Droplet Dist. (Histogram)", "hist", grids['rad'], 0, 30)
    ]
    
    for r, c, title, cmap, arr, vmin, vmax in plots:
        ax = axes[r,c]
        ax.set_facecolor('#0e1117')
        
        if cmap == "hist":
            ax.hist(arr.flatten(), bins=20, color='cyan', alpha=0.7)
            ax.set_title(title, color="white", fontweight='bold')
            ax.tick_params(colors='white')
        else:
            # NEAREST = RAW PIXEL LOOK
            im = ax.imshow(arr, cmap=cmap, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_title(title, color="white", fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Center Target Crosshair
            cy, cx = arr.shape[0]//2, arr.shape[1]//2
            ax.plot(cx, cy, 'r+', markersize=20, markeredgewidth=3)
            
            if "Phase" in title:
                val = arr[cy, cx]
                txt = "LIQUID" if 0.5 < val < 1.5 else "ICE" if val > 1.5 else "CLEAR"
                ax.text(0.5, 0.5, txt, color="cyan" if "LIQ" in txt else "white", ha="center", va="center", transform=ax.transAxes, fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
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
    st.caption("v88.0 | AUTO-SCAN MODE")
    if ds_sat: st.success("Meteosat Online")
    if ds_era: st.success("ERA5 Online")
    
    st.write("---")
    api_key = st.text_input("Gemini API Key", type="password")

# --- MAIN UI ---
st.title("VisionRain Command Center")

if ds_sat:
    # 1. AUTO-SCAN WHOLE DATASET
    with st.spinner("SCANNING FULL KINGDOM DATASET (VECTORIZED)..."):
        # We pass dummy args to trick cache into working with file reload
        scan = scan_whole_kingdom_cached(1, 1)
        m = scan['metrics']
        
        # APPROXIMATE LAT/LON (Bounding Box Mapping)
        # Saudi Bbox: ~ Lat 16-32, Lon 34-56
        # Map pixel Y (0-MAX) to Lat (32-16) (Inverted Y)
        # Map pixel X (0-MAX) to Lon (34-56)
        py, px = scan['coords_px']
        h, w = scan['full_shape']
        
        # Simple linear interpolation for coordinates (Mock Projection)
        lat_est = 32.0 - (py / h) * (32.0 - 16.0)
        lon_est = 34.0 + (px / w) * (56.0 - 34.0)

    # 2. MAP VISUALIZATION (GEE STYLE LAYER)
    st.markdown(f"### 📍 Target Locked: {lat_est:.4f}, {lon_est:.4f} (Score: {m['score']:.1f})")
    
    row1_col1, row1_col2 = st.columns([2, 1])
    
    with row1_col1:
        # Create Folium Map centered on Target
        m_folium = folium.Map(location=[lat_est, lon_est], zoom_start=6, tiles="CartoDB dark_matter")
        
        # Overlay the "Score Map" (The heatmap of seedability)
        # We construct an image overlay using the bounds
        img_overlay = array_to_colormap(scan['map_layer'], cmap_name='jet')
        folium.raster_layers.ImageOverlay(
            image=img_overlay,
            bounds=[[16, 34], [32, 56]], # Saudi Approx Bounds
            opacity=0.6,
            name="Seedability Heatmap"
        ).add_to(m_folium)
        
        # Add Marker for the specific target found
        folium.Marker(
            [lat_est, lon_est], 
            popup=f"TARGET ALPHA\nProb: {m['prob']:.0f}%\nLWC: {m['lwc']:.4f}",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
        ).add_to(m_folium)
        
        st_folium(m_folium, height=400, use_container_width=True)

    with row1_col2:
        st.markdown("#### Real-Time Telemetry")
        st.metric("Cloud Probability", f"{m['prob']:.1f}%", delta="High Confidence")
        st.metric("Effective Radius", f"{m['rad']:.1f} µm", help="Ideal: 5-14 µm")
        st.metric("Liquid Water (LWC)", f"{m['lwc']:.4f} g/m³")
        st.metric("Cloud Phase", "Liquid" if 0.5 < m['phase'] < 1.5 else "Ice/Mix")

    # 3. SCIENTIFIC VISUALIZATION
    st.subheader("Target Sector Microphysics (Raw 10km Grid)")
    viz_img = plot_real_grids(scan['grids'])
    st.image(viz_img, use_column_width=True)

    # 4. AI CONFIRMATION
    st.subheader("Vertex AI Mission Commander")
    if st.button("RUN MISSION DIAGNOSTIC"):
        if not api_key: st.warning("Enter API Key")
        else:
            with st.status("Gemini 2.5 Flash Analyzing Physics...") as status:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt = f"""
                ACT AS A SENIOR CLOUD PHYSICIST. 
                Analyze this AUTOMATICALLY DETECTED TARGET at {lat_est}, {lon_est}.
                
                METRICS:
                - Probability: {m['prob']:.1f}%
                - Radius: {m['rad']:.1f} µm
                - Phase Index: {m['phase']:.1f} (1=Liquid)
                - Temp: {m['temp']:.1f} C
                - LWC: {m['lwc']:.4f}
                - Vertical Velocity: {m['w']:.2f} m/s
                
                LOGIC GATES:
                1. GHOST ECHO: Prob > 50 but Radius ~0? -> ABORT.
                2. PHASE: Ice (Index > 1.5)? -> ABORT.
                3. GROWTH: LWC > 0 and Updraft > 0? -> SUPPORT.
                
                DECISION: GO or NO-GO?
                ANALYSIS: Explain the microphysics.
                """
                
                try:
                    res = model.generate_content([prompt, viz_img])
                    text = res.text
                    decision = "GO" if "GO" in text.upper() and "NO-GO" not in text.upper() else "NO-GO"
                    
                    status.update(label="Complete", state="complete")
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown(f'<div class="analysis-text analysis-go">✅ <b>MISSION APPROVED</b><br>{text}</div>', unsafe_allow_html=True)
                        st.link_button("🛰️ LAUNCH DRONES (Google Maps)", f"https://www.google.com/maps/search/?api=1&query={lat_est},{lon_est}")
                    else:
                        st.markdown(f'<div class="analysis-text analysis-nogo">⛔ <b>MISSION ABORTED</b><br>{text}</div>', unsafe_allow_html=True)
                        
                except Exception as e: st.error(f"AI Error: {e}")

else:
    st.error("DATA FILES NOT FOUND. Please upload .nc and .grib files to root.")
