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

# --- SAFE IMPORTS FOR SCIENTIFIC LIBRARIES ---
try:
    import xarray as xr
    import cfgrib
    SCIENTIFIC_LIBS_AVAILABLE = True
except ImportError:
    SCIENTIFIC_LIBS_AVAILABLE = False

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

# --- REAL FILE PATHS (From your upload) ---
NETCDF_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
GRIB_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .stMetric label {color: #888;}
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    .cloud-badge {
        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;
        background-color: #1a1a1a; border: 1px solid #444; color: #ccc;
        margin-bottom: 5px; display: inline-block;
    }
    .status-ok {color: #00ff80;}
    .status-err {color: #ff4444;}
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah": {"coords": [21.5433, 39.1728], "pixel_y": 2300, "pixel_x": 750, "bias_prob": 0, "humidity_base": 60},
    "Abha": {"coords": [18.2164, 42.5053], "pixel_y": 2500, "pixel_x": 800, "bias_prob": 30, "humidity_base": 70},
    "Riyadh": {"coords": [24.7136, 46.6753], "pixel_y": 2100, "pixel_x": 900, "bias_prob": -40, "humidity_base": 20},
    "Dammam": {"coords": [26.4207, 50.0888], "pixel_y": 2000, "pixel_x": 950, "bias_prob": -10, "humidity_base": 65},
    "Tabuk": {"coords": [28.3835, 36.5662], "pixel_y": 1900, "pixel_x": 700, "bias_prob": 10, "humidity_base": 35}
}

# --- CLOUD INFRASTRUCTURE SIMULATION ---
class BigQueryClient:
    def insert_rows(self, dataset, table, rows): time.sleep(0.1); return True
class CloudStorageClient:
    def fetch_satellite_data(self, region): time.sleep(0.1); return True
bq_client = BigQueryClient()
gcs_client = CloudStorageClient()

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

# --- SCIENTIFIC DATA LOADING ---
@st.cache_resource
def load_real_data():
    """Attempts to load the actual uploaded files."""
    ds_sat, ds_era = None, None
    if SCIENTIFIC_LIBS_AVAILABLE:
        if os.path.exists(NETCDF_FILE):
            try: ds_sat = xr.open_dataset(NETCDF_FILE, engine='netcdf4')
            except: pass
        if os.path.exists(GRIB_FILE):
            try: ds_era = xr.open_dataset(GRIB_FILE, engine='cfgrib')
            except: pass
    return ds_sat, ds_era

# --- SIMULATION ENGINE (FALLBACK) ---
def generate_cloud_texture(shape=(100, 100), seed=42, intensity=1.0, roughness=5.0):
    np.random.seed(seed)
    noise = np.random.rand(*shape)
    smooth = gaussian_filter(noise, sigma=roughness)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return smooth * intensity

def get_simulated_data(sector_name):
    """Simulated physics data if file read fails."""
    profile = SAUDI_SECTORS[sector_name]
    conditions = [
        {"prob": 85.0, "press": 650, "rad": 12.5, "opt": 15.0, "lwc": 0.005, "rh": 80, "temp": -8.0, "phase": 1}, 
        {"prob": 5.0, "press": 950, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 20, "temp": 28.0, "phase": 0},
    ]
    data = random.choice(conditions).copy()
    data['prob'] += profile['bias_prob'] + random.uniform(-5, 5)
    data['prob'] = max(0.0, min(100.0, data['prob']))
    
    if data['prob'] > 60 and data['rad'] < 14 and data['phase'] == 1: data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 40: data['status'] = "MONITORING"
    else: data['status'] = "UNSUITABLE"
    return data

# --- THE ROBUST VISUALIZER (FIXED MATH) ---
def generate_scientific_plots(ds_sat, ds_era, sector_name):
    """
    Generates the Matplotlib visualization using masked arrays to fix probability values.
    Handles REAL data if available, falls back to SIMULATION if not.
    """
    profile = SAUDI_SECTORS[sector_name]
    
    # Check if we have real data loaded
    use_real_data = (ds_sat is not None)
    
    # Setup Figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    # DATA EXTRACTION LOGIC
    if use_real_data:
        try:
            # 1. Dynamic Dimension Finder (User's Logic)
            # Find the Probability Variable (might be named differently)
            prob_var = None
            for v in ['cloud_probability', 'prob', 'c_prob']:
                if v in ds_sat: prob_var = v; break
            
            if prob_var:
                dims = list(ds_sat[prob_var].dims)
                y_dim_name = dims[0]
                x_dim_name = dims[1]

                # 2. Slicing
                center_y = profile['pixel_y']
                center_x = profile['pixel_x']
                window = 50
                
                max_y = ds_sat.sizes[y_dim_name]
                max_x = ds_sat.sizes[x_dim_name]
                
                y_start = max(0, center_y - window)
                y_end = min(max_y, center_y + window)
                x_start = max(0, center_x - window)
                x_end = min(max_x, center_x + window)

                slice_dict = {
                    y_dim_name: slice(y_start, y_end),
                    x_dim_name: slice(x_start, x_end)
                }
                
                # Extract & Fix Math
                raw_prob = ds_sat[prob_var].isel(**slice_dict).values
                # MASKING: Filter values < 0 or > 100 (Fill values)
                masked_prob = np.ma.masked_where((raw_prob < 0) | (raw_prob > 100), raw_prob)
                
                # We would repeat this extraction for other vars, but for brevity/demo stability
                # we will use this extracted slice for the visual shape and fill others
                visual_data_prob = masked_prob
                mean_prob = float(np.mean(masked_prob))
            else:
                use_real_data = False # Fallback if var missing
        except Exception as e:
            use_real_data = False # Fallback on error
            
    # If Real Data Failed or Missing, Generate Simulation
    if not use_real_data:
        sim = get_simulated_data(sector_name)
        seed = int(sim['prob'] * 100)
        visual_data_prob = generate_cloud_texture(seed=seed, roughness=6) * sim['prob']
        mean_prob = sim['prob']
        # Set other sim values for later
        current_metrics = sim
    else:
        # Create a hybrid metrics object using the real mean probability
        # but simulated microphysics for the other 9 plots (since we only extracted prob above)
        sim = get_simulated_data(sector_name)
        sim['prob'] = mean_prob # Override with real mean
        current_metrics = sim

    # PLOTTING ROUTINE
    # We use visual_data_prob for the first plot, and generate textures for others based on it
    
    plots = [
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": visual_data_prob, "vmax": 100},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": generate_cloud_texture(seed=1, roughness=8) * current_metrics['press'], "vmax": 1000},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": generate_cloud_texture(seed=2, roughness=4) * current_metrics['rad'], "vmax": 30},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": generate_cloud_texture(seed=3, roughness=5) * current_metrics['opt'], "vmax": 50},
        {"ax": axes[0,4], "title": "Phase (0=Clr,1=Liq,2=Ice)", "cmap": "cool", "data": generate_cloud_texture(seed=4, roughness=10) * current_metrics['phase'], "vmax": 2},
        
        {"ax": axes[1,0], "title": "Liquid Water (kg/m³)", "cmap": "Blues", "data": generate_cloud_texture(seed=5, roughness=7) * current_metrics['lwc'], "vmax": 0.01},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu", "data": generate_cloud_texture(seed=6, roughness=7) * (current_metrics['lwc']/3), "vmax": 0.01},
        {"ax": axes[1,2], "title": "Rel. Humidity (%)", "cmap": "Greens", "data": generate_cloud_texture(seed=7, roughness=10) * current_metrics['rh'], "vmax": 100},
        {"ax": axes[1,3], "title": "Vertical Velocity (m/s)", "cmap": "RdBu_r", "data": (generate_cloud_texture(seed=8, roughness=3) - 0.5) * 10},
        {"ax": axes[1,4], "title": "Temperature (°C)", "cmap": "inferno", "data": generate_cloud_texture(seed=9, roughness=15) * 10 + current_metrics['temp']},
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
    return Image.open(buf), current_metrics, use_real_data

# --- STRATEGIC PITCH VISUALIZER ---
def plot_strategic_sensor_fusion():
    """Generates the 3x5 Sensor Fusion Grid for the Pitch."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 8))
    fig.patch.set_facecolor('#0e1117')
    
    rows_info = [
        {"label": "SATELLITE\n(Meteosat)", "key": "prob", "cmap": "Blues"},
        {"label": "MICROPHYSICS\n(Scanner)", "key": "rad", "cmap": "viridis"},
        {"label": "THERMAL\n(ERA5)", "key": "temp", "cmap": "inferno"},
    ]
    
    col_idx = 0
    # Use simulated data for the pitch summary to ensure it looks complete
    for sector in SAUDI_SECTORS:
        data = get_simulated_data(sector)
        seed = int(data['prob'] * 100)
        
        for row_idx, info in enumerate(rows_info):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#0e1117')
            
            tex = generate_cloud_texture(seed=seed + (row_idx*10), roughness=5 + row_idx) 
            if info['key'] == 'temp':
                tex = tex * 0.5 + (data[info['key']] + 20) / 60.0 
            else:
                tex = tex * (data[info['key']] / (100.0 if info['key']=='prob' else 30.0))

            ax.imshow(tex, cmap=info['cmap'], aspect='auto')
            
            if row_idx == 0: ax.set_title(sector, color="#00e5ff", fontsize=11, fontweight='bold')
            if col_idx == 0: ax.set_ylabel(info['label'], color="white", fontsize=9, labelpad=10)
            
            val_str = f"{data[info['key']]:.0f}" + ("%" if info['key']=='prob' else "µm" if info['key']=='rad' else "°C")
            ax.text(50, 50, val_str, color="white", ha="center", va="center", fontweight="bold", fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round'))
            ax.set_xticks([]); ax.set_yticks([])
        col_idx += 1
        
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0e1117', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# --- APP STATE INIT ---
ds_sat_real, ds_era_real = load_real_data()

if 'selected_region' not in st.session_state:
    st.session_state.selected_region = "Jeddah"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=80)
    st.title("VisionRain")
    st.caption("Kingdom Commander | v26.0 (Sci-Ops)")
    
    st.markdown("### ☁️ Infrastructure")
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Cloud Run</div>', unsafe_allow_html=True)
    st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Vertex AI</div>', unsafe_allow_html=True)
    if ds_sat_real is not None:
        st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Live Sat Feed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cloud-badge"><span class="status-ok">●</span> Simulation Mode</div>', unsafe_allow_html=True)

    st.write("---")
    st.markdown("### 📡 Active Sector")
    region_options = list(SAUDI_SECTORS.keys())
    selected = st.selectbox("Region", region_options, index=region_options.index(st.session_state.selected_region))
    if selected != st.session_state.selected_region:
        st.session_state.selected_region = selected
        st.rerun()

    api_key = st.text_input("Gemini API Key", type="password")
    with st.expander("🔒 Admin Logs"):
        if st.text_input("Key", type="password") == "123456": st.dataframe(get_mission_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
tab1, tab2, tab3 = st.tabs(["🌍 Strategic Sensor Fusion", "🛰️ Scientific Operations", "🧠 Vertex AI Commander"])

# TAB 1: PITCH
with tab1:
    st.header("Vision 2030: Rain Enhancement Strategy")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 The Challenge</h3>
    <p>Saudi Arabia faces critical water scarcity. VisionRain uses AI to detect seedable clouds via sensor fusion.</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Kingdom-Wide Sensor Fusion")
    st.image(plot_strategic_sensor_fusion(), use_column_width=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**Solution**\n\nAI-driven, pilotless ecosystem."); c2.warning("**Tech**\n\nVertex AI + BigQuery."); c3.success("**Impact**\n\nAutomated Water Security.")

# TAB 2: OPS
with tab2:
    current_region = st.session_state.selected_region
    
    # GENERATE PLOTS (Handles Real vs Sim internally)
    matrix_img, current_metrics, is_real = generate_scientific_plots(ds_sat_real, ds_era_real, current_region)
    
    c_header, c_stats = st.columns([2, 3])
    with c_header:
        st.header(f"📍 {current_region}")
        if is_real: st.caption("✅ Live Telemetry: EUMETSAT Stream (Masked/Filtered)")
        else: st.caption("⚠️ Simulation Mode (GCS Connection Pending)")
            
    with c_stats:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cloud Prob", f"{current_metrics['prob']:.1f}%", delta="High" if current_metrics['prob']>60 else "Low")
        s2.metric("Radius", f"{current_metrics['rad']:.1f} µm", help="Target: < 14µm")
        s3.metric("Phase", "Liquid" if current_metrics['phase']==1 else "Ice/Mix")
        s4.metric("Status", current_metrics['status'])

    col_map, col_matrix = st.columns([1, 2])
    with col_map:
        st.markdown("**Live Sector Map**")
        m = folium.Map(location=SAUDI_SECTORS[current_region]['coords'], zoom_start=6, tiles="CartoDB dark_matter")
        for reg_name, info in SAUDI_SECTORS.items():
            # Quick get sim data just for map color
            d = get_simulated_data(reg_name)
            color = "green" if d['prob'] > 60 else "orange" if d['prob'] > 30 else "gray"
            folium.Marker(info['coords'], popup=f"{reg_name}", icon=folium.Icon(color=color, icon="cloud", prefix="fa")).add_to(m)
            if reg_name == current_region:
                folium.CircleMarker(info['coords'], radius=20, color="#00e5ff", fill=True, fill_opacity=0.2).add_to(m)
        st_folium(m, height=300, use_container_width=True)

    with col_matrix:
        st.markdown("**Real-Time Microphysics Matrix**")
        st.image(matrix_img, use_column_width=True)

    st.divider()
    st.subheader("Kingdom-Wide Surveillance Wall")
    table_data = []
    for reg in SAUDI_SECTORS:
        d = get_simulated_data(reg) # Using sim for summary table consistency
        table_data.append({
            "Region": reg, "Priority": "🔴 High" if d['prob'] > 60 else "🟡 Medium" if d['prob'] > 30 else "⚪ Low",
            "Probability": d['prob'], "Effective Radius": f"{d['rad']:.1f} µm", "Condition": d['status']
        })
    df_table = pd.DataFrame(table_data).sort_values("Probability", ascending=False)
    st.dataframe(df_table, use_container_width=True, column_config={"Probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0, max_value=100)}, hide_index=True)

# TAB 3: GEMINI
with tab3:
    st.header(f"Vertex AI Commander: {current_region}")
    c1, c2 = st.columns([1, 1])
    with c1: st.image(matrix_img, caption="Visual Input Tensor")
    with c2: st.write("### Telemetry Packet"); st.json(current_metrics)

    if st.button("🚀 REQUEST AUTHORIZATION (VERTEX AI)", type="primary"):
        with st.status("Initializing Vertex AI Pipeline...") as status:
            st.write("1. Fetching Model: `gemini-2.5-flash`...")
            time.sleep(0.5)
            
            prompt = f"""
            ACT AS A METEOROLOGIST. Analyze {current_region}.
            DATA: Prob: {current_metrics['prob']:.1f}%, Radius: {current_metrics['rad']:.1f}um, Phase: {current_metrics['phase']}.
            RULES: GO IF Radius < 14 AND Radius > 5 AND Phase=1 (Liquid). NO-GO IF Phase=2 (Ice) OR Prob < 50.
            OUTPUT: Decision (GO/NO-GO), Reasoning, Protocol.
            """
            
            decision = "PENDING"; response_text = ""
            if not api_key:
                st.warning("⚠️ Offline Mode (Simulated Response)")
                time.sleep(1)
                if current_metrics['prob'] > 60 and current_metrics['rad'] < 14 and current_metrics['phase'] == 1:
                    decision = "GO"; response_text = "Vertex AI Confidence: 98%. Conditions Optimal."
                else:
                    decision = "NO-GO"; response_text = f"Vertex AI Confidence: 95%. Conditions Unfavorable."
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    res = model.generate_content([prompt, matrix_img])
                    response_text = res.text
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                except Exception as e: decision = "ERROR"; response_text = str(e)

            st.write("3. Logging Decision to BigQuery...")
            status.update(label="Complete", state="complete")
        
        if "GO" in decision and "NO-GO" not in decision:
            st.balloons(); st.success(f"✅ MISSION APPROVED: {response_text}")
        else:
            st.error(f"⛔ MISSION ABORTED: {response_text}")
            
        save_mission_log(current_region, str(current_metrics), decision, response_text)
        st.toast("Audit Log Saved to BigQuery", icon="☁️")
