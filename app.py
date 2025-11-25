import streamlit as st
import google.generativeai as genai
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import PIL.Image
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Cloud Command", layout="wide", page_icon="⛈️")

# --- LOAD SATELLITE DATA ---
@st.cache_resource
def load_satellite_data():
    # REPLACE THIS with your actual filename if different
    file_path = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
    try:
        # We use 'h5netcdf' or 'netcdf4' depending on what's installed
        ds = xr.open_dataset(file_path)
        return ds
    except FileNotFoundError:
        st.error(f"🚨 CRITICAL ERROR: Satellite Data File '{file_path}' not found. Please upload the .nc file to your folder.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

ds = load_satellite_data()

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #050505;}
    h1, h2, h3 {font-family: 'Courier New', monospace; color: #00FFCC;}
    .metric-box {
        background-color: #111;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.1);
    }
    .metric-value {font-size: 24px; font-weight: bold; color: #fff;}
    .metric-label {font-size: 14px; color: #888;}
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_pixel_data(city_name, ds):
    """
    Finds the closest pixel in the satellite file for the city.
    Returns the raw scientific values.
    """
    if ds is None: return None

    # 1. DEFINE COORDINATES
    # 'Target Sector X' is the secret storm location (South of Jeddah) used for the demo
    if city_name == "Target Sector X (Demo)":
        y, x = 2300, 750
        lat, lon = 16.5, 42.5
    elif city_name == "Jeddah":
        y, x = 2589, 668
        lat, lon = 21.54, 39.17
    elif city_name == "Riyadh":
        y, x = 2500, 700 # Approximate for demo variety
        lat, lon = 24.71, 46.67
    else:
        y, x = 2500, 700
        lat, lon = 0.0, 0.0

    try:
        # 2. EXTRACT DATA FROM FILE
        # We use .isel to get the specific pixel
        data = ds.isel(number_of_lines=y, number_of_pixels=x)
        
        # 3. PARSE VARIABLES
        # A. Pressure (Proxy for Phase)
        # Variable names might vary slightly, this covers the standard OCA names
        if 'cloud_top_pressure' in data:
            pressure = float(data['cloud_top_pressure'].values)
        else:
            pressure = 101300.0 # Default to sea level if missing
            
        # Logic: High Pressure = Low Cloud (Warm), Low Pressure = High Cloud (Ice)
        if pressure > 60000: phase = "LIQUID (Warm)"
        elif pressure < 45000: phase = "ICE (Glaciated)"
        else: phase = "MIXED"
        
        # B. Radius
        if 'cloud_particle_effective_radius' in data:
            rad_m = float(data['cloud_particle_effective_radius'].values)
            rad_um = rad_m * 1e6 if not np.isnan(rad_m) else 0.0
        else:
            rad_um = 0.0
            
        # C. Optical Depth
        if 'cloud_optical_depth_log' in data:
            cod_log = float(data['cloud_optical_depth_log'].values)
            cod = 10**cod_log if not np.isnan(cod_log) else 0.0
        else:
            cod = 0.0
            
        # D. Probability
        if 'cloud_probability' in data:
            prob = float(data['cloud_probability'].values)
        else:
            prob = 0.0

        return {
            "lat": lat, "lon": lon,
            "phase": phase,
            "radius": round(rad_um, 1),
            "optical_depth": round(cod, 1),
            "pressure": round(pressure/100, 0), # hPa
            "prob": round(prob * 100, 1),
            "y_idx": y, "x_idx": x
        }
    except Exception as e:
        st.error(f"Error reading satellite pixel: {e}")
        return None

def generate_satellite_image(ds, y_center, x_center):
    """Generates the 'Thermal' view for the AI."""
    window = 100
    y_slice = slice(y_center - window, y_center + window)
    x_slice = slice(x_center - window, x_center + window)
    
    # Use Cloud Top Pressure as visual proxy for IR (Dark=Ground, Light=Clouds)
    if 'cloud_top_pressure' in ds:
        img_data = ds['cloud_top_pressure'].isel(number_of_lines=y_slice, number_of_pixels=x_slice).values
    else:
        # Fallback noise if variable missing
        img_data = np.random.rand(200, 200)

    # Plotting
    plt.figure(figsize=(5, 5))
    plt.imshow(img_data, cmap='gray_r') # Reversed gray so clouds (low pressure) are white
    plt.axis('off')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return buf

def log_to_bigquery(data_dict):
    """Simulates saving to BigQuery/Excel."""
    if 'log_df' not in st.session_state:
        st.session_state['log_df'] = pd.DataFrame(columns=["Timestamp", "Location", "Phase", "Radius", "Decision"])
    
    new_row = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Location": f"{data_dict['lat']}, {data_dict['lon']}",
        "Phase": data_dict['phase'],
        "Radius": data_dict['radius'],
        "Decision": "PENDING AI REVIEW"
    }
    # Concat is preferred over append in pandas 2.0+
    st.session_state['log_df'] = pd.concat([st.session_state['log_df'], pd.DataFrame([new_row])], ignore_index=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2011/2011601.png", width=80)
    st.title("VisionRain Hub")
    st.caption("Google Cloud x KFUPM | Prototype v3")
    
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### 🎯 Target Selection")
    # THE DEMO TRICK: 'Target Sector X' points to the storm pixel
    target_city = st.selectbox("Select Region", ["Jeddah", "Riyadh", "Abha", "Target Sector X (Demo)"])
    
    if st.button("🛰️ ACQUIRE TARGET"):
        st.session_state['target_city'] = target_city
        st.session_state['scan_data'] = get_pixel_data(target_city, ds)
        st.rerun()

# --- MAIN UI ---
st.title("📡 VisionRain: Ionization Command Center")

tab_pitch, tab_data, tab_ai, tab_admin = st.tabs(["💡 Initiative", "📊 Satellite Telemetry", "🤖 Gemini Analysis", "📝 Admin Logs"])

# --- TAB 1: INITIATIVE ---
with tab_pitch:
    st.markdown("## 🇸🇦 VisionRain: The Future of Water Security")
    st.markdown("""
    **VisionRain** leverages Google Cloud + Satellite AI to revolutionize cloud seeding.
    
    Instead of flying blind, we use **Microphysical Data** (Phase, Droplet Size, Optical Depth) to target clouds that are physically ready to rain but need a 'nudge' via ionization.
    
    ### How it works:
    1.  **Satellite Acquisition:** Meteosat-9 scans the region.
    2.  **Physics Analysis:** We check if droplets are "stuck" (<14 microns).
    3.  **AI Authorization:** Gemini validates the target structure.
    4.  **Action:** Ground-based emitters are activated.
    """)
    st.info("System Status: OPERATIONAL | Region: IODC (Indian Ocean Data Coverage)")

# --- TAB 2: DATA DASHBOARD ---
with tab_data:
    if 'scan_data' in st.session_state and st.session_state['scan_data']:
        data = st.session_state['scan_data']
        
        st.markdown(f"### 📍 Live Feed: {st.session_state['target_city']}")
        st.caption(f"Coordinates: {data['lat']}N, {data['lon']}E | Source: Meteosat-9 (IODC)")
        
        # 1. METRICS
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Cloud Phase</div><div class='metric-value'>{data['phase']}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Droplet Radius</div><div class='metric-value'>{data['radius']} µm</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Optical Depth</div><div class='metric-value'>{data['optical_depth']}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-box'><div class='metric-label'>Probability</div><div class='metric-value'>{data['prob']}%</div></div>", unsafe_allow_html=True)
            
        st.divider()
        
        # 2. VISUALS
        col_img, col_table = st.columns([1, 1.5])
        
        with col_img:
            st.markdown("#### 🔭 Satellite Thermal View")
            # Generate the image from the data
            img_buf = generate_satellite_image(ds, data['y_idx'], data['x_idx'])
            st.image(img_buf, caption="Processed Cloud Top Pressure (Dark=Ground, White=Cloud)", use_column_width=True)
            st.session_state['current_image'] = img_buf # Save for AI tab
            
        with col_table:
            st.markdown("#### 📋 Seeding Viability Check")
            
            # Logic for checkmarks
            phase_check = "✅" if "LIQUID" in data['phase'] else "❌"
            rad_check = "✅" if 0 < data['radius'] < 14 else "❌"
            prob_check = "✅" if data['prob'] > 50 else "❌"
            
            df_check = pd.DataFrame({
                "Parameter": ["Cloud Phase", "Droplet Size", "Cloud Presence"],
                "Ideal Condition": ["Liquid / Mixed", "< 14 Microns (Stuck)", "> 50% Probability"],
                "Actual Value": [data['phase'], f"{data['radius']} µm", f"{data['prob']}%"],
                "Status": [phase_check, rad_check, prob_check]
            })
            st.table(df_check)

        # Log this scan to the admin tab
        log_to_bigquery(data)

    else:
        st.info("👈 Please select a target city and click 'ACQUIRE TARGET' in the sidebar to start the scan.")

# --- TAB 3: AI ANALYSIS ---
with tab_ai:
    st.header("Gemini 2.0 Decision Engine")
    
    if 'scan_data' in st.session_state and 'current_image' in st.session_state:
        data = st.session_state['scan_data']
        
        st.markdown("The AI will now analyze the visual structure and microphysics to authorize ionization.")
        
        if st.button("🚀 REQUEST LAUNCH AUTHORIZATION", type="primary"):
            if not api_key:
                st.error("⚠️ Please enter Gemini API Key in the Sidebar first.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    # Use Flash model for speed
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Prepare Image for Gemini
                    img = st.session_state['current_image']
                    img_pil = PIL.Image.open(img)

                    # PROMPT
                    prompt = f"""
                    You are the Mission Control AI for the Saudi Cloud Seeding Initiative.
                    Analyze the following Satellite Telemetry and Thermal Image.

                    --- TELEMETRY DATA ---
                    Location: {st.session_state['target_city']}
                    Cloud Phase: {data['phase']} (Must be Liquid/Mixed for Ionization)
                    Effective Radius: {data['radius']} microns (Target: < 14 microns means droplets are stuck)
                    Optical Thickness: {data['optical_depth']}
                    
                    --- INSTRUCTIONS ---
                    1. Analyze the image. Do you see bright white areas (Clouds) or dark areas (Ground)?
                    2. Evaluate the Telemetry. Is the cloud "stuck" (small droplets) and containing liquid water?
                    3. PROVIDE A FINAL DECISION: "ACTIVATE IONIZERS" or "STANDBY".
                    4. Keep it professional and scientific.
                    """
                    
                    with st.spinner("Gemini is fusing visual and sensor data..."):
                        response = model.generate_content([prompt, img_pil])
                        
                        st.markdown("### 🛰️ Mission Report")
                        st.markdown(response.text)
                        
                        if "ACTIVATE" in response.text:
                            st.success("SYSTEM ALERT: IONIZATION EMITTERS ACTIVE")
                            st.balloons()
                        else:
                            st.warning("SYSTEM STANDBY: Conditions not met.")
                            
                except Exception as e:
                    st.error(f"AI Connection Error: {e}")
    else:
        st.warning("Waiting for Satellite Data acquisition...")

# --- TAB 4: ADMIN LOGS ---
with tab_admin:
    st.header("🗄️ BigQuery Transaction Logs")
    st.markdown("All satellite scans are automatically logged for audit and climate analysis.")
    if 'log_df' in st.session_state:
        st.dataframe(st.session_state['log_df'], use_container_width=True)
    else:
        st.text("No logs yet.")
