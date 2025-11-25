import streamlit as st
import google.generativeai as genai
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import PIL.Image
from io import BytesIO

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Cloud Command", layout="wide", page_icon="⛈️")


# --- 2. LOAD SATELLITE DATA ---
@st.cache_resource
def load_satellite_data():
    # 👇👇👇 PASTE YOUR EXACT FILENAME INSIDE THE QUOTES BELOW 👇👇👇
    file_path = "C:\Users\nessp\OneDrive\Documents\GitHub\VisionRainWEB\W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
    
    try:
        # We use 'h5netcdf' engine for better performance/compatibility on cloud
        ds = xr.open_dataset(file_path, engine='h5netcdf')
        return ds
    except FileNotFoundError:
        st.error(f"🚨 FILE NOT FOUND: Could not find '{file_path}'. Make sure it is in the same folder as app.py!")
        return None
    except Exception as e:
        st.error(f"Error loading satellite file: {e}")
        return None

ds = load_satellite_data()

# --- 3. STYLING (Command Center Aesthetic) ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    h1, h2, h3 {font-family: 'Courier New', monospace; color: #00FFCC;}
    .metric-box {
        background-color: #1c212c;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {font-size: 26px; font-weight: bold; color: #00FFCC;}
    .metric-label {font-size: 14px; color: #aaa; text-transform: uppercase; letter-spacing: 1px;}
    .success-box {border: 1px solid #00FF00; background-color: rgba(0,255,0,0.1); padding: 10px; border-radius: 5px; color: #00FF00;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. DATA PROCESSING FUNCTIONS ---

def get_pixel_data(city_name, ds):
    """
    Finds the specific pixel in the satellite file for the chosen target.
    """
    if ds is None: return None

    # COORDINATE MAPPING (The "Demo Magic")
    # We map "Target Sector X" to a storm location south of Jeddah
    if city_name == "Target Sector X (Demo)":
        y, x = 2300, 750  # <-- THE STORM PIXEL
        lat, lon = 16.5, 42.5
    elif city_name == "Jeddah":
        y, x = 2589, 668  # Jeddah Pixel
        lat, lon = 21.54, 39.17
    elif city_name == "Riyadh":
        y, x = 2500, 700  # Approx Riyadh Pixel
        lat, lon = 24.71, 46.67
    else:
        y, x = 2500, 700
        lat, lon = 0.0, 0.0

    try:
        # Extract the specific pixel data
        data = ds.isel(number_of_lines=y, number_of_pixels=x)
        
        # --- PARSE SCIENTIFIC VARIABLES ---
        # 1. Pressure (Proxy for Phase)
        # Low Pressure (<450hPa) = Ice/High Cloud
        # High Pressure (>600hPa) = Liquid/Low Cloud
        if 'cloud_top_pressure' in data:
            pressure = float(data['cloud_top_pressure'].values)
        else:
            pressure = 101300.0 # Default Sea Level
            
        if pressure > 60000: phase = "LIQUID (Warm)"
        elif pressure < 45000: phase = "ICE (Glaciated)"
        else: phase = "MIXED / UNCERTAIN"
        
        # 2. Effective Radius (Droplet Size)
        # Target: < 14 microns (means droplets are small and "stuck")
        if 'cloud_particle_effective_radius' in data:
            rad_m = float(data['cloud_particle_effective_radius'].values)
            rad_um = rad_m * 1e6 if not np.isnan(rad_m) else 0.0
        else:
            rad_um = 0.0
            
        # 3. Optical Depth (Thickness)
        if 'cloud_optical_depth_log' in data:
            cod_log = float(data['cloud_optical_depth_log'].values)
            cod = 10**cod_log if not np.isnan(cod_log) else 0.0
        else:
            cod = 0.0
            
        # 4. Cloud Probability
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
        st.error(f"Pixel Extraction Error: {e}")
        return None

def generate_satellite_image(ds, y_center, x_center):
    """Generates the 'Thermal' view using Cloud Top Pressure."""
    window = 100 # Zoom level
    y_slice = slice(y_center - window, y_center + window)
    x_slice = slice(x_center - window, x_center + window)
    
    # Extract image patch
    if 'cloud_top_pressure' in ds:
        img_data = ds['cloud_top_pressure'].isel(number_of_lines=y_slice, number_of_pixels=x_slice).values
    else:
        img_data = np.zeros((200, 200))

    # Render with Matplotlib
    plt.figure(figsize=(5, 5))
    # cmap='gray_r' (Reversed Gray) makes Low Pressure (Clouds) = White, High Pressure (Ground) = Dark
    plt.imshow(img_data, cmap='gray_r') 
    plt.axis('off')
    
    # Save to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return buf

def log_to_bigquery(data_dict):
    """Simulates logging data to BigQuery for Admin review."""
    if 'log_df' not in st.session_state:
        st.session_state['log_df'] = pd.DataFrame(columns=["Timestamp", "Location", "Phase", "Radius", "Status"])
    
    new_row = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Location": f"{data_dict['lat']}N, {data_dict['lon']}E",
        "Phase": data_dict['phase'],
        "Radius": f"{data_dict['radius']} µm",
        "Status": "SCANNED"
    }
    # Use concat instead of append (Pandas 2.0 best practice)
    st.session_state['log_df'] = pd.concat([st.session_state['log_df'], pd.DataFrame([new_row])], ignore_index=True)

# --- 5. SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2011/2011601.png", width=80)
    st.title("VisionRain Hub")
    st.caption("Google Cloud x KFUPM | Prototype v3")
    
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### 🎯 Target Selection")
    
    # SELECTION LOGIC
    target_city = st.selectbox("Select Mission Sector", ["Jeddah", "Riyadh", "Target Sector X (Demo)"])
    
    if st.button("🛰️ ACQUIRE TARGET"):
        st.session_state['target_city'] = target_city
        st.session_state['scan_data'] = get_pixel_data(target_city, ds)
        st.rerun() # Refresh page to show data

# --- 6. MAIN DASHBOARD UI ---
st.title("📡 VisionRain: Ionization Command Center")

tab_pitch, tab_data, tab_ai, tab_admin = st.tabs(["💡 Initiative", "📊 Satellite Telemetry", "🤖 Gemini Analysis", "📝 Admin Logs"])

# --- TAB 1: PITCH ---
with tab_pitch:
    st.markdown("## 🇸🇦 VisionRain: The Future of Water Security")
    st.markdown("""
    **VisionRain** leverages **Google Cloud + Meteosat AI** to revolutionize cloud seeding.
    
    Instead of 'spray and pray', we use **Microphysical Data** to target clouds that are physically ready to rain but need an ionic 'nudge'.
    
    ### The Science:
    * **Phase Discrimination:** We target **Liquid/Mixed** clouds (Ice clouds cannot be seeded).
    * **Droplet Analysis:** We look for droplets < 14 microns (Stalled Coalescence).
    * **AI Validation:** Gemini validates the convective structure visually.
    """)
    st.info("System Status: OPERATIONAL | Data Source: EUMETSAT Meteosat-9 (IODC)")

# --- TAB 2: DATA DASHBOARD ---
with tab_data:
    if 'scan_data' in st.session_state and st.session_state['scan_data']:
        data = st.session_state['scan_data']
        
        st.markdown(f"### 📍 Live Feed: {st.session_state['target_city']}")
        st.caption(f"Coordinates: {data['lat']}N, {data['lon']}E | Source: Meteosat-9 (IODC)")
        
        # METRICS ROW
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
        
        # VISUALS ROW
        col_img, col_table = st.columns([1, 1.5])
        
        with col_img:
            st.markdown("#### 🔭 Satellite Thermal View")
            img_buf = generate_satellite_image(ds, data['y_idx'], data['x_idx'])
            st.image(img_buf, caption="Processed Cloud Top Pressure (Dark=Ground, White=Cloud)", use_column_width=True)
            st.session_state['current_image'] = img_buf # Save for AI
            
        with col_table:
            st.markdown("#### 📋 Seeding Viability Check")
            # Logic for Checkmarks
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

        log_to_bigquery(data)

    else:
        st.info("👈 Please select a target city and click 'ACQUIRE TARGET' in the sidebar.")

# --- TAB 3: AI ANALYSIS ---
with tab_ai:
    st.header("Gemini 2.0 Decision Engine")
    
    if 'scan_data' in st.session_state and 'current_image' in st.session_state:
        data = st.session_state['scan_data']
        
        st.markdown("The AI will now fuse the Visual Data with the Numerical Telemetry to make a final Go/No-Go decision.")
        
        if st.button("🚀 REQUEST LAUNCH AUTHORIZATION", type="primary"):
            if not api_key:
                st.error("⚠️ Please enter Gemini API Key in the Sidebar first.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Prepare Image
                    img = st.session_state['current_image']
                    img_pil = PIL.Image.open(img)

                    # SUPERPROMPT
                    prompt = f"""
                    You are the Mission Control AI for the Saudi Cloud Seeding Initiative.
                    Analyze the following Satellite Telemetry and Thermal Image.

                    --- TELEMETRY DATA ---
                    Location: {st.session_state['target_city']}
                    Cloud Phase: {data['phase']} (Must be Liquid/Mixed)
                    Effective Radius: {data['radius']} microns (Target < 14 microns = Seedable)
                    Optical Thickness: {data['optical_depth']}
                    
                    --- INSTRUCTIONS ---
                    1. Analyze the Image: Do you see bright white convective structures?
                    2. Analyze the Physics: Is the cloud liquid and are droplets small ("stuck")?
                    3. DECISION: If Phase is LIQUID/MIXED and Radius < 14, output "ACTIVATE IONIZERS". Otherwise "STANDBY".
                    4. Provide brief scientific reasoning.
                    """
                    
                    with st.spinner("Gemini is analyzing atmospheric microphysics..."):
                        response = model.generate_content([prompt, img_pil])
                        
                        st.markdown("### 🛰️ Mission Report")
                        st.markdown(response.text)
                        
                        if "ACTIVATE" in response.text:
                            st.markdown("<div class='success-box'>✅ SYSTEM ALERT: IONIZATION EMITTERS ACTIVE</div>", unsafe_allow_html=True)
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
    st.markdown("Real-time logging of all scan events for climate auditing.")
    if 'log_df' in st.session_state:
        st.dataframe(st.session_state['log_df'], use_container_width=True)
    else:
        st.text("No logs yet.")