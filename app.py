import streamlit as st
import google.generativeai as genai
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import PIL.Image
from io import BytesIO
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Enterprise Core", layout="wide", page_icon="⛈️")

# --- 2. LOAD SATELLITE DATA (ROBUST) ---
@st.cache_resource
def load_satellite_data():
    # 👇👇👇 MAKE SURE THIS MATCHES YOUR UPLOADED FILE NAME EXACTLY 👇👇👇
    file_name = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
    
    # Dynamic Pathing (Works on Cloud and Local)
    file_path = os.path.join(os.getcwd(), file_name)
    
    try:
        # Using 'h5netcdf' engine (Fastest for Streamlit Cloud)
        ds = xr.open_dataset(file_path, engine='h5netcdf')
        return ds
    except FileNotFoundError:
        st.error(f"🚨 FILE MISSING: The app could not find '{file_name}' in {os.getcwd()}")
        return None
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

ds = load_satellite_data()

# --- 3. SCIENTIFIC PROCESSING ---
def get_pixel_data(city_name, ds):
    if ds is None: return None

    # COORDINATE SYSTEM
    # We map targets to specific pixels in the historical file
    coords = {
        "Jeddah": (2589, 668),
        "Riyadh": (2500, 700),
        "Target Sector X (Demo)": (2300, 750) # <-- THE STORM
    }
    
    y, x = coords.get(city_name, (2589, 668))
    # Fake lat/lon for display purposes since we are using pixel coordinates
    lat_disp = 21.54 if city_name == "Jeddah" else 16.5 

    try:
        # Extract Raw Data
        data = ds.isel(number_of_lines=y, number_of_pixels=x)
        
        # A. Phase Discrimination (via Pressure)
        # Pressure < 45000 Pa (450 hPa) = High Ice Cloud
        # Pressure > 60000 Pa (600 hPa) = Low Liquid Cloud
        pres = float(data.get('cloud_top_pressure', 101300).values)
        if pres > 60000: phase = "LIQUID (Warm)"
        elif pres < 45000: phase = "ICE (Glaciated)"
        else: phase = "MIXED"

        # B. Droplet Radius (The "Stuck" Factor)
        # Convert Meters to Microns (x 1,000,000)
        rad_val = float(data.get('cloud_particle_effective_radius', 0).values)
        radius = rad_val * 1e6 if not np.isnan(rad_val) else 0.0

        # C. Optical Thickness
        cod_log = float(data.get('cloud_optical_depth_log', 0).values)
        cod = 10**cod_log if not np.isnan(cod_log) else 0.0

        return {
            "lat": lat_disp, 
            "phase": phase, 
            "radius": round(radius, 1),
            "optical_depth": round(cod, 1), 
            "pressure": round(pres/100, 0),
            "y": y, "x": x
        }
    except Exception as e:
        st.error(f"Extraction Failed: {e}")
        return None

def generate_thermal_image(ds, y, x):
    """Generates a synthetic Thermal Image from pressure data for the AI."""
    window = 80 # Zoom level
    try:
        # Slice the dataset
        subset = ds['cloud_top_pressure'].isel(
            number_of_lines=slice(y-window, y+window), 
            number_of_pixels=slice(x-window, x+window)
        )
        
        # Plot with Matplotlib
        fig, ax = plt.subplots(figsize=(4, 4))
        # Gray_r means Low Pressure (Clouds) = White, High Pressure (Ground) = Black
        ax.imshow(subset, cmap='gray_r') 
        ax.axis('off')
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return buf
    except:
        return None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2011/2011601.png", width=80)
    st.title("VisionRain")
    st.caption("Google Cloud x EUMETSAT")
    
    # API KEY INPUT
    api_key = st.text_input("Gemini API Key", type="password", help="Paste your AI Studio key here")
    
    st.markdown("---")
    st.markdown("### 🎯 Target Sector")
    target = st.selectbox("Select Region", ["Jeddah", "Riyadh", "Target Sector X (Demo)"])
    
    if st.button("🛰️ ACQUIRE SATELLITE LOCK"):
        st.session_state['target'] = target
        st.session_state['data'] = get_pixel_data(target, ds)
        st.rerun()

# --- 5. MAIN UI ---
st.title("📡 VisionRain: Ionization Command Center")
st.markdown(f"### *Mission Status: ONLINE | Target: {st.session_state.get('target', 'None')}*")

tab1, tab2, tab3 = st.tabs(["📊 Microphysics Telemetry", "🧠 Gemini AI Core", "📝 Admin Logs"])

# --- TAB 1: TELEMETRY ---
with tab1:
    if 'data' in st.session_state and st.session_state['data']:
        d = st.session_state['data']
        
        # 1. METRICS ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cloud Phase", d['phase'], help="Liquid is required for ionization")
        c2.metric("Droplet Radius", f"{d['radius']} µm", help="<14µm means droplets are stuck")
        c3.metric("Optical Depth", d['optical_depth'])
        c4.metric("Top Pressure", f"{d['pressure']} hPa")
        
        st.divider()
        
        # 2. VISUALS & LOGIC
        c_img, c_logic = st.columns([1, 1.5])
        
        with c_img:
            st.markdown("#### 🔭 Satellite Thermal Feed")
            img_buf = generate_thermal_image(ds, d['y'], d['x'])
            if img_buf:
                st.image(img_buf, caption="Meteosat-9 Infrared Analysis", use_column_width=True)
                st.session_state['ai_image'] = img_buf # Save for AI tab
            else:
                st.warning("Imaging System Offline")
                
        with c_logic:
            st.markdown("#### ✅ Seeding Viability Check")
            
            # Auto-Logic
            phase_ok = "LIQUID" in d['phase']
            rad_ok = 0 < d['radius'] < 14
            
            logic_data = {
                "Parameter": ["Cloud Phase", "Droplet Size"],
                "Requirement": ["Liquid / Mixed", "< 14 Microns"],
                "Actual": [d['phase'], f"{d['radius']} µm"],
                "Status": ["✅ PASS" if phase_ok else "❌ FAIL", "✅ PASS" if rad_ok else "❌ FAIL"]
            }
            st.table(pd.DataFrame(logic_data))

    else:
        st.info("👈 Select 'Target Sector X (Demo)' in the sidebar and click ACQUIRE.")

# --- TAB 2: GEMINI AI ---
with tab2:
    st.header("Gemini 1.5 Pro Decision Engine")
    st.caption("Fuses Visual Thermal Data with Numerical Microphysics")
    
    if 'data' in st.session_state and 'ai_image' in st.session_state:
        d = st.session_state['data']
        
        if st.button("🚀 REQUEST LAUNCH AUTHORIZATION", type="primary"):
            if not api_key:
                st.error("⚠️ API Key Missing! Please enter it in the sidebar.")
            else:
                try:
                    # CONFIG
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # LOAD IMAGE
                    img = PIL.Image.open(st.session_state['ai_image'])
                    
                    # PROMPT
                    prompt = f"""
                    ACT AS A LEAD METEOROLOGIST.
                    Analyze this Satellite Thermal Image and Telemetry for Cloud Seeding.
                    
                    --- TELEMETRY ---
                    Target: {st.session_state['target']}
                    Cloud Phase: {d['phase']} (Must be Liquid/Mixed)
                    Droplet Radius: {d['radius']} microns (Must be < 14 microns)
                    
                    --- MISSION RULES ---
                    1. **Ionization** works on WARM LIQUID clouds with SMALL droplets (stalled coalescence).
                    2. It does NOT work on ICE clouds (Glaciated) or clouds that are already raining (Large droplets).
                    
                    --- OUTPUT ---
                    1. Analyze the visual structure (Bright areas = cold clouds).
                    2. Apply the Mission Rules to the Telemetry.
                    3. FINAL COMMAND: "ACTIVATE IONIZERS" or "STANDBY".
                    """
                    
                    with st.spinner("Gemini is analyzing atmospheric conditions..."):
                        res = model.generate_content([prompt, img])
                        
                        st.markdown("### 🛰️ Mission Report")
                        st.markdown(res.text)
                        
                        if "ACTIVATE" in res.text:
                            st.balloons()
                            st.success("✅ MISSION APPROVED: Ionization Emitters Activated")
                        else:
                            st.error("⛔ MISSION ABORTED: Conditions Unsuitable")
                            
                except Exception as e:
                    st.error(f"AI Error: {e}")
                    st.info("Check your API Key and internet connection.")
    else:
        st.warning("Waiting for Satellite Data...")

# --- TAB 3: ADMIN LOGS ---
with tab3:
    st.header("🗄️ BigQuery Audit Logs")
    st.markdown("Simulation of Google BigQuery Data Stream")
    
    if 'data' in st.session_state:
        # Simulate a log entry
        log_entry = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Sector": st.session_state['target'],
            "Phase": st.session_state['data']['phase'],
            "Radius": st.session_state['data']['radius'],
            "Action": "SCANNED"
        }
        st.dataframe(pd.DataFrame([log_entry]))
    else:
        st.text("No active logs.")
