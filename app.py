import streamlit as st
import google.generativeai as genai
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from io import BytesIO

# --- CONFIGURATION ---
# 1. SETUP PAGE
st.set_page_config(page_title="VisionRain | Cloud Command", layout="wide", page_icon="⛈️")

# 2. LOAD THE SATELLITE DATA (The "Secret" Historical File)
# We cache this so it doesn't reload every time you click a button
@st.cache_resource
def load_satellite_data():
    # REPLACE THIS WITH YOUR ACTUAL .NC FILENAME
    file_path = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc" 
    try:
        ds = xr.open_dataset(file_path)
        return ds
    except FileNotFoundError:
        st.error(f"🚨 CRITICAL ERROR: Satellite Data File '{file_path}' not found. Please upload the .nc file.")
        return None

ds = load_satellite_data()

# --- STYLING (Cyberpunk/Command Center Look) ---
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
    .status-go {color: #00FF00; font-weight: bold;}
    .status-nogo {color: #FF0000; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_pixel_data(city_name, ds):
    """
    Finds the closest pixel in the satellite file for the city.
    Returns the raw scientific values.
    """
    # Hardcoded coords for the demo (You can add more)
    locations = {
        "Jeddah": (21.54, 39.17),
        "Riyadh": (24.71, 46.67),
        "Dammam": (26.42, 50.08),
        "Abha": (18.22, 42.51), # Good for rain
        "Target Sector X (Demo)": (16.5, 42.5) # The "Secret" Storm location we found earlier
    }
    
    lat, lon = locations.get(city_name, (21.54, 39.17)) # Default to Jeddah
    
    # Locate pixel (Using the logic we built previously)
    # Note: In a real app, we'd use the proper lat/lon arrays. 
    # For this demo, we mock the pixel lookup based on our previous findings or fallback.
    
    if city_name == "Target Sector X (Demo)":
        # Force the storm pixel we found
        y, x = 2300, 750
    elif city_name == "Jeddah":
        y, x = 2589, 668
    else:
        # Random offset for other cities just to vary the data for the demo
        y, x = 2500, 700 

    try:
        data = ds.isel(number_of_lines=y, number_of_pixels=x)
        
        # EXTRACT SCIENTIFIC VARIABLES
        # 1. Phase (0=Clear, 1=Liquid, 2=Ice) - Using Pressure proxy if Phase missing
        pressure = float(data['cloud_top_pressure'].values)
        if pressure > 60000: phase = "LIQUID (Warm)"
        elif pressure < 45000: phase = "ICE (Glaciated)"
        else: phase = "MIXED"
        
        # 2. Radius
        rad_m = float(data['cloud_particle_effective_radius'].values)
        rad_um = rad_m * 1e6 if not np.isnan(rad_m) else 0.0
        
        # 3. Optical Depth
        cod_log = float(data['cloud_optical_depth_log'].values)
        cod = 10**cod_log if not np.isnan(cod_log) else 0.0
        
        # 4. Cloud Probability
        prob = float(data['cloud_probability'].values)

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
    
    # Use Cloud Top Pressure as visual proxy for IR
    img_data = ds['cloud_top_pressure'].isel(number_of_lines=y_slice, number_of_pixels=x_slice).values
    
    # Normalize for display (invert so clouds are white)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_data, cmap='gray_r') 
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
    st.session_state['log_df'] = pd.concat([st.session_state['log_df'], pd.DataFrame([new_row])], ignore_index=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2011/2011601.png", width=80)
    st.title("VisionRain Hub")
    st.caption("Google Cloud x KFUPM | Prototype v3")
    
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### 🎯 Target Selection")
    target_city = st.selectbox("Select Region", ["Jeddah", "Riyadh", "Abha", "Target Sector X (Demo)"])
    
    if st.button("🛰️ ACQUIRE TARGET"):
        st.session_state['target_city'] = target_city
        st.session_state['scan_data'] = get_pixel_data(target_city, ds)
        st.rerun()

# --- MAIN UI ---
st.title("📡 VisionRain: Ionization Command Center")

# TABS
tab_pitch, tab_data, tab_ai, tab_admin = st.tabs(["💡 Initiative", "📊 Satellite Telemetry", "🤖 Gemini Analysis", "📝 Admin Logs"])

# --- TAB 1: INITIATIVE (Your Pitch) ---
with tab_data:
    if 'scan_data' in st.session_state and st.session_state['scan_data']:
        data = st.session_state['scan_data']
        
        st.markdown(f"### 📍 Live Feed: {st.session_state['target_city']}")
        st.caption(f"Coordinates: {data['lat']}N, {data['lon']}E | Source: Meteosat-9 (IODC)")
        
        # 1. THE NUMERICAL DASHBOARD
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
        
        # 2. THE VISUALS & COMPARISON TABLE
        col_img, col_table = st.columns([1, 1.5])
        
        with col_img:
            st.markdown("#### 🔭 Satellite Thermal View")
            img_buf = generate_satellite_image(ds, data['y_idx'], data['x_idx'])
            st.image(img_buf, caption="Processed Cloud Top Pressure (Dark=Ground, White=Cloud)", use_column_width=True)
            st.session_state['current_image'] = img_buf # Save for AI
            
        with col_table:
            st.markdown("#### 📋 Seeding Viability Check")
            
            # Logic for status
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

        # Log to BigQuery automatically
        log_to_bigquery(data)

    else:
        st.info("👈 Please select a target city and click 'ACQUIRE TARGET' in the sidebar.")

# --- TAB 3: GEMINI ANALYSIS ---
with tab_ai:
    st.header("Gemini 2.0 Decision Engine")
    
    if 'scan_data' in st.session_state:
        data = st.session_state['scan_data']
        
        st.markdown("The AI will analyze the satellite image and the numerical microphysics data to authorize ionization.")
        
        if st.button("🚀 REQUEST LAUNCH AUTHORIZATION"):
            if not api_key:
                st.error("Please enter Gemini API Key in Sidebar")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Retrieve Image
                img = st.session_state['current_image']
                img_pil = plt.imread(img) # Convert back to format AI can read if needed, or re-open
                # Re-generate simple PIL for Gemini
                import PIL.Image
                img_pil = PIL.Image.open(img)

                # THE SUPERPROMPT
                prompt = f"""
                You are the Mission Control AI for the Saudi Cloud Seeding Initiative.
                Analyze the following Satellite Telemetry and Image.

                --- TELEMETRY DATA ---
                Location: {st.session_state['target_city']}
                Cloud Phase: {data['phase']} (Must be Liquid/Mixed)
                Effective Radius: {data['radius']} microns (Must be < 14 for seeding)
                Optical Thickness: {data['optical_depth']}
                
                --- INSTRUCTIONS ---
                1. Analyze the visual satellite image. Does it show convective activity (bright white spots)?
                2. Evaluate the Telemetry. Is the cloud "stuck" (small droplets) and containing liquid water?
                3. PROVIDE A FINAL DECISION: "ACTIVATE IONIZERS" or "STANDBY".
                4. Explain your scientific reasoning briefly.
                """
                
                with st.spinner("Analyzing microphysics..."):
                    response = model.generate_content([prompt, img_pil])
                    st.markdown(response.text)
                    
                    if "ACTIVATE" in response.text:
                        st.success("SYSTEM ALERT: IONIZATION EMITTERS ACTIVE")
                    else:
                        st.warning("SYSTEM STANDBY: Conditions not met.")

# --- TAB 4: ADMIN LOGS (BigQuery Sim) ---
with tab_admin:
    st.header("🗄️ BigQuery Transaction Logs")
    st.markdown("All satellite scans are logged for audit and analysis.")
    if 'log_df' in st.session_state:
        st.dataframe(st.session_state['log_df'], use_container_width=True)
    else:
        st.text("No logs yet.")

# --- TAB 1: PITCH (Kept Original) ---
with tab_pitch:
    st.markdown("## 🇸🇦 VisionRain: The Future of Water Security")
    st.markdown("""
    **VisionRain** leverages Google Cloud + Satellite AI to revolutionize cloud seeding.
    Instead of flying blind, we use **Microphysical Data** to target clouds that are physically ready to rain but need a 'nudge'.
    """)
    

[Image of cloud seeding diagram]
