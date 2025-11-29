import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import csv
import requests
from io import BytesIO
import random

# --- SAFELY IMPORT SCIENTIFIC LIBS ---
try:
    import xarray as xr
    import cfgrib
except ImportError:
    xr = None
    st.error("⚠️ Scientific Libraries (xarray/cfgrib) Missing! Check requirements.txt")

# --- CONFIGURATION ---
DEFAULT_API_KEY = "" 
LOG_FILE = "mission_logs.csv"

# FILE PATHS (Must match what you uploaded to GitHub)
SAT_FILE = "W_XX-EUMETSAT-Darmstadt,OCA+MSG4+SEVIRI_C_EUMG_20190831234500_1_OR_FES_E0000_0100.nc"
ERA5_FILE = "ce636265319242f2fef4a83020b30ecf.grib"

st.set_page_config(page_title="VisionRain | Scientific Core", layout="wide", page_icon="⛈️")

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    .stMetric {
        background-color: #1a1a1a; 
        border: 1px solid #333; 
        border-radius: 12px; 
        padding: 15px;
    }
    .pitch-box {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #00e5ff;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: rgba(0, 255, 128, 0.1); 
        border: 1px solid #00ff80; 
        color: #00ff80; 
        padding: 15px; 
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADERS ---
@st.cache_resource
def load_data_files():
    """Loads both Satellite and ERA5 files"""
    ds_sat, ds_era = None, None
    
    if xr:
        # Load Satellite (.nc)
        if os.path.exists(SAT_FILE):
            try:
                ds_sat = xr.open_dataset(SAT_FILE, engine='netcdf4')
            except Exception as e:
                st.error(f"Error reading Satellite NC: {e}")
        
        # Load ERA5 (.grib)
        if os.path.exists(ERA5_FILE):
            try:
                ds_era = xr.open_dataset(ERA5_FILE, engine='cfgrib')
            except Exception as e:
                st.warning(f"Error reading ERA5 Grib: {e}")
    
    return ds_sat, ds_era

# --- 2. PLOT GENERATORS ---
def generate_satellite_plots(ds):
    """Plots Meteosat Cloud Pressure & Probability"""
    if ds is None: return None, 0, 0, 0, 0
    
    try:
        # Zoom Sector (Jeddah Area approx)
        # Adjust these indices if your file covers a different crop
        y_slice = slice(2200, 2400)
        x_slice = slice(700, 900)
        
        # Extract & Mask Data
        # Pressure (hPa)
        press = ds['cloud_top_pressure'].isel(y=y_slice, x=x_slice).values / 100.0
        press = np.where(press < 1100, press, np.nan)
        
        # Probability (0-100)
        prob = ds['cloud_probability'].isel(y=y_slice, x=x_slice).values
        prob = np.where(prob <= 100, prob, np.nan)
        
        # Radius (Microns) - Handle if missing
        if 'cloud_particle_effective_radius' in ds:
            rad = ds['cloud_particle_effective_radius'].isel(y=y_slice, x=x_slice).values * 1e6
            rad = np.where(rad < 100, rad, np.nan)
        else:
            rad = np.zeros_like(press)
            
        # Optical Depth - Handle if missing
        if 'cloud_optical_thickness' in ds:
            cot = ds['cloud_optical_thickness'].isel(y=y_slice, x=x_slice).values
        else:
            cot = np.zeros_like(press)

        # Metrics
        avg_press = np.nanmean(press)
        avg_prob = np.nanmean(prob)
        avg_rad = np.nanmean(rad)
        avg_cot = np.nanmean(cot)
        
        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0e1117')
        
        # Plot 1: Pressure (Height)
        im1 = ax[0].imshow(press, cmap='gray_r')
        ax[0].set_title("Cloud Top Pressure (Height)", color="white")
        ax[0].axis('off')
        plt.colorbar(im1, ax=ax[0], label="hPa").ax.yaxis.set_tick_params(color='white')
        
        # Plot 2: Probability (AI Confidence)
        im2 = ax[1].imshow(prob, cmap='Blues', vmin=0, vmax=100)
        ax[1].set_title("AI Cloud Probability", color="white")
        ax[1].axis('off')
        plt.colorbar(im2, ax=ax[1], label="%").ax.yaxis.set_tick_params(color='white')
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor='#0e1117')
        buf.seek(0)
        return Image.open(buf), avg_press, avg_prob, avg_rad, avg_cot

    except Exception as e:
        st.error(f"Plotting Error: {e}")
        return None, 0, 0, 0, 0

def generate_era5_plot(ds):
    """Plots ERA5 Atmospheric Physics"""
    if ds is None: return None, 0
    
    try:
        # Find variables (ERA5 names vary)
        # Looking for Liquid Water (clwc) or Temperature (t)
        var = 'clwc' if 'clwc' in ds else 't' if 't' in ds else list(ds.data_vars)[0]
        data = ds[var].values
        
        # Flatten dimensions
        while data.ndim > 2: data = data[0]
        
        avg_val = float(np.nanmean(data))
        
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0e1117')
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(f"ERA5 Physics: {var.upper()}", color="white")
        ax.axis('off')
        plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color='white')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor='#0e1117')
        buf.seek(0)
        return Image.open(buf), avg_val
    except: return None, 0

# --- 3. LOGGING ---
def log_mission(decision, metrics):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f: f.write("Timestamp,Decision,Metrics\n")
    with open(LOG_FILE, 'a') as f:
        f.write(f"{ts},{decision},{metrics}\n")

def load_logs():
    if os.path.exists(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=90)
    st.title("VisionRain")
    st.caption("Scientific Core | v8.0")
    
    api_key = st.text_input("Google AI Key", value=DEFAULT_API_KEY, type="password")
    
    st.markdown("### 📍 Mission Target")
    st.info("Locked: **Jeddah (File Data)**")
    
    with st.expander("🔒 Admin Portal"):
        if st.text_input("Password", type="password") == "123456":
            st.dataframe(load_logs())

# --- MAIN UI ---
st.title("VisionRain Command Center")
st.markdown("### *Data Source: EUMETSAT NetCDF + ERA5 GRIB*")

tab1, tab2, tab3 = st.tabs(["🌍 Strategic Vision", "📡 Sensor Array", "🧠 Gemini Fusion"])

# --- TAB 1: PITCH ---
with tab1:
    st.header("Strategic Framework")
    st.markdown("""
    <div class="pitch-box">
    <h3>🚨 1. Problem Statement</h3>
    <p>Globally, regions such as <b>Saudi Arabia</b> face escalating environmental crises: water scarcity, prolonged droughts, and wildfire escalation. 
    Current cloud seeding operations are <b>manual, expensive ($8k/hr), and reactive</b>.</p>
    <p>This aligns critically with <b>Saudi Vision 2030</b> and the <b>Saudi Green Initiative</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Solution:** VisionRain - An AI-driven decision support platform analyzing satellite microphysics for precision seeding.")
    with c2:
        st.success("**Impact:** Enables low-cost, safer deployment and scales globally to support emergency climate-response.")

# --- DATA PROCESSING ---
ds_sat, ds_era = load_data_files()

# Generate Plots
sat_img, press, prob, radius, cot = generate_satellite_plots(ds_sat)
era_img, era_val = generate_era5_plot(ds_era)

# --- TAB 2: SENSORS ---
with tab2:
    st.header("Real-Time Hydro-Meteorological Fusion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("A. Meteosat (Optical/IR)")
        if sat_img: st.image(sat_img, caption="EUMETSAT OCA Analysis", use_column_width=True)
        else: st.warning("Satellite File Not Loaded")
        
    with col2:
        st.subheader("B. ERA5 (Atmospheric)")
        if era_img: st.image(era_img, caption="ERA5 Physics Model", use_column_width=True)
        else: st.warning("ERA5 File Not Loaded")

    st.divider()
    
    # TELEMETRY
    st.subheader("Microphysical Telemetry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cloud Probability", f"{prob:.1f}%", "AI Confidence")
    c2.metric("Cloud Top Pressure", f"{press:.0f} hPa", "Altitude")
    c3.metric("Effective Radius", f"{radius:.1f} µm", "Target < 14µm")
    c4.metric("Optical Depth", f"{cot:.1f}", "Density")

# --- TAB 3: GEMINI FUSION ---
with tab3:
    st.header("Gemini Fusion Engine")
    
    st.markdown("### 🔬 Physics-Informed Logic (The Master Table)")
    table_data = {
        "Parameter": ["Effective Radius", "Cloud Probability", "Top Pressure", "Optical Depth"],
        "Seedable Sweet Spot": ["< 14 µm", "> 70%", "400-700 hPa", "> 10"],
        "Current Reading": [f"{radius:.1f} µm", f"{prob:.1f}%", f"{press:.0f} hPa", f"{cot:.1f}"]
    }
    st.table(pd.DataFrame(table_data))

    # Visual Evidence for AI
    st.caption("Visual Evidence Sent to Vertex AI:")
    if sat_img and era_img:
        c1, c2 = st.columns(2)
        c1.image(sat_img, caption="Satellite")
        c2.image(era_img, caption="ERA5")

    st.divider()

    if st.button("RUN STRATEGIC ANALYSIS", type="primary"):
        if not api_key:
            st.error("🔑 Google API Key Missing!")
        elif not sat_img:
            st.error("⚠️ No Data.")
        else:
            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                ACT AS A LEAD METEOROLOGIST. Analyze this Multi-Modal Scientific Data.
                
                --- MISSION CONTEXT ---
                Location: Jeddah Sector
                Objective: Hygroscopic Cloud Seeding.
                
                --- MICROPHYSICS DATA ---
                - Cloud Probability: {prob:.1f}% (0-100 scale)
                - Cloud Top Pressure: {press:.0f} hPa
                - Effective Radius: {radius:.1f} microns
                - Optical Depth: {cot:.1f}
                - ERA5 Liquid Water Factor: {era_val:.2e}
                
                --- LOGIC RULES (Physics) ---
                1. IF Radius < 14 microns AND Optical Depth > 10 -> "GO" (Stuck cloud, needs nuclei).
                2. IF Radius > 15 microns -> "NO-GO" (Already raining).
                3. IF Pressure > 800hPa (Too Low) OR < 300hPa (Too High) -> "NO-GO".
                
                --- OUTPUT ---
                1. **Visual Analysis:** Describe the cloud structure in the images.
                2. **Physics Check:** Evaluate the Radius and Optical Depth against the Sweet Spot.
                3. **Decision:** **GO** or **NO-GO**?
                4. **Reasoning:** Scientific justification.
                """
                
                with st.spinner("Vertex AI is calculating microphysics..."):
                    inputs = [prompt, sat_img]
                    if era_img: inputs.append(era_img)
                    
                    res = model.generate_content(inputs)
                    
                    decision = "GO" if "GO" in res.text.upper() else "NO-GO"
                    log_mission(f"Jeddah", f"R:{radius} OD:{cot}", decision, "AI Authorized")
                    
                    st.markdown("### 🛰️ Mission Command Report")
                    st.write(res.text)
                    
                    if decision == "GO":
                        st.balloons()
                        st.markdown("<div class='success-box'>✅ MISSION APPROVED</div>", unsafe_allow_html=True)
                    else:
                        st.error("⛔ MISSION ABORTED")

            except Exception as e:
                st.error(f"AI Error: {e}")
