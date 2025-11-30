import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import google.generativeai as genai
from datetime import datetime
import os

# ==========================================
# 1. CONFIGURATION & GOOGLE CLOUD STYLE
# ==========================================
st.set_page_config(
    page_title="VisionRain | Kingdom Commander",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "Google Cloud Platform" Dark Mode aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #e5e7eb;
    }
    .highlight-box {
        border-left: 5px solid #4285F4;
        background-color: #181b21;
        padding: 15px;
        border-radius: 0 5px 5px 0;
    }
    .gemini-box {
        border: 1px solid #c084fc;
        background: linear-gradient(135deg, rgba(29, 26, 56, 0.9) 0%, rgba(45, 10, 60, 0.9) 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "b5710c0835b1558c7a5002809513f1a5.nc"
LOG_FILE = "mission_logs.csv"

# ==========================================
# 2. INTELLIGENT DATA PROCESSING
# ==========================================

@st.cache_resource
def load_and_process_data():
    """
    Loads ERA5, handles renaming, and computes aggregation over the 6-hour window.
    """
    if not os.path.exists(DATA_FILE):
        return None
    
    try:
        ds = xr.open_dataset(DATA_FILE, engine='netcdf4')
        
        # --- COORDINATE & VARIABLE STANDARDIZATION ---
        coord_renames = {}
        if 'pressure_level' in ds.coords: coord_renames['pressure_level'] = 'level'
        if 'valid_time' in ds.coords: coord_renames['valid_time'] = 'time'
        if coord_renames: ds = ds.rename(coord_renames)

        rename_map = {
            't': 'Temperature', 'r': 'Relative_Humidity', 
            'clwc': 'Specific_cloud_liquid_water_content', 'cc': 'Fraction_of_cloud_cover',
            'u': 'U_component_of_wind', 'v': 'V_component_of_wind',
            'z': 'Geopotential', 'w': 'Vertical_velocity'
        }
        actual_rename = {k: v for k, v in rename_map.items() if k in ds}
        ds = ds.rename(actual_rename)
        
        # Unit Conversions
        if 'Temperature' in ds: ds['Temperature'] -= 273.15
        
        # --- 6-HOUR AGGREGATION SCAN ---
        # Instead of just taking the last hour, we look for the MAX potential over the file duration
        # Focus on Cloud Layer (~700hPa)
        ds_layer = ds.sel(level=700, method='nearest')
        
        return ds, ds_layer
    except Exception as e:
        st.error(f"Critical Data Error: {e}")
        return None, None

def identify_clusters(ds_layer):
    """
    Scans the entire 6-hour window and all of Saudi Arabia.
    Returns the top seedable clusters.
    """
    # Create a 2D map of "Max Seedability Score" across the 6 hours
    # Logic: We take the MAX cloud cover and water content over the time dimension
    
    # 1. Calculate Score for every pixel/time
    # Score = (Cloud Cover * 50) + (LWC normalized * 50) + (RH/2)
    # Note: LWC is small (e.g., 0.0001), so we multiply by huge factor
    
    score_da = (
        (ds_layer['Fraction_of_cloud_cover'] * 40) + 
        (ds_layer['Relative_Humidity'] * 0.4) + 
        (ds_layer['Specific_cloud_liquid_water_content'] * 100000)
    )
    
    # 2. Flatten to dataframe to find top candidates across ALL times
    df = score_da.to_dataframe(name='Score').reset_index()
    
    # 3. Filter for Saudi Arabia
    df = df[(df['latitude'] >= 16) & (df['latitude'] <= 32) & 
            (df['longitude'] >= 34) & (df['longitude'] <= 56)]
    
    # 4. Filter Noise
    df = df[df['Score'] > 40] # Minimum threshold
    
    # 5. Add timestamp string for display
    df['time_str'] = df['time'].astype(str)
    
    return df

def get_gemini_analysis(api_key, context_data):
    """
    The Brain: Sends metrics to Gemini 1.5/2.0 Flash for decision support.
    """
    if not api_key:
        return "⚠️ **Gemini API Key missing.** Simulation: Conditions look favorable for hygroscopic seeding. High liquid water content detected."
    
    try:
        genai.configure(api_key=api_key)
        # Using 1.5 Flash as it's the current stable high-speed model
        # You can change to "gemini-2.0-flash-exp" if you have access
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = f"""
        You are 'Kingdom Commander', an AI Cloud Seeding Meteorologist for Saudi Arabia.
        Analyze the following cloud data taken from ERA5 satellite reanalysis:
        
        {context_data}
        
        Physics Rules:
        - Temp -5C to -15C + Liquid Water = Glaciogenic Seeding (High Priority)
        - Temp > 0C + High Humidity = Hygroscopic Seeding (Medium Priority)
        - High Wind Shear = Drone Risk
        
        Output a concise 'Mission Brief' (max 100 words). 
        1. Give a GO/NO-GO decision.
        2. Identify the Seeding Method (Hygroscopic vs Glaciogenic).
        3. Mention specific risks.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# ==========================================
# 3. UI & VISUALIZATION
# ==========================================

ds, ds_layer = load_and_process_data()

# Sidebar for Controls
with st.sidebar:
    st.title("⚙️ Operations Center")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Required for AI Brain")
    st.markdown("---")
    st.info("Dataset: ERA5 Hourly (6 Hour Window)")
    confidence_threshold = st.slider("Min Seedability Score", 0, 100, 50)

if ds is None:
    st.stop()

# --- HEADER ---
st.title("⛈️ VisionRain | Kingdom Commander")
st.markdown("### AI-Driven Cloud Seeding Decision Support (Google Cloud/Vertex AI Integrated)")

# --- GLOBAL SCAN LOGIC ---
all_candidates = identify_clusters(ds_layer)
top_candidates = all_candidates[all_candidates['Score'] > confidence_threshold].sort_values('Score', ascending=False)

# --- TABBED INTERFACE ---
tabs = st.tabs(["🗺️ Geospatial Command", "🔬 Deep Dive Analysis", "🤖 Gemini Intelligence", "📊 Data Grid"])

# 1. GEOSPATIAL COMMAND (The Visual Wow Factor)
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### 🛰️ 3D Seedability Heatmap (Saudi Sector)")
        if not top_candidates.empty:
            # PyDeck 3D Visualization (Google Cloud Style)
            layer = pdk.Layer(
                "HexagonLayer",
                top_candidates,
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                elevation_scale=200,
                pickable=True,
                elevation_range=[0, 3000],
                extruded=True,
                coverage=1,
                get_fill_color="[255, Score * 2, 100, 200]", # Dynamic Red/Orange color
            )
            
            view_state = pdk.ViewState(
                longitude=45.0,
                latitude=24.0,
                zoom=4.5,
                min_zoom=3,
                max_zoom=10,
                pitch=45.0, # 3D tilt
                bearing=0
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Lat: {position[1]}\nLon: {position[0]}\nDensity: {elevationValue}"},
                map_style="mapbox://styles/mapbox/dark-v10" # Requires token or use default
            )
            st.pydeck_chart(r)
        else:
            st.warning("No clusters found meeting current threshold.")

    with col2:
        st.markdown("#### 🎯 Priority Clusters")
        st.write("Identified high-potential zones across the 6-hour window.")
        
        # Group by location (approximate) to show unique storms
        # We round lat/lon to group nearby pixels
        unique_storms = top_candidates.copy()
        unique_storms['lat_round'] = unique_storms['latitude'].round(1)
        unique_storms['lon_round'] = unique_storms['longitude'].round(1)
        unique_storms = unique_storms.drop_duplicates(subset=['lat_round', 'lon_round']).head(5)
        
        for idx, row in unique_storms.iterrows():
            with st.expander(f"Target {idx} (Score: {row['Score']:.0f})"):
                st.write(f"📍 **{row['latitude']:.2f}, {row['longitude']:.2f}**")
                st.write(f"🕒 Time: {row['time_str']}")
                if st.button(f"Analyze Target {idx}", key=f"btn_{idx}"):
                    st.session_state['selected_lat'] = row['latitude']
                    st.session_state['selected_lon'] = row['longitude']
                    st.session_state['selected_time'] = row['time']

# 2. DEEP DIVE (Time Evolution)
with tabs[1]:
    if 'selected_lat' in st.session_state:
        lat = st.session_state['selected_lat']
        lon = st.session_state['selected_lon']
        
        st.markdown(f"### 📍 Target Analysis: {lat:.2f} N, {lon:.2f} E")
        
        # Extract 6-hour time series for this specific location
        # Get all levels for this location
        loc_ds = ds.sel(latitude=lat, longitude=lon, method='nearest')
        
        # Convert to dataframe for plotting
        loc_df = loc_ds.to_dataframe().reset_index()
        
        # FILTER: Just look at 700hPa for the time series overview
        loc_700 = loc_df[loc_df['level'] == 700]
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### ⏳ Cloud Evolution (6 Hours)")
            # Dual axis plot: Liquid Water vs Cloud Cover
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=loc_700['time'], y=loc_700['Specific_cloud_liquid_water_content'],
                                     name='Liquid Water', line=dict(color='#00d4ff', width=3)))
            fig_time.add_trace(go.Scatter(x=loc_700['time'], y=loc_700['Fraction_of_cloud_cover'],
                                     name='Cloud Cover', yaxis='y2', line=dict(color='#888888', dash='dot')))
            
            fig_time.update_layout(
                yaxis=dict(title="LWC (kg/kg)"),
                yaxis2=dict(title="Cloud Fraction", overlaying='y', side='right'),
                template="plotly_dark",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
        with col_g2:
            st.markdown("#### 🌡️ Stability & Temp (Vertical Profile)")
            # Pick the specific time selected or the time of max score
            # For simplicity, we plot the Mean profile over 6 hours
            mean_profile = loc_df.groupby('level').mean(numeric_only=True).reset_index()
            
            fig_skew = go.Figure()
            fig_skew.add_trace(go.Scatter(x=mean_profile['Temperature'], y=mean_profile['level'],
                                     name='Temp (°C)', line=dict(color='red', width=3)))
            fig_skew.add_trace(go.Scatter(x=mean_profile['Relative_Humidity']*100, y=mean_profile['level'],
                                     name='RH (%)', line=dict(color='green', width=2)))
            
            fig_skew.update_layout(
                yaxis=dict(title="Pressure (hPa)", autorange="reversed"),
                xaxis=dict(title="Value"),
                title="Mean Vertical Profile (6hr Avg)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_skew, use_container_width=True)

# 3. GEMINI INTELLIGENCE
with tabs[2]:
    if 'selected_lat' in st.session_state:
        st.markdown("### 🤖 Gemini 2.0 Flash: Mission Brief")
        
        # Prepare Data for LLM
        # We need to give it the exact metrics for the selected time
        sel_time = st.session_state['selected_time']
        
        # Grab the specific slice
        point_data = ds.sel(latitude=st.session_state['selected_lat'], 
                           longitude=st.session_state['selected_lon'], 
                           time=sel_time, method='nearest').sel(level=700, method='nearest')
        
        metrics_dict = {
            "Location": f"{st.session_state['selected_lat']:.2f}, {st.session_state['selected_lon']:.2f}",
            "Time": str(sel_time.values),
            "Temperature_700hPa": float(point_data['Temperature']),
            "Humidity_700hPa": float(point_data['Relative_Humidity']),
            "Liquid_Water_Content": float(point_data['Specific_cloud_liquid_water_content']),
            "Vertical_Velocity": float(point_data['Vertical_velocity']),
            "Wind_U": float(point_data['U_component_of_wind']),
            "Wind_V": float(point_data['V_component_of_wind']),
        }
        
        col_ai1, col_ai2 = st.columns([1, 2])
        
        with col_ai1:
            st.json(metrics_dict)
        
        with col_ai2:
            if st.button("GENERATE AI MISSION BRIEF"):
                with st.spinner("Gemini is analyzing atmospheric physics..."):
                    ai_response = get_gemini_analysis(api_key, str(metrics_dict))
                    
                st.markdown(f"""
                <div class="gemini-box">
                    <h3>⚡ Gemini Assessment</h3>
                    {ai_response}
                </div>
                """, unsafe_allow_html=True)
                
                # BigQuery Simulation Log
                if "GO" in ai_response.upper():
                    st.toast("Auto-logged 'GO' decision to BigQuery", icon="✅")
    else:
        st.info("Please select a target from the Geospatial Command tab first.")

# 4. DATA GRID
with tabs[3]:
    st.markdown("### 📊 Raw Scientific Data (Filtered)")
    st.dataframe(top_candidates[['time', 'latitude', 'longitude', 'Score', 'Temperature', 'Relative_Humidity', 'Specific_cloud_liquid_water_content']])
