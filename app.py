import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import google.generativeai as genai
import os

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="VisionRain | Kingdom Commander",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Mode / Google Cloud Console Aesthetic
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 15px;}
    .big-header {font-size: 32px; font-weight: bold; color: #e5e7eb; margin-bottom: 20px;}
    .sub-header {font-size: 24px; font-weight: bold; color: #4285F4; margin-top: 20px;}
    .drone-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid #10b981; padding: 20px; border-radius: 10px; color: #d1fae5;
    }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "b5710c0835b1558c7a5002809513f1a5.nc"

# ==========================================
# 2. DATA ENGINE
# ==========================================

@st.cache_resource
def load_data():
    """Loads and standardizes the ERA5 dataset."""
    if not os.path.exists(DATA_FILE):
        return None
    
    try:
        ds = xr.open_dataset(DATA_FILE, engine='netcdf4')
        
        # Rename coords
        coord_renames = {}
        if 'pressure_level' in ds.coords: coord_renames['pressure_level'] = 'level'
        if 'valid_time' in ds.coords: coord_renames['valid_time'] = 'time'
        if coord_renames: ds = ds.rename(coord_renames)

        # Rename variables to your specific list
        rename_map = {
            't': 'Temperature', 'r': 'Relative_Humidity', 
            'clwc': 'Specific_cloud_liquid_water_content', 'cc': 'Fraction_of_cloud_cover',
            'u': 'U_component_of_wind', 'v': 'V_component_of_wind',
            'z': 'Geopotential', 'w': 'Vertical_velocity', 'd': 'Divergence', 'q': 'Specific_humidity'
        }
        actual_rename = {k: v for k, v in rename_map.items() if k in ds}
        ds = ds.rename(actual_rename)
        
        # Unit Conversions
        if 'Temperature' in ds: ds['Temperature'] -= 273.15 # K to C
        
        return ds
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

def process_time_slice(ds, time_index):
    """Extracts a specific hour and calculates Seedability Scores."""
    try:
        # Select specific time and 700hPa level (Cloud Center)
        ds_slice = ds.isel(time=time_index).sel(level=700, method='nearest')
        
        # Calculate Score
        # Logic: High Cloud Cover + High Humidity + High LWC = Good
        cc = ds_slice['Fraction_of_cloud_cover'] if 'Fraction_of_cloud_cover' in ds_slice else 0
        rh = ds_slice['Relative_Humidity'] if 'Relative_Humidity' in ds_slice else 0
        lwc = ds_slice['Specific_cloud_liquid_water_content'] if 'Specific_cloud_liquid_water_content' in ds_slice else 0
        
        # Heuristic Score (0-100)
        score = (cc * 50) + (rh * 0.5) + (lwc * 200000)
        
        # Convert to DataFrame
        df = ds_slice.to_dataframe().reset_index()
        df['Score'] = score.values.flatten() # Add score column
        
        # Filter for Saudi Arabia Region
        df = df[(df['latitude'] >= 16) & (df['latitude'] <= 32) & 
                (df['longitude'] >= 34) & (df['longitude'] <= 56)]
        
        # Drop NaN
        df = df.dropna(subset=['Score'])
        
        return df, ds_slice.time.values
    except Exception as e:
        st.error(f"Processing Error: {e}")
        return pd.DataFrame(), None

def get_gemini_brief(api_key, metrics, decision, formation):
    """Generates the AI Explanation."""
    if not api_key: return "⚠️ **AI Offline:** Simulation Mode. Cloud structure indicates high viability."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp') # Or gemini-1.5-flash
        
        prompt = f"""
        Act as the 'Kingdom Commander' AI System for Saudi Arabia Cloud Seeding.
        
        **Mission Status:** {decision}
        **Drone Formation:** {formation}
        
        **Atmospheric Data:**
        {metrics}
        
        1. Explain WHY this location is seedable (or not) based on the physics (LWC, Temp, Divergence).
        2. Justify the Drone Formation (Why {formation}?). 
           - Formation A (6 Drones) = High Cloud Spread/Divergence.
           - Formation B (4 Drones) = Medium.
           - Formation C (2 Drones) = Concentrated.
        3. Keep it scientific but actionable. Max 150 words.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ AI Connection Failed. Proceeding with manual protocols."

# ==========================================
# 3. UI ARCHITECTURE
# ==========================================

ds = load_data()
if ds is None: st.stop()

# Sidebar (Hidden by default, used for API Key)
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")

# TABS
tab1, tab2, tab3 = st.tabs(["🌍 Mission Control", "🎯 Target Acquisition", "🤖 AI & Drone Ops"])

# --- TAB 1: MISSION CONTROL (The Initiative) ---
with tab1:
    st.markdown('<div class="big-header">🇸🇦 VisionRain: The Initiative</div>', unsafe_allow_html=True)
    
    # 1. Problem & Solution
    col_text, col_stat = st.columns([2, 1])
    with col_text:
        st.markdown("""
        **The Challenge:** Saudi Arabia faces water scarcity and reactive, costly cloud seeding operations ($8,000/hr).
        **The Solution:** An AI-Driven, Pilotless Drone Swarm system aligned with **Vision 2030**.
        **The Goal:** Increase rainfall efficiency, reduce costs, and ensure water security.
        """)
    with col_stat:
        st.metric("Global Drought Impact", "1.4 Billion People")
        st.metric("Water Sustainability", "Critical Priority")

    st.divider()
    
    # 2. Grand Overview (Satellite View of Saudi)
    st.markdown('<div class="sub-header">🛰️ Kingdom-Wide Atmospheric Scan (Cloud Tops)</div>', unsafe_allow_html=True)
    
    # Get max cloud cover over the whole dataset for the map
    ds_max = ds.max(dim='time').sel(level=600, method='nearest') # 600hPa = Cloud Tops
    df_max = ds_max.to_dataframe().reset_index()
    df_max = df_max[(df_max['latitude'] >= 16) & (df_max['latitude'] <= 32) & (df_max['longitude'] >= 34) & (df_max['longitude'] <= 56)]
    
    # Simple Density Map
    st.map(df_max[df_max['Fraction_of_cloud_cover'] > 0.5], latitude='latitude', longitude='longitude', size=20, color='#4285F4')
    st.caption("Blue points indicate persistent cloud cover detected > 600hPa across the scanning window.")


# --- TAB 2: TARGET ACQUISITION (The Data) ---
with tab2:
    st.markdown('<div class="sub-header">⏳ Temporal Scan (Hour 1 - 6)</div>', unsafe_allow_html=True)
    
    # 1. Time Slider
    hour_idx = st.slider("Select Scan Hour (1 = Start, 6 = End)", 0, 5, 0)
    
    # 2. Process Data for this hour
    df_hour, timestamp = process_time_slice(ds, hour_idx)
    st.write(f"**Scanning Time:** {str(timestamp)}")
    
    # 3. Seedability Map (Highlighted Regions)
    st.markdown("#### 📍 Seedability Heatmap")
    
    # PyDeck Map for "Google Cloud" feel
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_hour[df_hour['Score'] > 20],
        get_position=["longitude", "latitude"],
        get_color="[255, Score * 2, 50, 160]",
        get_radius=15000,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=24.0, longitude=45.0, zoom=4.5)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Score: {Score}"})
    st.pydeck_chart(r)
    
    # 4. Regional Table
    st.markdown("#### 📋 Priority Targets (High Seedability)")
    # Sort and show top
    top_targets = df_hour.sort_values(by='Score', ascending=False).head(10)
    
    # Interactive Table
    st.dataframe(top_targets[['latitude', 'longitude', 'Score', 'Fraction_of_cloud_cover', 'Specific_cloud_liquid_water_content', 'Temperature']].style.background_gradient(cmap='Reds'))
    
    # 5. Selected Region Deep Dive
    st.markdown('<div class="sub-header">🔬 Target Deep Dive (All Metrics)</div>', unsafe_allow_html=True)
    
    selected_target_idx = st.selectbox("Select Target from List:", top_targets.index, format_func=lambda x: f"Lat: {top_targets.loc[x,'latitude']:.2f}, Score: {top_targets.loc[x,'Score']:.0f}")
    target_row = top_targets.loc[selected_target_idx]
    
    # Store selection for Tab 3
    st.session_state['target_row'] = target_row
    st.session_state['target_time'] = timestamp
    
    # VISUALS FOR ALL METRICS
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric("Temp (°C)", f"{target_row['Temperature']:.2f}")
        st.metric("Liquid Water (kg/kg)", f"{target_row['Specific_cloud_liquid_water_content']:.5f}")
        st.metric("Cloud Cover (0-1)", f"{target_row['Fraction_of_cloud_cover']:.2f}")
    
    with m_col2:
        st.metric("Rel. Humidity (0-1)", f"{target_row['Relative_Humidity']:.2f}")
        div_val = target_row['Divergence'] if 'Divergence' in target_row else 0
        st.metric("Divergence (s^-1)", f"{div_val:.2e}")
        w_val = target_row['Vertical_velocity'] if 'Vertical_velocity' in target_row else 0
        st.metric("Vertical Velocity (Pa/s)", f"{w_val:.2f}")

    with m_col3:
        u = target_row['U_component_of_wind'] if 'U_component_of_wind' in target_row else 0
        v = target_row['V_component_of_wind'] if 'V_component_of_wind' in target_row else 0
        speed = np.sqrt(u**2 + v**2)
        st.metric("Wind Speed (m/s)", f"{speed:.2f}")
        st.metric("Geopotential", f"{target_row['Geopotential']:.0f}")
        
    # Graphical Deep Dive (Temperature & LWC)
    st.markdown("##### 📉 Atmospheric Profile")
    chart_df = pd.DataFrame({
        "Metric": ["Cloud Cover", "Humidity", "Liquid Water (Scaled)"],
        "Value": [target_row['Fraction_of_cloud_cover'], target_row['Relative_Humidity'], target_row['Specific_cloud_liquid_water_content']*10000]
    })
    st.bar_chart(chart_df.set_index("Metric"))


# --- TAB 3: AI & DRONE OPS ---
with tab3:
    if 'target_row' in st.session_state:
        target = st.session_state['target_row']
        
        st.markdown('<div class="sub-header">🤖 AI Mission Commander</div>', unsafe_allow_html=True)
        
        col_ai_viz, col_ai_desc = st.columns([1, 1])
        
        # 1. Determine Seeding Logic
        is_seedable = target['Score'] > 40
        
        # 2. Determine Drone Formation
        cc = target['Fraction_of_cloud_cover']
        if cc > 0.8:
            formation = "Formation A (6 Drones)"
            drones = 6
            spread_desc = "High Cloud Spread detected. Max coverage required."
        elif cc > 0.4:
            formation = "Formation B (4 Drones)"
            drones = 4
            spread_desc = "Moderate Cloud Spread. Standard grid required."
        else:
            formation = "Formation C (2 Drones)"
            drones = 2
            spread_desc = "Concentrated Cloud Core. Precision strike required."

        with col_ai_viz:
            st.markdown("#### 📡 Drone Formation Preview")
            
            # Create synthetic drone coordinates for the plot
            if drones == 6:
                drone_pos = pd.DataFrame({'x': [-1, 1, -2, 2, 0, 0], 'y': [1, 1, 0, 0, -1, -2]})
            elif drones == 4:
                drone_pos = pd.DataFrame({'x': [-1, 1, -1, 1], 'y': [1, 1, -1, -1]})
            else:
                drone_pos = pd.DataFrame({'x': [-0.5, 0.5], 'y': [0, 0]})
            
            fig_drones = px.scatter(drone_pos, x='x', y='y', title=f"{formation}", size_max=20)
            fig_drones.update_traces(marker=dict(size=20, symbol="triangle-up", color="#00ff00"))
            
            # --- FIX: Updated layout properties to be valid ---
            fig_drones.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_drones, use_container_width=True)

        with col_ai_desc:
            st.markdown(f"**Target:** {target['latitude']:.2f}, {target['longitude']:.2f}")
            st.markdown(f"**Seedability Score:** {target['Score']:.0f}/100")
            
            if is_seedable:
                st.success("✅ STATUS: SEEDABLE")
                st.info(f"**Strategy:** {formation}")
                st.caption(f"Reasoning: {spread_desc}")
                
                if st.button("🚀 DEPLOY DRONES", type="primary"):
                    with st.spinner("Gemini 2.0 Calculating Trajectories..."):
                        # Get AI Analysis
                        metrics_str = f"Temp: {target['Temperature']}C, LWC: {target['Specific_cloud_liquid_water_content']}, CloudCover: {cc}"
                        brief = get_gemini_brief(api_key, metrics_str, "GO", formation)
                    
                    st.markdown("---")
                    st.markdown("### 🛰️ Mission Log (Gemini Analysis)")
                    st.write(brief)
                    st.toast("Drones Launched Successfully!", icon="🚁")
            else:
                st.error("❌ STATUS: NOT SUITABLE")
                st.write("Atmospheric conditions do not meet ionization thresholds.")

    else:
        st.info("Please select a target in Tab 2 to initialize Drone Ops.")
