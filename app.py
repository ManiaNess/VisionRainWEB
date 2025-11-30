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
    .city-card {
        border-left: 5px solid #4285F4;
        background-color: #181b21;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "b5710c0835b1558c7a5002809513f1a5.nc"

# SAUDI CITIES DATABASE
SAUDI_CITIES = {
    "Riyadh": {"lat": 24.7136, "lon": 46.6753},
    "Jeddah": {"lat": 21.5433, "lon": 39.1728},
    "Mecca": {"lat": 21.3891, "lon": 39.8579},
    "Medina": {"lat": 24.5247, "lon": 39.5692},
    "Dammam": {"lat": 26.4207, "lon": 50.0888},
    "Abha": {"lat": 18.2205, "lon": 42.5053},
    "Tabuk": {"lat": 28.3835, "lon": 36.5662},
    "Hail": {"lat": 27.5114, "lon": 41.7208},
    "Jizan": {"lat": 16.8894, "lon": 42.5706},
    "Najran": {"lat": 17.4924, "lon": 44.1277},
    "Al Baha": {"lat": 20.0129, "lon": 41.4677},
    "Taif": {"lat": 21.2885, "lon": 40.4167},
    "Buraydah": {"lat": 26.3592, "lon": 43.9818}
}

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

def process_cities_for_hour(ds, time_index):
    """Extracts data for Major Cities at a specific hour."""
    try:
        # Select specific time and 700hPa level
        ds_slice = ds.isel(time=time_index).sel(level=700, method='nearest')
        
        city_data_list = []
        
        for city_name, coords in SAUDI_CITIES.items():
            # Snap to nearest grid point in ERA5
            try:
                point = ds_slice.sel(latitude=coords['lat'], longitude=coords['lon'], method='nearest')
                
                # Extract values safely
                cc = float(point['Fraction_of_cloud_cover']) if 'Fraction_of_cloud_cover' in point else 0
                rh = float(point['Relative_Humidity']) if 'Relative_Humidity' in point else 0
                lwc = float(point['Specific_cloud_liquid_water_content']) if 'Specific_cloud_liquid_water_content' in point else 0
                temp = float(point['Temperature']) if 'Temperature' in point else 0
                
                # Heuristic Score (0-100)
                # Boosted sensitivity for demo purposes
                score = (cc * 50) + (rh * 0.5) + (lwc * 300000)
                score = min(100, score) # Cap at 100
                
                city_data_list.append({
                    "City": city_name,
                    "latitude": coords['lat'],
                    "longitude": coords['lon'],
                    "Score": score,
                    "Temperature": temp,
                    "Fraction_of_cloud_cover": cc,
                    "Specific_cloud_liquid_water_content": lwc,
                    "Relative_Humidity": rh,
                    "Vertical_velocity": float(point['Vertical_velocity']) if 'Vertical_velocity' in point else 0,
                    "Divergence": float(point['Divergence']) if 'Divergence' in point else 0,
                    "U_component_of_wind": float(point['U_component_of_wind']) if 'U_component_of_wind' in point else 0,
                    "V_component_of_wind": float(point['V_component_of_wind']) if 'V_component_of_wind' in point else 0,
                    "Geopotential": float(point['Geopotential']) if 'Geopotential' in point else 0
                })
            except:
                continue
                
        df = pd.DataFrame(city_data_list)
        return df, ds_slice.time.values
    except Exception as e:
        st.error(f"Processing Error: {e}")
        return pd.DataFrame(), None

def get_gemini_brief(api_key, metrics, decision, formation, city_name):
    """Generates the AI Explanation."""
    if not api_key: 
        return f"""
        **AI MISSION BRIEF (SIMULATION)**
        **Target:** {city_name} Sector
        
        **1. Physics Analysis:**
        Atmospheric scan over {city_name} indicates favorable hygroscopic conditions. Liquid Water Content (LWC) is sufficient for nucleation. 
        
        **2. Strategy ({formation}):**
        The cloud topology suggests a scattered convective system, requiring the {formation} for optimal coverage.
        
        **3. Risk Assessment:**
        Wind shear nominal. Clearance granted.
        """
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        
        prompt = f"""
        Act as 'Kingdom Commander' AI for Saudi Cloud Seeding.
        Target City: {city_name}
        Mission Status: {decision}
        Formation: {formation}
        Data: {metrics}
        
        1. Explain WHY {city_name} is seedable based on physics (LWC, Temp).
        2. Justify {formation} based on cloud spread.
        3. Scientific but concise (Max 150 words).
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ AI Connection Error. Displaying cached mission profile."

# ==========================================
# 3. UI ARCHITECTURE
# ==========================================

ds = load_data()
if ds is None: st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")

# TABS
tab1, tab2, tab3 = st.tabs(["🌍 Mission Control", "🏙️ City Targets", "🤖 AI & Drone Ops"])

# --- TAB 1: MISSION CONTROL ---
with tab1:
    st.markdown('<div class="big-header">🇸🇦 VisionRain: The Initiative</div>', unsafe_allow_html=True)
    
    col_text, col_stat = st.columns([2, 1])
    with col_text:
        st.markdown("""
        **The Challenge:** Saudi Arabia faces water scarcity and reactive, costly cloud seeding operations ($8,000/hr).
        **The Solution:** An AI-Driven, Pilotless Drone Swarm system targeting major population centers.
        **The Goal:** Increase rainfall efficiency, reduce costs, and ensure water security.
        """)
    with col_stat:
        st.metric("Target Cities", f"{len(SAUDI_CITIES)}")
        st.metric("Water Sustainability", "Vision 2030")

    st.divider()
    
    st.markdown('<div class="sub-header">🛰️ Kingdom-Wide Scan (Cloud Tops)</div>', unsafe_allow_html=True)
    # Background Map
    ds_max = ds.max(dim='time').sel(level=600, method='nearest')
    df_max = ds_max.to_dataframe().reset_index()
    df_max = df_max[(df_max['latitude'] >= 16) & (df_max['latitude'] <= 32) & (df_max['longitude'] >= 34) & (df_max['longitude'] <= 56)]
    st.map(df_max[df_max['Fraction_of_cloud_cover'] > 0.4], latitude='latitude', longitude='longitude', size=20, color='#4285F4')


# --- TAB 2: CITY TARGETS ---
with tab2:
    st.markdown('<div class="sub-header">⏳ Temporal Scan (Hour 1 - 6)</div>', unsafe_allow_html=True)
    
    hour_idx = st.slider("Select Scan Hour", 0, 5, 0)
    
    # Process Data
    df_cities, timestamp = process_cities_for_hour(ds, hour_idx)
    st.write(f"**Scanning Time:** {str(timestamp)}")
    
    # Sort by Score
    df_cities = df_cities.sort_values(by='Score', ascending=False)
    
    # 1. Map of Cities (Color coded by Score)
    st.markdown("#### 📍 City Seedability Status")
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_cities,
        get_position=["longitude", "latitude"],
        get_color="[255, 255 - (Score * 2.5), 100]", # Green to Red gradient
        get_radius=30000,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=24.0, longitude=45.0, zoom=4.5)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{City}\nScore: {Score:.1f}"})
    st.pydeck_chart(r)
    
    # 2. Leaderboard Table
    st.markdown("#### 🏆 Seedability Leaderboard")
    st.dataframe(
        df_cities[['City', 'Score', 'Temperature', 'Fraction_of_cloud_cover', 'Specific_cloud_liquid_water_content']].style.background_gradient(subset=['Score'], cmap='Greens')
    )
    
    # 3. Selection
    st.markdown('<div class="sub-header">🔬 City Deep Dive</div>', unsafe_allow_html=True)
    
    selected_city_name = st.selectbox("Select City for Analysis:", df_cities['City'])
    target_row = df_cities[df_cities['City'] == selected_city_name].iloc[0]
    
    # Store for Tab 3
    st.session_state['target_row'] = target_row
    st.session_state['target_time'] = timestamp
    
    # Visuals
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Temp (°C)", f"{target_row['Temperature']:.2f}")
        st.metric("Liquid Water", f"{target_row['Specific_cloud_liquid_water_content']:.5f}")
    with m_col2:
        st.metric("Cloud Cover", f"{target_row['Fraction_of_cloud_cover']:.2f}")
        st.metric("Humidity", f"{target_row['Relative_Humidity']:.2f}")
    with m_col3:
        wind = np.sqrt(target_row['U_component_of_wind']**2 + target_row['V_component_of_wind']**2)
        st.metric("Wind Speed", f"{wind:.2f} m/s")
        st.metric("Vertical Vel.", f"{target_row['Vertical_velocity']:.3f}")

    st.bar_chart(pd.DataFrame({
        "Metric": ["Cloud Cover", "Humidity", "LWC (x10k)"], 
        "Value": [target_row['Fraction_of_cloud_cover'], target_row['Relative_Humidity'], target_row['Specific_cloud_liquid_water_content']*10000]
    }).set_index("Metric"))


# --- TAB 3: AI OPS ---
with tab3:
    if 'target_row' in st.session_state:
        target = st.session_state['target_row']
        
        st.markdown('<div class="sub-header">🤖 AI Mission Commander</div>', unsafe_allow_html=True)
        
        col_ai_viz, col_ai_desc = st.columns([1, 1])
        
        # Logic
        is_seedable = target['Score'] > 25 # Lower threshold for demo
        cc = target['Fraction_of_cloud_cover']
        
        if cc > 0.7:
            formation = "Formation A (6 Drones)"
            drones = 6
            spread_desc = "High Spread. Max coverage."
        elif cc > 0.3:
            formation = "Formation B (4 Drones)"
            drones = 4
            spread_desc = "Moderate Spread."
        else:
            formation = "Formation C (2 Drones)"
            drones = 2
            spread_desc = "Precision Strike."

        with col_ai_viz:
            st.markdown(f"#### ☁️ Cloud Topology: {target['City']}")
            # Simulated local cloud spread for visual
            # In a real app, we'd slice grid pixels around the city coords
            # Here we just show the city point context
            
            # Drone Blueprint
            if drones == 6:
                drone_pos = pd.DataFrame({'x': [-1, 1, -2, 2, 0, 0], 'y': [1, 1, 0, 0, -1, -2]})
            elif drones == 4:
                drone_pos = pd.DataFrame({'x': [-1, 1, -1, 1], 'y': [1, 1, -1, -1]})
            else:
                drone_pos = pd.DataFrame({'x': [-0.5, 0.5], 'y': [0, 0]})
            
            fig_drones = px.scatter(drone_pos, x='x', y='y', title=f"Drone Blueprint: {formation}", size_max=20)
            fig_drones.update_traces(marker=dict(size=20, symbol="triangle-up", color="#00ff00"))
            fig_drones.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_drones, use_container_width=True)

        with col_ai_desc:
            st.markdown(f"**Target:** {target['City']}")
            st.markdown(f"**Score:** {target['Score']:.1f}")
            
            if is_seedable:
                st.success("✅ STATUS: SEEDABLE")
                st.info(f"**Strategy:** {formation}")
                
                if st.button("🚀 DEPLOY DRONES", type="primary"):
                    with st.spinner("Gemini 2.0 Calculating..."):
                        metrics_str = f"Temp: {target['Temperature']:.1f}, LWC: {target['Specific_cloud_liquid_water_content']:.5f}"
                        brief = get_gemini_brief(api_key, metrics_str, "GO", formation, target['City'])
                    
                    st.markdown("### 🛰️ Mission Log")
                    st.write(brief)
                    st.toast("Drones Launched!", icon="🚁")
            else:
                st.error("❌ NOT SUITABLE")
                st.write("Conditions below threshold.")
    else:
        st.info("Select a city in Tab 2 first.")
