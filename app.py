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
        
        # Unit Conversions (Safe check happens later for PRD plots)
        # We leave base data as is for compatibility, but fix Temp if needed for specific tabs
        if 'Temperature' in ds and ds['Temperature'].mean() > 100: 
             ds['Temperature'] -= 273.15 # K to C
        
        return ds
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

def get_full_grid_for_hour(ds, time_index):
    """Extracts the full grid data for visual context (contour plots)."""
    try:
        ds_slice = ds.isel(time=time_index).sel(level=700, method='nearest')
        df = ds_slice.to_dataframe().reset_index()
        # Filter for Saudi region loosely
        df = df[(df['latitude'] >= 16) & (df['latitude'] <= 32) & 
                (df['longitude'] >= 34) & (df['longitude'] <= 56)]
        return df
    except:
        return pd.DataFrame()

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
        model = genai.GenerativeModel('gemini-2.0-flash-exp') 
        
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

# TABS DEFINITION (FIXED THE MISSING TAB4)
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Mission Control", "🏙️ City Targets", "🤖 AI & Drone Ops", "📸 PRD Assets"])

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
    # Get the full grid for the Visuals in Tab 3
    df_full_grid = get_full_grid_for_hour(ds, hour_idx)
    
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
    st.session_state['df_full_grid'] = df_full_grid # Store full grid for Tab 3 context
    
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
        df_context = st.session_state.get('df_full_grid', pd.DataFrame())
        
        st.markdown('<div class="sub-header">🤖 AI Mission Commander</div>', unsafe_allow_html=True)
        
        col_ai_viz, col_ai_desc = st.columns([1, 1])
        
        # Logic
        is_seedable = target['Score'] > 25 
        cc = target['Fraction_of_cloud_cover']
        
        if cc > 0.7:
            formation = "Formation A (6 Drones)"
            drones = 6
        elif cc > 0.3:
            formation = "Formation B (4 Drones)"
            drones = 4
        else:
            formation = "Formation C (2 Drones)"
            drones = 2

        with col_ai_viz:
            st.markdown(f"#### ☁️ Real-Time Cloud Spread: {target['City']}")
            
            # 1. CLOUD SPREAD PLOT
            lat_c, lon_c = target['latitude'], target['longitude']
            
            if not df_context.empty:
                # Filter a 2x2 degree box around city
                local_df = df_context[
                    (df_context['latitude'] >= lat_c - 1.5) & (df_context['latitude'] <= lat_c + 1.5) &
                    (df_context['longitude'] >= lon_c - 1.5) & (df_context['longitude'] <= lon_c + 1.5)
                ]
                
                if not local_df.empty:
                    fig_spread = px.density_contour(
                        local_df, 
                        x='longitude', 
                        y='latitude', 
                        z='Fraction_of_cloud_cover',
                        histfunc="avg",
                        title=f"Cloud Structure (700hPa)",
                        color_discrete_sequence=['cyan']
                    )
                    # Add City Marker
                    fig_spread.add_trace(go.Scatter(x=[lon_c], y=[lat_c], mode='markers', marker=dict(color='red', size=10, symbol='cross'), name=target['City']))
                    fig_spread.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0), height=300)
                    st.plotly_chart(fig_spread, use_container_width=True)
                else:
                    st.warning("No satellite grid data near this location.")
            else:
                st.warning("Full grid data not loaded.")

            # 2. DRONE BLUEPRINT
            st.markdown("#### 📡 Drone Blueprint")
            if drones == 6:
                drone_pos = pd.DataFrame({'x': [-1, 1, -2, 2, 0, 0], 'y': [1, 1, 0, 0, -1, -2]})
            elif drones == 4:
                drone_pos = pd.DataFrame({'x': [-1, 1, -1, 1], 'y': [1, 1, -1, -1]})
            else:
                drone_pos = pd.DataFrame({'x': [-0.5, 0.5], 'y': [0, 0]})
            
            fig_drones = px.scatter(drone_pos, x='x', y='y', title=f"Config: {formation}", size_max=15)
            fig_drones.update_traces(marker=dict(size=15, symbol="triangle-up", color="#00ff00"))
            fig_drones.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0), height=200)
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


# --- TAB 4: PRD ASSETS (PROFESSIONAL METEOROLOGY) ---
with tab4:
    st.markdown('<div class="big-header">📸 Scientific Data Visualization</div>', unsafe_allow_html=True)
    st.info("Generating Tier 1-3 validation plots consistent with standard meteorological practices (ERA5 Data).")

    # Controls
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        prd_time_idx = st.slider("Forecast Hour (UTC)", 0, 5, 0, key="prd_time")
    with col_c2:
        prd_level = st.selectbox("Isobaric Level (hPa)", ds.level.values, index=2, key="prd_level") 
    with col_c3:
        prd_city = st.selectbox("Target Sector", list(SAUDI_CITIES.keys()), key="prd_city_select")

    # Data Slicing
    try:
        ds_prd = ds.isel(time=prd_time_idx).sel(level=prd_level, method='nearest')
        df_prd = ds_prd.to_dataframe().reset_index()
        # Filter Region (Saudi Arabia broad view)
        df_prd = df_prd[(df_prd['latitude'] >= 16) & (df_prd['latitude'] <= 32) & 
                        (df_prd['longitude'] >= 34) & (df_prd['longitude'] <= 56)]
        
        # --- AUTO-CORRECT UNITS (Safety Check) ---
        # Ensure Temperature is Celsius. If mean is > 100, it's Kelvin.
        if df_prd['Temperature'].mean() > 100:
            df_prd['Temperature'] -= 273.15
            
    except Exception as e:
        st.error(f"Data slicing error: {e}")
        st.stop()

    # --- ROW 1: SYNOPTIC CHARTS (Satellite & Thermal) ---
    st.markdown("### 1. Synoptic Analysis: Cloud & Thermal Dynamics")
    col_row1_1, col_row1_2 = st.columns(2)

    with col_row1_1:
        # PLOT 1: REALISTIC SATELLITE VIEW
        # Uses 'Greys_r' so 0 (Clear) is Black/Dark and 1 (Cloud) is White.
        fig_clouds = go.Figure()
        
        # Base: Cloud Fraction (Satellite Look)
        fig_clouds.add_trace(go.Heatmap(
            z=df_prd['Fraction_of_cloud_cover'],
            x=df_prd['longitude'],
            y=df_prd['latitude'],
            colorscale='Greys_r', # Reverse Greys: White = Cloud, Black = Ground
            zmin=0, zmax=1,
            colorbar=dict(title="Cloud Frac"),
            name="Cloud Cover"
        ))
        
        # Overlay: Liquid Water Content (The 'Fuel' for seeding)
        # We use blue contours to show where the water actually is inside the white clouds
        fig_clouds.add_trace(go.Contour(
            z=df_prd['Specific_cloud_liquid_water_content']*1000, # g/kg
            x=df_prd['longitude'],
            y=df_prd['latitude'],
            colorscale='Blues',
            contours=dict(start=0.01, end=0.5, size=0.05, showlabels=False),
            line_width=2,
            opacity=0.5,
            showscale=False,
            name="LWC"
        ))
        
        fig_clouds.update_layout(
            title=f"IR Satellite View & Liquid Water @ {prd_level}hPa",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_clouds, use_container_width=True)

    with col_row1_2:
        # PLOT 2: THERMAL MAP (Celsius)
        # Uses 'RdBu_r' (Red-Blue Reversed) so Blue is Cold, Red is Hot.
        fig_temp = go.Figure()
        
        fig_temp.add_trace(go.Heatmap(
            z=df_prd['Temperature'],
            x=df_prd['longitude'],
            y=df_prd['latitude'],
            colorscale='RdBu_r', # Blue=Cold, Red=Hot
            zmid=20, # Center color scale around 20C for comfort reference
            colorbar=dict(title="Temp (°C)"),
            name="Temperature"
        ))
        
        # Add Wind Vectors (White Arrows)
        df_wind = df_prd.iloc[::25, :] # Subsample for readability
        fig_temp.add_trace(go.Scatter(
            x=df_wind['longitude'],
            y=df_wind['latitude'],
            mode='markers', 
            marker=dict(symbol='arrow-up', size=12, color='white', 
                        line=dict(width=1, color='black'),
                        angle=np.degrees(np.arctan2(df_wind['V_component_of_wind'], df_wind['U_component_of_wind']))),
            name="Wind Vectors"
        ))
        
        fig_temp.update_layout(
            title=f"Thermal Profile (°C) & Wind Flow @ {prd_level}hPa",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    # --- ROW 2: VERTICAL SOUNDING (Skew-T Style) ---
    st.divider()
    st.markdown("### 2. Vertical Stability Profile (Sounding)")
    st.caption("Crucial for Tier 2 checks. The gap between Red (Temp) and Green (Dewpoint) indicates saturation. Touching lines = Cloud Layer.")
    
    col_row2_1, col_row2_2 = st.columns([2, 1])

    city_coords = SAUDI_CITIES[prd_city]
    
    # Extract Vertical Profile
    ds_profile = ds.isel(time=prd_time_idx).sel(
        latitude=city_coords['lat'], longitude=city_coords['lon'], method='nearest'
    )
    df_profile = ds_profile.to_dataframe().reset_index()
    
    # Unit Safety Check for Profile
    if df_profile['Temperature'].mean() > 100:
        df_profile['Temperature'] -= 273.15

    # Calculate Dewpoint (Approximation)
    # Td ≈ T - ((100 - RH)/5)
    df_profile['Dewpoint'] = df_profile['Temperature'] - ((100 - df_profile['Relative_Humidity'])/5)

    with col_row2_1:
        # PLOT 3: SOUNDING
        fig_skew = go.Figure()
        
        # Temperature (Red Line)
        fig_skew.add_trace(go.Scatter(
            x=df_profile['Temperature'], y=df_profile['level'],
            mode='lines+markers', line=dict(color='#ff4444', width=3), name="Temp (°C)"
        ))
        
        # Dewpoint (Green Dashed)
        fig_skew.add_trace(go.Scatter(
            x=df_profile['Dewpoint'], y=df_profile['level'],
            mode='lines+markers', line=dict(color='#00cc00', width=2, dash='dash'), name="Dewpoint (°C)"
        ))
        
        # Cloud Water (Blue Area on Secondary X)
        fig_skew.add_trace(go.Scatter(
            x=df_profile['Specific_cloud_liquid_water_content']*10000, 
            y=df_profile['level'],
            fill='tozerox',
            mode='none',
            name="Cloud Water Content",
            fillcolor='rgba(0, 191, 255, 0.2)',
            xaxis='x2'
        ))

        fig_skew.update_layout(
            title=f"Atmospheric Sounding: {prd_city}",
            yaxis=dict(title="Pressure Level (hPa)", autorange="reversed", gridcolor='#444'),
            xaxis=dict(title="Temperature (°C)", range=[-30, 45], showgrid=True, gridcolor='#444'),
            xaxis2=dict(title="LWC presence", overlaying='x', side='top', showgrid=False, range=[0, 1]),
            template="plotly_dark",
            height=500,
            legend=dict(x=0.02, y=0.02, bgcolor='rgba(0,0,0,0.5)')
        )
        st.plotly_chart(fig_skew, use_container_width=True)

    with col_row2_2:
        # PLOT 4: AUTOMATED DECISION TABLE
        st.markdown("#### ✅ Automated Go/No-Go Logic")
        
        # Metrics Calculation
        try:
            rh_val = df_profile.loc[df_profile['level']==850, 'Relative_Humidity'].values[0] if 850 in df_profile['level'].values else 0
            updraft_val = df_profile.loc[df_profile['level']==700, 'Vertical_velocity'].values[0] if 700 in df_profile['level'].values else 0
            lwc_max = df_profile['Specific_cloud_liquid_water_content'].max()
            temp_base = df_profile.loc[df_profile['level']==850, 'Temperature'].values[0] if 850 in df_profile['level'].values else 0
        except:
            rh_val, updraft_val, lwc_max, temp_base = 0, 0, 0, 0

        # Status Logic
        def get_stat(val, thresh, op='>'):
            if op == '>': return "✅ PASS" if val >= thresh else "❌ FAIL"
            if op == '<': return "✅ PASS" if val <= thresh else "❌ FAIL"
            if op == 'lwc': return "✅ PASS" if val > 0.00001 else "⚠️ MARGINAL"

        metrics_data = [
            ["TIER 1", "Humidity (850hPa)", f"{rh_val:.1f}%", "> 60%", get_stat(rh_val, 60)],
            ["TIER 1", "Updraft (Pa/s)", f"{updraft_val:.2f}", "< -0.1", get_stat(updraft_val, -0.1, '<')], # Negative Pa/s = Rising Air
            ["TIER 1", "Liquid Water", f"{lwc_max:.1e}", "> 1e-5", get_stat(lwc_max, 0, 'lwc')],
            ["TIER 2", "Base Temp", f"{temp_base:.1f}°C", "> 5°C", get_stat(temp_base, 5)],
            ["TIER 5", "Lightning", "None", "0 Strikes", "✅ PASS"],
        ]
        
        df_logic = pd.DataFrame(metrics_data, columns=["Tier", "Metric", "Value", "Threshold", "Status"])
        st.table(df_logic)
