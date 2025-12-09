import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time

# --- CONFIGURATION (Dark Mode & Wide Layout) ---
st.set_page_config(
    page_title="Project Aether",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match your "Dark Blue Background" and "Aether Symbol" style
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117; /* Dark Background */
        color: white;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
    }
    .tier-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
        text-align: center;
    }
    .tier-green { background-color: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .tier-grey { background-color: #374151; color: #9ca3af; border: 1px dashed #6b7280; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'safety_scan_complete' not in st.session_state:
    st.session_state.safety_scan_complete = False
if 'mission_active' not in st.session_state:
    st.session_state.mission_active = False

# --- PAGE 1: LOGIN SCREEN ---
def login_page():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("PROJECT AETHER")
        st.markdown("*Intelligent Atmospheric Decision System*")
        
        with st.form("login_form"):
            st.text_input("Operator ID", value="VISION_RAIN_01")
            st.text_input("Password", type="password")
            submitted = st.form_submit_button("LOGIN")
            
            if submitted:
                st.session_state.page = 'dashboard'
                st.rerun()

# --- PAGE 2: MAIN DASHBOARD ---
def dashboard_page():
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("📍 Jeddah, Saudi Arabia")
        st.caption("Live Feed • EUMETSAT-9 Satellite Link • Google Vertex AI")
    with c2:
        if st.button("LOGOUT"):
            st.session_state.page = 'login'
            st.rerun()

    # SECTION: The Map & Timeline (Your Handwritten Page 1)
    col_map, col_stats = st.columns([2, 1])

    with col_map:
        # Simulating the Satellite Imagery of Jeddah with Cloud Cover
        st.markdown("### 🛰️ Real-Time Satellite Imagery")
        
        # Coordinates for Jeddah
        jeddah_lat = 21.4858
        jeddah_lon = 39.1925
        
        # Create a "Cloud" layer (Scatterplot)
        cloud_data = pd.DataFrame({
            'lat': [jeddah_lat + 0.02, jeddah_lat - 0.01, jeddah_lat + 0.05],
            'lon': [jeddah_lon + 0.03, jeddah_lon - 0.02, jeddah_lon + 0.01],
            'type': ['Seedable', 'Scattered', 'Seedable']
        })

        layer = pdk.Layer(
            "ScatterplotLayer",
            cloud_data,
            get_position='[lon, lat]',
            get_color='[0, 255, 128, 160]', # Greenish for seedable
            get_radius=3000,
        )

        view_state = pdk.ViewState(latitude=jeddah_lat, longitude=jeddah_lon, zoom=10, pitch=0)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    with col_stats:
        st.markdown("### 🕒 Previous 16 Hours")
        # Your specific timeline request (Red/Green slots)
        timeline_data = {
            "03:00": "No", "04:00": "No", "05:00": "No", 
            "06:00": "No", "07:00": "YES" 
        }
        
        for time_slot, status in timeline_data.items():
            color = "green" if status == "YES" else "red"
            st.markdown(f"**{time_slot}**: :{color}[{status}]")
        
        st.info("Current Hour Status: **CLOUD SEEDABLE**")
        
        if st.button(">> ANALYZE TARGET (Drill Down)"):
            st.session_state.page = 'analysis'
            st.rerun()

# --- PAGE 3: THE 5-TIER ANALYSIS (Your Handwritten Page 2) ---
def analysis_page():
    st.button("← Back to Dashboard", on_click=lambda: st.session_state.update(page='dashboard'))
    st.title("Target Analysis: Cloud Cell #JED-04")
    
    col_tiers, col_action = st.columns([1, 1])
    
    with col_tiers:
        st.subheader("5-Tier Validation Logic")
        
        # Tier 1-4 (Green/Satisfied)
        st.markdown('<div class="tier-box tier-green">✅ TIER 1: Humidity (78%) & Updraft (3 m/s)</div>', unsafe_allow_html=True)
        st.markdown('<div class="tier-box tier-green">✅ TIER 2: Cloud Depth (2.1 km)</div>', unsafe_allow_html=True)
        st.markdown('<div class="tier-box tier-green">✅ TIER 3: Wind Velocity (Surface 6 m/s)</div>', unsafe_allow_html=True)
        st.markdown('<div class="tier-box tier-green">✅ TIER 4: Aerosol Concentration (Moderate)</div>', unsafe_allow_html=True)
        
        # Tier 5 (Grey -> Green Logic)
        if not st.session_state.safety_scan_complete:
            st.markdown('<div class="tier-box tier-grey">⏳ TIER 5: Safety Override (PENDING SCAN)</div>', unsafe_allow_html=True)
        else:
             st.markdown('<div class="tier-box tier-green">✅ TIER 5: Safety Override (CLEAR)</div>', unsafe_allow_html=True)

    with col_action:
        st.subheader("Mission Control")
        
        # The "Terminal" Check Stats Feature
        with st.expander("💻 Open Terminal / Check Stats"):
            st.code("""
# CONNECTING TO GOOGLE BIGQUERY...
> SELECT * FROM `aether.sensor_stream` WHERE loc='JED-04'
------------------------------------------------
TIMESTAMP       | LWC (g/m3) | UPDRAFT | CHARGE
------------------------------------------------
12:00:01 UTC    | 0.08       | 3.2 m/s | NEUTRAL
12:00:05 UTC    | 0.09       | 3.4 m/s | NEUTRAL
------------------------------------------------
> STATUS: OPTIMAL FOR IONIZATION
            """, language="sql")

        st.markdown("---")
        
        # The Action Button
        if not st.session_state.safety_scan_complete:
            if st.button("🚀 INITIATE DRONE DEPLOYMENT"):
                with st.status("Running Tier 5 Safety Protocols...", expanded=True) as status:
                    st.write("Checking Restricted Airspace...")
                    time.sleep(1) # Dramatic pause for video
                    st.write("Scanning for Lightning (<50km)...")
                    time.sleep(1)
                    st.write("Verifying Google Earth Terrain Data...")
                    time.sleep(1)
                    status.update(label="Safety Scan Complete. Skies Clear.", state="complete")
                
                st.session_state.safety_scan_complete = True
                st.rerun()
        else:
            st.success("TIER 5 CLEARED. FLIGHT PATH APPROVED.")
            if st.button("CONFIRM LAUNCH"):
                st.session_state.mission_active = True
                st.balloons()
                st.toast("Drones Deployed! Monitoring via Vertex AI.")

    # Show "Mission Active" View if launched
    if st.session_state.mission_active:
        st.markdown("### 🛸 Live Drone Telemetry")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Big_Data_Prob.svg/1200px-Big_Data_Prob.svg.png", caption="Simulating Drone Swarm Formation (Source: Mock)", width=600)
        st.metric(label="Estimated Rainfall Yield Increase", value="+23%", delta="High Confidence")

# --- MAIN ROUTER ---
if st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'dashboard':
    dashboard_page()
elif st.session_state.page == 'analysis':
    analysis_page()
