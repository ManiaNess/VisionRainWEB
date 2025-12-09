import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import random

# ==========================================
# 1. VISUAL STYLE (Matching Handwritten Page 1)
# ==========================================
st.set_page_config(
    page_title="Project Aether | VisionRain",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Dark Blue Background" [Handwritten Page 1, Source: 6-7]
st.markdown("""
    <style>
    .stApp {
        background-color: #0b1120; /* Deep Dark Blue */
        color: white;
    }
    /* Tier Boxes matching the Sketch */
    .tier-box {
        padding: 12px;
        margin-bottom: 8px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
    }
    .tier-pass {
        background-color: #065f46; /* Green */
        border: 1px solid #34d399;
        color: #d1fae5;
    }
    .tier-pending {
        background-color: #374151; /* Grey */
        border: 1px dashed #9ca3af;
        color: #9ca3af;
    }
    /* Terminal Style for "Check Stats" */
    .terminal {
        background-color: #000;
        color: #0f0;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border: 1px solid #333;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'safety_scan' not in st.session_state:
    st.session_state.safety_scan = False # Tier 5 starts false
if 'terminal_open' not in st.session_state:
    st.session_state.terminal_open = False

# ==========================================
# PAGE 1: LOGIN (Handwritten Page 1)
# ==========================================
def render_login():
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("⚡ PROJECT AETHER")
        st.markdown("*Intelligent Atmospheric Decision System*")
        
        with st.form("login"):
            st.text_input("ID", value="VR_OPERATOR_1")
            st.text_input("Password", type="password")
            
            if st.form_submit_button("LOGIN"):
                st.session_state.page = 'dashboard'
                st.rerun()

# ==========================================
# PAGE 2: DASHBOARD (Handwritten Page 1 Bottom)
# ==========================================
def render_dashboard():
    # Header: "Jeddah Saudi Arabia" [Source: 11]
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("📍 Jeddah, Saudi Arabia")
        st.caption("Live Feed • EUMETSAT MSG-4 • Google Vertex AI")
    with c2:
        if st.button("LOGOUT"):
            st.session_state.page = 'login'
            st.rerun()

    col_map, col_timeline = st.columns([2, 1])

    # --- LEFT: SATELLITE IMAGERY [Source: 12] ---
    with col_map:
        st.markdown("### 🛰️ Satellite Imagery (IR)")
        
        # Simulated Cloud Map over Jeddah
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({'lat': [21.5433], 'lon': [39.1728], 'status': ['TARGET']}),
            get_position='[lon, lat]',
            get_color='[0, 255, 128, 140]', # Green Highlight
            get_radius=5000,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=21.5433, longitude=39.1728, zoom=9)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    # --- RIGHT: 16-HOUR TIMELINE [Source: 20-26] ---
    with col_timeline:
        st.markdown("### 🕒 Previous 16 Hours")
        
        # EXACT DATA FROM SKETCH [Source: 21-26]
        # "5am No", "6am No", "7am Yes"
        timeline = [
            ("03:00", "NO", "red"),
            ("04:00", "NO", "red"),
            ("05:00", "NO", "red"),
            ("06:00", "NO", "red"),
            ("07:00", "YES", "green"), # The Seedable Event
            ("08:00", "PENDING", "grey"),
        ]
        
        for time_val, status, color in timeline:
            if color == "green":
                st.markdown(f"**{time_val}**: :green[CLOUD SEEDABLE] ✅")
            elif color == "red":
                st.markdown(f"**{time_val}**: :red[No]")
            else:
                st.markdown(f"**{time_val}**: {status}")

        st.divider()
        st.success("CURRENT STATUS: **SEEDABLE**")
        
        # "New button to open that tab" [Source: 93]
        if st.button(">> ANALYZE TARGET (Drill Down)"):
            st.session_state.page = 'analysis'
            st.rerun()

# ==========================================
# PAGE 3: DRILL DOWN (Handwritten Page 2 + Tech Doc Pg 6)
# ==========================================
def render_analysis():
    if st.button("← Back"):
        st.session_state.page = 'dashboard'
        st.rerun()

    st.title("Target Analysis: Cloud JED-04")
    
    col_tiers, col_action = st.columns([1.5, 1])

    # --- LEFT: THE 5 TIERS [Source: 42-50, Tech Doc Pg 6] ---
    with col_tiers:
        st.subheader("5-Tier Validation Logic")
        
        # TIER 1: PHYSICAL CAPABILITY [Tech Doc Pg 6]
        st.markdown(f"""
        <div class="tier-box tier-pass">
            <b>✅ TIER 1: PHYSICAL CAPABILITY</b><br>
            Humidity: 78% (>75%) | Updraft: 3.2 m/s (>2m/s)
        </div>
        """, unsafe_allow_html=True)

        # TIER 2: MICROPHYSICS [Tech Doc Pg 6]
        st.markdown(f"""
        <div class="tier-box tier-pass">
            <b>✅ TIER 2: MICROPHYSICS</b><br>
            Cloud Depth: 2.4 km (>2km) | LWC: 0.09 g/m³
        </div>
        """, unsafe_allow_html=True)

        # TIER 3: DYNAMICS [Tech Doc Pg 6]
        st.markdown(f"""
        <div class="tier-box tier-pass">
            <b>✅ TIER 3: DYNAMICS</b><br>
            Cloud Base: 1200m (<1500m) | Wind: 5 m/s
        </div>
        """, unsafe_allow_html=True)

        # TIER 4: OPTIMIZATION [Tech Doc Pg 6]
        st.markdown(f"""
        <div class="tier-box tier-pass">
            <b>✅ TIER 4: OPTIMIZATION</b><br>
            Stage: Growing | Aerosols: Moderate
        </div>
        """, unsafe_allow_html=True)

        # TIER 5: SAFETY (THE LOGIC FROM SKETCH) [Source: 69-70]
        # "Tier 5 is grey because it will be pending"
        if not st.session_state.safety_scan:
            st.markdown(f"""
            <div class="tier-box tier-pending">
                <b>⏳ TIER 5: SAFETY OVERRIDE</b><br>
                Status: PENDING SCAN
            </div>
            """, unsafe_allow_html=True)
        else:
            # TURNS GREEN AFTER SCAN [Source: 109]
            st.markdown(f"""
            <div class="tier-box tier-pass">
                <b>✅ TIER 5: SAFETY OVERRIDE</b><br>
                Lightning < 50km: NONE | Airspace: CLEAR
            </div>
            """, unsafe_allow_html=True)

    # --- RIGHT: TERMINAL & ACTIONS [Source: 57-68] ---
    with col_action:
        st.subheader("Mission Control")
        
        # "Check Stats via Terminal" [Source: 57, 62]
        if st.button("💻 OPEN TERMINAL / CHECK STATS"):
            st.session_state.terminal_open = not st.session_state.terminal_open
            
        if st.session_state.terminal_open:
            st.markdown("""
            <div class="terminal">
            user@aether:~$ ./check_microphysics<br>
            > CONNECTING TO GOOGLE BIGQUERY...<br>
            > TARGET: JED-04<br>
            --------------------------------<br>
            LWC: 0.09 g/m3 [OK]<br>
            UPDRAFT: 3.2 m/s [OK]<br>
            TEMP_TOP: -12 C [OK]<br>
            --------------------------------<br>
            > READY FOR TIER 5 CHECK.
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # "Send Drone" Button Logic [Source: 58-59]
        if not st.session_state.safety_scan:
            st.info("Validation complete. Initiate Safety Scan to clear Tier 5.")
            if st.button("🚀 INITIATE SAFETY SCAN (TIER 5)", type="primary"):
                with st.spinner("Scanning for Lightning (<50km)..."): # [Tech Doc Pg 11]
                    time.sleep(2)
                st.session_state.safety_scan = True
                st.rerun()
        else:
            # ONCE TIER 5 IS GREEN -> CONFIRM LAUNCH
            st.success("TIER 5 CLEARED. AIRSPACE SAFE.")
            
            # FORMATION SELECTION [Tech Doc Pg 11]
            formation = st.selectbox("Select Swarm Formation", 
                                   ["Formation A (Wide Coverage)", 
                                    "Formation B (Focused)", 
                                    "Formation C (Small)"])
            
            if st.button("CONFIRM LAUNCH"):
                st.session_state.page = 'mission_active'
                st.rerun()

# ==========================================
# PAGE 4: MISSION ACTIVE [Source: 17, 35]
# ==========================================
def render_mission_active():
    st.balloons()
    st.title("MISSION ACTIVE")
    st.success("Swarm Deployed. Monitoring via Vertex AI.")
    
    st.metric("Est. Rainfall Increase", "+23%") # [Source: 191]
    
    if st.button("Return to Dashboard"):
        st.session_state.page = 'dashboard'
        st.session_state.safety_scan = False
        st.rerun()

# ==========================================
# MAIN ROUTER
# ==========================================
if st.session_state.page == 'login':
    render_login()
elif st.session_state.page == 'dashboard':
    render_dashboard()
elif st.session_state.page == 'analysis':
    render_analysis()
elif st.session_state.page == 'mission_active':
    render_mission_active()
