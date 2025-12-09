import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import google.generativeai as genai
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION & STYLING (The "Dark Blue" Theme)
# ==========================================
st.set_page_config(
    page_title="Project Aether | Command",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match your Handwritten PDF (Dark Blue/Black aesthetic)
st.markdown("""
    <style>
    .stApp {
        background-color: #050a14; /* Deep Dark Blue/Black */
        color: #e0f2fe;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 4px;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    /* Tier Box Styling */
    .tier-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-family: monospace;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .tier-green {
        background-color: #064e3b; 
        border: 1px solid #34d399;
        color: #6ee7b7;
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.2);
    }
    .tier-grey {
        background-color: #374151; 
        border: 1px dashed #9ca3af;
        color: #d1d5db;
        opacity: 0.7;
    }
    .terminal-window {
        background-color: #000000;
        border: 1px solid #333;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 15px;
        border-radius: 5px;
        font-size: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #111827;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #1f2937;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'safety_scan_complete' not in st.session_state:
    st.session_state.safety_scan_complete = False
if 'terminal_open' not in st.session_state:
    st.session_state.terminal_open = False
if 'mission_active' not in st.session_state:
    st.session_state.mission_active = False

# ==========================================
# 3. GEMINI AI INTEGRATION (Report Generation)
# ==========================================
def generate_ai_report(api_key, context_data):
    if not api_key:
        return "⚠️ API Key missing. Showing simulation mode."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp') # Or 1.5-flash
        prompt = f"""
        Act as the 'Project Aether' Operating System. 
        Generate a mission success report based on this telemetry:
        {context_data}
        
        Format:
        1. Conditions Verified (Bullet points)
        2. Tier 5 Safety Check Status
        3. Predicted Rainfall Yield (%)
        Keep it technical, military-grade concise.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI System Offline: {str(e)}"

# ==========================================
# PAGE 1: LOGIN (Handwritten Page 1 Top)
# ==========================================
def render_login():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        # Placeholder for the Aether Symbol (Circle/Triangle from your sketch)
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 60px; margin-bottom: 0;">⚡</h1>
            <h1 style="margin-top: 0;">PROJECT AETHER</h1>
            <p style="color: #94a3b8;">Autonomous Atmospheric Ionization System</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.text_input("OPERATOR ID", value="VISION_RAIN_001")
            st.text_input("PASSWORD", type="password", value="password")
            submitted = st.form_submit_button("AUTHENTICATE")
            
            if submitted:
                with st.spinner("Connecting to Google Cloud Vertex AI..."):
                    time.sleep(1.5) # Dramatic pause
                st.session_state.page = 'dashboard'
                st.rerun()

# ==========================================
# PAGE 2: MAIN DASHBOARD (Handwritten Page 1 Bottom)
# ==========================================
def render_dashboard():
    # Sidebar for API Key
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/768px-Google_%22G%22_logo.svg.png", width=30)
        st.markdown("**System Backend:**")
        st.caption("• Vertex AI (Decision)")
        st.caption("• Earth Engine (Spatial)")
        st.caption("• BigQuery (Storage)")
        st.divider()
        api_key = st.text_input("Gemini API Key", type="password")

    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("📍 Jeddah, Saudi Arabia")
        st.markdown("`LIVE SATELLITE FEED` • `EUMETSAT MSG-4`")
    with c2:
        if st.button("LOGOUT"):
            st.session_state.page = 'login'
            st.rerun()

    # --- MAP SECTION ---
    col_map, col_timeline = st.columns([2, 1])
    
    with col_map:
        # Simulating Jeddah Cloud Cover
        jeddah_lat, jeddah_lon = 21.5433, 39.1728
        
        # PyDeck Map (Dark Mode Satellite)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({
                'lat': [21.58, 21.52, 21.60],
                'lon': [39.20, 39.15, 39.25],
                'status': ['SEEDABLE', 'DRY', 'SEEDABLE']
            }),
            get_position='[lon, lat]',
            get_color='[0, 255, 128, 150]', # Green Glow
            get_radius=4000,
            pickable=True,
        )
        
        view_state = pdk.ViewState(latitude=jeddah_lat, longitude=jeddah_lon, zoom=9.5, pitch=0)
        st.pydeck_chart(pdk.Deck(
            layers=[layer], 
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v10'
        ))

    # --- TIMELINE SECTION (Handwritten Note: "Previous 16 hours") ---
    with col_timeline:
        st.markdown("### 🕒 Historic Scan (16H)")
        
        # Creating the Red/Green list from your sketch
        history_data = [
            ("03:00", "NO", "red"), ("04:00", "NO", "red"),
            ("05:00", "NO", "red"), ("06:00", "NO", "red"),
            ("07:00", "YES", "green"), # This matches your note "7am Yes"
            ("08:00", "NO", "red"),
            ("09:00", "PENDING", "grey")
        ]
        
        for time_val, status, color in history_data:
            if color == "green":
                st.markdown(f"**{time_val}** : <span style='color:#34d399; font-weight:bold'>CLOUD SEEDABLE</span>", unsafe_allow_html=True)
            elif color == "red":
                st.markdown(f"**{time_val}** : <span style='color:#ef4444'>Conditions Not Met</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{time_val}** : {status}", unsafe_allow_html=True)

        st.divider()
        st.markdown("### ⚡ Current Status")
        st.success("**TARGET DETECTED: SECTOR JED-04**")
        
        if st.button(">> ANALYZE TARGET (Drill Down)"):
            st.session_state.page = 'analysis'
            st.rerun()

# ==========================================
# PAGE 3: 5-TIER ANALYSIS (Handwritten Page 2)
# ==========================================
def render_analysis():
    # Sidebar API access
    with st.sidebar:
        api_key = st.text_input("Gemini API Key", type="password")

    # Header
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← BACK"):
            st.session_state.page = 'dashboard'
            st.rerun()
    with col_title:
        st.subheader("TARGET ANALYSIS: CLOUD JED-04")

    col_tiers, col_actions = st.columns([1.5, 1])

    # --- LEFT COL: THE 5 TIERS (Your Sketch) ---
    with col_tiers:
        st.markdown("### 🛡️ 5-Tier Validation Logic (Vertex AI)")
        
        # Tiers 1-4 are GREEN (Conditions Satisfied)
        st.markdown('<div class="tier-box tier-green"><span>TIER 1: PHYSICAL CAPABILITY</span> <span>✅ PASS</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="tier-box tier-green"><span>TIER 2: MICROPHYSICS (LWC > 0.05)</span> <span>✅ PASS</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="tier-box tier-green"><span>TIER 3: DYNAMICS (Updraft 3m/s)</span> <span>✅ PASS</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="tier-box tier-green"><span>TIER 4: OPTIMIZATION (Aerosols)</span> <span>✅ PASS</span></div>', unsafe_allow_html=True)

        # TIER 5 (The Dynamic Part from your Sketch)
        if not st.session_state.safety_scan_complete:
            # GREY - PENDING
            st.markdown('<div class="tier-box tier-grey"><span>TIER 5: SAFETY OVERRIDE</span> <span>⏳ PENDING SCAN</span></div>', unsafe_allow_html=True)
            st.caption("⚠️ Lightning scan required before drone launch.")
        else:
            # GREEN - CLEARED
            st.markdown('<div class="tier-box tier-green"><span>TIER 5: SAFETY OVERRIDE</span> <span>✅ CLEARED</span></div>', unsafe_allow_html=True)

    # --- RIGHT COL: ACTION CENTER ---
    with col_actions:
        st.markdown("### 💻 Command Terminal")
        
        # Button: Check Stats (Opens Terminal)
        if st.button("CHECK STATS (Open Terminal)"):
            st.session_state.terminal_open = not st.session_state.terminal_open

        if st.session_state.terminal_open:
            st.markdown("""
            <div class="terminal-window">
            user@aether-command:~$ ./check_cloud_stats --target=JED04<br>
            > CONNECTING TO BIGQUERY...<br>
            > FETCHING SENSOR DATA...<br>
            ----------------------------------------<br>
            HUMIDITY: 78% | TEMP: 18°C | WIND: 6 m/s<br>
            LWC: 0.09 g/m3| UPDRAFT: 3.2 m/s<br>
            ----------------------------------------<br>
            > STATUS: OPTIMAL FOR IONIZATION.<br>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        
        # ACTION LOGIC
        if not st.session_state.safety_scan_complete:
            # Button: Send Drone (Triggers Scan)
            if st.button("🚀 INITIATE SAFETY SCAN (TIER 5)", type="primary"):
                with st.status("Deploying Scout Drone...", expanded=True) as status:
                    st.write("Checking Restricted Airspace (GACA Database)...")
                    time.sleep(1)
                    st.write("Scanning for Lightning (<50km)...")
                    time.sleep(1)
                    st.write("Verifying Topography (Earth Engine)...")
                    time.sleep(1)
                    status.update(label="Safety Checks Passed", state="complete")
                
                st.session_state.safety_scan_complete = True
                st.rerun()
        else:
            # If Tier 5 is Green -> Show Launch
            st.success("ALL SYSTEMS GO. READY FOR IONIZATION.")
            
            # Formations (From TDD)
            formation = st.selectbox("Select Swarm Formation:", ["Formation A (Wide)", "Formation B (Focused)", "Formation C (Precision)"])
            
            if st.button("CONFIRM LAUNCH & GENERATE REPORT"):
                st.session_state.mission_active = True
                
                # Call Gemini for the report (if Key exists)
                if api_key:
                    with st.spinner("Gemini is writing mission report..."):
                        report = generate_ai_report(api_key, f"Target: JED04, Humidity: 78%, Tier 5: Cleared, Formation: {formation}")
                        st.session_state.report_text = report
                
                st.rerun()

# ==========================================
# PAGE 4: MISSION ACTIVE (The "Success" View)
# ==========================================
def render_mission():
    st.balloons()
    st.title("🛸 MISSION ACTIVE")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🛰️ Live Telemetry")
        # Placeholder for Drone Formation Visual
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Big_Data_Prob.svg/1200px-Big_Data_Prob.svg.png", caption="Swarm Formation Active", use_column_width=True)
    
    with col2:
        st.markdown("### 📝 AI Mission Report")
        if 'report_text' in st.session_state:
            st.info(st.session_state.report_text)
        else:
            st.info("Mission ID: #A99-JED\nStatus: Ionization Emitters Active\nEst. Yield: +23%")
            
        if st.button("RETURN TO DASHBOARD"):
            st.session_state.page = 'dashboard'
            st.session_state.safety_scan_complete = False # Reset
            st.session_state.mission_active = False
            st.rerun()

# ==========================================
# MAIN ROUTER
# ==========================================
if st.session_state.page == 'login':
    render_login()
elif st.session_state.page == 'dashboard':
    render_dashboard()
elif st.session_state.page == 'analysis':
    if st.session_state.mission_active:
        render_mission()
    else:
        render_analysis()
