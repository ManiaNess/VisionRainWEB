import streamlit as st
import google.generativeai as genai
import ee
import geemap.foliumap as geemap
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from io import BytesIO
from scipy.ndimage import gaussian_filter
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {background-color: #050505;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    .stMetric {background-color: #111; border: 1px solid #333; border-radius: 8px;}
    .status-badge {padding: 5px; border-radius: 5px; font-weight: bold;}
    .badge-ok {background-color: #00ff80; color: black;}
    .badge-err {background-color: #ff0055; color: white;}
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION DEBUGGER ---
GEE_ACTIVE = False
AUTH_ERROR = "Unknown Error"
PROJECT_ID = 'ee-karmaakabane701' # Matches your JSON

try:
    # Check if the secret exists
    if "earth_engine" not in st.secrets:
        raise Exception("Secrets file missing [earth_engine] section.")
    
    if "service_account" not in st.secrets["earth_engine"]:
        raise Exception("Secrets missing 'service_account' key inside [earth_engine].")

    # Load the JSON
    service_account_info = json.loads(st.secrets["earth_engine"]["service_account"])
    
    # Create credentials
    credentials = ee.ServiceAccountCredentials(
        email=service_account_info['client_email'],
        key_data=service_account_info['private_key'],
        project=PROJECT_ID
    )
    
    # Initialize
    ee.Initialize(credentials=credentials, project=PROJECT_ID)
    GEE_ACTIVE = True

except Exception as e:
    GEE_ACTIVE = False
    AUTH_ERROR = str(e)

# --- UI ---
st.title("VisionRain | Kingdom Commander")

if GEE_ACTIVE:
    st.markdown(f'<span class="status-badge badge-ok">✅ ONLINE: {PROJECT_ID}</span>', unsafe_allow_html=True)
else:
    st.markdown(f'<span class="status-badge badge-err">❌ OFFLINE: {AUTH_ERROR}</span>', unsafe_allow_html=True)
    st.error(f"🛑 CRITICAL AUTH ERROR: {AUTH_ERROR}")
    st.info("Check the Streamlit Secrets formatting (Step 3 below).")

# ... (Rest of app logic handles fallback) ...
