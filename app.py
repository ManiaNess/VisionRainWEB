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

# --- AUTHENTICATION (Service Account Method) ---
GEE_ACTIVE = False
PROJECT_ID = 'oceanic-craft-479120-c3'

try:
    if "earth_engine" in st.secrets:
        # Load the Service Account JSON from Secrets
        service_account_info = json.loads(st.secrets["earth_engine"]["service_account"])
        
        # Authenticate using the Service Account Credentials
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info['client_email'],
            key_data=service_account_info['private_key'],
            project=PROJECT_ID
        )
        ee.Initialize(credentials=credentials, project=PROJECT_ID)
        GEE_ACTIVE = True
    else:
        # Fallback for local testing (if you have the file locally)
        # You can set GOOGLE_APPLICATION_CREDENTIALS env var locally
        ee.Initialize(project=PROJECT_ID)
        GEE_ACTIVE = True
except Exception as e:
    GEE_ACTIVE = False
    AUTH_ERROR = str(e)

# --- DATA ENGINE ---
def get_data(lat, lon):
    if not GEE_ACTIVE:
        # Simulation Fallback
        return {"temp": 30, "prob": 50, "rad": 10, "phase": 1, "source": "SIMULATION (Auth Failed)"}

    try:
        point = ee.Geometry.Point([lon, lat])
        
        # REAL SATELLITE FETCH
        # ERA5: Temperature
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate('2023-01-01', '2024-01-01').first()
        # MODIS: Cloud Physics
        modis = ee.ImageCollection("MODIS/006/MOD06_L2").filterDate('2023-01-01', '2024-01-01').first()
        
        combined = era5.addBands(modis)
        # Reduce region to get numbers
        val = combined.reduceRegion(ee.Reducer.mean(), point, 5000).getInfo()
        
        # Parse Results
        temp = (val.get('temperature_2m', 300) - 273.15)
        rad = val.get('Cloud_Effective_Radius', 0) or 0
        
        # Physics Logic
        prob = 85 if rad > 0 else 10
        if temp < 0 and temp > -15: prob += 20
        
        return {
            "temp": temp,
            "prob": min(prob, 100),
            "rad": rad,
            "phase": 1 if temp > -10 else 2,
            "source": "SATELLITE (Active)"
        }
    except Exception as e:
        return {"temp": 0, "prob": 0, "rad": 0, "phase": 0, "source": f"ERROR: {str(e)}"}

# --- TEXTURE GENERATOR ---
def make_texture(intensity, cmap):
    np.random.seed(int(intensity*100))
    data = gaussian_filter(np.random.rand(50,50), sigma=3) * intensity
    fig, ax = plt.subplots(figsize=(2,1.5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.imshow(data, cmap=cmap)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor='#0a0a0a', bbox_inches='tight', pad_inches=0)
    return Image.open(buf)

# --- UI ---
st.title("VisionRain | Kingdom Commander")

if GEE_ACTIVE:
    st.markdown(f'<span class="status-badge badge-ok">✅ GEE CONNECTED: {PROJECT_ID}</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-badge badge-err">⚠️ OFFLINE (Check Secrets)</span>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    m = geemap.Map(center=[24.7, 46.6], zoom=5, basemap="CartoDB.DarkMatter")
    m.to_streamlit(height=400)

with col2:
    if st.button("Scan Riyadh (Real Data)"):
        with st.spinner("Satellite Handshake..."):
            d = get_data(24.7, 46.6)
            st.metric("Source", d['source'])
            st.metric("Temp", f"{d['temp']:.1f}°C")
            st.metric("Prob", f"{d['prob']:.0f}%")
            st.image(make_texture(d['prob']/100, "Blues"), caption="Cloud Structure")
