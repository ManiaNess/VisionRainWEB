import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import time
import random
from io import BytesIO
import folium
from streamlit_folium import st_folium
from scipy.ndimage import gaussian_filter
import ee  # Google Earth Engine
import json

# --- FIREBASE / FIRESTORE IMPORTS (SAFE MODE) ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- GLOBAL STYLES ---
st.markdown("""
    <style>
    .stApp {background-color: #0a0a0a;}
    h1 {color: #00e5ff; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #e0e0e0;}
    /* Metrics */
    .stMetric {background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;}
    .stMetric label {color: #888;}
    
    /* Pitch Box */
    .pitch-box {background: linear-gradient(145deg, #1e1e1e, #252525); padding: 25px; border-radius: 15px; border-left: 6px solid #00e5ff; margin-bottom: 20px;}
    
    /* Cloud Status Badge */
    .cloud-badge {
        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;
        background-color: #1a1a1a; border: 1px solid #444; color: #ccc;
        margin-bottom: 5px; display: inline-block;
    }
    .status-ok {color: #00ff80;}
    .status-warn {color: #ffaa00;}
    
    /* Map Container */
    iframe {border-radius: 10px; border: 1px solid #444;}
    </style>
    """, unsafe_allow_html=True)

# --- SAUDI SECTOR CONFIGURATION ---
SAUDI_SECTORS = {
    "Jeddah (Red Sea Coast)": {
        "coords": [21.5433, 39.1728], 
        "bias_prob": 0, "bias_temp": 0, "humidity_base": 60
    },
    "Abha (Asir Mountains)": {
        "coords": [18.2164, 42.5053], 
        "bias_prob": 30, "bias_temp": -10, "humidity_base": 70
    },
    "Riyadh (Central Arid)": {
        "coords": [24.7136, 46.6753], 
        "bias_prob": -40, "bias_temp": 5, "humidity_base": 20
    },
    "Dammam (Gulf Coast)": {
        "coords": [26.4207, 50.0888], 
        "bias_prob": -10, "bias_temp": 2, "humidity_base": 65
    },
    "Tabuk (Northern Region)": {
        "coords": [28.3835, 36.5662], 
        "bias_prob": 10, "bias_temp": -5, "humidity_base": 35
    }
}

# --- GOOGLE CLOUD ARCHITECTURE SIMULATION ---
class BigQueryClient:
    """Simulates pushing logs to Google BigQuery."""
    def insert_rows(self, dataset, table, rows):
        time.sleep(0.1) 
        return True

class CloudStorageClient:
    """Simulates fetching data from GCS Buckets."""
    def fetch_satellite_data(self, region):
        time.sleep(0.2)
        return True

bq_client = BigQueryClient()
gcs_client = CloudStorageClient()

# --- FIRESTORE SETUP ---
if "firestore_db" not in st.session_state:
    st.session_state.firestore_db = []

def init_firebase():
    if not FIREBASE_AVAILABLE: return None
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                initialize_app(cred)
            else:
                return None
        return firestore.client()
    except Exception: return None

db = init_firebase()

def save_mission_log(region, stats, decision, reasoning):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "region": region,
        "stats": str(stats),
        "decision": decision,
        "reasoning": reasoning,
        "engine": "VertexAI/Gemini-2.5-Flash"
    }
    st.session_state.firestore_db.append(entry)
    if db:
        try: db.collection("mission_logs").add(entry)
        except: pass
    bq_client.insert_rows("visionrain_logs", "mission_audit", [entry])

def get_mission_logs():
    return pd.DataFrame(st.session_state.firestore_db)

# --- EARTH ENGINE INITIALIZATION ---
@st.cache_resource
def init_ee():
    try:
        service_account_info = json.loads(st.secrets["earth_engine"]["service_account"])
        credentials = ee.ServiceAccountCredentials(
            service_account_info["client_email"],
            ee.ServiceAccountCredentials.from_p12_keyfile_buffer(
                service_account_info["client_email"],
                service_account_info["private_key"].encode(),
                private_key_id=service_account_info["private_key_id"]
            )
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Earth Engine initialization failed: {e}")
        return False

ee_initialized = init_ee()

# --- SCIENTIFIC DATA ENGINE USING GEE ---
@st.cache_data(ttl=3600)
def fetch_gee_data_for_region(region_name):
    if not ee_initialized:
        return None
    coords = SAUDI_SECTORS[region_name]['coords']
    lat, lon = coords
    bbox = ee.Geometry.Rectangle([lon-2.5, lat-2.5, lon+2.5, lat+2.5])
    point = ee.Geometry.Point([lon, lat])
    date = ee.Date(datetime.datetime.now() - datetime.timedelta(days=1))
    
    modis_collection = ee.ImageCollection('MODIS/061/MOD08_D3').filterDate(date, date.advance(1, 'day'))
    modis = modis_collection.first()
    era5_collection = ee.ImageCollection('ECMWF/ERA5/DAILY').filterDate(date, date.advance(1, 'day'))
    era5 = era5_collection.first()
    
    prob_val = modis.select('Cloud_Fraction_Day').reduceRegion(ee.Reducer.mean(), point, 1000).get('Cloud_Fraction_Day').getInfo()
    press_val = modis.select('Cloud_Top_Pressure_Day').reduceRegion(ee.Reducer.mean(), point, 1000).get('Cloud_Top_Pressure_Day').getInfo()
    rad_val = modis.select('Cloud_Effective_Radius_Liquid_Mean').reduceRegion(ee.Reducer.mean(), point, 1000).get('Cloud_Effective_Radius_Liquid_Mean').getInfo()
    opt_val = modis.select('Cloud_Optical_Thickness_Liquid_Mean').reduceRegion(ee.Reducer.mean(), point, 1000).get('Cloud_Optical_Thickness_Liquid_Mean').getInfo()
    phase_val = 1 if rad_val and rad_val > 0 else 2
    
    temp_val = era5.select('mean_2m_air_temperature').reduceRegion(ee.Reducer.mean(), point, 1000).get('mean_2m_air_temperature').getInfo()
    rh_val = era5.select('total_column_water_vapour').reduceRegion(ee.Reducer.mean(), point, 1000).get('total_column_water_vapour').getInfo()
    w_val = era5.select('v_component_of_wind_10m').reduceRegion(ee.Reducer.mean(), point, 1000).get('v_component_of_wind_10m').getInfo()
    lwc_val = rh_val
    iwc_val = lwc_val * 0.3 if lwc_val else 0
    
    data = {
        'prob': prob_val or 0,
        'press': press_val or 950,
        'rad': rad_val or 0,
        'opt': opt_val or 0,
        'phase': phase_val,
        'temp': temp_val or 25,
        'rh': rh_val or 50,
        'w': w_val or 0,
        'lwc': lwc_val or 0,
        'iwc': iwc_val
    }
    
    profile = SAUDI_SECTORS[region_name]
    data['prob'] += profile['bias_prob']
    data['temp'] += profile['bias_temp']
    data['rh'] = profile['humidity_base'] + random.uniform(-10, 10)
    data['prob'] = max(0.0, min(100.0, data['prob']))
    data['rh'] = max(5.0, min(100.0, data['rh']))
    
    if data['prob'] > 60 and data['rad'] < 14 and data['phase'] == 1:
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 40:
        data['status'] = "MONITORING"
    else:
        data['status'] = "UNSUITABLE"
    
    return data

@st.cache_data(ttl=3600)
def fetch_gee_image_for_metric(region_name, metric_name):
    if not ee_initialized:
        return np.random.rand(100, 100)
    
    coords = SAUDI_SECTORS[region_name]['coords']
    lat, lon = coords
    bbox = ee.Geometry.Rectangle([lon-2.5, lat-2.5, lon+2.5, lat+2.5])
    date = ee.Date(datetime.datetime.now() - datetime.timedelta(days=1))
    
    if metric_name in ['prob', 'press', 'rad', 'opt']:
        collection = ee.ImageCollection('MODIS/061/MOD08_D3').filterDate(date, date.advance(1, 'day'))
        img = collection.first()
        band_map = {'prob': 'Cloud_Fraction_Day', 'press': 'Cloud_Top_Pressure_Day', 'rad': 'Cloud_Effective_Radius_Liquid_Mean', 'opt': 'Cloud_Optical_Thickness_Liquid_Mean'}
        band = band_map[metric_name]
    elif metric_name == 'phase':
        modis = ee.ImageCollection('MODIS/061/MOD08_D3').filterDate(date, date.advance(1, 'day')).first()
        img = ee.Image(1).where(modis.select('Cloud_Effective_Radius_Ice_Mean').gt(0), 2)
        band = 'constant'
    else:
        collection = ee.ImageCollection('ECMWF/ERA5/DAILY').filterDate(date, date.advance(1, 'day'))
        img = collection.first()
        band_map = {'lwc': 'total_column_water_vapour', 'iwc': 'total_column_water_vapour', 'rh': 'total_column_water_vapour', 'w': 'v_component_of_wind_10m', 'temp': 'mean_2m_air_temperature'}
        band = band_map[metric_name]
        if metric_name == 'iwc':
            img = img.select(band).multiply(0.3)
    
    img_clipped = img.select(band).clip(bbox)
    try:
        array = img_clipped.sampleRectangle(region=bbox, defaultValue=0).get(band).getInfo()
        return np.array(array)
    except:
        return np.random.rand(100, 100)

def generate_cloud_texture_from_gee(region_name, metric_name, intensity=1.0):
    array = fetch_gee_image_for_metric(region_name, metric_name)
    smooth = gaussian_filter(array, sigma=5.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min()) if smooth.max() > smooth.min() else smooth
    return smooth * intensity

def run_kingdom_wide_scan():
    if not ee_initialized:
        st.error("GEE not initialized. Using simulated data.")
        return {sector: scan_single_sector(sector) for sector in SAUDI_SECTORS}
    
    with st.spinner("Fetching data from Google Earth Engine..."):
        results = {}
        for sector in SAUDI_SECTORS:
            results[sector] = fetch_gee_data_for_region(sector)
    return results

def scan_single_sector(sector_name):
    profile = SAUDI_SECTORS[sector_name]
    conditions = [
        {"prob": 85.0, "press": 650, "rad": 12.5, "opt": 15.0, "lwc": 0.005, "rh": 80, "temp": -8.0, "w": 2.5, "phase": 1}, 
        {"prob": 5.0, "press": 950, "rad": 0.0, "opt": 0.5, "lwc": 0.000, "rh": 20, "temp": 28.0, "w": 0.1, "phase": 0},
        {"prob": 70.0, "press": 350, "rad": 25.0, "opt": 5.0, "lwc": 0.001, "rh": 60, "temp": -35.0, "w": 0.5, "phase": 2},
    ]
    data = random.choice(conditions).copy()
    data['prob'] += profile['bias_prob']
    data['rh'] = profile['humidity_base'] + random.uniform(-10, 10)
    data['temp'] += profile['bias_temp']
    data['prob'] += random.uniform(-5, 5)
    data['prob'] = max(0.0, min(100.0, data['prob'])) 
    data['rh'] = max(5.0, min(100.0, data['rh']))
    if data['prob'] > 60 and data['rad'] < 14 and data['phase'] == 1:
        data['status'] = "SEEDABLE TARGET"
    elif data['prob'] > 40:
        data['status'] = "MONITORING"
    else:
        data['status'] = "UNSUITABLE"
    return data

# --- VISUALIZATION ENGINE (2x5 Matrix) ---
def plot_scientific_matrix(data_points, region_name):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.patch.set_facecolor('#0e1117')
    
    plots = [
        {"ax": axes[0,0], "title": "Cloud Probability (%)", "cmap": "Blues", "data": generate_cloud_texture_from_gee(region_name, 'prob', data_points['prob']), "vmax": 100},
        {"ax": axes[0,1], "title": "Cloud Top Pressure (hPa)", "cmap": "gray_r", "data": generate_cloud_texture_from_gee(region_name, 'press', data_points['press']), "vmax": 1000},
        {"ax": axes[0,2], "title": "Effective Radius (µm)", "cmap": "viridis", "data": generate_cloud_texture_from_gee(region_name, 'rad', data_points['rad']), "vmax": 30},
        {"ax": axes[0,3], "title": "Optical Depth", "cmap": "magma", "data": generate_cloud_texture_from_gee(region_name, 'opt', data_points['opt']), "vmax": 50},
        {"ax": axes[0,4], "title": "Phase (0=Clr,1=Liq,2=Ice)", "cmap": "cool", "data": generate_cloud_texture_from_gee(region_name, 'phase', data_points['phase']), "vmax": 2},
        {"ax": axes[1,0], "title": "Liquid Water (kg/m³)", "cmap": "Blues", "data": generate_cloud_texture_from_gee(region_name, 'lwc', data_points['lwc']), "vmax": 0.01},
        {"ax": axes[1,1], "title": "Ice Water Content", "cmap": "PuBu",
