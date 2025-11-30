import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import datetime
import math
import json

# --- CONFIG ---
st.set_page_config(page_title="VisionRain | Kingdom Commander", layout="wide", page_icon="⛈️")

# --- SAUDI BOUNDS ---
LAT_N, LAT_S = 32, 16
LON_W, LON_E = 34, 56

# --- LOAD ERA5 DATA ---
@st.cache_data
def load_nc_data(file_path):
    ds = xr.open_dataset(file_path)
    # Subset for Saudi Arabia
    saudi_ds = ds.sel(latitude=slice(LAT_N, LAT_S), longitude=slice(LON_W, LON_E))
    return saudi_ds

ds = load_nc_data("b5710c0835b1558c7a5002809513f1a5.nc")

# --- DEFINE SECTORS (same as original, can adjust granularity) ---
SAUDI_SECTORS = {
    "Jeddah": [21.5433, 39.1728],
    "Abha": [18.2164, 42.5053],
    "Riyadh": [24.7136, 46.6753],
    "Dammam": [26.4207, 50.0888],
    "Tabuk": [28.3835, 36.5662]
}

# --- EXTRACT METRICS FUNCTION ---
def extract_metrics(ds, lat, lon):
    # Find nearest grid point
    lat_idx = np.abs(ds.latitude - lat).argmin().item()
    lon_idx = np.abs(ds.longitude - lon).argmin().item()
    
    metrics = {
        "prob": float(ds['cloud_probability'].isel(latitude=lat_idx, longitude=lon_idx).values),
        "rad": float(ds['cloud_radius'].isel(latitude=lat_idx, longitude=lon_idx).values),
        "temp": float(ds['temperature'].isel(latitude=lat_idx, longitude=lon_idx).values),
        "rh": float(ds['relative_humidity'].isel(latitude=lat_idx, longitude=lon_idx).values),
        "phase": int(ds['phase'].isel(latitude=lat_idx, longitude=lon_idx).values),
        "press": float(ds['pressure'].isel(latitude=lat_idx, longitude=lon_idx).values),
    }
    
    # Status based on Master Table
    if metrics['prob']>60 and metrics['rad']<14 and metrics['phase']==1:
        metrics['status']="SEEDABLE"
    elif metrics['prob']>40:
        metrics['status']="MONITOR"
    else:
        metrics['status']="UNSUITABLE"
    
    return metrics

# --- RUN SCAN ---
@st.cache_data
def kingdom_scan(ds):
    results = {}
    for region, coords in SAUDI_SECTORS.items():
        results[region] = extract_metrics(ds, coords[0], coords[1])
    return results

st.session_state.all_sector_data = kingdom_scan(ds)

# --- MISSION MAP ---
def mission_map(sector_data):
    center_lat = np.mean([c[0] for c in SAUDI_SECTORS.values()])
    center_lon = np.mean([c[1] for c in SAUDI_SECTORS.values()])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB dark_matter")
    
    for region, coords in SAUDI_SECTORS.items():
        data = sector_data[region]
        color = "green" if data['prob']>60 else "orange" if data['prob']>40 else "gray"
        folium.CircleMarker(
            location=coords, radius=10, color=color, fill=True,
            fill_color=color, popup=f"{region}: {data['status']}"
        ).add_to(m)
    return m

st.subheader("Kingdom-Wide Cloud Map")
st_folium(mission_map(st.session_state.all_sector_data), height=450, use_container_width=True)

# --- MATRIX OF PLOTS ---
st.subheader("Metrics Matrix")
selected_region = st.selectbox("Select Region", list(SAUDI_SECTORS.keys()))
metrics = st.session_state.all_sector_data[selected_region]

fig, axs = plt.subplots(2, 3, figsize=(12,6))
sns.barplot(x=list(metrics.keys())[:3], y=list(metrics.values())[:3], ax=axs[0,0])
axs[0,0].set_title("Prob / Rad / Temp")
sns.barplot(x=list(metrics.keys())[3:], y=list(metrics.values())[3:], ax=axs[0,1])
axs[0,1].set_title("RH / Phase / Press")
axs[1,0].axis('off')
axs[1,1].axis('off')
axs[1,2].axis('off')
st.pyplot(fig)

# --- NUMERICAL TABLE ---
st.subheader("Saudi Arabia Cloud Metrics Table")
table_data = pd.DataFrame.from_dict(st.session_state.all_sector_data, orient='index')
st.dataframe(table_data)
