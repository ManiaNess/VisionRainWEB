import requests
from PIL import Image
from io import BytesIO
import os

# --- CONFIGURATION ---
# Jeddah Coordinates
LAT = 21.5433
LON = 39.1728
SAVE_DIR = "generated_images"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# --- THE "GENERATOR" FUNCTION (Same as in your app) ---
def generate_scientific_image(layer_type, date_str, lat, lon):
    print(f"⚙️ Generating {layer_type} image for {date_str}...")
    
    bbox = f"{lon-10},{lat-10},{lon+10},{lat+10}"
    
    # NASA Layer IDs
    layer_map = {
        "Satellite_Visual": "Meteosat_MSG_SEVIRI_Band03_Visible",
        "Radar_Precipitation": "GPM_3IMERGHH_06_Precipitation",
        "Thermal_LandTemp": "MODIS_Terra_Land_Surface_Temperature_Day",
    }
    
    selected_layer = layer_map.get(layer_type)
    
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": selected_layer,
        "STYLES": "", "FORMAT": "image/jpeg", "CRS": "EPSG:4326",
        "BBOX": bbox, "WIDTH": "1024", "HEIGHT": "1024", # High Res
        "TIME": date_str
    }
    
    try:
        # Mimic browser to avoid blocking
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            # Save image to disk
            filename = f"{SAVE_DIR}/{date_str}_{layer_type}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ SUCCESS: Saved to {filename}")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
    return False

# --- MAIN AUTOMATION LOOP ---
# Add any dates you want to generate images for
target_dates = [
    "2022-11-24", # Jeddah Floods
    "2023-01-01", # New Year
    "2023-08-15", # Summer Day
    "2024-05-10", # Recent date
]

print("🚀 Starting Automatic Image Generation...\n")

for date in target_dates:
    # Generate a set of images for each date
    generate_scientific_image("Satellite_Visual", date, LAT, LON)
    generate_scientific_image("Radar_Precipitation", date, LAT, LON)
    # Add a small delay to be nice to NASA servers
    import time
    time.sleep(1) 

print("\n✨ Generation Complete! Check the 'generated_images' folder.")
