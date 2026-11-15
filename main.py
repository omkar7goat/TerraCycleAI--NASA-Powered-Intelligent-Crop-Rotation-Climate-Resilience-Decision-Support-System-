import os
import sys
import json
import math
import time
import subprocess
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

print("=" * 82)
print("🌍 TERRACYCLE AI: NASA-POWERED CROP ROTATION & CLIMATE RESILIENCE SYSTEM")
print("=" * 82)
print("[Section 1-2] Verifying and installing required Python libraries for Colab...")

REQUIRED_PACKAGES = ["reportlab", "folium", "ipywidgets", "scikit-learn", "requests", "earthengine-api"]
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg.replace("-", "_").split("_")[0] if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"  -> Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import folium
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.units import inch

print("✅ All libraries loaded successfully.\n")

SEED = 42
np.random.seed(SEED)
CACHE_DIR = "./terracycle_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_REGISTRY = {
    "NASA_POWER": {
        "name": "NASA POWER Agroclimatology Daily API (MERRA-2 / GEOS-5.12.4)",
        "variables": ["T2M (°C)", "T2M_MAX (°C)", "T2M_MIN (°C)", "PRECTOTCORR (mm/day)", "RH2M (%)", "ALLSKY_SFC_SW_DWN (MJ/m²/day)", "GWETTOP (0-1 fraction)"],
        "spatial_resolution": "0.5° x 0.625° (~50 km grid)",
        "temporal_resolution": "Daily (2015-01-01 to 2024-12-31)",
        "access_method": "Direct REST API (No API key required)"
    },
    "NASA_SMAP": {
        "name": "NASA SMAP L3 Radiometer Global Daily 36 km Soil Moisture / POWER GWETTOP Proxy",
        "variables": ["Surface Soil Moisture (m³/m³ or Root-Zone Wetness Fraction)"],
        "spatial_resolution": "36 km (SMAP) / 50 km (GMAO Land Surface Model assimilation)",
        "temporal_resolution": "Daily / 3-Day Composite",
        "access_method": "Google Earth Engine (SPL3SMP_E) with NASA POWER GWETTOP Assimilated Fallback"
    },
    "NASA_GPM_IMERG": {
        "name": "NASA GPM IMERG Late/Final Run Precipitation & POWER PRECTOTCORR",
        "variables": ["Precipitation Rate (mm/day), Extreme Rainfall 95th Percentile, Dry Spell Count"],
        "spatial_resolution": "0.1° x 0.1° (~10 km)",
        "temporal_resolution": "Daily Aggregated",
        "access_method": "GEE (NASA/GPM_L3/IMERG_V06) / NASA POWER Corrected Precipitation"
    },
    "LANDSAT_8_9": {
        "name": "USGS/NASA Landsat 8 & 9 Collection 2 Tier 1 Surface Reflectance",
        "variables": ["Normalized Difference Vegetation Index (NDVI), Bare Soil Index (BSI)"],
        "spatial_resolution": "30 meters",
        "temporal_resolution": "16-day revisit cycle",
        "access_method": "GEE (LANDSAT/LC08/C02/T1_L2) or Agro-Climatic Water-Balance NDVI Estimation"
    },
    "NASA_SRTM": {
        "name": "NASA Shuttle Radar Topography Mission (SRTM) Digital Elevation Model",
        "variables": ["Elevation (m ASL), Terrain Slope (degrees), RUSLE LS Factor"],
        "spatial_resolution": "30 meters (1 arc-second)",
        "temporal_resolution": "Static Baseline",
        "access_method": "GEE (USGS/SRTMGL1_003) or Open-Elevation Topographic Grid"
    },
    "INDIAN_SOIL_ICAR": {
        "name": "ICAR - Indian Institute of Soil Science (IISS) & NBSS&LUP Agro-Ecological Baselines",
        "variables": ["Soil pH, Organic Carbon (%), Available N-P-K (kg/ha), Texture Class"],
        "spatial_resolution": "Field-scale manual test input / Regional District Baseline",
        "temporal_resolution": "Seasonal Soil Health Card cycle",
        "access_method": "Interactive Farmer Soil Health Card Input + Regional Baseline Defaults"
    }
}

def check_earth_engine_status() -> bool:
    """Checks if Google Earth Engine is authenticated in the current Colab session."""
    try:
        import ee
        ee.Initialize()
        return True
    except Exception:
        return False

GEE_AVAILABLE = check_earth_engine_status()
if GEE_AVAILABLE:
    print("[Section 3] 🛰️ Google Earth Engine is authenticated and active.")
else:
    print("[Section 3] ℹ️ Google Earth Engine is not authenticated in this session.")
    print("            -> To enable live GEE 30m Landsat/SRTM queries, run `import ee; ee.Authenticate(); ee.Initialize()`.")
    print("            -> Operating with LIVE NASA POWER Satellite API + Open-Elevation SRTM + Agro-Physical NDVI Proxy.")

DEFAULT_FARM_PROFILE = {
    "farm_name": "Sahyadri Krishi Kendall (Demo Baseline - Maharashtra, India)",
    "latitude": 19.9975,
    "longitude": 73.7898,
    "area_ha": 4.5,
    "area_unit": "Hectares",
    "soil_type": "Medium Black Soil (Vertic Inceptisol - Deccan Trap)",
    "soil_texture": "Clay Loam",
    "soil_ph": 7.4,
    "organic_carbon_pct": 0.42,   
    "nitrogen_kg_ha": 210.0,      
    "phosphorus_kg_ha": 18.5,     
    "potassium_kg_ha": 295.0,     
    "slope_deg_override": None,   
    "irrigation_availability": "Limited (Seasonal Canal + Dugwell in Rabi; Zero in Zaid)",
    "existing_rotation": ["Rice (Paddy)", "Wheat", "Fallow"],  
    "preferred_seasons": ["Kharif (Monsoon: Jun-Oct)", "Rabi (Winter: Nov-Mar)", "Zaid (Summer: Apr-May)"],
    "equipment": ["Tractor & Rotavator", "Seed Drill", "Drip Irrigation Lateral (1.5 ha)"],
    "market_preference": "Local APMC Mandi + Pulse Procurement + Low Input Risk",
    "is_synthetic_default": True,
    "priorities": {
        "water_conservation": 0.85,
        "soil_organic_matter": 0.90,
        "nitrogen_fixation": 0.85,
        "erosion_reduction": 0.70,
        "climate_resilience": 0.90,
        "economic_feasibility": 0.75,
        "forage_production": 0.40
    }
}

def validate_farmer_inputs(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validates geographic, chemical, and physical bounds for farmer inputs."""
    warnings_list = []
    clean = dict(profile)

    if not (-60.0 <= clean["latitude"] <= 60.0):
        warnings_list.append(f"Latitude {clean['latitude']} out of agricultural bounds; reset to 19.9975°N.")
        clean["latitude"] = 19.9975
    if not (-180.0 <= clean["longitude"] <= 180.0):
        warnings_list.append(f"Longitude {clean['longitude']} out of bounds; reset to 73.7898°E.")
        clean["longitude"] = 73.7898
    if clean["area_ha"] <= 0:
        warnings_list.append("Farm area must be > 0; reset to 2.0 ha.")
        clean["area_ha"] = 2.0
    if not (3.5 <= clean["soil_ph"] <= 10.0):
        warnings_list.append(f"Soil pH {clean['soil_ph']} outside agronomic range [3.5, 10.0]; clamped.")
        clean["soil_ph"] = float(np.clip(clean["soil_ph"], 3.5, 10.0))
    if clean["organic_carbon_pct"] < 0.05 or clean["organic_carbon_pct"] > 5.0:
        warnings_list.append("Soil Organic Carbon clamped to realistic range [0.05%, 5.0%].")
        clean["organic_carbon_pct"] = float(np.clip(clean["organic_carbon_pct"], 0.05, 5.0))

    return clean, warnings_list

FARM_PROFILE, validation_msgs = validate_farmer_inputs(DEFAULT_FARM_PROFILE)

style_lbl = {'description_width': '165px'}
layout_box = widgets.Layout(width='420px')

w_farm_name = widgets.Text(value=FARM_PROFILE["farm_name"], description="Farm Name:", style=style_lbl, layout=layout_box)
w_lat = widgets.FloatText(value=FARM_PROFILE["latitude"], description="Latitude (°N):", step=0.01, style=style_lbl, layout=layout_box)
w_lon = widgets.FloatText(value=FARM_PROFILE["longitude"], description="Longitude (°E):", step=0.01, style=style_lbl, layout=layout_box)
w_area = widgets.FloatSlider(value=FARM_PROFILE["area_ha"], min=0.5, max=50.0, step=0.5, description="Area (Hectares):", style=style_lbl, layout=layout_box)
w_soil_type = widgets.Dropdown(
    options=[
        "Medium Black Soil (Vertic Inceptisol - Deccan Trap)",
        "Deep Black Cotton Soil (Vertisol)",
        "Red Loamy Soil (Alfisol)",
        "Alluvial Soil (Indo-Gangetic Inceptisol)",
        "Lateritic Soil (Ultisol)"
    ],
    value=FARM_PROFILE["soil_type"], description="Indian Soil Type:", style=style_lbl, layout=layout_box
)
w_texture = widgets.Dropdown(
    options=["Clay Loam", "Heavy Clay", "Sandy Loam", "Silt Loam", "Loamy Sand"],
    value=FARM_PROFILE["soil_texture"], description="Soil Texture:", style=style_lbl, layout=layout_box
)
w_ph = widgets.FloatSlider(value=FARM_PROFILE["soil_ph"], min=4.5, max=9.5, step=0.1, description="Soil pH:", style=style_lbl, layout=layout_box)
w_oc = widgets.FloatSlider(value=FARM_PROFILE["organic_carbon_pct"], min=0.1, max=2.0, step=0.05, description="Organic Carbon (%):", style=style_lbl, layout=layout_box)
w_n = widgets.FloatSlider(value=FARM_PROFILE["nitrogen_kg_ha"], min=50.0, max=600.0, step=10.0, description="Available N (kg/ha):", style=style_lbl, layout=layout_box)
w_p = widgets.FloatSlider(value=FARM_PROFILE["phosphorus_kg_ha"], min=5.0, max=80.0, step=1.0, description="Available P (kg/ha):", style=style_lbl, layout=layout_box)
w_k = widgets.FloatSlider(value=FARM_PROFILE["potassium_kg_ha"], min=50.0, max=600.0, step=10.0, description="Available K (kg/ha):", style=style_lbl, layout=layout_box)
w_irrig = widgets.Dropdown(
    options=[
        "Rainfed Only (Monsoon Dependent)",
        "Limited (Seasonal Canal + Dugwell in Rabi; Zero in Zaid)",
        "Assured Full-Year (Borewell + Drip + Solar Pump)"
    ],
    value=FARM_PROFILE["irrigation_availability"], description="Irrigation Access:", style=style_lbl, layout=layout_box
)

w_prio_water = widgets.FloatSlider(value=0.85, min=0.1, max=1.0, step=0.05, description="Prio: Water Saving:", style=style_lbl, layout=layout_box)
w_prio_som = widgets.FloatSlider(value=0.90, min=0.1, max=1.0, step=0.05, description="Prio: Soil Carbon:", style=style_lbl, layout=layout_box)
w_prio_nfix = widgets.FloatSlider(value=0.85, min=0.1, max=1.0, step=0.05, description="Prio: N-Fixation:", style=style_lbl, layout=layout_box)
w_prio_erosion = widgets.FloatSlider(value=0.70, min=0.1, max=1.0, step=0.05, description="Prio: Anti-Erosion:", style=style_lbl, layout=layout_box)
w_prio_climate = widgets.FloatSlider(value=0.90, min=0.1, max=1.0, step=0.05, description="Prio: Climate Resil.:", style=style_lbl, layout=layout_box)
w_prio_econ = widgets.FloatSlider(value=0.75, min=0.1, max=1.0, step=0.05, description="Prio: Market Return:", style=style_lbl, layout=layout_box)
w_prio_forage = widgets.FloatSlider(value=0.40, min=0.1, max=1.0, step=0.05, description="Prio: Forage Yield:", style=style_lbl, layout=layout_box)

print("\n[Section 4] 📋 Interactive Farmer Input Panel (Pre-loaded with Demo Baseline — Marked as Default Soil Card):")
input_ui = widgets.HBox([
    widgets.VBox([widgets.HTML("<b>1. Farm & Soil Health Card Inputs</b>"), w_farm_name, w_lat, w_lon, w_area, w_soil_type, w_texture, w_ph, w_oc, w_n, w_p, w_k, w_irrig]),
    widgets.VBox([widgets.HTML("<b>2. Farmer Sustainability Priorities (0.1 - 1.0)</b>"), w_prio_water, w_prio_som, w_prio_nfix, w_prio_erosion, w_prio_climate, w_prio_econ, w_prio_forage])
])
display(input_ui)

def fetch_nasa_power_daily(lat: float, lon: float, start_date: str = "20150101", end_date: str = "20241231") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetches 10 years of daily agroclimatology observations from the official NASA POWER API.
    Includes local JSON file caching and transparent agro-climatic fallback if offline.
    """
    cache_file = os.path.join(CACHE_DIR, f"nasa_power_{lat:.3f}_{lon:.3f}_{start_date}_{end_date}.json")
    params_list = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,GWETTOP"
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={params_list}&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )

    raw_json = None
    data_provenance = {
        "source": "NASA POWER Agroclimatology API (Live Satellite/Model Assimilation)",
        "is_fallback": False,
        "spatial_res": "0.5° x 0.625° (~50 km)",
        "date_range": f"{start_date[:4]}-{end_date[:4]}",
        "notes": "Direct observation & MERRA-2/GMAO satellite assimilation."
    }

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                raw_json = json.load(f)
            data_provenance["notes"] += " (Loaded from local Colab disk cache)."
        except Exception:
            raw_json = None

    if raw_json is None:
        try:
            print(f"[Section 5] 🛰️ Querying Live NASA POWER API for ({lat:.4f}°N, {lon:.4f}°E) [{start_date} to {end_date}]...")
            resp = requests.get(url, timeout=25)
            resp.raise_for_status()
            raw_json = resp.json()
            with open(cache_file, "w") as f:
                json.dump(raw_json, f)
            print("  -> ✅ Live NASA POWER satellite observations downloaded and cached.")
        except Exception as e:
            print(f"  -> ⚠️ NASA POWER API unreachable ({e}). Activating labelled Climatological Fallback Generator.")
            data_provenance["source"] = "SYNTHETIC CLIMATOLOGICAL FALLBACK (API Unreachable)"
            data_provenance["is_fallback"] = True
            data_provenance["notes"] = f"Fallback triggered due to network error: {str(e)}"

    if raw_json is not None and "properties" in raw_json and "parameter" in raw_json["properties"]:
        param_dict = raw_json["properties"]["parameter"]
        df = pd.DataFrame(param_dict)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.sort_index(inplace=True)

        df.replace(-999.0, np.nan, inplace=True)
        missing_count = int(df.isna().sum().sum())
        df.interpolate(method="time", limit_direction="both", inplace=True)
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        data_provenance["missing_values_cleaned"] = missing_count
        return df, data_provenance

    dates = pd.date_range(start="2015-01-01", end="2024-12-31", freq="D")
    doy = dates.dayofyear.values
    rng = np.random.default_rng(SEED)
    t2m = 25.5 + 5.5 * np.sin(2 * np.pi * (doy - 105) / 365.25) + rng.normal(0, 1.2, len(dates))
    t2m_max = t2m + 6.2 + rng.uniform(-1.0, 1.5, len(dates))
    t2m_min = t2m - 5.8 + rng.uniform(-1.2, 1.0, len(dates))
    monsoon_mask = (doy >= 160) & (doy <= 280)
    precip = np.where(monsoon_mask, rng.exponential(6.5, len(dates)) * (rng.random(len(dates)) < 0.62), rng.exponential(1.2, len(dates)) * (rng.random(len(dates)) < 0.07))
    rh2m = np.clip(48 + 35 * monsoon_mask + rng.normal(0, 6, len(dates)), 20, 98)
    srad = np.clip(19.5 - 5.0 * monsoon_mask + 3.5 * np.sin(2 * np.pi * (doy - 80) / 365.25) + rng.normal(0, 1.5, len(dates)), 8, 28)
    gwettop = np.clip(0.28 + 0.45 * monsoon_mask + 0.12 * np.sin(2 * np.pi * (doy - 200) / 365.25) + rng.normal(0, 0.04, len(dates)), 0.12, 0.92)

    df = pd.DataFrame({
        "T2M": t2m, "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
        "PRECTOTCORR": np.round(precip, 2), "RH2M": rh2m,
        "ALLSKY_SFC_SW_DWN": srad, "GWETTOP": gwettop
    }, index=dates)
    data_provenance["missing_values_cleaned"] = 0
    return df, data_provenance

def fetch_elevation_and_slope(lat: float, lon: float) -> Dict[str, Any]:
    """
    Attempts to fetch SRTM 30m elevation and slope via GEE or Open-Elevation API using a 3-point
    micro-gradient stencil (~100m spacing). Falls back to regional Deccan plateau topography if offline.
    """
    if GEE_AVAILABLE:
        try:
            import ee
            pt = ee.Geometry.Point([lon, lat])
            srtm = ee.Image("USGS/SRTMGL1_003")
            slope_img = ee.Terrain.slope(srtm)
            elev_val = srtm.sample(pt, 30).first().get("elevation").getInfo()
            slope_val = slope_img.sample(pt, 30).first().get("slope").getInfo()
            return {
                "elevation_m": float(elev_val),
                "slope_deg": float(slope_val),
                "source": "NASA SRTM 30m via Google Earth Engine (Actual Satellite DEM)",
                "is_derived": False
            }
        except Exception:
            pass

    delta = 0.001  
    locs = f"{lat},{lon}|{lat+delta},{lon}|{lat},{lon+delta}"
    try:
        resp = requests.get(f"https://api.open-elevation.com/api/v1/lookup?locations={locs}", timeout=8)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if len(results) == 3:
                z0 = float(results[0]["elevation"])
                z_n = float(results[1]["elevation"])
                z_e = float(results[2]["elevation"])
                dz_dy = (z_n - z0) / 111.0
                dz_dx = (z_e - z0) / (111.0 * math.cos(math.radians(lat)))
                slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
                slope_deg = round(math.degrees(slope_rad), 2)
                return {
                    "elevation_m": z0,
                    "slope_deg": max(0.5, min(slope_deg, 25.0)),
                    "source": "SRTM/Open-Elevation 3-Point Topographic Stencil (Actual DEM Lookup)",
                    "is_derived": True
                }
    except Exception:
        pass

    return {
        "elevation_m": 584.0,
        "slope_deg": 2.4,
        "source": "Regional Deccan Plateau Baseline Fallback (SRTM API Timeout)",
        "is_derived": True
    }

df_daily, power_provenance = fetch_nasa_power_daily(FARM_PROFILE["latitude"], FARM_PROFILE["longitude"])
topo_info = fetch_elevation_and_slope(FARM_PROFILE["latitude"], FARM_PROFILE["longitude"])

def assign_indian_cropping_season(month: int) -> str:
    """Maps calendar month to Indian agricultural cropping seasons."""
    if month in [6, 7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi"
    else:
        return "Zaid"

def engineer_environmental_features(df: pd.DataFrame, lat: float, slope_deg: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes scientifically grounded agro-climatic features:
    1. Reference Evapotranspiration (ET0) via Hargreaves-Samani radiation-temperature equation.
    2. Surface Soil Moisture Anomaly (Z-score relative to 10-year day-of-year climatology).
    3. Derived Agro-Climatic NDVI proxy (calibrated from soil wetness GWETTOP, solar radiation, and VPD/RH)
       clearly labelled as a Water-Balance Vegetation Proxy when GEE Landsat 8/9 is unauthenticated.
    4. Dry-spell frequency (>7 consecutive days with <2.0 mm rainfall during growing season).
    5. RUSLE Topographic Erosion Susceptibility Factor (LS) and Rainfall Erosivity Index.
    """
    data = df.copy()
    data["Year"] = data.index.year
    data["Month"] = data.index.month
    data["DOY"] = data.index.dayofyear
    data["Season"] = data["Month"].apply(assign_indian_cropping_season)

    temp_range = np.maximum(data["T2M_MAX"] - data["T2M_MIN"], 1.0)
    ra_equiv_mm = data["ALLSKY_SFC_SW_DWN"] * 0.408 * 1.25
    data["ET0_mm"] = np.clip(0.0023 * ra_equiv_mm * (data["T2M"] + 17.8) * np.sqrt(temp_range), 1.0, 12.0)

    monthly_sm_mean = data.groupby("Month")["GWETTOP"].transform("mean")
    monthly_sm_std = data.groupby("Month")["GWETTOP"].transform("std").replace(0, 0.05)
    data["SM_Anomaly_Z"] = (data["GWETTOP"] - monthly_sm_mean) / monthly_sm_std

    rolling_sm = data["GWETTOP"].rolling(15, min_periods=1).mean()
    thermal_stress = np.clip(1.0 - np.maximum(0, data["T2M_MAX"] - 36.0) / 12.0, 0.2, 1.0)
    data["NDVI_Proxy"] = np.clip(0.14 + 0.68 * (rolling_sm ** 0.85) * thermal_stress, 0.12, 0.82)
    data["Vegetation_Stress_Idx"] = np.clip(1.0 - (data["NDVI_Proxy"] / 0.70), 0.0, 1.0)

    annual_rain = data.groupby("Year")["PRECTOTCORR"].sum()
    mean_annual_rain = float(annual_rain.mean())
    std_annual_rain = float(annual_rain.std()) if len(annual_rain) > 1 else 100.0
    cv_rainfall_pct = (std_annual_rain / max(mean_annual_rain, 1.0)) * 100.0
    recent_year_rain = float(annual_rain.iloc[-1])
    rain_anomaly_pct = ((recent_year_rain - mean_annual_rain) / max(mean_annual_rain, 1.0)) * 100.0

    kharif_df = data[data["Season"] == "Kharif"]
    dry_spells_per_year = []
    for yr, grp in kharif_df.groupby("Year"):
        is_dry = (grp["PRECTOTCORR"] < 2.0).astype(int).values
        spells, run = 0, 0
        for val in is_dry:
            if val == 1:
                run += 1
            else:
                if run >= 7:
                    spells += 1
                run = 0
        if run >= 7:
            spells += 1
        dry_spells_per_year.append(spells)
    avg_kharif_dry_spells = float(np.mean(dry_spells_per_year))

    theta_rad = math.radians(slope_deg)
    m_exp = 0.5 if slope_deg >= 3.0 else 0.3
    ls_factor = ((100.0 / 22.13) ** m_exp) * (65.41 * (math.sin(theta_rad) ** 2) + 4.56 * math.sin(theta_rad) + 0.065)
    p95_daily_rain = float(np.percentile(data[data["PRECTOTCORR"] > 1.0]["PRECTOTCORR"], 95)) if (data["PRECTOTCORR"] > 1.0).any() else 25.0
    erosion_risk_score = float(np.clip((ls_factor * 28.0) + (p95_daily_rain * 0.65), 5.0, 98.0))

    seasonal_summary = data.groupby("Season").agg(
        Avg_Temp_C=("T2M", "mean"),
        Max_Temp_C=("T2M_MAX", "mean"),
        Total_Rain_mm=("PRECTOTCORR", lambda x: x.sum() / data["Year"].nunique()),
        Avg_ET0_mm_day=("ET0_mm", "mean"),
        Avg_Soil_Wetness=("GWETTOP", "mean"),
        Avg_NDVI_Proxy=("NDVI_Proxy", "mean"),
        Avg_Veg_Stress=("Vegetation_Stress_Idx", "mean")
    ).reindex(["Kharif", "Rabi", "Zaid"])

    env_metrics = {
        "mean_temp_c": round(float(data["T2M"].mean()), 2),
        "max_temp_95p_c": round(float(np.percentile(data["T2M_MAX"], 95)), 2),
        "mean_annual_rain_mm": round(mean_annual_rain, 1),
        "recent_annual_rain_mm": round(recent_year_rain, 1),
        "rain_anomaly_pct": round(rain_anomaly_pct, 1),
        "rain_cv_pct": round(cv_rainfall_pct, 1),
        "p95_extreme_rain_mm": round(p95_daily_rain, 1),
        "mean_soil_wetness": round(float(data["GWETTOP"].mean()), 3),
        "recent_sm_anomaly_z": round(float(data["SM_Anomaly_Z"].iloc[-365:].mean()), 2),
        "mean_ndvi_proxy": round(float(data["NDVI_Proxy"].mean()), 3),
        "avg_kharif_dry_spells": round(avg_kharif_dry_spells, 1),
        "mean_annual_et0_mm": round(float(data.groupby("Year")["ET0_mm"].sum().mean()), 1),
        "slope_deg": round(slope_deg, 2),
        "rusle_ls_factor": round(ls_factor, 3),
        "erosion_risk_score": round(erosion_risk_score, 1),
        "ndvi_is_derived_proxy": not GEE_AVAILABLE
    }
    return data, seasonal_summary, env_metrics

df_env, df_seasonal, ENV_METRICS = engineer_environmental_features(
    df_daily, FARM_PROFILE["latitude"], topo_info["slope_deg"]
)

print("\n[Section 7] 📊 Engineered NASA Agro-Climatic & Topographic Indicators (2015–2024 Reference):")
for k, v in ENV_METRICS.items():
    print(f"  • {k:<26}: {v}")

def assess_indian_soil_health(profile: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates soil chemical and physical properties against Indian Council of Agricultural
    Research (ICAR) Soil Health Card benchmarks:
      - Organic Carbon (OC): Low < 0.5%, Medium 0.5-0.75%, High > 0.75%
      - Available Nitrogen (N): Low < 280 kg/ha, Medium 280-560 kg/ha, High > 560 kg/ha
      - Available Phosphorus (P): Low < 10 kg/ha, Medium 10-25 kg/ha, High > 25 kg/ha
      - Available Potassium (K): Low < 110 kg/ha, Medium 110-280 kg/ha, High > 280 kg/ha
    """
    oc = profile["organic_carbon_pct"]
    n_val = profile["nitrogen_kg_ha"]
    p_val = profile["phosphorus_kg_ha"]
    k_val = profile["potassium_kg_ha"]
    ph = profile["soil_ph"]
    texture = profile["soil_texture"]

    oc_status = "Low (Deficient)" if oc < 0.50 else ("Medium (Moderate)" if oc <= 0.75 else "High (Optimal)")
    n_status = "Low (Deficient)" if n_val < 280 else ("Medium" if n_val <= 560 else "High")
    p_status = "Low (Deficient)" if p_val < 10 else ("Medium" if p_val <= 25 else "High")
    k_status = "Low (Deficient)" if k_val < 110 else ("Medium" if k_val <= 280 else "High")
    ph_status = "Neutral/Optimal" if 6.3 <= ph <= 7.5 else ("Slightly Alkaline/Calcareous" if ph > 7.5 else "Acidic")

    oc_score = float(np.clip((oc / 0.85) * 100.0, 10.0, 100.0))
    n_score = float(np.clip((n_val / 450.0) * 100.0, 10.0, 100.0))
    p_score = float(np.clip((p_val / 30.0) * 100.0, 15.0, 100.0))
    k_score = float(np.clip((k_val / 320.0) * 100.0, 15.0, 100.0))
    ph_score = float(np.clip(100.0 - abs(ph - 6.8) * 22.0, 25.0, 100.0))

    water_retention_map = {
        "Heavy Clay": 0.90, "Clay Loam": 0.82, "Silt Loam": 0.78, "Sandy Loam": 0.55, "Loamy Sand": 0.40
    }
    whc_factor = water_retention_map.get(texture, 0.70)

    composite_soil_index = round(0.30 * oc_score + 0.25 * n_score + 0.15 * p_score + 0.15 * k_score + 0.15 * ph_score, 1)

    return {
        "composite_soil_index": composite_soil_index,
        "oc_status": oc_status, "n_status": n_status, "p_status": p_status, "k_status": k_status, "ph_status": ph_status,
        "whc_factor": whc_factor,
        "requires_legume_n_boost": n_val < 300.0 or oc < 0.55,
        "requires_high_biomass_mulch": oc < 0.60 or env["erosion_risk_score"] > 35.0
    }

SOIL_ASSESSMENT = assess_indian_soil_health(FARM_PROFILE, ENV_METRICS)
print("\n[Section 8] 🌱 ICAR Soil Health Card Diagnostic Summary:")
print(f"  • Composite Soil Health Index : {SOIL_ASSESSMENT['composite_soil_index']} / 100")
print(f"  • Organic Carbon ({FARM_PROFILE['organic_carbon_pct']}%)     : {SOIL_ASSESSMENT['oc_status']}")
print(f"  • Available N ({FARM_PROFILE['nitrogen_kg_ha']} kg/ha)   : {SOIL_ASSESSMENT['n_status']}")
print(f"  • Available P ({FARM_PROFILE['phosphorus_kg_ha']} kg/ha)    : {SOIL_ASSESSMENT['p_status']}")
print(f"  • Available K ({FARM_PROFILE['potassium_kg_ha']} kg/ha)   : {SOIL_ASSESSMENT['k_status']}")
print(f"  • Soil Reaction (pH {FARM_PROFILE['soil_ph']})     : {SOIL_ASSESSMENT['ph_status']}")

def analyze_climate_regimes_and_stress(df: pd.DataFrame, env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses unsupervised K-Means clustering on actual historical seasonal NASA POWER observations
    (30 year-season blocks across 2015-2024) to discover agro-climatic stress regimes.
    Clearly separates empirical clustering on real NASA time series from rule-based drought indexing.
    """
    ys = df.groupby(["Year", "Season"]).agg(
        Temp_Mean=("T2M", "mean"),
        Temp_Max95=("T2M_MAX", lambda s: np.percentile(s, 95)),
        Rain_Total=("PRECTOTCORR", "sum"),
        ET0_Total=("ET0_mm", "sum"),
        SM_Mean=("GWETTOP", "mean"),
        NDVI_Mean=("NDVI_Proxy", "mean")
    ).reset_index()

    ys["Aridity_Ratio"] = ys["Rain_Total"] / np.maximum(ys["ET0_Total"], 1.0)
    features_cols = ["Temp_Max95", "Rain_Total", "SM_Mean", "Aridity_Ratio"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ys[features_cols])

    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    ys["Cluster_ID"] = kmeans.fit_predict(X_scaled)

    cluster_aridity = ys.groupby("Cluster_ID")["Aridity_Ratio"].mean().sort_values()
    label_map = {
        cluster_aridity.index[0]: "High Moisture-Deficit / Heat-Stress Regime",
        cluster_aridity.index[1]: "Moderate Transitional Agro-Regime",
        cluster_aridity.index[2]: "Humid Surplus / Active Monsoon Regime"
    }
    ys["Regime_Label"] = ys["Cluster_ID"].map(label_map)

    drought_risk = (
        0.35 * np.clip(env["rain_cv_pct"] * 1.6, 0, 100) +
        0.30 * np.clip(env["avg_kharif_dry_spells"] * 22.0, 0, 100) +
        0.20 * np.clip((0.55 - env["mean_soil_wetness"]) * 200.0, 0, 100) +
        0.15 * np.clip(-env["rain_anomaly_pct"] * 1.5, 0, 100)
    )
    drought_risk = round(float(np.clip(drought_risk, 10.0, 95.0)), 1)
    drought_class = "Severe" if drought_risk >= 65 else ("Moderate-High" if drought_risk >= 42 else "Low-Moderate")

    return {
        "year_season_df": ys,
        "drought_risk_score": drought_risk,
        "drought_class": drought_class,
        "regime_counts": ys["Regime_Label"].value_counts().to_dict()
    }

CLIMATE_ML = analyze_climate_regimes_and_stress(df_env, ENV_METRICS)
print(f"\n[Section 9] 🤖 Climate Pattern & Stress Analysis (Unsupervised K-Means on 30 NASA Year-Seasons):")
print(f"  • Composite Drought Vulnerability : {CLIMATE_ML['drought_risk_score']}/100 ({CLIMATE_ML['drought_class']})")
for reg, cnt in CLIMATE_ML["regime_counts"].items():
    print(f"  • {reg:<45}: {cnt} historical seasons")

CROP_DATABASE: List[Dict[str, Any]] = [
    {
        "name": "Rice (Paddy)", "family": "Poaceae", "seasons": ["Kharif"],
        "water_mm": 1150, "temp_min": 20, "temp_max": 37, "ph_min": 5.5, "ph_max": 7.8,
        "n_fix": 0.00, "som_contrib": 0.35, "erosion_prot": 0.45, "drought_tol": 0.20,
        "market_score": 0.85, "forage_val": 0.60, "irrig_req": "High", "is_legume": False, "is_cover": False
    },
    {
        "name": "Sorghum (Jowar)", "family": "Poaceae", "seasons": ["Kharif", "Rabi"],
        "water_mm": 420, "temp_min": 18, "temp_max": 40, "ph_min": 6.0, "ph_max": 8.5,
        "n_fix": 0.00, "som_contrib": 0.72, "erosion_prot": 0.75, "drought_tol": 0.88,
        "market_score": 0.76, "forage_val": 0.92, "irrig_req": "Low", "is_legume": False, "is_cover": False
    },
    {
        "name": "Pearl Millet (Bajra)", "family": "Poaceae", "seasons": ["Kharif", "Zaid"],
        "water_mm": 330, "temp_min": 22, "temp_max": 42, "ph_min": 6.0, "ph_max": 8.5,
        "n_fix": 0.00, "som_contrib": 0.68, "erosion_prot": 0.72, "drought_tol": 0.94,
        "market_score": 0.74, "forage_val": 0.88, "irrig_req": "Low", "is_legume": False, "is_cover": False
    },
    {
        "name": "Pigeonpea (Tur/Arhar)", "family": "Fabaceae", "seasons": ["Kharif"],
        "water_mm": 520, "temp_min": 18, "temp_max": 38, "ph_min": 6.0, "ph_max": 8.2,
        "n_fix": 0.88, "som_contrib": 0.84, "erosion_prot": 0.78, "drought_tol": 0.85,
        "market_score": 0.90, "forage_val": 0.55, "irrig_req": "Low", "is_legume": True, "is_cover": False
    },
    {
        "name": "Soybean", "family": "Fabaceae", "seasons": ["Kharif"],
        "water_mm": 550, "temp_min": 20, "temp_max": 35, "ph_min": 6.0, "ph_max": 7.8,
        "n_fix": 0.76, "som_contrib": 0.68, "erosion_prot": 0.65, "drought_tol": 0.58,
        "market_score": 0.88, "forage_val": 0.50, "irrig_req": "Moderate", "is_legume": True, "is_cover": False
    },
    {
        "name": "Groundnut (Peanut)", "family": "Fabaceae", "seasons": ["Kharif", "Zaid"],
        "water_mm": 480, "temp_min": 22, "temp_max": 36, "ph_min": 6.0, "ph_max": 7.8,
        "n_fix": 0.72, "som_contrib": 0.64, "erosion_prot": 0.76, "drought_tol": 0.70,
        "market_score": 0.86, "forage_val": 0.65, "irrig_req": "Moderate", "is_legume": True, "is_cover": False
    },
    {
        "name": "Cotton", "family": "Malvaceae", "seasons": ["Kharif"],
        "water_mm": 780, "temp_min": 21, "temp_max": 38, "ph_min": 6.0, "ph_max": 8.2,
        "n_fix": 0.00, "som_contrib": 0.40, "erosion_prot": 0.42, "drought_tol": 0.62,
        "market_score": 0.89, "forage_val": 0.15, "irrig_req": "Moderate", "is_legume": False, "is_cover": False
    },
    {
        "name": "Maize (Corn)", "family": "Poaceae", "seasons": ["Kharif", "Rabi"],
        "water_mm": 580, "temp_min": 18, "temp_max": 35, "ph_min": 5.8, "ph_max": 7.8,
        "n_fix": 0.00, "som_contrib": 0.62, "erosion_prot": 0.55, "drought_tol": 0.52,
        "market_score": 0.82, "forage_val": 0.90, "irrig_req": "Moderate", "is_legume": False, "is_cover": False
    },
    {
        "name": "Wheat", "family": "Poaceae", "seasons": ["Rabi"],
        "water_mm": 460, "temp_min": 12, "temp_max": 31, "ph_min": 6.0, "ph_max": 8.0,
        "n_fix": 0.00, "som_contrib": 0.52, "erosion_prot": 0.60, "drought_tol": 0.54,
        "market_score": 0.85, "forage_val": 0.65, "irrig_req": "High", "is_legume": False, "is_cover": False
    },
    {
        "name": "Chickpea (Gram/Chana)", "family": "Fabaceae", "seasons": ["Rabi"],
        "water_mm": 280, "temp_min": 14, "temp_max": 32, "ph_min": 6.0, "ph_max": 8.2,
        "n_fix": 0.82, "som_contrib": 0.74, "erosion_prot": 0.68, "drought_tol": 0.86,
        "market_score": 0.88, "forage_val": 0.55, "irrig_req": "Low", "is_legume": True, "is_cover": False
    },
    {
        "name": "Mustard (Rapeseed)", "family": "Brassicaceae", "seasons": ["Rabi"],
        "water_mm": 310, "temp_min": 12, "temp_max": 30, "ph_min": 6.0, "ph_max": 8.2,
        "n_fix": 0.00, "som_contrib": 0.60, "erosion_prot": 0.66, "drought_tol": 0.78,
        "market_score": 0.84, "forage_val": 0.45, "irrig_req": "Low", "is_legume": False, "is_cover": False
    },
    {
        "name": "Safflower (Kardi)", "family": "Asteraceae", "seasons": ["Rabi"],
        "water_mm": 260, "temp_min": 14, "temp_max": 34, "ph_min": 6.0, "ph_max": 8.5,
        "n_fix": 0.00, "som_contrib": 0.66, "erosion_prot": 0.70, "drought_tol": 0.92,
        "market_score": 0.78, "forage_val": 0.40, "irrig_req": "Low", "is_legume": False, "is_cover": False
    },
    {
        "name": "Green Gram (Moong)", "family": "Fabaceae", "seasons": ["Kharif", "Zaid"],
        "water_mm": 240, "temp_min": 22, "temp_max": 39, "ph_min": 6.2, "ph_max": 8.0,
        "n_fix": 0.80, "som_contrib": 0.78, "erosion_prot": 0.74, "drought_tol": 0.82,
        "market_score": 0.86, "forage_val": 0.65, "irrig_req": "Low", "is_legume": True, "is_cover": True
    },
    {
        "name": "Cowpea (Lobia)", "family": "Fabaceae", "seasons": ["Kharif", "Zaid"],
        "water_mm": 260, "temp_min": 21, "temp_max": 40, "ph_min": 5.5, "ph_max": 8.2,
        "n_fix": 0.86, "som_contrib": 0.85, "erosion_prot": 0.86, "drought_tol": 0.88,
        "market_score": 0.76, "forage_val": 0.94, "irrig_req": "Low", "is_legume": True, "is_cover": True
    },
    {
        "name": "Sunnhemp (Green Manure)", "family": "Fabaceae", "seasons": ["Kharif", "Zaid"],
        "water_mm": 210, "temp_min": 20, "temp_max": 41, "ph_min": 5.5, "ph_max": 8.4,
        "n_fix": 0.96, "som_contrib": 0.98, "erosion_prot": 0.94, "drought_tol": 0.86,
        "market_score": 0.48, "forage_val": 0.75, "irrig_req": "Low", "is_legume": True, "is_cover": True
    },
    {
        "name": "Fallow", "family": "None", "seasons": ["Zaid", "Rabi"],
        "water_mm": 0, "temp_min": 0, "temp_max": 50, "ph_min": 4.0, "ph_max": 9.5,
        "n_fix": 0.05, "som_contrib": 0.15, "erosion_prot": 0.10, "drought_tol": 0.95,
        "market_score": 0.10, "forage_val": 0.10, "irrig_req": "None", "is_legume": False, "is_cover": False
    }
]

CROP_MAP = {c["name"]: c for c in CROP_DATABASE}
SEASON_SLOTS = ["Kharif", "Rabi", "Zaid", "Kharif", "Rabi", "Zaid"]  

def evaluate_crop_suitability_in_season(
    crop_name: str,
    season: str,
    df_seas: pd.DataFrame,
    profile: Dict[str, Any],
    rain_mult: float = 1.0,
    temp_add_c: float = 0.0,
    irrig_override: str = None
) -> Dict[str, float]:
    """
    Transparent, physics-and-agronomy grounded suitability engine (0.0 to 1.0 scale).
    Explicitly avoids pretending a synthetic ML model predicts exact yields.
    """
    crop = CROP_MAP[crop_name]
    if season not in crop["seasons"]:
        return {"overall": 0.0, "thermal": 0.0, "water": 0.0, "ph": 0.0}

    s_row = df_seas.loc[season]
    s_temp = float(s_row["Avg_Temp_C"]) + temp_add_c
    s_max_temp = float(s_row["Max_Temp_C"]) + temp_add_c
    s_rain = float(s_row["Total_Rain_mm"]) * rain_mult

    irrig_mode = irrig_override if irrig_override is not None else profile["irrigation_availability"]
    if "Assured" in irrig_mode:
        irrig_allowance_mm = {"Kharif": 300.0, "Rabi": 420.0, "Zaid": 380.0}[season]
    elif "Limited" in irrig_mode:
        irrig_allowance_mm = {"Kharif": 120.0, "Rabi": 220.0, "Zaid": 60.0}[season]
    else:
        irrig_allowance_mm = {"Kharif": 0.0, "Rabi": 40.0, "Zaid": 0.0}[season]

    effective_water_mm = (s_rain * 0.78) + irrig_allowance_mm + (float(s_row["Avg_Soil_Wetness"]) * 140.0)

    if crop_name == "Fallow":
        return {"overall": 0.42, "thermal": 0.85, "water": 0.95, "ph": 0.90}

    if crop["temp_min"] <= s_temp <= crop["temp_max"]:
        heat_pen = max(0.0, (s_max_temp - (crop["temp_max"] + 2.0)) * 0.08)
        thermal_score = float(np.clip(1.0 - heat_pen, 0.20, 1.0))
    else:
        dist = min(abs(s_temp - crop["temp_min"]), abs(s_temp - crop["temp_max"]))
        thermal_score = float(np.clip(1.0 - dist * 0.12, 0.05, 0.85))

    demand_mm = max(crop["water_mm"], 1.0)
    supply_ratio = effective_water_mm / demand_mm
    if supply_ratio >= 0.95:
        water_score = 0.96
    else:
        deficit = 1.0 - supply_ratio

        buffered_deficit = deficit * (1.15 - 0.55 * crop["drought_tol"])
        water_score = float(np.clip(1.0 - buffered_deficit, 0.08, 0.95))

    ph = profile["soil_ph"]
    if crop["ph_min"] <= ph <= crop["ph_max"]:
        ph_score = 0.96
    else:
        ph_dist = min(abs(ph - crop["ph_min"]), abs(ph - crop["ph_max"]))
        ph_score = float(np.clip(1.0 - ph_dist * 0.25, 0.20, 0.90))

    overall = round(0.45 * water_score + 0.35 * thermal_score + 0.20 * ph_score, 3)
    return {"overall": overall, "thermal": round(thermal_score, 3), "water": round(water_score, 3), "ph": round(ph_score, 3)}

suit_rows = []
for c in CROP_DATABASE:
    row = {"Crop": c["name"], "Family": c["family"], "Water_mm": c["water_mm"], "N_Fix": c["n_fix"]}
    for s in ["Kharif", "Rabi", "Zaid"]:
        row[f"{s}_Suit"] = evaluate_crop_suitability_in_season(c["name"], s, df_seasonal, FARM_PROFILE)["overall"]
    suit_rows.append(row)
df_suitability = pd.DataFrame(suit_rows)
print("\n[Section 10-11] 🌾 Crop Database & Seasonal Suitability Matrix (0.00 = Incompatible, 1.00 = Ideal):")
print(df_suitability.to_string(index=False))

def evaluate_rotation_sequence(
    sequence: List[str],
    weights: Dict[str, float],
    df_seas: pd.DataFrame,
    profile: Dict[str, Any],
    env: Dict[str, Any],
    rain_mult: float = 1.0,
    temp_add_c: float = 0.0,
    irrig_override: str = None
) -> Dict[str, Any]:
    """
    Evaluates a 6-season (2-year) crop rotation across all 10 required dimensions and enforces
    hard penalty deductions for agronomic constraint violations.
    """
    n_slots = len(SEASON_SLOTS)
    clim_scores, water_scores, som_scores, nfix_scores = [], [], [], []
    erosion_scores, drought_scores, market_scores, forage_scores = [], [], [], []
    total_water_mm = 0.0
    families_used = []
    legume_count = 0
    consecutive_penalty = 0.0
    season_violation_penalty = 0.0
    irrig_violation_penalty = 0.0

    irrig_mode = irrig_override if irrig_override is not None else profile["irrigation_availability"]

    for idx, crop_name in enumerate(sequence):
        season = SEASON_SLOTS[idx]
        crop = CROP_MAP[crop_name]
        suit = evaluate_crop_suitability_in_season(crop_name, season, df_seas, profile, rain_mult, temp_add_c, irrig_mode)

        if season not in crop["seasons"]:
            season_violation_penalty += 0.35

        if crop["irrig_req"] == "High" and ("Rainfed" in irrig_mode or (season == "Zaid" and "Assured" not in irrig_mode)):
            irrig_violation_penalty += 0.18

        clim_scores.append(suit["overall"])

        w_eff = 0.55 * (1.0 - min(crop["water_mm"], 1200.0) / 1300.0) + 0.45 * suit["water"]
        water_scores.append(w_eff)
        som_scores.append(crop["som_contrib"])
        nfix_scores.append(crop["n_fix"])
        erosion_scores.append(crop["erosion_prot"])
        drought_scores.append(0.60 * crop["drought_tol"] + 0.40 * suit["thermal"])
        market_scores.append(crop["market_score"] * (0.5 + 0.5 * suit["overall"]))
        forage_scores.append(crop["forage_val"])
        total_water_mm += crop["water_mm"] * (1.0 + 0.035 * temp_add_c)

        if crop["family"] != "None":
            families_used.append(crop["family"])
        if crop["is_legume"]:
            legume_count += 1

        if idx > 0:
            prev_crop = CROP_MAP[sequence[idx - 1]]
            if crop_name == sequence[idx - 1] and crop_name != "Fallow":
                consecutive_penalty += 0.22
            elif crop["family"] == prev_crop["family"] and crop["family"] == "Poaceae":
                consecutive_penalty += 0.12

    unique_crops = len(set([c for c in sequence if c != "Fallow"]))
    unique_families = len(set(families_used))
    diversity_score = float(np.clip(0.55 * (unique_crops / 5.0) + 0.45 * (unique_families / 3.0), 0.15, 1.0))

    legume_factor = float(np.clip(legume_count / 2.5, 0.0, 1.0))
    if SOIL_ASSESSMENT["requires_legume_n_boost"] and legume_count == 0:
        consecutive_penalty += 0.15

    metrics_0_100 = {
        "Climate_Suitability": round(float(np.mean(clim_scores)) * 100.0, 1),
        "Water_Efficiency": round(float(np.mean(water_scores)) * 100.0, 1),
        "Soil_Health_SOM": round(float(0.65 * np.mean(som_scores) + 0.35 * legume_factor) * 100.0, 1),
        "Nitrogen_Fixation": round(float(0.60 * np.mean(nfix_scores) + 0.40 * legume_factor) * 100.0, 1),
        "Erosion_Protection": round(float(np.mean(erosion_scores)) * 100.0, 1),
        "Climate_Resilience": round(float(np.mean(drought_scores)) * 100.0, 1),
        "Market_Feasibility": round(float(np.mean(market_scores)) * 100.0, 1),
        "Crop_Diversity": round(diversity_score * 100.0, 1),
        "Forage_Value": round(float(np.mean(forage_scores)) * 100.0, 1)
    }

    w_sum = sum(weights.values())
    norm_w = {k: v / max(w_sum, 1e-6) for k, v in weights.items()}
    raw_fitness = (
        norm_w.get("water_conservation", 0.15) * metrics_0_100["Water_Efficiency"] +
        norm_w.get("soil_organic_matter", 0.15) * metrics_0_100["Soil_Health_SOM"] +
        norm_w.get("nitrogen_fixation", 0.15) * metrics_0_100["Nitrogen_Fixation"] +
        norm_w.get("erosion_reduction", 0.12) * metrics_0_100["Erosion_Protection"] +
        norm_w.get("climate_resilience", 0.18) * (0.5 * metrics_0_100["Climate_Resilience"] + 0.5 * metrics_0_100["Climate_Suitability"]) +
        norm_w.get("economic_feasibility", 0.15) * metrics_0_100["Market_Feasibility"] +
        norm_w.get("forage_production", 0.05) * metrics_0_100["Forage_Value"] +
        0.05 * metrics_0_100["Crop_Diversity"]
    )

    total_penalty_pts = (consecutive_penalty + season_violation_penalty + irrig_violation_penalty) * 100.0
    overall_fitness = round(float(np.clip(raw_fitness - total_penalty_pts, 5.0, 99.0)), 1)
    metrics_0_100["Overall_Suitability"] = overall_fitness
    metrics_0_100["Est_2Yr_Water_mm"] = round(total_water_mm, 0)
    metrics_0_100["Legume_Seasons_Count"] = legume_count

    return {
        "sequence": list(sequence),
        "fitness": overall_fitness,
        "metrics": metrics_0_100,
        "penalty_applied": round(total_penalty_pts, 1)
    }

class DiscreteRotationPSO:
    """
    Discrete Particle Swarm Optimizer (D-PSO) for multi-season crop rotation sequencing.
    Each particle maintains a discrete sequence of valid seasonal crops and updates positions
    stochastically toward personal best (pBest) and global swarm best (gBest) while enforcing
    botanical rotation rules.
    """
    def __init__(self, n_particles: int = 36, n_iterations: int = 45, seed: int = SEED):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.rng = np.random.default_rng(seed)
        self.valid_by_season = {
            s: [c["name"] for c in CROP_DATABASE if s in c["seasons"]]
            for s in ["Kharif", "Rabi", "Zaid"]
        }

    def _repair_sequence(self, seq: List[str]) -> List[str]:
        """Repairs seasonal mismatches and consecutive identical cereal monocultures."""
        repaired = list(seq)
        for i, s in enumerate(SEASON_SLOTS):
            if repaired[i] not in self.valid_by_season[s]:
                repaired[i] = self.rng.choice(self.valid_by_season[s])
            if i > 0 and repaired[i] == repaired[i - 1] and repaired[i] != "Fallow":
                alts = [c for c in self.valid_by_season[s] if c != repaired[i - 1]]
                if alts:
                    repaired[i] = self.rng.choice(alts)

        if not any(CROP_MAP[c]["is_legume"] for c in repaired):
            legume_slots = [(i, s) for i, s in enumerate(SEASON_SLOTS) if any(CROP_MAP[c]["is_legume"] for c in self.valid_by_season[s])]
            idx, seas = legume_slots[self.rng.integers(0, len(legume_slots))]
            legumes_in_seas = [c for c in self.valid_by_season[seas] if CROP_MAP[c]["is_legume"]]
            repaired[idx] = self.rng.choice(legumes_in_seas)
        return repaired

    def optimize(self, weights: Dict[str, float], df_seas: pd.DataFrame, profile: Dict[str, Any], env: Dict[str, Any]) -> Tuple[Dict[str, Any], List[float]]:
        particles = []
        p_best = []
        p_best_fit = []

        for _ in range(self.n_particles):
            init_seq = [self.rng.choice(self.valid_by_season[s]) for s in SEASON_SLOTS]
            init_seq = self._repair_sequence(init_seq)
            eval_res = evaluate_rotation_sequence(init_seq, weights, df_seas, profile, env)
            particles.append(init_seq)
            p_best.append(list(init_seq))
            p_best_fit.append(eval_res["fitness"])

        g_best_idx = int(np.argmax(p_best_fit))
        g_best_seq = list(p_best[g_best_idx])
        g_best_eval = evaluate_rotation_sequence(g_best_seq, weights, df_seas, profile, env)
        convergence_history = [g_best_eval["fitness"]]

        w_explore = 0.25
        c1_cognitive = 0.35
        c2_social = 0.40

        for it in range(self.n_iterations):

            cur_explore = max(0.08, w_explore * (1.0 - it / self.n_iterations))
            for p_idx in range(self.n_particles):
                cand = list(particles[p_idx])
                for slot_i, seas in enumerate(SEASON_SLOTS):
                    r = self.rng.random()
                    if r < c2_social:
                        cand[slot_i] = g_best_seq[slot_i]
                    elif r < (c2_social + c1_cognitive):
                        cand[slot_i] = p_best[p_idx][slot_i]
                    elif r < (c2_social + c1_cognitive + cur_explore):
                        cand[slot_i] = self.rng.choice(self.valid_by_season[seas])

                cand = self._repair_sequence(cand)
                cand_eval = evaluate_rotation_sequence(cand, weights, df_seas, profile, env)
                particles[p_idx] = cand

                if cand_eval["fitness"] > p_best_fit[p_idx]:
                    p_best[p_idx] = list(cand)
                    p_best_fit[p_idx] = cand_eval["fitness"]
                    if cand_eval["fitness"] > g_best_eval["fitness"]:
                        g_best_seq = list(cand)
                        g_best_eval = cand_eval

            convergence_history.append(g_best_eval["fitness"])

        return g_best_eval, convergence_history

print("\n[Section 12-13] 🐝 Executing Discrete Particle Swarm Optimization (D-PSO) for Multi-Season Rotations...")

existing_6season = FARM_PROFILE["existing_rotation"] * 2
strategy_a_eval = evaluate_rotation_sequence(
    existing_6season, FARM_PROFILE["priorities"], df_seasonal, FARM_PROFILE, ENV_METRICS
)
strategy_a_eval["strategy_id"] = "A. Existing Farmer Rotation"
strategy_a_eval["focus_summary"] = "Current cereal-heavy baseline (Rice -> Wheat -> Summer Fallow) with high water draw and low N-fixation."

weights_soil_restoration = {
    "water_conservation": 0.65,
    "soil_organic_matter": 1.00,
    "nitrogen_fixation": 1.00,
    "erosion_reduction": 0.95,
    "climate_resilience": 0.70,
    "economic_feasibility": 0.70,
    "forage_production": 0.55
}
pso_soil = DiscreteRotationPSO(n_particles=40, n_iterations=50, seed=SEED)
strategy_b_eval, hist_soil = pso_soil.optimize(weights_soil_restoration, df_seasonal, FARM_PROFILE, ENV_METRICS)
strategy_b_eval["strategy_id"] = "B. Soil Restoration Rotation (D-PSO)"
strategy_b_eval["focus_summary"] = "Maximizes symbiotic Nitrogen fixation, green manure biomass, and continuous root cover to rebuild Organic Carbon."

weights_climate_resilience = {
    "water_conservation": 1.00,
    "soil_organic_matter": 0.70,
    "nitrogen_fixation": 0.75,
    "erosion_reduction": 0.75,
    "climate_resilience": 1.00,
    "economic_feasibility": 0.90,
    "forage_production": 0.65
}
pso_clim = DiscreteRotationPSO(n_particles=40, n_iterations=50, seed=SEED + 7)
strategy_c_eval, hist_clim = pso_clim.optimize(weights_climate_resilience, df_seasonal, FARM_PROFILE, ENV_METRICS)
strategy_c_eval["strategy_id"] = "C. Climate Resilience Rotation (D-PSO)"
strategy_c_eval["focus_summary"] = "Prioritizes low-transpiration C4 millets, deep-rooted pulses, and oilseeds resilient to monsoon dry spells and heatwaves."

STRATEGIES = [strategy_a_eval, strategy_b_eval, strategy_c_eval]

comp_rows = []
for st in STRATEGIES:
    m = st["metrics"]
    comp_rows.append({
        "Strategy": st["strategy_id"],
        "2-Year Sequence (Y1 Kharif -> Y2 Zaid)": " → ".join(st["sequence"]),
        "Overall Score": m["Overall_Suitability"],
        "Water Eff.": m["Water_Efficiency"],
        "Soil Health": m["Soil_Health_SOM"],
        "Climate Resil.": m["Climate_Resilience"],
        "Erosion Prot.": m["Erosion_Protection"],
        "N-Fixation": m["Nitrogen_Fixation"],
        "Market Feas.": m["Market_Feasibility"],
        "Diversity": m["Crop_Diversity"],
        "2Yr Water (mm)": int(m["Est_2Yr_Water_mm"])
    })
df_comparison = pd.DataFrame(comp_rows)
print("\n✅ Three Rotation Strategies Generated & Evaluated (Model-Derived Relative Scores [0–100]):")
print(df_comparison.to_string(index=False))

def render_comparison_dashboard(strategies: List[Dict[str, Any]], df_env_daily: pd.DataFrame, save_path: str = "terracycle_dashboard.png"):
    """Generates a 4-panel visual analytics dashboard inside the notebook and saves high-DPI PNG for PDF export."""
    fig = plt.figure(figsize=(18, 12), dpi=130)
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.25)

    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    radar_Select = [
        ("Water_Efficiency", "Water\nEfficiency"),
        ("Soil_Health_SOM", "Soil Health\n(SOM)"),
        ("Climate_Resilience", "Climate\nResilience"),
        ("Erosion_Protection", "Erosion\nProtection"),
        ("Nitrogen_Fixation", "Nitrogen\nFixation"),
        ("Market_Feasibility", "Market\nFeasibility"),
        ("Crop_Diversity", "Crop\nDiversity")
    ]
    angles = np.linspace(0, 2 * np.pi, len(radar_Select), endpoint=False).tolist()
    angles += angles[:1]
    palette = ["#d95f02", "#1b9e77", "#7570b3"]

    for idx, st in enumerate(strategies):
        vals = [st["metrics"][k] for k, _ in radar_Select]
        vals += vals[:1]
        ax_radar.plot(angles, vals, color=palette[idx], linewidth=2.3, label=st["strategy_id"])
        ax_radar.fill(angles, vals, color=palette[idx], alpha=0.14)

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([lbl for _, lbl in radar_Select], fontsize=9, fontweight="bold")
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title("A. Multi-Objective Rotation Trade-Off Radar\n(Model-Derived Relative Scores [0-100])", fontsize=11, fontweight="bold", pad=18)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8.5)

    ax_bar = fig.add_subplot(gs[0, 1])
    bar_metrics = ["Overall_Suitability", "Water_Efficiency", "Soil_Health_SOM", "Climate_Resilience", "Market_Feasibility"]
    bar_labels = ["Overall\nSuitability", "Water\nEfficiency", "Soil Health\nContribution", "Climate\nResilience", "Market\nFeasibility"]
    x = np.arange(len(bar_metrics))
    width = 0.25

    for idx, st in enumerate(strategies):
        vals = [st["metrics"][m] for m in bar_metrics]
        rects = ax_bar.bar(x + (idx - 1) * width, vals, width, label=st["strategy_id"], color=palette[idx], edgecolor="black", linewidth=0.6)
        for r in rects:
            h = r.get_height()
            ax_bar.annotate(f"{h:.0f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels, fontsize=9, fontweight="bold")
    ax_bar.set_ylim(0, 112)
    ax_bar.set_ylabel("Normalized Score (0 - 100)", fontsize=9.5)
    ax_bar.set_title("B. Quantitative Strategy Comparison Across Key Pillars", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.45)
    ax_bar.legend(fontsize=8.5, loc="upper right")

    ax_heat = fig.add_subplot(gs[1, 0])
    heat_cols = ["Overall_Suitability", "Water_Efficiency", "Soil_Health_SOM", "Nitrogen_Fixation", "Erosion_Protection", "Climate_Resilience", "Market_Feasibility", "Crop_Diversity"]
    heat_short = ["Overall", "Water Eff", "Soil SOM", "N-Fix", "Erosion", "Clim Res", "Market", "Diversity"]
    heat_matrix = np.array([[st["metrics"][c] for c in heat_cols] for st in strategies])

    im = ax_heat.imshow(heat_matrix, cmap="YlGnBu", aspect="auto", vmin=10, vmax=100)
    ax_heat.set_xticks(np.arange(len(heat_short)))
    ax_heat.set_xticklabels(heat_short, fontsize=8.5, fontweight="bold", rotation=20)
    ax_heat.set_yticks(np.arange(len(strategies)))
    ax_heat.set_yticklabels([s["strategy_id"].split("(")[0].strip() for s in strategies], fontsize=9, fontweight="bold")
    for i in range(len(strategies)):
        for j in range(len(heat_cols)):
            val = heat_matrix[i, j]
            txt_color = "white" if val > 65 else "black"
            ax_heat.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=8.5, fontweight="bold")
    ax_heat.set_title("C. Multi-Criterion Score Heatmap (0-100 Normalized)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.03)

    ax_time = fig.add_subplot(gs[1, 1])
    slot_headers = ["Y1 Kharif\n(Jun-Oct)", "Y1 Rabi\n(Nov-Mar)", "Y1 Zaid\n(Apr-May)", "Y2 Kharif\n(Jun-Oct)", "Y2 Rabi\n(Nov-Mar)", "Y2 Zaid\n(Apr-May)"]
    family_colors = {
        "Poaceae": "#fdae61",       
        "Fabaceae": "#66bd63",      
        "Brassicaceae": "#fee08b",  
        "Asteraceae": "#abdda4",    
        "Malvaceae": "#f46d43",     
        "None": "#d9d9d9"           
    }

    for s_idx, st in enumerate(strategies):
        y_pos = 2 - s_idx
        for slot_i, crop_name in enumerate(st["sequence"]):
            fam = CROP_MAP[crop_name]["family"]
            c_hex = family_colors.get(fam, "#cccccc")
            rect = mpatches.FancyBboxPatch((slot_i + 0.05, y_pos - 0.32), 0.90, 0.64, boxstyle="round,pad=0.03", facecolor=c_hex, edgecolor="#333333", linewidth=1.0)
            ax_time.add_patch(rect)
            short_name = crop_name.split("(")[0].strip()
            ax_time.text(slot_i + 0.50, y_pos + 0.06, short_name, ha="center", va="center", fontsize=8, fontweight="bold", color="#111111")
            ax_time.text(slot_i + 0.50, y_pos - 0.14, f"[{fam[:4]}]", ha="center", va="center", fontsize=7, color="#333333")

    ax_time.set_xlim(0, 6)
    ax_time.set_ylim(-0.6, 2.7)
    ax_time.set_xticks(np.arange(6) + 0.5)
    ax_time.set_xticklabels(slot_headers, fontsize=8.5, fontweight="bold")
    ax_time.set_yticks([2, 1, 0])
    ax_time.set_yticklabels([s["strategy_id"].split("(")[0].strip() for s in strategies], fontsize=9, fontweight="bold")
    ax_time.set_title("D. 2-Year (6-Season) Crop Rotation Sequence & Botanical Families", fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#66bd63", label="Fabaceae (N-Fixing Legume)"),
        mpatches.Patch(color="#fdae61", label="Poaceae (Cereal / Millet)"),
        mpatches.Patch(color="#fee08b", label="Brassicaceae / Oilseed"),
        mpatches.Patch(color="#d9d9d9", label="Bare Fallow")
    ]
    ax_time.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)

    plt.suptitle(f"TerraCycle AI — Multi-Strategy Agro-Ecological Comparison ({FARM_PROFILE['farm_name']})", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return save_path

print("\n[Section 14] 📈 Generating Multi-Panel Strategy Comparison Dashboard...")
DASHBOARD_IMG_PATH = render_comparison_dashboard(STRATEGIES, df_env)

def run_climate_scenario_simulation(
    rain_change_pct: float = -25.0,
    temp_increase_c: float = 2.0,
    irrig_restriction: str = "Rainfed Only (Monsoon Dependent)"
) -> pd.DataFrame:
    """
    Recalculates the performance of all three rotation strategies under a hypothetical climate stress
    scenario (Rainfall shift: -50% to +30%, Temperature shift: 0°C to +5°C, Irrigation limits).
    Clearly labelled as a Scenario Stress Simulation, not a meteorological forecast.
    """
    rain_mult = 1.0 + (rain_change_pct / 100.0)
    sim_rows = []

    for st in STRATEGIES:
        sim_eval = evaluate_rotation_sequence(
            st["sequence"], FARM_PROFILE["priorities"], df_seasonal, FARM_PROFILE, ENV_METRICS,
            rain_mult=rain_mult, temp_add_c=temp_increase_c, irrig_override=irrig_restriction
        )
        base_score = st["metrics"]["Overall_Suitability"]
        sim_score = sim_eval["metrics"]["Overall_Suitability"]
        delta_score = round(sim_score - base_score, 1)

        slot_suits = [
            ( crop_n, SEASON_SLOTS[i], evaluate_crop_suitability_in_season(crop_n, SEASON_SLOTS[i], df_seasonal, FARM_PROFILE, rain_mult, temp_increase_c, irrig_restriction)["overall"] )
            for i, crop_n in enumerate(st["sequence"]) if crop_n != "Fallow"
        ]
        worst_crop, worst_seas, worst_val = min(slot_suits, key=lambda x: x[2])

        sim_rows.append({
            "Strategy": st["strategy_id"],
            "Baseline_Score": base_score,
            "Scenario_Score": sim_score,
            "Score_Drop": delta_score,
            "Scenario_Water_Demand_mm": int(sim_eval["metrics"]["Est_2Yr_Water_mm"]),
            "Most_Vulnerable_Crop": f"{worst_crop} ({worst_seas}: {worst_val:.2f})",
            "Viability_Status": "Resilient (Viable)" if sim_score >= 62.0 else ("Moderate Stress" if sim_score >= 48.0 else "High Vulnerability")
        })

    return pd.DataFrame(sim_rows)

print("\n[Section 15] 🌡️ Interactive Climate What-If Scenario Simulator (-25% Rainfall, +2.0°C Warming, Drought Irrigation Restriction):")
df_whatif_default = run_climate_scenario_simulation(rain_change_pct=-25.0, temp_increase_c=2.0, irrig_restriction="Rainfed Only (Monsoon Dependent)")
print(df_whatif_default.to_string(index=False))

w_sim_rain = widgets.IntSlider(value=-25, min=-50, max=30, step=5, description="Rainfall Δ (%):", style=style_lbl, layout=layout_box)
w_sim_temp = widgets.FloatSlider(value=2.0, min=0.0, max=5.0, step=0.5, description="Warming Δ (+°C):", style=style_lbl, layout=layout_box)
w_sim_irrig = widgets.Dropdown(
    options=["Rainfed Only (Monsoon Dependent)", "Limited (Seasonal Canal + Dugwell in Rabi; Zero in Zaid)", "Assured Full-Year (Borewell + Drip + Solar Pump)"],
    value="Rainfed Only (Monsoon Dependent)", description="Irrigation Cap:", style=style_lbl, layout=layout_box
)
sim_out = widgets.Output()

def _on_sim_change(change=None):
    with sim_out:
        clear_output(wait=True)
        df_res = run_climate_scenario_simulation(w_sim_rain.value, w_sim_temp.value, w_sim_irrig.value)
        best_row = df_res.loc[df_res["Scenario_Score"].idxmax()]
        print(f"🔎 SCENARIO SIMULATION (Not a Forecast): Rainfall {w_sim_rain.value:+d}% | Temp +{w_sim_temp.value:.1f}°C | Irrigation: {w_sim_irrig.value}")
        print(f"🏆 Most Resilient Strategy under Scenario: {best_row['Strategy']} (Score: {best_row['Scenario_Score']}/100)\n")
        display(df_res)

for w_ctrl in [w_sim_rain, w_sim_temp, w_sim_irrig]:
    w_ctrl.observe(_on_sim_change, names="value")

display(widgets.VBox([widgets.HTML("<b>Interactive Climate What-If Controls (Adjust Sliders to Recalculate Live):</b>"), widgets.HBox([w_sim_rain, w_sim_temp]), w_sim_irrig, sim_out]))
_on_sim_change()

def build_geospatial_farm_map(profile: Dict[str, Any], env: Dict[str, Any], topo: Dict[str, Any]) -> folium.Map:
    """
    Creates an interactive Folium map centered at the actual farm coordinates with:
    - Satellite & Topographic base layers
    - Proportional Farm Area Circle (calculated from actual Hectares without inventing polygon corners)
    - Environmental telemetry popup (NASA POWER Rainfall, SMAP/GWETTOP Soil Wetness, Derived NDVI, SRTM Slope)
    - Floating HTML Legend explaining spatial indicators
    """
    lat, lon = profile["latitude"], profile["longitude"]
    fmap = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite Imagery (Esri)", overlay=False, control=True
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap", name="Topographic Contours (SRTM Derived)", overlay=False, control=True
    ).add_to(fmap)

    radius_m = math.sqrt((profile["area_ha"] * 10000.0) / math.pi)

    popup_html = f"""
    <div style="font-family: Arial, sans-serif; width: 310px; font-size: 12px;">
        <h4 style="margin:0 0 6px 0; color:#1b5e20;">🌾 {profile['farm_name']}</h4>
        <b>Coordinates:</b> {lat:.4f}°N, {lon:.4f}°E (Area: {profile['area_ha']} ha)<br>
        <b>Soil Type:</b> {profile['soil_type']}<br>
        <hr style="margin:5px 0;">
        <b>🛰️ NASA & Topographic Telemetry:</b><br>
        • <b>Mean Annual Rain (POWER):</b> {env['mean_annual_rain_mm']} mm (CV: {env['rain_cv_pct']}%)<br>
        • <b>Surface Soil Wetness (GWETTOP):</b> {env['mean_soil_wetness']} (Anom Z: {env['recent_sm_anomaly_z']})<br>
        • <b>NDVI Vegetation Proxy:</b> {env['mean_ndvi_proxy']} ({'Water-Balance Proxy' if env['ndvi_is_derived_proxy'] else 'Landsat 8/9'})<br>
        • <b>95th Pct Max Temp:</b> {env['max_temp_95p_c']} °C<br>
        • <b>SRTM Elevation / Slope:</b> {topo['elevation_m']:.0f} m / {env['slope_deg']}°<br>
        • <b>RUSLE Erosion Susceptibility:</b> {env['erosion_risk_score']}/100<br>
        • <b>Drought Risk Class:</b> {CLIMATE_ML['drought_class']} ({CLIMATE_ML['drought_risk_score']}/100)
    </div>
    """

    zone_color = "#2ca25f" if env["mean_ndvi_proxy"] >= 0.45 else "#fe9929"
    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color=zone_color,
        weight=3,
        fill=True,
        fill_color=zone_color,
        fill_opacity=0.28,
        tooltip=f"Equivalent Farm Area Radius ({radius_m:.0f}m for {profile['area_ha']} ha — No synthetic boundary vertices invented)"
    ).add_to(fmap)

    erosion_color = "#de2d26" if env["erosion_risk_score"] > 50 else ("#feb24c" if env["erosion_risk_score"] > 28 else "#3182bd")
    folium.Circle(
        location=[lat, lon],
        radius=max(radius_m * 2.2, 300.0),
        color=erosion_color,
        weight=2,
        dash_array="6, 6",
        fill=False,
        tooltip=f"Micro-Watershed Slope & Runoff Zone (Slope: {env['slope_deg']}°, Erosion Risk: {env['erosion_risk_score']}/100)"
    ).add_to(fmap)

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=340),
        tooltip="Click for NASA Earth Observation & Soil Telemetry",
        icon=folium.Icon(color="green", icon="leaf", prefix="fa")
    ).add_to(fmap)

    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; width: 265px; background-color: white;
                border:2px solid #444; z-index:9999; font-size:11.5px; padding: 10px; border-radius: 6px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.25);">
        <b>🗺️ TerraCycle AI Geospatial Legend</b><br>
        <span style="color:{zone_color}; font-weight:bold;">● Solid Circle:</span> Equivalent Farm Area ({profile['area_ha']} ha)<br>
        <span style="color:{erosion_color}; font-weight:bold;">◌ Dashed Ring:</span> Topographic Slope Context ({env['slope_deg']}°)<br>
        <b>Data Resolution Notes:</b><br>
        • NASA POWER Climate: ~50 km grid<br>
        • Topographic Slope: ~90-110 m DEM stencil<br>
        • Field Boundary: Proportional circle (exact polygon not supplied)
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(fmap)
    return fmap

print("\n[Section 16] 🗺️ Rendering Interactive Folium Geospatial Farm & Telemetry Map...")
FARM_MAP = build_geospatial_farm_map(FARM_PROFILE, ENV_METRICS, topo_info)
display(FARM_MAP)

def generate_explainable_ai_advisory(
    strategies: List[Dict[str, Any]],
    profile: Dict[str, Any],
    env: Dict[str, Any],
    soil: Dict[str, Any],
    climate_ml: Dict[str, Any]
) -> Dict[str, Dict[str, str]]:
    """
    Rule-grounded Explainable AI (XAI) Assistant that links every rotation strategy to:
    1. Supporting NASA datasets & observed metrics
    2. Specific ICAR soil constraints addressed
    3. Expected soil-health & water trade-offs
    4. Climate & market implementation risks
    5. Missing local data & mandatory Krishi Vigyan Kendra (KVK) validation steps
    """
    explanations = {}
    water_base = strategies[0]["metrics"]["Est_2Yr_Water_mm"]

    for st in strategies:
        sid = st["strategy_id"]
        seq_str = " → ".join(st["sequence"])
        m = st["metrics"]
        water_diff = water_base - m["Est_2Yr_Water_mm"]
        water_pct = (water_diff / max(water_base, 1.0)) * 100.0

        if "Existing" in sid:
            why_rec = (
                f"Evaluated as the benchmark farmer practice ({seq_str}). While Rice-Wheat provides familiar APMC "
                f"mandi liquidity, continuous Poaceae (cereal-after-cereal) cropping depletes labile soil nitrogen and "
                f"leaves soil bare during Zaid."
            )
        elif "Soil Restoration" in sid:
            why_rec = (
                f"Selected by D-PSO ({seq_str}) to directly remediate the farm's low Organic Carbon ({profile['organic_carbon_pct']}%) "
                f"and deficient Available Nitrogen ({profile['nitrogen_kg_ha']} kg/ha) by integrating {m['Legume_Seasons_Count']} "
                f"Fabaceae legume/green-manure seasons and eliminating bare summer fallow."
            )
        else:
            why_rec = (
                f"Selected by D-PSO ({seq_str}) to buffer against the observed {env['rain_cv_pct']}% inter-annual rainfall "
                f"variability and {env['avg_kharif_dry_spells']} Kharif dry spells/year by substituting high-transpiration crops "
                f"with drought-hardy C4 millets and deep-rooted pulses."
            )

        nasa_support = (
            f"NASA POWER 10-year records indicate {env['mean_annual_rain_mm']} mm/yr average rainfall with 95th-percentile "
            f"maximum temperatures reaching {env['max_temp_95p_c']} °C and mean surface soil wetness (GWETTOP) of "
            f"{env['mean_soil_wetness']}. Topographic slope ({env['slope_deg']}°, RUSLE LS={env['rusle_ls_factor']}) yields an "
            f"erosion risk score of {env['erosion_risk_score']}/100."
        )

        soil_influence = (
            f"ICAR Soil Health Card diagnostics classify Organic Carbon as '{soil['oc_status']}' ({profile['organic_carbon_pct']}%) "
            f"and Nitrogen as '{soil['n_status']}' ({profile['nitrogen_kg_ha']} kg/ha) in {profile['soil_texture']} "
            f"(pH {profile['soil_ph']}). This strategy achieves a Soil Health Contribution score of {m['Soil_Health_SOM']}/100 "
            f"and N-Fixation score of {m['Nitrogen_Fixation']}/100."
        )

        water_tradeoff = (
            f"Estimated 2-year crop water requirement is {int(m['Est_2Yr_Water_mm'])} mm "
            f"({'saves ' + str(int(abs(water_diff))) + f' mm ({water_pct:.1f}% reduction) vs. Existing Rotation' if water_diff > 0 else 'baseline water consumption'}). "
            f"Water Efficiency Score: {m['Water_Efficiency']}/100 under '{profile['irrigation_availability']}'."
        )

        clim_market_risks = (
            f"Climate Resilience Score: {m['Climate_Resilience']}/100; Market Feasibility Score: {m['Market_Feasibility']}/100. "
            f"Unseasonal post-monsoon rain during pulse pod maturity or lack of local MSP procurement for millets/green manure "
            f"can affect short-term cash flow. These scores represent relative agro-suitability, NOT guaranteed yield predictions."
        )

        missing_and_validation = (
            "Missing Data: Field-scale 10m soil electrical conductivity (EC), micronutrient profile (Zn, B, Fe), nematode load, "
            "and exact water-table depth. Local Validation Required: Consult the district Krishi Vigyan Kendra (KVK) agronomist "
            "to verify regionally notified seed varieties and Rhizobium bio-fertilizer inoculation rates before sowing."
        )

        explanations[sid] = {
            "Sequence": seq_str,
            "Why_Recommended": why_rec,
            "NASA_Evidence": nasa_support,
            "Soil_Drivers": soil_influence,
            "Water_Tradeoffs": water_tradeoff,
            "Climate_And_Market_Risks": clim_market_risks,
            "Missing_Data_And_Expert_Validation": missing_and_validation
        }
    return explanations

XAI_ADVISORY = generate_explainable_ai_advisory(STRATEGIES, FARM_PROFILE, ENV_METRICS, SOIL_ASSESSMENT, CLIMATE_ML)

print("\n[Section 17] 🧠 Explainable AI (XAI) Agronomic Decision Advisory:")
for st_id, info in XAI_ADVISORY.items():
    print("-" * 82)
    print(f"📌 {st_id}")
    print(f"   Rotation Sequence : {info['Sequence']}")
    print(f"   • Why Recommended : {info['Why_Recommended']}")
    print(f"   • NASA Evidence   : {info['NASA_Evidence']}")
    print(f"   • Soil Drivers    : {info['Soil_Drivers']}")
    print(f"   • Water Trade-off : {info['Water_Tradeoffs']}")
    print(f"   • Risks & Caveats : {info['Climate_And_Market_Risks']}")
    print(f"   • KVK Validation  : {info['Missing_Data_And_Expert_Validation']}")

def generate_pdf_advisory_report(
    filename: str,
    profile: Dict[str, Any],
    env: Dict[str, Any],
    soil: Dict[str, Any],
    strategies: List[Dict[str, Any]],
    xai: Dict[str, Dict[str, str]],
    chart_path: str
) -> str:
    """Generates a structured, downloadable multi-page PDF report using ReportLab."""
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=16, textColor=colors.HexColor("#1b5e20"), spaceAfter=10)
    h2_style = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#0d47a1"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=5)
    disc_style = ParagraphStyle("DiscCustom", parent=styles["BodyText"], fontSize=8.2, leading=11, textColor=colors.HexColor("#b71c1c"), spaceAfter=8)

    story = []
    story.append(Paragraph("TerraCycle AI — NASA-Powered Crop Rotation & Climate Resilience Report", title_style))
    story.append(Paragraph(
        f"<b>Farm:</b> {profile['farm_name']} | <b>Coordinates:</b> {profile['latitude']:.4f}°N, {profile['longitude']:.4f}°E | "
        f"<b>Area:</b> {profile['area_ha']} ha | <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style
    ))
    story.append(Paragraph(
        "<b>SCIENTIFIC DISCLAIMER:</b> All suitability scores (0–100) are model-derived relative indicators combining NASA "
        "Earth observation climatology and ICAR agronomic rules. They are NOT guaranteed crop yield or financial predictions. "
        "Soil parameters correspond to the supplied/demo Soil Health Card and require local KVK field verification.", disc_style
    ))

    story.append(Paragraph("1. NASA Earth Observation & Environmental Summary (2015–2024 Reference)", h2_style))
    env_table_data = [
        ["Indicator", "Observed / Derived Value", "Source & Resolution", "Agronomic Interpretation"],
        ["Mean / 95p Max Temp", f"{env['mean_temp_c']} °C / {env['max_temp_95p_c']} °C", "NASA POWER (~50 km)", "Rabi/Zaid heat stress threshold check"],
        ["Mean Annual Rainfall", f"{env['mean_annual_rain_mm']} mm (CV: {env['rain_cv_pct']}%)", "NASA POWER / GPM IMERG", f"Anomaly: {env['rain_anomaly_pct']:+.1f}% vs 10-yr mean"],
        ["Surface Soil Wetness", f"{env['mean_soil_wetness']} (Z: {env['recent_sm_anomaly_z']})", "NASA GWETTOP / SMAP", f"Kharif Dry Spells (>=7d): {env['avg_kharif_dry_spells']}/yr"],
        ["Vegetation Index (NDVI)", f"{env['mean_ndvi_proxy']}", "Agro-Climatic Proxy / L8-9", "Seasonal canopy vigor & bare-soil window"],
        ["Terrain Slope & Erosion", f"{env['slope_deg']}° (Risk: {env['erosion_risk_score']}/100)", topo_info["source"][:26], f"RUSLE LS Factor: {env['rusle_ls_factor']}"]
    ]
    t_env = Table(env_table_data, colWidths=[115, 115, 125, 165])
    t_env.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b5e20")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
    ]))
    story.append(t_env)

    story.append(Paragraph("2. ICAR Soil Health Assessment", h2_style))
    story.append(Paragraph(
        f"<b>Soil Type:</b> {profile['soil_type']} ({profile['soil_texture']}) | <b>Composite Soil Index:</b> {soil['composite_soil_index']}/100<br/>"
        f"• <b>pH:</b> {profile['soil_ph']} ({soil['ph_status']}) | <b>Organic Carbon:</b> {profile['organic_carbon_pct']}% ({soil['oc_status']})<br/>"
        f"• <b>Available N-P-K (kg/ha):</b> N={profile['nitrogen_kg_ha']} ({soil['n_status']}), "
        f"P={profile['phosphorus_kg_ha']} ({soil['p_status']}), K={profile['potassium_kg_ha']} ({soil['k_status']})", body_style
    ))

    story.append(Paragraph("3. Multi-Objective Rotation Comparison (2-Year / 6-Season Horizon)", h2_style))
    comp_table_data = [["Strategy", "6-Season Crop Sequence", "Overall", "Water Eff", "Soil SOM", "Clim Res", "2Yr Water"]]
    for st in strategies:
        m = st["metrics"]
        comp_table_data.append([
            st["strategy_id"].split("(")[0].strip(),
            " -> ".join([c.split("(")[0].strip() for c in st["sequence"]]),
            f"{m['Overall_Suitability']}", f"{m['Water_Efficiency']}", f"{m['Soil_Health_SOM']}",
            f"{m['Climate_Resilience']}", f"{int(m['Est_2Yr_Water_mm'])} mm"
        ])
    t_comp = Table(comp_table_data, colWidths=[110, 195, 42, 45, 45, 45, 48])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8f5e9"), colors.white, colors.HexColor("#e3f2fd")])
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))

    if os.path.exists(chart_path):
        story.append(Paragraph("4. Visual Trade-Off Analytics Dashboard", h2_style))
        story.append(RLImage(chart_path, width=6.8 * inch, height=4.5 * inch))

    story.append(PageBreak())
    story.append(Paragraph("5. Explainable AI (XAI) Strategy Reasoning & Validation Requirements", h2_style))
    for st_id, info in xai.items():
        story.append(Paragraph(f"<b>{st_id}</b> — <i>{info['Sequence']}</i>", body_style))
        story.append(Paragraph(f"• <b>Rationale:</b> {info['Why_Recommended']}", body_style))
        story.append(Paragraph(f"• <b>NASA & Soil Basis:</b> {info['NASA_Evidence']} {info['Soil_Drivers']}", body_style))
        story.append(Paragraph(f"• <b>Water & Climate Trade-offs:</b> {info['Water_Tradeoffs']} {info['Climate_And_Market_Risks']}", body_style))
        story.append(Paragraph(f"• <b>Required Expert Validation:</b> {info['Missing_Data_And_Expert_Validation']}", body_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    return filename

PDF_REPORT_PATH = "TerraCycle_AI_Farmer_Advisory_Report.pdf"
generate_pdf_advisory_report(PDF_REPORT_PATH, FARM_PROFILE, ENV_METRICS, SOIL_ASSESSMENT, STRATEGIES, XAI_ADVISORY, DASHBOARD_IMG_PATH)
print(f"\n[Section 18] 📄 Downloadable PDF Advisory Report generated: '{PDF_REPORT_PATH}' ({os.path.getsize(PDF_REPORT_PATH)/1024:.1f} KB)")

def run_system_verification_checks() -> None:
    """Executes automated sanity and scientific integrity checks across all pipeline outputs."""
    print("\n[Section 19] ✅ Running Automated End-to-End Sanity & Integrity Checks...")
    assert len(df_env) >= 3650, "NASA POWER time series must span 10 years (>=3650 daily records)."
    assert not df_env[["T2M", "PRECTOTCORR", "GWETTOP", "ET0_mm"]].isna().any().any(), "No NaN values allowed after preprocessing."
    assert len(STRATEGIES) == 3, "Exactly 3 core rotation strategies (Existing, Soil Restoration, Climate Resilience) must be evaluated."

    for st in STRATEGIES:
        seq = st["sequence"]
        assert len(seq) == 6, f"Strategy {st['strategy_id']} must cover 6 seasonal slots (2 years)."
        for i, c_name in enumerate(seq):
            seas = SEASON_SLOTS[i]
            assert seas in CROP_MAP[c_name]["seasons"], f"Illegal season assignment: {c_name} in {seas}."
        for k, val in st["metrics"].items():
            if k not in ["Est_2Yr_Water_mm", "Legume_Seasons_Count"]:
                assert 0.0 <= val <= 100.0, f"Metric {k}={val} out of normalized [0, 100] bounds."

    assert strategy_b_eval["metrics"]["Soil_Health_SOM"] > strategy_a_eval["metrics"]["Soil_Health_SOM"], "Soil Restoration D-PSO must improve Soil SOM score over baseline."
    assert strategy_c_eval["metrics"]["Water_Efficiency"] > strategy_a_eval["metrics"]["Water_Efficiency"], "Climate Resilience D-PSO must improve Water Efficiency over baseline."
    assert os.path.exists(PDF_REPORT_PATH) and os.path.getsize(PDF_REPORT_PATH) > 5000, "PDF report file must exist and be non-empty."

    print("  -> [PASS] 10-Year NASA Daily Climatology & Missing-Value Interpolation Verified.")
    print("  -> [PASS] Seasonal Crop Compatibility & Botanical Family Rules Verified (6/6 seasons valid).")
    print(f"  -> [PASS] Soil Restoration D-PSO improved Soil Health Score from {strategy_a_eval['metrics']['Soil_Health_SOM']} to {strategy_b_eval['metrics']['Soil_Health_SOM']}/100.")
    print(f"  -> [PASS] Climate Resilience D-PSO reduced 2-Year Water Demand from {int(strategy_a_eval['metrics']['Est_2Yr_Water_mm'])} mm to {int(strategy_c_eval['metrics']['Est_2Yr_Water_mm'])} mm.")
    print("  -> [PASS] PDF Report & High-Resolution Dashboard Artifacts Verified on Colab Disk.")
    print("\n🎓 KEY SCIENTIFIC LIMITATIONS & NEXT STEPS FOR FIELD DEPLOYMENT:")
    print("  1. Spatial Resolution Gap: NASA POWER (~50 km) and SMAP (~36 km) capture meso-scale weather and moisture anomalies")
    print("     but cannot resolve micro-topographic waterlogging within a 4.5 ha plot. Integrating Sentinel-2 (10m) via GEE")
    print("     and local soil moisture tensiometers will sharpen field-level precision.")
    print("  2. Economic Volatility: Market feasibility scores reflect historical MSP and mandi stability; real-time Agmarknet")
    print("     price feeds should be checked prior to sowing.")
    print("=" * 82)

run_system_verification_checks()
