# src/dataloader/road_embeddings.py

import os
import sys
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from tqdm import tqdm
from shapely.geometry import box
from rasterio.features import rasterize
from s2sphere import CellId, LatLng, Cell

sys.path.append(os.path.abspath("src"))
from embedding.road_encoder import RoadEncoder

# === Config ===
SHP_PATH = "./data/shapefiles/switzerland_roads.shp"
RASTER_REF = "./data/tif_files/terrain/switzerland_dem.tif"
LOCATION_PARQUET = "./data/geographic_data/terrain_embeddings.parquet"
MODEL_PATH = "./models/road_encoder.pt"
OUTPUT_PARQUET = "./data/geographic_data/road_embeddings.parquet"
PATCH_DIR = "./data/geographic_data/road_patches"
PATCH_SIZE = 64
LEVEL = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PATCH_DIR, exist_ok=True)

roads = gpd.read_file(SHP_PATH)
ref_raster = rasterio.open(RASTER_REF)
transform = ref_raster.transform
encoder = RoadEncoder()
encoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
encoder.to(DEVICE)
encoder.eval()

def s2_bounds(lat, lon, level=LEVEL):
    cell = Cell(CellId.from_lat_lng(LatLng.from_degrees(lat, lon)).parent(level))
    lats, lons = [], []
    for i in range(4):
        vertex = cell.get_vertex(i)
        latlng = LatLng.from_point(vertex)
        lats.append(latlng.lat().degrees)
        lons.append(latlng.lng().degrees)
    return (min(lons), min(lats), max(lons), max(lats))

def extract_patch(lat, lon):
    bounds = s2_bounds(lat, lon)
    geom = box(*bounds)
    clipped = roads[roads.intersects(geom)]
    if clipped.empty:
        return None
    try:
        patch = rasterize(
            ((g, 1) for g in clipped.geometry),
            out_shape=(PATCH_SIZE, PATCH_SIZE),
            transform=rasterio.transform.from_bounds(*bounds, PATCH_SIZE, PATCH_SIZE),
            fill=0,
            dtype=np.uint8
        ).astype(np.float32)
        return patch
    except:
        return None

loc_df = pd.read_parquet(LOCATION_PARQUET)
records = []

for _, row in tqdm(loc_df.iterrows(), total=len(loc_df)):
    db_id = row["id"]
    lat = row["lat"]
    lon = row["lon"]

    patch = extract_patch(lat, lon)
    if patch is None:
        continue

    patch_id = f"{db_id}_{round(lat, 4)}_{round(lon, 4)}"
    patch_path = os.path.join(PATCH_DIR, f"{patch_id}.npy")
    np.save(patch_path, patch)

    with torch.no_grad():
        patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
        embedding, _ = encoder(patch_tensor)
        embedding = embedding.squeeze(0).cpu().numpy()

    records.append({
        "id": db_id,
        "lat": lat,
        "lon": lon,
        "road_patch_id": patch_id,
        "road_embedding": embedding
    })

df = pd.DataFrame(records)
df.to_parquet(OUTPUT_PARQUET)
