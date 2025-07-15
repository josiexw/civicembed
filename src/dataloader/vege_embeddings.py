# src/dataloader/vege_embeddings.py

import os
import sys
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from s2sphere import CellId, LatLng, Cell

sys.path.append(os.path.abspath("src"))
from embedding.water_encoder import WaterEncoder

# === Config ===
DEM_PATH = "./data/tif_files/vegetation/switzerland_vegetation_cover.tif"
LOCATION_PARQUET = "./data/geographic_data/terrain_embeddings.parquet"
MODEL_PATH = "./models/vegetation_encoder.pt"
OUTPUT_PARQUET = "./data/geographic_data/vegetation_embeddings.parquet"
PATCH_DIR = "./data/geographic_data/vegetation_patches"
PATCH_SIZE = 64
LEVEL = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PATCH_DIR, exist_ok=True)

dem = rasterio.open(DEM_PATH)
encoder = WaterEncoder()
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
    try:
        window = rasterio.windows.from_bounds(*bounds, transform=dem.transform)
        patch = dem.read(1, window=window, out_shape=(PATCH_SIZE, PATCH_SIZE),
                         resampling=rasterio.enums.Resampling.bilinear)
        if patch.shape != (PATCH_SIZE, PATCH_SIZE) or np.isnan(patch).any():
            return None
        return patch.astype(np.float32)
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
        "vegetation_patch_id": patch_id,
        "vegetation_embedding": embedding
    })

df = pd.DataFrame(records)
df.to_parquet(OUTPUT_PARQUET)
