# src/dataloader/terrain_embeddings.py

import json
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from s2sphere import CellId, LatLng, Cell
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import os
import sys

sys.path.append(os.path.abspath("src"))
from embedding.terrain_encoder import TerrainEncoder

# === Config ===
DEM_PATH = "./data/tif_files/terrain/switzerland_dem.tif"
METADATA_JSONL = "./data/opendata/opendataswiss_metadata.jsonl"
LOCATION_PARQUET = "./data/opendata/opendataswiss_locations.parquet"
MODEL_PATH = "./models/terrain_encoder.pt"
OUTPUT_PARQUET = "./data/geographic_data/terrain_embeddings.parquet"
PATCH_DIR = "./data/geographic_data/terrain_patches"
PATCH_SIZE = 64
LEVEL = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PATCH_DIR, exist_ok=True)

with open(METADATA_JSONL, "r", encoding="utf-8") as f:
    metadata = {entry["id"]: entry for entry in map(json.loads, f)}

dem = rasterio.open(DEM_PATH)
geolocator = Nominatim(user_agent="fusion_encoder_lookup", timeout=10, scheme='http', domain='localhost:8080')

encoder = TerrainEncoder()
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
grouped = loc_df.groupby("id")["location_name"].apply(list).to_dict()

records = []
for db_id, locations in tqdm(grouped.items()):
    meta = metadata.get(db_id, {})
    found = False

    for loc in locations:
        try:
            result = geolocator.geocode(loc)
            if result is None:
                continue
            lat, lon = result.latitude, result.longitude
            patch = extract_patch(lat, lon)
            if patch is None:
                continue

            patch_id = f"{db_id}_{round(lat, 4)}_{round(lon, 4)}"
            patch_path = os.path.join(PATCH_DIR, f"{patch_id}.npy")
            np.save(patch_path, patch)

            with torch.no_grad():
                patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
                embedding = encoder(patch_tensor).squeeze(0).cpu().numpy()

            records.append({
                "id": db_id,
                "lat": lat,
                "lon": lon,
                "location": loc,
                "terrain_patch_id": patch_id,
                "terrain_embedding": embedding,
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "keywords": " ".join(meta.get("keywords", []))
            })
            found = True
        except (GeocoderTimedOut, GeocoderServiceError):
            continue

df = pd.DataFrame(records)
df.to_parquet(OUTPUT_PARQUET)
