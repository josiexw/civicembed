import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
from rasterio.transform import xy
import os
import sys
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath("src"))
from embedding.terrain_encoder import TerrainEncoder

# === Config ===
DEM_PATH = "./data/tif_files/terrain/switzerland_dem.tif"
MODEL_PATH = "./models/terrain_encoder.pt"
OUTPUT_PARQUET = "/mnt/ldm1/scratch/all_terrain_embeddings.parquet"
PATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEGREES_PER_100M = 100 / 111320

# Load DEM
dem = rasterio.open(DEM_PATH)
nodata = dem.nodata
arr = dem.read(1)
res_x, res_y = dem.res
stride_x = int(round(DEGREES_PER_100M / res_x))
stride_y = int(round(DEGREES_PER_100M / res_y))
print(f"Stride in pixels: {stride_x} x {stride_y}")

# Load encoder
encoder = TerrainEncoder()
encoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
encoder.to(DEVICE)
encoder.eval()

coords = []
embeddings = []
rows, cols = arr.shape
half = PATCH_SIZE // 2

for row in tqdm(range(0, rows, stride_y), desc="Rows"):
    for col in range(0, cols, stride_x):
        if row - half < 0 or row + half > rows or col - half < 0 or col + half > cols:
            continue

        patch = arr[row - half:row + half, col - half:col + half]

        if np.any(patch == nodata) or np.isnan(patch).any():
            continue

        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = encoder(patch_tensor)
            if isinstance(output, tuple):
                output = output[0]
            embedding = output.squeeze(0).cpu().numpy()

        lon, lat = xy(dem.transform, row, col, offset='center')
        coords.append((lat, lon))
        embeddings.append(embedding)

# === PCA-based similarity ===
embeddings_array = np.vstack(embeddings)
pca = PCA(n_components=1)
similarities = pca.fit_transform(embeddings_array).flatten()
similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())

# === Save as single-list embeddings column ===
df = pd.DataFrame({
    "lat": [lat for lat, _ in coords],
    "lon": [lon for _, lon in coords],
    "embedding": [e.tolist() for e in embeddings],
    "similarity": similarities
})

df.to_parquet(OUTPUT_PARQUET)
print(f"Saved {len(df)} grid cells to {OUTPUT_PARQUET}")
