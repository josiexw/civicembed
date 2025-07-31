import rasterio
import numpy as np

INPUT_DEM = "./data/tif_files/terrain/switzerland_dem.tif"
OUTPUT_DEM = "./data/tif_files/terrain/switzerland_dem_padded.tif"

# === Load DEM ===
with rasterio.open(INPUT_DEM) as src:
    data = src.read(1).astype(np.float32)
    profile = src.profile

# === Replace NaNs with min value ===
if np.isnan(data).any():
    valid_min = np.nanmin(data)
    data = np.nan_to_num(data, nan=valid_min)

profile.update(dtype="float32", nodata=None)

with rasterio.open(OUTPUT_DEM, "w", **profile) as dst:
    dst.write(data, 1)

print(f"Padded DEM written to: {OUTPUT_DEM}")
