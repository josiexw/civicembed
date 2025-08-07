import rasterio
import numpy as np

INPUT_DEM = "./data/tif_files/terrain/switzerland_dem.tif"
OUTPUT_DEM = "./data/tif_files/terrain/switzerland_dem_no_zeros.tif"

with rasterio.open(INPUT_DEM) as src:
    data = src.read(1).astype(np.float32)
    profile = src.profile

mask_nonzero = data > 0
if not np.any(mask_nonzero):
    raise ValueError("DEM contains no non-zero values.")
next_min = np.min(data[mask_nonzero])
data[data == 0] = next_min
profile.update(dtype="float32")

with rasterio.open(OUTPUT_DEM, "w", **profile) as dst:
    dst.write(data, 1)

print(f"Saved to {OUTPUT_DEM}")
