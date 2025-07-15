import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# === Config ===
INPUT_TIF = "./data/tif_files/water/switzerland_water_cover.tif"
OUTPUT_TIF = "./data/tif_files/water/switzerland_water_proximity_km.tif"
DEG2KM = 111.32
TILE_SIZE = 1024
PADDING = 128  # overlap to reduce edge effects

with rasterio.open(INPUT_TIF) as src:
    transform = src.transform
    crs = src.crs
    profile = src.profile.copy()
    width, height = src.width, src.height

    pixel_size_deg = transform.a
    pixel_size_km = pixel_size_deg * DEG2KM

    profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        for row in tqdm(range(0, height, TILE_SIZE)):
            for col in range(0, width, TILE_SIZE):
                # Compute window with padding
                row_off = max(row - PADDING, 0)
                col_off = max(col - PADDING, 0)
                row_end = min(row + TILE_SIZE + PADDING, height)
                col_end = min(col + TILE_SIZE + PADDING, width)

                read_height = row_end - row_off
                read_width = col_end - col_off

                window = Window(col_off, row_off, read_width, read_height)
                water_tile = src.read(1, window=window) > 0

                # Compute distance transform (on padded region)
                dist_km = distance_transform_edt(~water_tile) * pixel_size_km

                # Trim to central (unpadded) region
                r0 = row - row_off
                c0 = col - col_off
                r1 = r0 + min(TILE_SIZE, height - row)
                c1 = c0 + min(TILE_SIZE, width - col)

                dist_crop = dist_km[r0:r1, c0:c1]

                write_window = Window(col, row, dist_crop.shape[1], dist_crop.shape[0])
                dst.write(dist_crop.astype(np.float32), 1, window=write_window)

print(f"Saved chunked proximity map to: {OUTPUT_TIF}")
