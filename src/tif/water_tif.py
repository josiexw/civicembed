import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
import tempfile, os

tifs = ["./data/tif_files/water/occurrence_0E_50Nv1_4_2021.tif", "./data/tif_files/water/occurrence_10E_50Nv1_4_2021.tif"]
out_tif = "./data/tif_files/water/switzerland_water_cover.tif"
bbox = (5.1, 45.0, 12.1, 49.5)  # lon_min, lat_min, lon_max, lat_max (slightly larger)
dst_crs = "EPSG:4326"
tmp_files = []

# Clip each tile to the bbox in its native CRS
for src_path in tifs:
    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(4326) 
        left, bottom, right, top = rasterio._warp._transform_bounds(
            dst_crs, src.crs, *bbox, densify_pts=21
        )

        window = rasterio.windows.from_bounds(
            left, bottom, right, top, transform=src.transform
        ).round_offsets().round_lengths()

        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
        })

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        tmp_files.append(tmp)
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)

srcs = [rasterio.open(fp) for fp in tmp_files]
mosaic, transform = merge(srcs, resampling=rasterio.enums.Resampling.nearest)

meta = srcs[0].meta.copy()
meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform,
    "crs": dst_crs,
    "compress": "lzw",
    "tiled": True,
})

with rasterio.open(out_tif, "w", **meta) as dst:
    dst.write(mosaic)

print(f"Wrote {out_tif}")

for fp in tmp_files:
    os.remove(fp)
