import rasterio
from rasterio.merge import merge

tifs = ["./data/tif_files/water/occurrence_0E_50Nv1_4_2021.tif", "./data/tif_files/water/occurrence_10E_50Nv1_4_2021.tif"]
out_tif = "./data/tif_files/water/switzerland_water_cover.tif"
bbox = (5.96, 45.82, 10.49, 47.81)  # lon_min, lat_min, lon_max, lat_max
dst_crs = "EPSG:4326"
srcs = [rasterio.open(fp) for fp in tifs]

# Merge with bounding box and CRS reprojection
mosaic, transform = merge(
    srcs,
    bounds=bbox,
    resampling=rasterio.enums.Resampling.nearest
)

# Use metadata from the first source
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

# Write the merged output
with rasterio.open(out_tif, "w", **meta) as dst:
    dst.write(mosaic)

print(f"Wrote {out_tif}")
