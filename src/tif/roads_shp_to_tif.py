import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# === Config ===
shp_path = "./data/tif_files/roads/switzerland_roads.shp"
output_tif = "./data/tif_files/roads/switzerland_roads.tif"
pixel_size = 0.0001  # ~10 meters per pixel
gdf = gpd.read_file(shp_path)
bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
width = int((bounds[2] - bounds[0]) / pixel_size)
height = int((bounds[3] - bounds[1]) / pixel_size)

transform = from_origin(bounds[0], bounds[3], pixel_size, pixel_size)

# === Rasterize ===
raster = rasterize(
    ((geom, 1) for geom in gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

with rasterio.open(
    output_tif,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="uint8",
    crs=gdf.crs,
    transform=transform,
) as dst:
    dst.write(raster, 1)

print(f"Saved raster to {output_tif}")
