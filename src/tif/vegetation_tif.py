import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling

DECIDUOUS_TIF = "./data/tif_files/vegetation/vege_copernicus_deciduous_100.tif"
CONIFEROUS_TIF = "./data/tif_files/vegetation/vege_copernicus_coniferous_100.tif"
TMP_TIF = "./data/tif_files/vegetation/switzerland_vegetation_cover_2056"
OUT_TIF = "./data/tif_files/vegetation/switzerland_vegetation_cover.tif"

src_dec = rasterio.open(DECIDUOUS_TIF)
src_con = rasterio.open(CONIFEROUS_TIF)

if src_dec.crs != src_con.crs:
    raise ValueError("CRS mismatch; reproject one file first.")

# resample smaller pixel size to larger
if src_dec.res != src_con.res:
    high_res_src, low_res_src = (src_dec, src_con) if src_dec.res[0] < src_con.res[0] else (src_con, src_dec)
    data = low_res_src.read(
        out_shape=(low_res_src.count,
                   int(high_res_src.height),
                   int(high_res_src.width)),
        resampling=Resampling.nearest)
    transform = high_res_src.transform
    profile = high_res_src.profile
    profile.update(height=high_res_src.height, width=high_res_src.width, transform=transform)
    tmp_path = "_tmp_resampled.tif"
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
    low_res_src = rasterio.open(tmp_path)
    src_dec, src_con = (high_res_src, low_res_src) if src_dec.res[0] < src_con.res[0] else (low_res_src, high_res_src)

mosaic_arr, mosaic_transform = merge([src_dec, src_con], method='max')
meta = src_dec.meta.copy()
meta.update({
    "height": mosaic_arr.shape[1],
    "width": mosaic_arr.shape[2],
    "transform": mosaic_transform,
    "driver": "GTiff",
    "compress": "lzw",
    "tiled": True
})

with rasterio.open(TMP_TIF, "w", **meta) as dst:
    dst.write(mosaic_arr)

print(f"Merged vegetation raster saved to {TMP_TIF}")

# === Project output to CRS = EPSG:4326 ===
def reproject_to_wgs84(input_path, output_path):
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update({
            "crs": "EPSG:4326",
            "transform": transform,
            "width": width,
            "height": height,
            "driver": "GTiff",
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": src.nodata
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear
                )

        print(f"Saved reprojected raster to: {output_path}")

reproject_to_wgs84(TMP_TIF, OUT_TIF)

with rasterio.open(OUT_TIF) as src:
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bounds:", src.bounds)
