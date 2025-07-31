import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import rasterio
from rasterio.plot import show
from pyproj import Transformer

# === Config ===
PARQUET_PATH = "./data/geographic_data/all_terrain_embeddings.parquet"
MAP_TIF = "./data/tif_files/swiss-map-raster500_500_kgrs_25_2056.tif"
OUTPUT_PATH = "./experiments/embedding_space/terrain_overlay.png"
MAX_POINTS = 80000
POINT_SIZE = 20
ALPHA = 0.5
COLORMAP = "plasma"

df = pd.read_parquet(PARQUET_PATH, columns=["lat", "lon", "terrain_embedding"])
if len(df) > MAX_POINTS:
    df = df.sample(MAX_POINTS, random_state=42).reset_index(drop=True)
coords_wgs84 = df[["lon", "lat"]].to_numpy()
embeddings = np.stack(df["terrain_embedding"].tolist())

# Reproject to EPSG:2056
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
coords_2056 = np.array([transformer.transform(lon, lat) for lon, lat in coords_wgs84])

pca = PCA(n_components=1)
values_1d = pca.fit_transform(embeddings).flatten()
norm = Normalize(vmin=values_1d.min(), vmax=values_1d.max())
cmap = cm.get_cmap(COLORMAP)
colors = cmap(norm(values_1d))
colors[:, 3] = ALPHA

with rasterio.open(MAP_TIF) as src:
    fig, ax = plt.subplots(figsize=(12, 10))
    show(src, ax=ax)
    ax.scatter(coords_2056[:, 0], coords_2056[:, 1], color=colors, s=POINT_SIZE, marker='s')
    ax.set_title("Terrain Embedding Space on Swiss Map")
    ax.set_xlim(src.bounds.left, src.bounds.right)
    ax.set_ylim(src.bounds.bottom, src.bounds.top)
    ax.set_aspect("equal")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Terrain Similarity (1D PCA)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()
