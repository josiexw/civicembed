import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.decomposition import PCA

# === Config ===
PARQUET_PATH = "./data/geographic_data/road_embeddings.parquet"
OUTPUT_PATH = "./experiments/embedding_space/road_embedding_space.png"
MAX_POINTS = 80000
POINT_SIZE = 20
ALPHA = 0.5
COLORMAP = "plasma"

df = pd.read_parquet(PARQUET_PATH, columns=["lat", "lon", "road_embedding"])
if len(df) > MAX_POINTS:
    df = df.sample(MAX_POINTS, random_state=42).reset_index(drop=True)
coords = df[["lon", "lat"]].to_numpy()
embeddings = np.stack(df["road_embedding"].tolist())
pca = PCA(n_components=1)
values_1d = pca.fit_transform(embeddings).flatten()
norm = Normalize(vmin=values_1d.min(), vmax=values_1d.max())
cmap = cm.get_cmap(COLORMAP)
colors = cmap(norm(values_1d))
colors[:, 3] = ALPHA

# === Plot ===
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(coords[:, 0], coords[:, 1], color=colors, s=POINT_SIZE, marker='s')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Road Embedding Similarity (1D PCA)")
ax.set_aspect("equal")

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="Similarity (1D PCA)")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()
