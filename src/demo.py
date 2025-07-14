# src/demo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import os
import rasterio
from s2sphere import CellId, LatLng, Cell
from dataloader.keyword_embeddings import TEXT_MODEL
from embedding.fused_encoder import EMBED_DIM, FusedEncoder
from embedding.terrain_encoder import DEM_PATH, LEVEL, PATCH_SIZE

# === Config ===
MODEL_PATH = "models/fused_encoder.pt"
PATCH_DIR = "data/geographic_data/terrain_patches"
DATA_PATH = "data/geographic_data/terrain_embeddings.parquet"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 5

# === DEM ===
dem = rasterio.open(DEM_PATH)

# === Override text_encoder for inference ===
class InferenceTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.bert = AutoModel.from_pretrained(TEXT_MODEL)
        self.proj = nn.Linear(self.bert.config.hidden_size, EMBED_DIM)

    def forward(self, texts):
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        out = self.bert(**encoded).last_hidden_state[:, 0]
        return self.proj(out)

# === Patch extraction ===
def s2_cell_bounds(lat, lon):
    cell = CellId.from_lat_lng(LatLng.from_degrees(lat, lon)).parent(LEVEL)
    cell = Cell(cell)
    lats, lons = [], []
    for i in range(4):
        vertex = cell.get_vertex(i)
        latlng = LatLng.from_point(vertex)
        lats.append(latlng.lat().degrees)
        lons.append(latlng.lng().degrees)
    return (min(lons), min(lats), max(lons), max(lats))

def extract_patch(lat, lon, patch_id):
    path = os.path.join(PATCH_DIR, f"{patch_id}.npy")
    if os.path.exists(path):
        patch = np.load(path, mmap_mode='r')
        return torch.tensor(patch).unsqueeze(0).float() / 1000.0
    else:
        try:
            bounds = s2_cell_bounds(lat, lon)
            window = rasterio.windows.from_bounds(*bounds, transform=dem.transform)
            patch = dem.read(1, window=window, out_shape=(PATCH_SIZE, PATCH_SIZE),
                             resampling=rasterio.enums.Resampling.bilinear)
            if patch.shape != (PATCH_SIZE, PATCH_SIZE) or np.isnan(patch).any():
                raise ValueError("Invalid patch")
            os.makedirs(PATCH_DIR, exist_ok=True)
            np.save(path, patch)
            return torch.tensor(patch).unsqueeze(0).float() / 1000.0
        except Exception as e:
            print(f"Failed to extract patch for {patch_id}: {e}")
            return None

# === Load model and override text_encoder ===
empty_keyword_vecs = torch.zeros(1, EMBED_DIM).to(DEVICE)
model = FusedEncoder(empty_keyword_vecs).to(DEVICE)
model.text_encoder = InferenceTextEncoder().to(DEVICE)
model.eval()

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith("text_encoder.")  # skip incompatible text_encoder weights
}

# === Load data ===
df = pd.read_parquet(DATA_PATH)

# === Compute embeddings ===
embeddings = []
with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding DB entries"):
        text = row["keywords"]
        coords = torch.tensor([[row["lat"], row["lon"]]], dtype=torch.float32).to(DEVICE)
        patch = extract_patch(row["lat"], row["lon"], row["terrain_patch_id"])
        if patch is None:
            embeddings.append(torch.zeros(model.fuse[-1].out_features))  # fallback
            continue
        patch = patch.to(DEVICE).unsqueeze(0)
        emb = model(text, coords, patch)
        embeddings.append(emb.squeeze(0).cpu())

embeddings = torch.stack(embeddings)


def find_top_k_similar(query_id, k=K):
    idx = df[df["id"] == query_id].index[0]
    query_emb = embeddings[idx].unsqueeze(0)
    sims = F.cosine_similarity(query_emb, embeddings)
    sorted_indices = torch.argsort(sims, descending=True)
    seen_ids = set([query_id])
    topk_indices = []

    for i in sorted_indices:
        row_id = df.iloc[i]["id"]
        if row_id not in seen_ids:
            topk_indices.append(i.item())
            seen_ids.add(row_id)
        if len(topk_indices) == k:
            break

    return df.iloc[topk_indices]


if __name__ == "__main__":
    test_id = df.iloc[0]["id"]
    print(f"Top {K} similar to {test_id}:")
    print(find_top_k_similar(test_id))
