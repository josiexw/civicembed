# src/dataloader/contrastive_triplets.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import pandas as pd

# === Save contrastive learning datasets ===
TRIPLET_SAVE_PATH = "./data/triplets.pt"
TERRAIN_PARQUET_PATH = "./data/geographic_data/terrain_embeddings.parquet"
KEYWORD_PATH = "./data/keyword_vecs.pt"

if __name__ == "__main__":
    df = pd.read_parquet(TERRAIN_PARQUET_PATH)
    keyword_vecs = torch.load(KEYWORD_PATH)

    print("Precomputing anchor, positive, negative triplets...")
    triplets = []

    for idx in tqdm(range(len(df))):
        anchor_row = df.iloc[idx]
        anchor_kw = keyword_vecs[idx]
        anchor_terrain = torch.tensor(anchor_row["terrain_embedding"], dtype=torch.float32)

        # Find positive
        distances = ((df["lat"] - anchor_row["lat"]) ** 2 + (df["lon"] - anchor_row["lon"]) ** 2).pow(0.5)
        nearby_idxs = distances[distances < 0.05].index.tolist()
        pos_idx = None

        for i in random.sample(nearby_idxs, min(10, len(nearby_idxs))):
            if i == idx:
                continue
            kw_sim = F.cosine_similarity(anchor_kw.unsqueeze(0), keyword_vecs[i].unsqueeze(0)).item()
            terrain_sim = F.cosine_similarity(anchor_terrain.unsqueeze(0), torch.tensor(df.loc[i, "terrain_embedding"]).unsqueeze(0)).item()
            if kw_sim > 0.7 and terrain_sim > 0.7:
                pos_idx = i
                break

        if pos_idx is None:
            fallback = [i for i in nearby_idxs if i != idx]
            if not fallback:
                continue
            pos_idx = random.choice(fallback)

        # Find negative
        neg_idx = None
        for _ in range(10):
            candidate = random.randint(0, len(df) - 1)
            if candidate in [idx, pos_idx]:
                continue
            kw_sim = F.cosine_similarity(anchor_kw.unsqueeze(0), keyword_vecs[candidate].unsqueeze(0)).item()
            dist = ((anchor_row["lat"] - df.loc[candidate, "lat"]) ** 2 + (anchor_row["lon"] - df.loc[candidate, "lon"]) ** 2) ** 0.5
            if kw_sim < 0.3 and dist > 0.1:
                neg_idx = candidate
                break

        if neg_idx is None:
            far_idxs = [i for i in range(len(df)) if i not in [idx, pos_idx] and
                        ((anchor_row["lat"] - df.loc[i, "lat"]) ** 2 + (anchor_row["lon"] - df.loc[i, "lon"]) ** 2) ** 0.5 > 0.1]
            if not far_idxs:
                continue
            neg_idx = random.choice(far_idxs)

        triplets.append((idx, pos_idx, neg_idx))

    # Save triplets
    print(f"Saving {len(triplets)} triplets to {TRIPLET_SAVE_PATH}...")
    torch.save(triplets, TRIPLET_SAVE_PATH)
