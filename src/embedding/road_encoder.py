import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys

# Reusing terrain encoder to encode other features
sys.path.append(os.path.abspath("src"))
from embedding.terrain_encoder import TerrainPatchDataset, TerrainEncoder, info_nce


# === Config ===
DEM_PATH = "./data/tif_files/roads/switzerland_map.tif"
OUTPUT_MODEL = "./models/road_encoder.pt"
PATCH_SIZE = 64
LEVEL = 16
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Training ===
if __name__ == "__main__":
    dataset = TerrainPatchDataset(DEM_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = TerrainEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for anchor, pos, neg in loader:
            anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            emb_anchor = model(anchor)
            emb_pos = model(pos)
            emb_neg = model(neg)

            loss = info_nce(emb_anchor, emb_pos, emb_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        with torch.no_grad():
            pos_sim_val = F.cosine_similarity(emb_anchor, emb_pos).mean().item()
            neg_sim_val = F.cosine_similarity(emb_anchor, emb_neg).mean().item()
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | PosSim: {pos_sim_val:.3f} | NegSim: {neg_sim_val:.3f}")

    torch.save(model.state_dict(), OUTPUT_MODEL)
