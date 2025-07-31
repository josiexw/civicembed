import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from s2sphere import CellId, LatLng, Cell
from shapely.geometry import box
import os
import pickle
import shutil

# === Config ===
SHP = "./data/tif_files/roads/switzerland_roads.shp"
RASTER_REF = "./data/tif_files/terrain/switzerland_dem.tif"
PATCH_SIZE = 64
LEVEL = 16
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Road Patch Dataset ===
class RoadPatchDataset(Dataset):
    def __init__(self, shp_path, raster_ref=RASTER_REF, level=LEVEL, cache_path="./data/geographic_data/road_patches.pkl"):
        self.cache_path = cache_path
        self.gdf = gpd.read_file(shp_path)
        self.level = level
        self.ref = rasterio.open(raster_ref)
        self.transform = self.ref.transform
        self.shape = (PATCH_SIZE, PATCH_SIZE)
        self.cells = self._generate_cells()

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                features, bins, patches = pickle.load(f)
            self.patches = [torch.tensor(p, dtype=torch.float32).unsqueeze(0) for p in patches]
            self.features = torch.tensor(features, dtype=torch.float32)
            self.bins = bins
        else:
            features, bins, patches = self._precompute_features()
            self.patches = [torch.tensor(p, dtype=torch.float32).unsqueeze(0) for p in patches]
            self.features = torch.tensor(features, dtype=torch.float32)
            self.bins = bins
            with open(cache_path, "wb") as f:
                pickle.dump((features, bins, patches), f)
            backup_path = "/mnt/ldm1/scratch/road_patches.pkl"
            shutil.copy(cache_path, backup_path)
            print("Wrote patches to scratch")

    def _generate_cells(self):
        left, bottom, right, top = self.ref.bounds
        lat = np.linspace(bottom, top, 300)
        lon = np.linspace(left, right, 300)
        return [
            (CellId.from_lat_lng(LatLng.from_degrees(la, lo)).parent(self.level).id(),
             (la, lo))
            for la in lat for lo in lon
        ]

    def _bounds(self, cid):
        cell = Cell(CellId(cid))
        lat, lon = zip(*[(LatLng.from_point(cell.get_vertex(i)).lat().degrees,
                          LatLng.from_point(cell.get_vertex(i)).lng().degrees)
                         for i in range(4)])
        return min(lon), min(lat), max(lon), max(lat)

    def _read_patch(self, bounds):
        geom = box(*bounds)
        clipped = self.gdf[self.gdf.intersects(geom)]
        if clipped.empty:
            return None
        try:
            out = rasterize(
                ((g, 1) for g in clipped.geometry),
                out_shape=self.shape,
                transform=rasterio.transform.from_bounds(*bounds, *self.shape),
                fill=0,
                dtype=np.uint8
            )
            return out.astype(np.float32)
        except Exception:
            return None

    def _road_density(self, patch):
        return patch.sum() / (PATCH_SIZE * PATCH_SIZE)

    def _road_structure(self, patch):
        v, h = np.gradient(patch)
        angle_map = np.arctan2(v, h)
        return np.histogram(angle_map, bins=8, range=(-np.pi, np.pi))[0].astype(np.float32)

    def _precompute_features(self):
        print("Precomputing features")
        features = []
        valid_idx = []
        patches = []
        for i, (cid, _) in enumerate(self.cells):
            patch = self._read_patch(self._bounds(cid))
            if patch is not None:
                density = self._road_density(patch)
                structure = self._road_structure(patch)
                feat = np.concatenate([[density], structure])
                features.append(feat)
                patches.append(patch)
                valid_idx.append(i)
            else:
                features.append(np.full(9, np.nan))
                patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
        features = np.vstack(features)
        valid = ~np.isnan(features[:, 0])
        low = np.where((features[:, 0] < 0.01) & valid)[0]
        med = np.where((features[:, 0] >= 0.01) & (features[:, 0] < 0.05) & valid)[0]
        high = np.where((features[:, 0] >= 0.05) & valid)[0]
        bins = dict(low=low, med=med, high=high)
        return features, bins, patches

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        return self.patches[idx], self.features[idx]

def triplet_sampler(dataset: RoadPatchDataset):
    bin_key = np.random.choice(["low", "med"])
    neg_key = "high" if bin_key == "low" else "low"

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    a_idx = np.random.choice(dataset.bins[bin_key]).item()
    anchor_feat = dataset.features[a_idx]

    same_bin = dataset.bins[bin_key]
    pos_idx = max(same_bin, key=lambda i: cosine_sim(anchor_feat, dataset.features[i]))
    neg_idx = int(np.random.choice(dataset.bins[neg_key]))

    return (*dataset[a_idx], *dataset[pos_idx], *dataset[neg_idx])

# === Road Encoder ===
class RoadEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, dim)
        self.reg_head = nn.Linear(128, 9)

    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.proj(f), dim=-1), self.reg_head(f)

# === InfoNCE Loss ===
def info_nce(anchor, positive, negative, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    pos_sim = torch.exp(torch.sum(anchor * positive, dim=-1) / temperature)
    neg_sim = torch.exp(torch.sum(anchor * negative, dim=-1) / temperature)
    return -torch.log(pos_sim / (pos_sim + neg_sim)).mean()

# === Training ===
if __name__ == "__main__":
    log_path = "/mnt/ldm1/scratch/road_encoder_logs.txt"
    ds = RoadPatchDataset(SHP)
    print("Finished creating dataset")
    model = RoadEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lam = 1.0
    steps = len(ds) // BATCH_SIZE

    with open(log_path, "w") as log_file:
        for epoch in range(EPOCHS):
            total_loss = total_p = total_n = 0.0
            for _ in range(steps):
                batch = [triplet_sampler(ds) for _ in range(BATCH_SIZE)]
                a, ad, p, pd, n, nd = map(torch.stack, zip(*batch))
                a, p, n, ad, pd, nd = [t.to(DEVICE) for t in (a, p, n, ad, pd, nd)]

                e_a, d_a = model(a)
                e_p, d_p = model(p)
                e_n, d_n = model(n)

                loss = info_nce(e_a, e_p, e_n) + lam * (
                    F.mse_loss(d_a, ad) + F.mse_loss(d_p, pd) + F.mse_loss(d_n, nd))

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                total_p += F.cosine_similarity(e_a, e_p, dim=-1).mean().item()
                total_n += F.cosine_similarity(e_a, e_n, dim=-1).mean().item()

            log_line = (
                f"Epoch {epoch+1:02d} | Loss={total_loss/steps:.4f} "
                f"| PosSim={total_p/steps:.3f} | NegSim={total_n/steps:.3f}\n"
            )
            print(log_line.strip())
            log_file.write(log_line)
            log_file.flush()

    torch.save(model.state_dict(), "/mnt/ldm1/scratch/road_encoder.pt")
