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
    def __init__(self, shp_path, raster_ref=RASTER_REF, level=LEVEL):
        self.gdf = gpd.read_file(shp_path)
        self.level = level
        self.ref = rasterio.open(raster_ref)
        self.transform = self.ref.transform
        self.shape = (PATCH_SIZE, PATCH_SIZE)
        self.cells = self._generate_cells()
        self.features, self.bins = self._precompute_features()

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
        features = []
        valid_idx = []
        for i, (cid, _) in enumerate(self.cells):
            patch = self._read_patch(self._bounds(cid))
            if patch is not None:
                density = self._road_density(patch)
                structure = self._road_structure(patch)
                feat = np.concatenate([[density], structure])
                features.append(feat)
                valid_idx.append(i)
            else:
                features.append(np.full(9, np.nan))
        features = np.vstack(features)
        valid = ~np.isnan(features[:, 0])
        low = np.where((features[:, 0] < 0.01) & valid)[0]
        med = np.where((features[:, 0] >= 0.01) & (features[:, 0] < 0.05) & valid)[0]
        high = np.where((features[:, 0] >= 0.05) & valid)[0]
        bins = dict(low=low, med=med, high=high)
        return features, bins

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        patch = self._read_patch(self._bounds(self.cells[idx][0]))
        if patch is None:
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        return patch_t, torch.tensor(self.features[idx], dtype=torch.float32)

def triplet_sampler(dataset: RoadPatchDataset):
    bin_key = np.random.choice(["low", "med"])
    neg_key = "high" if bin_key == "low" else "low"

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    a_idx = int(np.random.choice(dataset.bins[bin_key]))
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
        self.reg_head = nn.Linear(128, 1)

    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.proj(f), dim=-1), self.reg_head(f).squeeze(-1)

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
    ds = RoadPatchDataset(SHP)
    model = RoadEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lam = 1.0
    steps = len(ds) // BATCH_SIZE

    for epoch in range(EPOCHS):
        total_loss = total_p = total_n = 0.0
        for _ in range(steps):
            print("Starting")
            batch = [triplet_sampler(ds) for _ in range(BATCH_SIZE)]
            a, p, n, ad, pd, nd = map(torch.stack, zip(*batch))
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

        print(f"Epoch {epoch+1:02d} | Loss={total_loss/steps:.4f} | PosSim={total_p/steps:.3f} | NegSim={total_n/steps:.3f}")

    torch.save(model.state_dict(), "./models/road_encoder.pt")
