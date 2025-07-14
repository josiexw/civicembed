import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
from s2sphere import CellId, LatLng, Cell

# === Config ===
TIF = "./data/tif_files/water/switzerland_water_proximity_km.tif"
PATCH_SIZE = 64
LEVEL = 12
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Water Patch Dataset ===
class WaterPatchDataset(Dataset):
    def __init__(self, tif_path: str, level: int = LEVEL):
        self.src = rasterio.open(tif_path)
        self.level = level
        self.cells = self._generate_cells()
        self.mean_km, self.bins = self._precompute_distances()

    def _generate_cells(self):
        left, bottom, right, top = self.src.bounds
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
        try:
            win = rasterio.windows.from_bounds(*bounds, transform=self.src.transform)
            win = win.round_offsets().round_lengths()

            if (win.col_off >= self.src.width or win.row_off >= self.src.height or
                win.col_off + win.width <= 0 or win.row_off + win.height <= 0):
                print("Out of bounds")
                return None

            # Clip to raster
            win = win.intersection(rasterio.windows.Window(0, 0, self.src.width, self.src.height))
            if win.width < 2 or win.height < 2:
                return None

            patch = self.src.read(
                1,
                window=win,
                out_shape=(PATCH_SIZE, PATCH_SIZE),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            if patch.shape != (PATCH_SIZE, PATCH_SIZE) or np.isnan(patch).any():
                return None
            return patch
        except Exception:
            return None

    def _precompute_distances(self):
        mean_km = []
        valid_idx = []
        for i, (cid, _) in enumerate(self.cells):
            patch = self._read_patch(self._bounds(cid))
            if patch is not None:
                mean_km.append(patch.mean())
                valid_idx.append(i)
            else:
                mean_km.append(np.nan)
        mean_km = np.asarray(mean_km)

        # Build bins only from valid indices
        valid = ~np.isnan(mean_km)
        near = np.where((mean_km < 0.2) & valid)[0]
        mid  = np.where((mean_km >= 0.2) & (mean_km < 1.0) & valid)[0]
        far  = np.where((mean_km >= 1.0) & valid)[0]
        bins = dict(near=near, mid=mid, far=far)
        return mean_km, bins

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        patch = self._read_patch(self._bounds(self.cells[idx][0]))
        if patch is None:
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # (1,64,64)
        return patch_t / 10.0, torch.tensor(self.mean_km[idx] if not np.isnan(self.mean_km[idx]) else 0.0,
                                           dtype=torch.float32)

def triplet_sampler(dataset: WaterPatchDataset):
    anchor_bin = np.random.choice(["near", "mid"])
    neg_bin    = "far" if anchor_bin == "near" else "near"

    a_idx = int(np.random.choice(dataset.bins[anchor_bin]))
    p_idx = int(np.random.choice(dataset.bins[anchor_bin]))
    n_idx = int(np.random.choice(dataset.bins[neg_bin]))

    a_patch, a_d = dataset[a_idx]
    p_patch, p_d = dataset[p_idx]
    n_patch, n_d = dataset[n_idx]
    return a_patch, p_patch, n_patch, a_d, p_d, n_d

# === Water Encoder ===
class WaterEncoder(nn.Module):
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
    ds = WaterPatchDataset(TIF)
    model = WaterEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lam = 1.0
    steps = len(ds) // BATCH_SIZE

    for epoch in range(EPOCHS):
        total_loss = total_p = total_n = 0.0
        for _ in range(steps):
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

    torch.save(model.state_dict(), "./models/water_encoder.pt")
