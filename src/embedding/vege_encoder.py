import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
from s2sphere import CellId, LatLng, Cell

# === Config ===
TIF = "./data/tif_files/vegetation/switzerland_vegetation_cover.tif"
PATCH_SIZE = 64
LEVEL = 16
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Vegetation Patch Dataset ===
class VegetationPatchDataset(Dataset):
    def __init__(self, tif_path, level=LEVEL):
        self.src = rasterio.open(tif_path)
        self.level = level
        self.cells, self.mean_veg = self._generate_cells()

        # Create bins based on mean vegetation
        self.low_bin = np.where((self.mean_veg >= 0) & (self.mean_veg < 5))[0]
        self.mid_bin = np.where((self.mean_veg >= 5) & (self.mean_veg <= 40))[0]
        self.high_bin = np.where(self.mean_veg > 40)[0]
        
    def _generate_cells(self):
        left, bottom, right, top = self.src.bounds
        lat = np.linspace(bottom, top, 300)
        lon = np.linspace(left, right, 300)
        cells, means = [], []

        for la in lat:
            for lo in lon:
                cid = CellId.from_lat_lng(LatLng.from_degrees(la, lo)).parent(self.level).id()
                bounds = self._bounds(cid)
                patch = self._read_patch(bounds)
                if patch is not None:
                    cells.append((cid, (la, lo)))
                    means.append(patch.mean())
        return cells, np.array(means)

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
                # print("Out of bounds")
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

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        cid, _ = self.cells[idx]
        patch = self._read_patch(self._bounds(cid))
        mean_veg = patch.mean()
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / 1000.0
        return patch, torch.tensor(mean_veg, dtype=torch.float32)

def triplet_sampler(dataset):
    anchor_idx = np.random.choice(dataset.mid_bin)
    pos_idx = np.random.choice(dataset.mid_bin)
    neg_idx = np.random.choice(dataset.low_bin if np.random.rand() < 0.5 else dataset.high_bin)

    a_patch, a_val = dataset[anchor_idx]
    p_patch, p_val = dataset[pos_idx]
    n_patch, n_val = dataset[neg_idx]
    return a_patch, p_patch, n_patch, a_val, p_val, n_val

# === Vegetation Encoder ===
class VegetationEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, dim)
        self.reg_head = nn.Linear(128, 1)

    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)
        emb = F.normalize(self.proj(f), dim=-1)
        reg = self.reg_head(f).squeeze(-1)
        return emb, reg

def info_nce(anchor, positive, negative, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    pos_sim = torch.exp(torch.sum(anchor * positive, dim=-1) / temperature)
    neg_sim = torch.exp(torch.sum(anchor * negative, dim=-1) / temperature)
    return -torch.log(pos_sim / (pos_sim + neg_sim)).mean()

# === Training ===
if __name__ == "__main__":
    ds = VegetationPatchDataset(TIF)
    model = VegetationEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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

    torch.save(model.state_dict(), "./models/vegetation_encoder.pt")
