import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
from scipy.ndimage import zoom
from s2sphere import CellId, LatLng, Cell

TIF = "./data/tif_files/vegetation/switzerland_vegetation_cover.tif"
PATCH_SIZE = 64
LEVEL = 16
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VegetationPatchDataset(Dataset):
    def __init__(self, tif_path, level=LEVEL):
        self.src = rasterio.open(tif_path)
        self.veg_map = self.src.read(1).astype(np.float32)
        self.veg_map = np.nan_to_num(self.veg_map, nan=0.0)
        self.level = level
        self.cells = self._generate_cells()

    def _generate_cells(self):
        lat_min, lat_max = 45.82, 47.81
        lon_min, lon_max = 5.96, 10.49
        lat = np.linspace(lat_min, lat_max, 300)
        lon = np.linspace(lon_min, lon_max, 300)
        cells = []
        for la in lat:
            for lo in lon:
                cid = CellId.from_lat_lng(LatLng.from_degrees(la, lo)).parent(self.level).id()
                cells.append((cid, (la, lo)))
        return cells

    def _bounds(self, cid):
        cell = Cell(CellId(cid))
        lats, lons = [], []
        for i in range(4):
            v = cell.get_vertex(i)
            p = LatLng.from_point(v)
            lats.append(p.lat().degrees)
            lons.append(p.lng().degrees)
        return min(lons), min(lats), max(lons), max(lats)

    def _extract(self, bounds):
        try:
            window = rasterio.windows.from_bounds(*bounds, transform=self.src.transform)
            row_off, col_off = int(window.row_off), int(window.col_off)
            height, width = int(window.height), int(window.width)

            patch = self.veg_map[row_off:row_off+height, col_off:col_off+width]
            if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
                return None

            zoom_factors = (PATCH_SIZE / patch.shape[0], PATCH_SIZE / patch.shape[1])
            patch = zoom(patch, zoom_factors, order=1)
            return patch
        except:
            return None

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        cid, _ = self.cells[idx]
        patch = None
        while patch is None:
            bounds = self._bounds(cid)
            patch = self._extract(bounds)
            if patch is None:
                idx = (idx + 1) % len(self)
                cid, _ = self.cells[idx]
        mean_veg = patch.mean()
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / 1000.0
        return patch, torch.tensor(mean_veg, dtype=torch.float32)

def triplet_sampler(dataset):
    while True:
        a_idx = np.random.randint(len(dataset))
        p_idx = (a_idx + np.random.randint(1, 10)) % len(dataset)
        n_idx = np.random.randint(len(dataset))
        a_patch, a_v = dataset[a_idx]
        p_patch, p_v = dataset[p_idx]
        n_patch, n_v = dataset[n_idx]
        if abs(a_v - p_v) < 50 and abs(a_v - n_v) > 150:
            return (a_patch, p_patch, n_patch, a_v, p_v, n_v)

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

if __name__ == "__main__":
    dataset = VegetationPatchDataset(TIF)
    model = VegetationEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for _ in range(len(dataset) // BATCH_SIZE):
            a, p, n, av, pv, nv = zip(*[triplet_sampler(dataset) for _ in range(BATCH_SIZE)])
            a = torch.stack(a).to(DEVICE)
            p = torch.stack(p).to(DEVICE)
            n = torch.stack(n).to(DEVICE)
            av = torch.tensor(av, device=DEVICE)
            pv = torch.tensor(pv, device=DEVICE)
            nv = torch.tensor(nv, device=DEVICE)

            ea, ra = model(a)
            ep, rp = model(p)
            en, rn = model(n)

            loss_con = info_nce(ea, ep, en)
            loss_reg = F.mse_loss(ra, av) + F.mse_loss(rp, pv) + F.mse_loss(rn, nv)
            loss = loss_con + loss_reg

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        pos_sim = F.cosine_similarity(ea, ep, dim=-1).mean().item()
        neg_sim = F.cosine_similarity(ea, en, dim=-1).mean().item()
        print(f"Epoch {epoch+1}: Loss={total / (len(dataset) // BATCH_SIZE):.4f} | PosSim={pos_sim:.3f} | NegSim={neg_sim:.3f}")

    torch.save(model.state_dict(), "./models/vegetation_encoder.pt")
