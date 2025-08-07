import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
from s2sphere import CellId, LatLng, Cell

RASTER_PATH = "./data/tif_files/roads/switzerland_roads.tif"
PATCH_SIZE = 64
LEVEL = 16
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoadPatchDataset(Dataset):
    def __init__(self, tif_path=RASTER_PATH, level=LEVEL):
        self.tif = rasterio.open(tif_path)
        self.shape = (PATCH_SIZE, PATCH_SIZE)
        self.level = level
        self.cells = self._generate_cells()
        self.features, self.bins = self._precompute_features()

    def _generate_cells(self):
        left, bottom, right, top = self.tif.bounds
        lat = np.linspace(bottom, top, 300)
        lon = np.linspace(left, right, 300)
        cells = []
        for la in lat:
            for lo in lon:
                try:
                    cid = CellId.from_lat_lng(LatLng.from_degrees(la, lo)).parent(self.level).id()
                    cells.append((cid, (la, lo)))
                except Exception as e:
                    print(f"Failed to generate CellId for ({la}, {lo}): {e}")
        return cells

    def _bounds(self, cid):
        try:
            cell = Cell(CellId(cid))
            lat, lon = zip(*[(LatLng.from_point(cell.get_vertex(i)).lat().degrees,
                              LatLng.from_point(cell.get_vertex(i)).lng().degrees)
                             for i in range(4)])
            return min(lon), min(lat), max(lon), max(lat)
        except Exception as e:
            print(f"Failed to compute bounds for cid {cid}: {e}")
            return None

    def _read_patch(self, bounds):
        try:
            if bounds is None:
                return None
            win = rasterio.windows.from_bounds(*bounds, transform=self.tif.transform)
            win = win.round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, self.tif.width, self.tif.height))
            if win.width < 2 or win.height < 2:
                return None
            patch = self.tif.read(1, window=win, out_shape=self.shape,
                                  resampling=rasterio.enums.Resampling.nearest)
            if patch.shape != self.shape or np.isnan(patch).any():
                return None
            return patch.astype(np.float32)
        except Exception as e:
            print(f"Failed to read patch: {e}")
            return None

    def _road_density(self, patch):
        return patch.sum() / (PATCH_SIZE * PATCH_SIZE)

    def _road_structure(self, patch):
        v, h = np.gradient(patch)
        angle_map = np.arctan2(v, h)
        return np.histogram(angle_map, bins=8, range=(-np.pi, np.pi))[0].astype(np.float32)

    def _precompute_features(self):
        features = []
        for cid, _ in self.cells:
            patch = self._read_patch(self._bounds(cid))
            if patch is not None:
                density = self._road_density(patch)
                structure = self._road_structure(patch)
                feat = np.concatenate([[density], structure])
                features.append(feat)
            else:
                features.append(np.full(9, np.nan))

        features = np.vstack(features)
        valid = ~np.isnan(features[:, 0])
        low = np.where((features[:, 0] < 0.01) & valid)[0]
        med = np.where((features[:, 0] >= 0.01) & (features[:, 0] < 0.05) & valid)[0]
        high = np.where((features[:, 0] >= 0.05) & valid)[0]
        bins = dict(low=low, med=med, high=high)
        return torch.tensor(features, dtype=torch.float32), bins

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        for _ in range(5):
            bounds = self._bounds(self.cells[idx][0])
            patch = self._read_patch(bounds)
            if patch is not None and not torch.isnan(self.features[idx][0]):
                patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
                return patch_t, self.features[idx]
            idx = (idx + 5) % len(self)
        raise RuntimeError("Failed to fetch a valid patch after 5 retries")

def triplet_sampler(dataset: RoadPatchDataset):
    bin_key = np.random.choice(["low", "med"])
    neg_key = "high" if bin_key == "low" else "low"
    a_idx = int(np.random.choice(dataset.bins[bin_key]))
    anchor_feat = dataset.features[a_idx].numpy()
    same_bin = dataset.bins[bin_key]
    feats = dataset.features[same_bin]
    sims = torch.nn.functional.cosine_similarity(torch.tensor(anchor_feat), feats, dim=1)
    pos_idx = same_bin[torch.argmax(sims).item()]
    neg_idx = int(np.random.choice(dataset.bins[neg_key]))
    return (*dataset[a_idx], *dataset[pos_idx], *dataset[neg_idx])

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

def info_nce(anchor, positive, negative, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    pos_sim = torch.exp(torch.sum(anchor * positive, dim=-1) / temperature)
    neg_sim = torch.exp(torch.sum(anchor * negative, dim=-1) / temperature)
    return -torch.log(pos_sim / (pos_sim + neg_sim)).mean()

if __name__ == "__main__":
    log_path = "./road_encoder_logs.txt"
    try:
        ds = RoadPatchDataset()
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
                    f"Epoch {epoch+1:02d} | Loss={total_loss/steps:.4f} | "
                    f"PosSim={total_p/steps:.3f} | NegSim={total_n/steps:.3f}\n"
                )
                print(log_line.strip())
                log_file.write(log_line)
                log_file.flush()

        torch.save(model.state_dict(), "./road_encoder.pt")
    except Exception as e:
        print(f"Fatal error during training: {e}")
