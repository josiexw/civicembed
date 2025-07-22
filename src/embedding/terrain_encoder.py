# src/embedding/terrain_encoder.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import rasterio
from rasterio.transform import Affine
from s2sphere import CellId, LatLng, Cell

# === Config ===
DEM_PATH = "./data/tif_files/terrain/switzerland_dem.tif"
OUTPUT_MODEL = "./models/terrain_encoder.pt"
PATCH_SIZE = 64
LEVEL = 16
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Terrain Patch Dataset ===
class TerrainPatchDataset(Dataset):
    def __init__(self, dem_path, level=LEVEL):
        self.dem = rasterio.open(dem_path)

        # Check DEM is not identity
        if self.dem.crs is None:
            raise ValueError(f"{dem_path} has no CRS metadata. ")
        if self.dem.transform == Affine.identity():
            raise ValueError(f"{dem_path} has an identity affine transform (pixel size 1, origin 0,0). ")

        self.level = level
        self.cells = self.generate_s2_cells()
        self.pos_neighbors = self.precompute_neighbors()

    def generate_s2_cells(self):
        lat_min, lat_max = 45.82, 47.81
        lon_min, lon_max = 5.96, 10.49
        lat_steps = np.linspace(lat_min, lat_max, 300)
        lon_steps = np.linspace(lon_min, lon_max, 300)

        cell_ids_coords = []
        for lat in lat_steps:
            for lon in lon_steps:
                cell = CellId.from_lat_lng(LatLng.from_degrees(lat, lon)).parent(self.level)
                cell_ids_coords.append((cell.id(), (lat, lon)))
        return cell_ids_coords

    def precompute_neighbors(self):
        coords = np.array([latlon for _, latlon in self.cells])
        neighbors = []

        for i, (lat1, lon1) in enumerate(coords):
            dlat = np.radians(coords[:, 0] - lat1)
            dlon = np.radians(coords[:, 1] - lon1)
            phi1 = np.radians(lat1)
            phi2 = np.radians(coords[:, 0])
            a = np.sin(dlat/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon/2)**2
            distances = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            mask = (distances > 1.0) & (distances < 5.0)
            neighbors.append(np.where(mask)[0].tolist())

        return neighbors

    def s2_cell_bounds(self, cell_id):
        cell = Cell(CellId(cell_id))
        latitudes, longitudes = [], []
        for i in range(4):
            vertex = cell.get_vertex(i)
            latlng = LatLng.from_point(vertex)
            latitudes.append(latlng.lat().degrees)
            longitudes.append(latlng.lng().degrees)
        return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))

    def extract_patch(self, bounds):
        try:
            window = rasterio.windows.from_bounds(*bounds, transform=self.dem.transform)
            patch = self.dem.read(1, window=window, out_shape=(PATCH_SIZE, PATCH_SIZE), resampling=rasterio.enums.Resampling.bilinear)
            if patch.shape != (PATCH_SIZE, PATCH_SIZE) or np.isnan(patch).any():
                return None
            return patch
        except:
            return None
        
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        anchor_id, (anchor_lat, anchor_lon) = self.cells[idx]
        bounds = self.s2_cell_bounds(anchor_id)
        anchor = self.extract_patch(bounds)

        if anchor is None:
            return self.__getitem__((idx + 1) % len(self))

        anchor_tensor = torch.tensor(anchor).unsqueeze(0).float() / 1000.0
        anchor_flat = anchor_tensor.flatten()

        # Sample positive patch: within 5km radius
        candidates = self.pos_neighbors[idx]
        np.random.shuffle(candidates)
        for pos_idx in candidates:
            pos_id, _ = self.cells[pos_idx]
            pos_bounds = self.s2_cell_bounds(pos_id)
            positive = self.extract_patch(pos_bounds)
            if positive is not None:
                positive_tensor = torch.tensor(positive).unsqueeze(0).float() / 1000.0
                break
        else:
            return self.__getitem__((idx + 3) % len(self))

        # Sample negative patch: cosine similarity < 0.2 and outside of 10km radius
        while True:
            neg_idx = np.random.randint(len(self))
            neg_id, (neg_lat, neg_lon) = self.cells[neg_idx]
            dist = self.haversine(anchor_lat, anchor_lon, neg_lat, neg_lon)
            if dist <= 10.0:
                continue

            neg_bounds = self.s2_cell_bounds(neg_id)
            negative = self.extract_patch(neg_bounds)
            if negative is None:
                continue

            negative_tensor = torch.tensor(negative).unsqueeze(0).float() / 1000.0
            if F.cosine_similarity(anchor_flat, negative_tensor.flatten(), dim=0) < 0.2:
                break

        return anchor_tensor, positive_tensor, negative_tensor

# === Terrain Encoder ===
class TerrainEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        base = models.resnet18(pretrained=True)

        weight = base.conv1.weight
        new_weight = weight.sum(dim=1, keepdim=True) / 3.0
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = new_weight

        self.encoder = nn.Sequential(*(list(base.children())[:-1]))
        self.proj = nn.Linear(512, dim)

    def forward(self, x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)

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
