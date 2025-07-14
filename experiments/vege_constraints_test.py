import numpy as np
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath("src"))
from embedding.vege_encoder import *

def collect_mean_distances(ds):
    """Return a numpy array of the mean_dist for every patch in ds."""
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    dists = []
    for patches, mean_d in loader:
        dists.extend(mean_d.numpy())
    return np.asarray(dists)

if __name__ == "__main__":
    dataset = VegetationPatchDataset(TIF, level=LEVEL)
    all_mean_dists = collect_mean_distances(dataset)
    print(all_mean_dists)

    count = 0
    for _ in range(100):
        a_d = np.random.choice(all_mean_dists)
        p_d = np.random.choice(all_mean_dists)
        n_d = np.random.choice(all_mean_dists)
        if abs(a_d - p_d) < 0.001 and abs(a_d - n_d) > 0.005:
            count += 1

    print(f"Valid triplets in 100 tries: {count}")
