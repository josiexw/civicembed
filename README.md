# CivicEmbed

**Geography-Aware Semantic Retrieval for Civic Datasets**

CivicEmbed is a multimodal embedding system that retrieves relevant civic datasets by topical, spatial, and geographical similarity. It is designed to support urban planners and developers by finding existing solutions from **geographically comparable regions**, improving on traditional keyword-based dataset search.

---

## Overview

Currently, users search for datasets using keyword and location queries (e.g., “transportation in Lausanne”), relying on direct text matches.

CivicEmbed improves this by introducing **geographic similarity**, which includes:
- **Spatial proximity**
- **Similar terrain/topography**
- **Similar vegetation coverage**
- **Similar water coverage**
- **City road network structure**

This enables new use cases in **urban development**, where planners can find comparable infrastructure, design patterns, and statistical reports from regions with similar physical landscapes.

**Input:** Civic dataset  
**Output:** Semantically and geographically similar civic datasets

---

## Implementation Overview

### Geographic Context Representation

Civic datasets include:
- `id`
- `title`, `description`
- `keywords`
- Miscellaneous metadata

We extract spatial metadata from descriptions, get coordinates from location names, and incorporate terrain features via S2 patching of DEMs (Digital Elevation Models).

# Multimodal Embedding Pipeline Overview

This pipeline generates fused embeddings for OpenDataSwiss datasets by combining text, geographic, and terrain features using a contrastively trained encoder.

---

## File Layout and Module Descriptions

### `opendata_fetch.py`
- Downloads metadata from OpenDataSwiss.
- **Output**:  
  - `opendataswiss_metadata.jsonl`

---

### `location_metadata.py`
- Extracts geographic mentions from dataset titles and descriptions.
- **Output**:  
  - `opendataswiss_locations.parquet`  
    *(fields: `id`, `locations`)*

---

### `terrain_encoder.py`
- Defines a terrain feature encoder using a **ResNet-18** backbone.
- Learns from patches in `switzerland_dem.tif` using **InfoNCE contrastive loss**:
  - **Positives**: patches within 5km
  - **Negatives**: patches outside of a 5km radius and with low cosine similarity
- **Output**:  
  - `terrain_encoder.pt`

---

### `terrain_embeddings.py`
- Extracts S2 cell patches at level 16 using `s2sphere`.
- Resolves location names to coordinates using Nominatim (via Docker).
- Applies `terrain_encoder.pt` to generate terrain embeddings.
- **Output**:  
  - `terrain_embeddings.parquet`  
    *(fields: `id`, `lat`, `lon`, `location`, `terrain_patch_id`, `terrain_embedding`, `title`, `description`, `keywords`)*
  - Terrain patches saved to: `data/terrain_patches/`

---

### `keyword_embeddings.py`
- Encodes dataset keywords using multilingual BERT (`bert-base-multilingual-uncased`).
- **Output**:  
  - `keyword_vecs.pt`  
    *(shape: `[N, 128]`)*

---

### `contrastive_triplets.py`
- Generates anchor/positive/negative triplets for contrastive training.
- **Positive/Negative selection based on**:
  - Geographic proximity
  - Text similarity
  - Terrain embedding similarity
- **Output**:  
  - `triplets.pt`  
    *(triplet index tensors for training)*

---

### `fused_encoder.py`
- Final encoder combining:
  - **Text encoder**: multilingual BERT → 128D projection  
  - **Geo encoder**: coordinates → 128D  
  - **Terrain encoder**: terrain patch → 128D  
  - **Fusion**: concatenation → 256 → 128D
- Trained using contrastive learning with precomputed triplets.
- **Input**:
  ```python
  text_descriptions: List[str]
  coordinates: Tensor[N, 2]
  terrain_embeddings: Tensor[N, C, H, W]

---

## Retrieval Pipeline

1. Precompute embeddings using the fused encoder
2. At query time, encode input metadata + coordinates + terrain
3. Retrieve similar datasets using cosine similarity (top-K search)

---

## Results

Query:
> id 22227598-... *Grossratswahlen 2020: Kandidatenstimmen nach Herkunft der Stimmen (Panaschierstatistik) Bezirk Münchwilen*

Top 5 retrieved:
```
id                                     keywords
f0e52fe8-...   panaschieren kantonsrat bezirk grossratswahlen
2787a927-...   kandidat gemeinde kantonale-wahlen stimmen
...
```

---

## Module Overview

```
civicembed/
├── data/
│   ├── opendata/
│   │   └── opendataswiss_metadata.jsonl
│   │   └── opendataswiss_locations.parquet
│   ├── geographic_data/
│   │   └── terrain_patches/
│   │   └── terrain_embeddings.parquet
│   ├── tif_files/
│   │   └── roads/
│   │   └── terrain/
│   │   └── vegetation/
│   │   └── water/
│   ├── keyword_vecs.pt
│   └── triplets.pt
│
├── models/
│   ├── terrain_encoder.pt
│   └── fused_encoder.pt
│
├── src/
│   ├── api/
│   │   └── opendata_fetch.py
│   │   └── location_metadata.py
│   ├── dataloader/
│   │   └── keyword_embeddings.py
│   │   └── contrastive_triplets.py
│   │   └── terrain_embeddings.py
│   ├── embedding/
│   │   └── road_encoder.py
│   │   └── terrain_encoder.py
│   │   └── vege_encoder.py
│   │   └── water_encoder.py
│   │   └── fused_encoder.py
│   └── demo.py
│
├── notebooks/
│   └── fused_encoder.ipynb
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/josiexw/civicembed.git
cd civicembed
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
