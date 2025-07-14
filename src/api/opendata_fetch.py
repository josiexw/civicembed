# src/api/opendata_fetch.py

import json
import math
import time
import re
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ——— CONFIG ———
BASE = "https://opendata.swiss/api/3/action"
ROWS = 1000
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "opendata"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "opendataswiss_metadata.jsonl"

# Recognized structured formats (all lowercase for consistent comparison)
STRUCTURED_FORMATS = {s.lower() for s in (
    "CSV","XLS","XLSX","JSON","GEOJSON","SHP","GPKG","GML","PARQUET","TSV"
)}

# ——— SET UP A SESSION WITH RETRIES ———
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ——— HELPERS ———

def _sanitize_text(s: str) -> str:
    """Remove Unicode line/paragraph separators."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[\u2028\u2029]", " ", s)

def _bulk_search(start=0, rows=ROWS):
    """
    Returns (batch_results, total_count) from package_search.
    Retries on network errors up to `retries.total` times.
    """
    url = f"{BASE}/package_search"
    params = {"start": start, "rows": rows}
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    result = resp.json()["result"]
    return result["results"], result["count"]

def _richness(meta: dict) -> float:
    """Simple completeness heuristic across core fields."""
    keys = [
        "title", "description", "tags", "keywords", "groups",
        "license_id", "spatial", "accrual_periodicity", "resources", "url"
    ]
    filled = sum(bool(meta.get(k)) for k in keys)
    return round(filled / len(keys), 2)

def _clean(meta: dict) -> dict:
    """Extract + enrich only the fields we care about."""
    # languages
    langs = set(meta.get("language", []))
    for r in meta.get("resources", []):
        langs.update(r.get("language", []))

    # resource formats & URLs
    formats, dlinks, lic = [], [], []
    for r in meta.get("resources", []):
        fmt = (r.get("format") or "").lower().strip()
        if fmt:
            formats.append(fmt)
        dl = r.get("download_url") or r.get("url")
        if dl:
            dlinks.append(dl)
        # collect resource-level license for open_license flag
        lic_url = r.get("license") or r.get("rights")
        if lic_url:
            lic.append(lic_url.lower())

    has_struct = any(f in STRUCTURED_FORMATS for f in formats)
    has_geo    = any(f in {"shp","gpkg","gml","geojson"} for f in formats)

    # groups: handle both dict entries and JSON-string values
    raw_groups = meta.get("groups", [])
    clean_groups = []
    for g in raw_groups:
        if isinstance(g, dict):
            title = g.get("title")
            if isinstance(title, dict):
                clean_groups.append(title.get("en") or next(iter(title.values())))
            elif isinstance(title, str):
                clean_groups.append(title)
        elif isinstance(g, str):
            try:
                obj = json.loads(g)
                clean_groups.append(obj.get("en") or next(iter(obj.values())))
            except Exception:
                clean_groups.append(g)

    # derive new fields
    open_license = any("cc-" in u or "terms_open" in u or "odc-" in u for u in lic)
    structured_formats = sorted({f for f in formats if f in STRUCTURED_FORMATS})
    resource_count_by_format = {fmt: formats.count(fmt) for fmt in set(formats)}
    issued_year = None
    if meta.get("issued"):
        try:
            issued_year = int(meta["issued"][:4])
        except:
            pass

    return {
        "id":                      meta.get("id"),
        "title":                   _sanitize_text(" ".join(meta.get("title", {}).values())),
        "description":             _sanitize_text(" ".join(meta.get("description", {}).values())),
        "tags":                    [t["name"] for t in meta.get("tags", [])],
        "keywords":                sum(meta.get("keywords", {}).values(), []),
        "groups":                  clean_groups,
        "publisher":               meta.get("organization", {}) \
                                          .get("title", {}) \
                                          .get("en", ""),
        "organization":            meta.get("organization", {}).get("name"),
        "license":                 meta.get("license_id") or meta.get("license_title"),
        "open_license":            open_license,
        "language":                sorted(langs),
        "num_resources":           meta.get("num_resources",
                                           len(meta.get("resources", []))),
        "resource_formats":        sorted(set(formats)),
        "structured_formats":      structured_formats,
        "resource_count_by_format":resource_count_by_format,
        "download_urls":           dlinks,
        "spatial":                 meta.get("spatial"),
        "periodicity":             meta.get("accrual_periodicity"),
        "issued":                  meta.get("issued"),
        "issued_year":             issued_year,
        "modified":                meta.get("metadata_modified"),
        "url":                     meta.get("url"),
        "has_structured_resource": has_struct,
        "has_geo":                 has_geo,
        "richness_score":          _richness(meta),
    }

def fetch_and_clean_all(limit=None, sleep=0.1):
    """
    Crawl up to `limit` datasets (all if None), clean them, write JSONL.
    """
    start, total = 0, math.inf
    with OUT_FILE.open("w", encoding="utf-8") as out_f:
        while start < total:
            batch, total = _bulk_search(start)
            if limit is not None:
                total = min(total, limit)

            for meta in batch:
                out_f.write(json.dumps(_clean(meta), ensure_ascii=False) + "\n")

            start += len(batch)
            print(f"→ fetched {start}/{total}")
            if limit is not None and start >= limit:
                break
            time.sleep(sleep)

    print(f"✅ Saved {min(start, total)} records to {OUT_FILE}")

if __name__ == "__main__":
    fetch_and_clean_all()
