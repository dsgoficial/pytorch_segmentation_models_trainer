---
sidebar_position: 12
title: GEE LULC Downloader
---

# GEE LULC Downloader

Download land-use/land-cover (LULC) rasters from Google Earth Engine for any
set of regions.

Three products are supported:

| Source | CLI key | Native resolution |
|---|---|---|
| MapBiomas | `mapbiomas` | 30 m |
| Dynamic World | `dynamic-world` | 10 m |
| ESRI LULC | `esri-lulc` | 10 m |

## Installation

```bash
pip install earthengine-api
```

## Authentication

Google Earth Engine does **not** support plain username/password login.
Two options are available:

### Service account (recommended for servers/CI)

Create a service account in the Google Cloud Console, grant it Earth Engine
access, and download its JSON private key.

```bash
pytorch-smt-tools download-gee-lulc \
  --source mapbiomas --year 2022 \
  --bbox -48.5 -23.5 -47.5 -22.5 \
  --output-dir /data/mapbiomas \
  --project my-gee-project \
  --service-account-email bot@my-project.iam.gserviceaccount.com \
  --service-account-key-file /secrets/gee_key.json
```

Credentials can also be provided via environment variables to avoid exposing
them in shell history:

```bash
export GEE_SERVICE_ACCOUNT_EMAIL=bot@my-project.iam.gserviceaccount.com
export GEE_SERVICE_ACCOUNT_KEY_FILE=/secrets/gee_key.json

pytorch-smt-tools download-gee-lulc \
  --source mapbiomas --year 2022 \
  --bbox -48.5 -23.5 -47.5 -22.5 \
  --output-dir /data/mapbiomas \
  --project my-gee-project
```

### OAuth2 (interactive — requires browser access)

Omitting the service account flags triggers `ee.Authenticate()`, which opens a
browser window (or prints an authorization URL for code-based flow).

```bash
pytorch-smt-tools download-gee-lulc \
  --source mapbiomas --year 2022 \
  --bbox -48.5 -23.5 -47.5 -22.5 \
  --output-dir /data/mapbiomas \
  --project my-gee-project
```

## Input: bounding box or vector file

### Bounding box

Downloads a single GeoTIFF covering the rectangle.

```bash
# xmin ymin xmax ymax (WGS84 degrees)
--bbox -48.5 -23.5 -47.5 -22.5
```

### Vector file (GeoPackage or GeoJSON)

Each polygon produces a separate GeoTIFF named after an attribute value.

```bash
--vector-path /data/municipalities.gpkg \
--name-attribute cod_ibge
```

If `--name-attribute` is omitted, files are named `region_0.tif`, `region_1.tif`, …

## Download modes

### direct (default)

Uses GEE's `getDownloadURL`. Works for regions up to roughly 1° × 1° at
native resolution. No Drive setup required.

```bash
--download-mode direct
```

### export-drive

Submits a `Export.image.toDrive` task. Suitable for large regions.
The file lands in Google Drive; download it from there after the task completes.

```bash
--download-mode export-drive \
--drive-folder GEE_Downloads
```

## MapBiomas collections

The default collection is **Brazil** (`--mapbiomas-collection brazil`).
Pass a different key for other regions:

| Key | Region |
|---|---|
| `brazil` | Brazil (default) |
| `amazon` | Pan-Amazonia |
| `chaco` | Gran Chaco |
| `atlantic_forest` | Atlantic Forest |
| `pampa` | Pampa |
| `peru` | Peru |
| `bolivia` | Bolivia |
| `colombia` | Colombia |
| `venezuela` | Venezuela |
| `ecuador` | Ecuador |
| `paraguay` | Paraguay |
| `argentina` | Argentina |
| `uruguay` | Uruguay |
| `indonesia` | Indonesia |

To use a custom collection not listed above, pass the full GEE asset ID:

```bash
--mapbiomas-collection-id projects/my-org/public/my_collection
```

## Parallelism

Downloads run in parallel using `ThreadPoolExecutor`. Control the number of
threads with `--max-workers` (default 4):

```bash
--max-workers 8
```

## Python API

```python
from pytorch_segmentation_models_trainer.tools.gee.gee_downloader import (
    initialize_gee,
    load_regions,
    download_regions,
)

initialize_gee(
    project="my-gee-project",
    service_account_email="bot@my-project.iam.gserviceaccount.com",
    service_account_key_file="/secrets/gee_key.json",
)

regions = load_regions(
    vector_path="/data/municipalities.gpkg",
    name_attribute="cod_ibge",
)

results = download_regions(
    regions=regions,
    output_dir="/data/mapbiomas",
    source="mapbiomas",
    year=2022,
    max_workers=8,
)
# results: {"cod_1": True, "cod_2": False, ...}
```

## CLI reference

```
Usage: pytorch-smt-tools download-gee-lulc [OPTIONS]

Options:
  --source          [mapbiomas|dynamic-world|esri-lulc]  (required)
  --year            INT                                  (required)
  --output-dir      PATH                                 (required)
  --bbox            XMIN YMIN XMAX YMAX
  --vector-path     PATH
  --name-attribute  TEXT
  --download-mode   [direct|export-drive]   default: direct
  --max-workers     INT                     default: 4
  --scale           INT                     default: product native
  --crs             TEXT                    default: EPSG:4326
  --drive-folder    TEXT                    default: GEE_Downloads
  --mapbiomas-collection   TEXT             default: brazil
  --mapbiomas-collection-id TEXT
  --project         TEXT
  --service-account-email    TEXT  env: GEE_SERVICE_ACCOUNT_EMAIL
  --service-account-key-file PATH  env: GEE_SERVICE_ACCOUNT_KEY_FILE
```
