# -*- coding: utf-8 -*-
"""Per-tile caches of auxiliary rasters read straight from public
Cloud-Optimized GeoTIFFs (no Earth Engine account needed).

- :mod:`.gem15`: GEM-15 surface/terrain elevation models (Google Research,
  15 m, CC-BY 4.0).
- :mod:`.forest_typology`: Forest Typology (ForTy) 2020 v1.0 forest-class
  probabilities (Google / Nature Trace, 10 m, CC-BY 4.0).
- :mod:`.common`: reprojection of one or more source COGs onto a tile's own
  WGS84 grid, and the threaded, resumable batch downloader both use.
"""
