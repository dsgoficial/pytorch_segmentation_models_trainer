# -*- coding: utf-8 -*-
"""Pluggable imagery source resolution.

``read_source_aligned_to_mask_window`` (see ``alignment.py``) has always accepted a
single file path (an MBTiles archive, VRT, or any rasterio-readable raster). This
module adds three more ways to point at imagery that doesn't ship as one pre-built
mosaic:

- A directory of tiles matched to the current mask tile by **spatial bounds**
  overlap (``DirectoryImageSource``, ``match_by="bounds"``, the default) — every
  candidate file's bounds are indexed once (a cheap, header-only open).
- A directory of tiles matched by **basename** — the image file with the same stem
  as the mask tile (``DirectoryImageSource``, ``match_by="basename"``) — no file
  needs to be opened at all to build the index, just listed.
- A **CSV manifest**, one row per tile (``CSVImageSource``) — a ``path`` column is
  required; an optional ``mask_path`` column enables the same direct tile-keyed
  lookup as basename matching (no bounds needed), and optional
  ``minx,miny,maxx,maxy[,crs]`` columns enable bounds-based matching as a fallback
  for rows without a ``mask_path`` match. This mirrors the wide
  ``tile_id,image_path,mask_path,<lulc columns>`` CSV convention already used by
  ``tools.soft_labels`` (``_normalize_sources_dataframe``).

Direct tile-keyed lookup (CSV ``mask_path`` column, or basename matching) is always
tried first when available — it's a plain dict lookup, no spatial computation at
all — falling back to bounds-based search only when no direct entry exists (or no
tile-keyed data was provided in the first place).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.warp import transform_bounds


@dataclass
class DirectoryImageSource:
    """A folder of raster tiles, searched by extension.

    Args:
        directory: Path to the folder of raster tiles.
        extensions: File extensions to include, with or without the leading dot
            (default ``[".tif", ".tiff"]``).
        recursive: Whether to search subdirectories (default False).
        match_by: How a mask tile is paired with a file in this folder:

            - ``"bounds"`` (default): every file's bounds/CRS are indexed once
              (opens each file's header); a window is matched to whichever
              file(s) spatially overlap it.
            - ``"basename"``: no file is opened to build the index — a mask tile
              named ``tile_001.tif`` is paired with ``tile_001.<any extension>``
              in this folder, by stem, with no spatial computation at all.
    """

    directory: str
    extensions: List[str] = field(default_factory=lambda: [".tif", ".tiff"])
    recursive: bool = False
    match_by: str = "bounds"


@dataclass
class CSVImageSource:
    """A CSV manifest, one row per tile.

    Args:
        csv_path: Path to the CSV file.
        path_column: Column holding the image file path (required column name;
            the column itself is required in the CSV).
        mask_path_column: Column holding the corresponding mask tile path, for
            direct tile-keyed lookup (matched by stem against the mask tile
            being processed). Optional — used only if present in the CSV.
        default_crs: CRS to use for rows without a ``crs`` column/value, when
            bounds columns (``minx,miny,maxx,maxy``) are present. Only needed
            for the bounds-based fallback path.
    """

    csv_path: str
    path_column: str = "path"
    mask_path_column: str = "mask_path"
    default_crs: str = ""


ImageSourceSpec = Union[str, dict, DirectoryImageSource, CSVImageSource]

_BOUNDS_COLUMNS = ("minx", "miny", "maxx", "maxy")


def _normalized_extensions(extensions: List[str]) -> set:
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}


def _list_candidate_files(source: DirectoryImageSource) -> List[Path]:
    root = Path(source.directory)
    walker = root.rglob if source.recursive else root.glob
    exts = _normalized_extensions(source.extensions)
    return sorted(p for p in walker("*") if p.is_file() and p.suffix.lower() in exts)


def _bounds_intersect(a: BoundingBox, b: BoundingBox) -> bool:
    return (
        a.left < b.right and a.right > b.left and a.bottom < b.top and a.top > b.bottom
    )


def _coerce_dict_spec(spec: dict) -> Union[DirectoryImageSource, CSVImageSource]:
    if "csv_path" in spec:
        return CSVImageSource(**spec)
    if "directory" in spec:
        return DirectoryImageSource(**spec)
    raise ValueError(
        "Directory/CSV image source dict must include 'directory' or 'csv_path', "
        f"got keys: {sorted(spec)}"
    )


class TiledImageSource:
    """A multi-file image source resolved to a tile lookup and/or bounds index.

    Built once (typically at corrector init time). See the module docstring for
    the three ways a spec can resolve to entries here.

    Args:
        spec: :class:`DirectoryImageSource`, :class:`CSVImageSource`, or an
            equivalent dict (discriminated by ``"directory"`` vs ``"csv_path"``).

    Raises:
        ValueError: If the spec resolves to no usable entries (no matching files
            in a directory, or a CSV with neither ``mask_path`` nor bounds columns).
    """

    def __init__(self, spec: Union[DirectoryImageSource, CSVImageSource, dict]) -> None:
        if isinstance(spec, dict):
            spec = _coerce_dict_spec(spec)
        self._spec = spec
        self._tile_lookup: Dict[str, Path] = {}
        self._entries: List[Tuple[Path, BoundingBox, CRS]] = []

        if isinstance(spec, CSVImageSource):
            self._init_from_csv(spec)
        elif isinstance(spec, DirectoryImageSource):
            self._init_from_directory(spec)
        else:
            raise TypeError(f"Unsupported image source spec: {spec!r}")

        if not self._tile_lookup and not self._entries:
            raise ValueError(f"Image source resolved no usable entries: {spec!r}")

    def _init_from_directory(self, spec: DirectoryImageSource) -> None:
        files = _list_candidate_files(spec)
        if not files:
            where = f"'{spec.directory}'" + (" (recursive)" if spec.recursive else "")
            raise ValueError(f"No files matching {spec.extensions} found in {where}.")

        if spec.match_by == "basename":
            self._tile_lookup = {p.stem: p for p in files}
        elif spec.match_by == "bounds":
            for p in files:
                with rasterio.open(p) as ds:
                    self._entries.append((p, ds.bounds, ds.crs))
        else:
            raise ValueError(
                f"Unknown match_by {spec.match_by!r}; expected 'bounds' or 'basename'."
            )

    def _init_from_csv(self, spec: CSVImageSource) -> None:
        df = pd.read_csv(spec.csv_path)
        if spec.path_column not in df.columns:
            raise ValueError(
                f"CSV image source '{spec.csv_path}' must contain a "
                f"'{spec.path_column}' column."
            )
        has_mask_col = spec.mask_path_column in df.columns
        has_bounds_cols = all(c in df.columns for c in _BOUNDS_COLUMNS)
        if not has_mask_col and not has_bounds_cols:
            raise ValueError(
                f"CSV image source '{spec.csv_path}' needs either a "
                f"'{spec.mask_path_column}' column (direct tile lookup) or "
                f"{_BOUNDS_COLUMNS} columns (spatial search)."
            )
        has_crs_col = "crs" in df.columns

        for _, row in df.iterrows():
            path = Path(row[spec.path_column])

            if has_mask_col and pd.notna(row[spec.mask_path_column]):
                self._tile_lookup[Path(row[spec.mask_path_column]).stem] = path

            if has_bounds_cols and all(pd.notna(row[c]) for c in _BOUNDS_COLUMNS):
                crs_val = (
                    row["crs"] if has_crs_col and pd.notna(row.get("crs")) else None
                )
                crs_str = crs_val if crs_val else spec.default_crs
                if not crs_str:
                    continue
                bounds = BoundingBox(
                    float(row["minx"]),
                    float(row["miny"]),
                    float(row["maxx"]),
                    float(row["maxy"]),
                )
                self._entries.append((path, bounds, CRS.from_user_input(crs_str)))

    def resolve_candidates(
        self, tile_name: str, dst_bounds: BoundingBox, dst_crs: CRS
    ) -> List[Path]:
        """Resolve candidate file(s) for the window being read.

        Tries a direct tile-keyed lookup first (CSV ``mask_path`` column or
        basename matching — an O(1) dict lookup, no spatial computation); falls
        back to bounds-based search only when no direct entry is available.

        Args:
            tile_name: The mask tile's path or filename (matched by stem).
            dst_bounds: Destination window bounds, in ``dst_crs``.
            dst_crs: CRS of ``dst_bounds``.

        Returns:
            Candidate file path(s) to composite into the destination window.
        """
        if self._tile_lookup:
            hit = self._tile_lookup.get(Path(tile_name).stem)
            if hit is not None:
                return [hit]
        if self._entries:
            return self.candidates_for_bounds(dst_bounds, dst_crs)
        return []

    def candidates_for_bounds(
        self, dst_bounds: BoundingBox, dst_crs: CRS
    ) -> List[Path]:
        """Return candidate file paths whose indexed bounds intersect ``dst_bounds``.

        Args:
            dst_bounds: Destination window bounds, in ``dst_crs``.
            dst_crs: CRS of ``dst_bounds``.

        Returns:
            Paths of candidate files overlapping the destination bounds.
        """
        hits = []
        for path, bounds, crs in self._entries:
            b = (
                dst_bounds
                if crs == dst_crs
                else BoundingBox(*transform_bounds(dst_crs, crs, *dst_bounds))
            )
            if _bounds_intersect(b, bounds):
                hits.append(path)
        return hits


def resolve_image_source(spec: ImageSourceSpec) -> Union[Path, TiledImageSource]:
    """Resolve a config value into a usable image source.

    Args:
        spec: A single file path (``str``, existing behavior — an MBTiles
            archive, VRT, or any rasterio-readable raster), or a multi-file
            spec — :class:`DirectoryImageSource`, :class:`CSVImageSource`, or an
            equivalent dict (see the module docstring for the three forms).

    Returns:
        ``Path`` for a single-file source, or :class:`TiledImageSource` for a
        multi-file source — both accepted directly by
        ``read_source_aligned_to_mask_window``.
    """
    if isinstance(spec, dict):
        return TiledImageSource(_coerce_dict_spec(spec))
    if isinstance(spec, (DirectoryImageSource, CSVImageSource)):
        return TiledImageSource(spec)
    return Path(spec)
