# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.tools.mbtiles.image_source."""

from pathlib import Path

import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")

from pytorch_segmentation_models_trainer.tools.mbtiles.image_source import (
    CSVImageSource,
    DirectoryImageSource,
    TiledImageSource,
    resolve_image_source,
)


def _write_tile(path: Path, minx: float, maxy: float, width=8, height=8, fill=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(minx, maxy, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((height, width), fill, dtype=np.uint8), 1)


class TestResolveImageSource:
    def test_string_resolves_to_path(self):
        result = resolve_image_source("/data/tiles.mbtiles")
        assert isinstance(result, Path)
        assert str(result) == "/data/tiles.mbtiles"

    def test_dict_resolves_to_tiled_source(self, tmp_path):
        _write_tile(tmp_path / "a.tif", 0, 8)
        result = resolve_image_source({"directory": str(tmp_path)})
        assert isinstance(result, TiledImageSource)

    def test_directory_image_source_resolves(self, tmp_path):
        _write_tile(tmp_path / "a.tif", 0, 8)
        spec = DirectoryImageSource(directory=str(tmp_path))
        result = resolve_image_source(spec)
        assert isinstance(result, TiledImageSource)

    def test_csv_dict_resolves_to_tiled_source(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("path,mask_path\n/data/img_001.tif,/data/mask_001.tif\n")
        result = resolve_image_source({"csv_path": str(csv_path)})
        assert isinstance(result, TiledImageSource)

    def test_dict_without_directory_or_csv_raises(self):
        with pytest.raises(ValueError, match="'directory' or 'csv_path'"):
            resolve_image_source({"some_other_key": "value"})

    def test_unsupported_spec_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported image source spec"):
            TiledImageSource(123)  # type: ignore[arg-type]

    def test_no_usable_entries_raises_value_error(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("path,mask_path\n/data/a.tif,\n")  # mask_path all-NaN
        with pytest.raises(ValueError, match="resolved no usable entries"):
            TiledImageSource(CSVImageSource(csv_path=str(csv_path)))


class TestTiledImageSource:
    def test_finds_files_by_extension(self, tmp_path):
        _write_tile(tmp_path / "a.tif", 0, 8)
        (tmp_path / "not_a_raster.txt").write_text("x")
        source = TiledImageSource(DirectoryImageSource(directory=str(tmp_path)))
        assert len(source._entries) == 1

    def test_custom_extensions(self, tmp_path):
        _write_tile(tmp_path / "a.tiff", 0, 8)
        source = TiledImageSource(
            DirectoryImageSource(directory=str(tmp_path), extensions=["tiff"])
        )
        assert len(source._entries) == 1

    def test_recursive_search(self, tmp_path):
        _write_tile(tmp_path / "sub" / "a.tif", 0, 8)
        with pytest.raises(ValueError, match="No files matching"):
            TiledImageSource(
                DirectoryImageSource(directory=str(tmp_path), recursive=False)
            )

        recursive = TiledImageSource(
            DirectoryImageSource(directory=str(tmp_path), recursive=True)
        )
        assert len(recursive._entries) == 1

    def test_raises_when_no_files_found(self, tmp_path):
        with pytest.raises(ValueError, match="No files matching"):
            TiledImageSource(DirectoryImageSource(directory=str(tmp_path)))

    def test_dict_spec_accepted(self, tmp_path):
        _write_tile(tmp_path / "a.tif", 0, 8)
        source = TiledImageSource({"directory": str(tmp_path)})
        assert len(source._entries) == 1

    def test_candidates_for_bounds_filters_by_overlap(self, tmp_path):
        from rasterio.coords import BoundingBox

        _write_tile(tmp_path / "left.tif", 0, 8, width=8, height=8)
        _write_tile(tmp_path / "right.tif", 100, 8, width=8, height=8)
        source = TiledImageSource(DirectoryImageSource(directory=str(tmp_path)))

        left_window = BoundingBox(0, 0, 8, 8)
        hits = source.candidates_for_bounds(left_window, "EPSG:3857")
        assert [p.name for p in hits] == ["left.tif"]

    def test_candidates_for_bounds_reprojects_when_crs_differs(self, tmp_path):
        from rasterio.coords import BoundingBox
        from rasterio.warp import transform_bounds

        _write_tile(tmp_path / "a.tif", 0, 8, width=8, height=8)
        source = TiledImageSource(DirectoryImageSource(directory=str(tmp_path)))

        # Query in EPSG:4326 for a bbox that maps back onto the EPSG:3857 tile.
        query_bounds_3857 = BoundingBox(0, 0, 8, 8)
        query_bounds_4326 = BoundingBox(
            *transform_bounds("EPSG:3857", "EPSG:4326", *query_bounds_3857)
        )
        hits = source.candidates_for_bounds(query_bounds_4326, "EPSG:4326")
        assert [p.name for p in hits] == ["a.tif"]

    def test_unknown_match_by_raises(self, tmp_path):
        _write_tile(tmp_path / "a.tif", 0, 8)
        with pytest.raises(ValueError, match="Unknown match_by"):
            TiledImageSource(
                DirectoryImageSource(directory=str(tmp_path), match_by="nonsense")
            )


class TestTiledImageSourceBasenameMatching:
    def test_basename_match_builds_stem_lookup_without_opening_files(self, tmp_path):
        # A file that is NOT a valid raster still works — basename matching never
        # opens it, just lists the directory.
        (tmp_path / "tile_001.tif").write_bytes(b"not a real raster")
        (tmp_path / "tile_002.tif").write_bytes(b"not a real raster either")
        source = TiledImageSource(
            DirectoryImageSource(directory=str(tmp_path), match_by="basename")
        )
        assert source._tile_lookup["tile_001"].name == "tile_001.tif"
        assert source._entries == []

    def test_resolve_candidates_matches_by_mask_tile_stem(self, tmp_path):
        (tmp_path / "tile_001.tif").write_bytes(b"x")
        source = TiledImageSource(
            DirectoryImageSource(directory=str(tmp_path), match_by="basename")
        )
        from rasterio.coords import BoundingBox

        hits = source.resolve_candidates(
            "/data/masks/tile_001.tif", BoundingBox(0, 0, 1, 1), "EPSG:3857"
        )
        assert [p.name for p in hits] == ["tile_001.tif"]

    def test_resolve_candidates_no_match_and_no_bounds_returns_empty(self, tmp_path):
        (tmp_path / "tile_001.tif").write_bytes(b"x")
        source = TiledImageSource(
            DirectoryImageSource(directory=str(tmp_path), match_by="basename")
        )
        from rasterio.coords import BoundingBox

        hits = source.resolve_candidates(
            "/data/masks/tile_999_no_match.tif", BoundingBox(0, 0, 1, 1), "EPSG:3857"
        )
        assert hits == []


class TestCSVImageSource:
    def test_requires_path_column(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("wrong_column\nfoo\n")
        with pytest.raises(ValueError, match="must contain a 'path' column"):
            TiledImageSource(CSVImageSource(csv_path=str(csv_path)))

    def test_requires_mask_path_or_bounds_columns(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("path\n/data/img_001.tif\n")
        with pytest.raises(ValueError, match="needs either a 'mask_path' column"):
            TiledImageSource(CSVImageSource(csv_path=str(csv_path)))

    def test_mask_path_column_enables_direct_lookup(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text(
            "path,mask_path\n"
            "/data/img_001.tif,/data/masks/tile_001.tif\n"
            "/data/img_002.tif,/data/masks/tile_002.tif\n"
        )
        source = TiledImageSource(CSVImageSource(csv_path=str(csv_path)))
        from rasterio.coords import BoundingBox

        hits = source.resolve_candidates(
            "tile_002.tif", BoundingBox(0, 0, 1, 1), "EPSG:3857"
        )
        assert [str(p) for p in hits] == ["/data/img_002.tif"]

    def test_bounds_columns_enable_spatial_fallback(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text(
            "path,minx,miny,maxx,maxy,crs\n"
            "/data/left.tif,0,0,8,8,EPSG:3857\n"
            "/data/right.tif,100,0,108,8,EPSG:3857\n"
        )
        source = TiledImageSource(CSVImageSource(csv_path=str(csv_path)))
        from rasterio.coords import BoundingBox

        # No mask_path column at all -> straight to bounds search.
        hits = source.resolve_candidates(
            "some_tile.tif", BoundingBox(0, 0, 8, 8), "EPSG:3857"
        )
        assert [str(p) for p in hits] == ["/data/left.tif"]

    def test_mask_path_lookup_tried_before_bounds_fallback(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text(
            "path,mask_path,minx,miny,maxx,maxy,crs\n"
            "/data/matched.tif,/data/masks/tile_a.tif,0,0,8,8,EPSG:3857\n"
            "/data/overlapping.tif,,0,0,8,8,EPSG:3857\n"
        )
        source = TiledImageSource(CSVImageSource(csv_path=str(csv_path)))
        from rasterio.coords import BoundingBox

        # tile_a matches by mask_path directly -> single hit, bounds ignored.
        hits = source.resolve_candidates(
            "tile_a.tif", BoundingBox(0, 0, 8, 8), "EPSG:3857"
        )
        assert [str(p) for p in hits] == ["/data/matched.tif"]

    def test_falls_back_to_bounds_when_tile_not_in_lookup(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text(
            "path,mask_path,minx,miny,maxx,maxy,crs\n"
            "/data/matched.tif,/data/masks/tile_a.tif,0,0,8,8,EPSG:3857\n"
            "/data/unmatched_but_overlapping.tif,,0,0,8,8,EPSG:3857\n"
        )
        source = TiledImageSource(CSVImageSource(csv_path=str(csv_path)))
        from rasterio.coords import BoundingBox

        # tile_z has no mask_path row -> falls back to bounds, hits the row
        # that DOES have bounds (the unmatched one; the matched row also has
        # bounds so both would hit, but this confirms the fallback path runs).
        hits = source.resolve_candidates(
            "tile_z.tif", BoundingBox(0, 0, 8, 8), "EPSG:3857"
        )
        assert {p.name for p in hits} == {
            "matched.tif",
            "unmatched_but_overlapping.tif",
        }

    def test_default_crs_used_when_crs_column_missing(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("path,minx,miny,maxx,maxy\n/data/a.tif,0,0,8,8\n")
        source = TiledImageSource(
            CSVImageSource(csv_path=str(csv_path), default_crs="EPSG:3857")
        )
        from rasterio.coords import BoundingBox

        hits = source.resolve_candidates(
            "tile.tif", BoundingBox(0, 0, 8, 8), "EPSG:3857"
        )
        assert [p.name for p in hits] == ["a.tif"]

    def test_row_skipped_when_no_crs_available(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text(
            "path,mask_path,minx,miny,maxx,maxy\n"
            "/data/a.tif,/data/masks/tile.tif,0,0,8,8\n"
        )
        # No crs column and no default_crs -> bounds row is skipped entirely,
        # but the mask_path lookup still works (independent of CRS).
        source = TiledImageSource(CSVImageSource(csv_path=str(csv_path)))
        assert source._entries == []
        from rasterio.coords import BoundingBox

        hits = source.resolve_candidates(
            "tile.tif", BoundingBox(0, 0, 8, 8), "EPSG:3857"
        )
        assert [p.name for p in hits] == ["a.tif"]

    def test_custom_path_and_mask_path_column_names(self, tmp_path):
        csv_path = tmp_path / "images.csv"
        csv_path.write_text("image_file,mask_file\n/data/a.tif,/data/masks/tile.tif\n")
        source = TiledImageSource(
            CSVImageSource(
                csv_path=str(csv_path),
                path_column="image_file",
                mask_path_column="mask_file",
            )
        )
        from rasterio.coords import BoundingBox

        hits = source.resolve_candidates(
            "tile.tif", BoundingBox(0, 0, 1, 1), "EPSG:3857"
        )
        assert [p.name for p in hits] == ["a.tif"]
