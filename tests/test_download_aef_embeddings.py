# -*- coding: utf-8 -*-
"""Tests for scripts/download_aef_embeddings.py."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import geopandas as gpd
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box
from unittest.mock import MagicMock, patch

from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
    _get_tile_bbox,
    _find_hf_cell_for_bbox,
    _href_to_public_sourcecoop_url,
    _infer_sourcecoop_year,
    _load_sourcecoop_index,
    _select_sourcecoop_item,
    _write_sourcecoop_crop,
    _window_covering_bounds,
    download_gcs_embeddings,
    download_hf_embeddings,
    download_sourcecoop_embeddings,
)

EPSG4326 = CRS.from_epsg(4326)


def _write_image(path: Path, h: int, w: int, transform, crs=EPSG4326):
    data = np.zeros((3, h, w), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_image_utm(path: Path, h: int, w: int):
    """Write a raster in UTM Zone 23S (EPSG:32723), covering ~Sao Paulo region."""
    from rasterio.crs import CRS as RioCRS

    utm_crs = RioCRS.from_epsg(32723)
    # UTM coordinates near Sao Paulo: ~315000E, 7395000N
    transform = from_bounds(315000, 7395000, 316000, 7396000, w, h)
    data = np.zeros((3, h, w), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
        crs=utm_crs,
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_aef_embedding(path: Path, bounds, crs=EPSG4326):
    data = np.arange(64 * 64 * 64, dtype=np.int32).reshape(64, 64, 64)
    data = (data % 127).astype(np.int8)
    transform = from_bounds(*bounds, 64, 64)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=64,
        dtype="int8",
        nodata=-128,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# _get_tile_bbox
# ---------------------------------------------------------------------------


class TestGetTileBbox:
    def test_returns_four_elements(self, tmp_path):
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 32, 32)
        img_path = tmp_path / "img.tif"
        _write_image(img_path, 32, 32, transform)
        bbox = _get_tile_bbox(str(img_path))
        assert len(bbox) == 4

    def test_returns_floats(self, tmp_path):
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 32, 32)
        img_path = tmp_path / "img.tif"
        _write_image(img_path, 32, 32, transform)
        bbox = _get_tile_bbox(str(img_path))
        assert all(isinstance(v, float) for v in bbox)

    def test_bbox_matches_raster_bounds(self, tmp_path):
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 32, 32)
        img_path = tmp_path / "img.tif"
        _write_image(img_path, 32, 32, transform)
        left, bottom, right, top = _get_tile_bbox(str(img_path))
        assert abs(left - 10.0) < 1e-5
        assert abs(bottom - 20.0) < 1e-5
        assert abs(right - 11.0) < 1e-5
        assert abs(top - 21.0) < 1e-5

    def test_left_less_than_right(self, tmp_path):
        transform = from_bounds(5.0, 15.0, 6.0, 16.0, 16, 16)
        img_path = tmp_path / "img.tif"
        _write_image(img_path, 16, 16, transform)
        left, bottom, right, top = _get_tile_bbox(str(img_path))
        assert left < right
        assert bottom < top


# ---------------------------------------------------------------------------
# _find_hf_cell_for_bbox
# ---------------------------------------------------------------------------


class TestFindHfCellForBbox:
    def _make_cells(self, lats, lons):
        return pd.DataFrame({"centre_lat": lats, "centre_lon": lons})

    def test_finds_nearest_cell(self):
        cells = self._make_cells([10.0, 50.0, 80.0], [20.0, 60.0, 90.0])
        # tile centre ~(lon=20, lat=10) → closest to row 0
        bbox = (19.5, 9.5, 20.5, 10.5)
        idx = _find_hf_cell_for_bbox(bbox, cells)
        assert idx == 0

    def test_returns_none_for_empty_df(self):
        cells = self._make_cells([], [])
        result = _find_hf_cell_for_bbox((0.0, 0.0, 1.0, 1.0), cells)
        assert result is None

    def test_exact_match(self):
        cells = self._make_cells([5.0, 15.0], [5.0, 15.0])
        # tile centre exactly at (lon=15, lat=15) → row 1
        bbox = (14.5, 14.5, 15.5, 15.5)
        idx = _find_hf_cell_for_bbox(bbox, cells)
        assert idx == 1

    def test_single_cell_always_matches(self):
        cells = self._make_cells([0.0], [0.0])
        idx = _find_hf_cell_for_bbox((100.0, 100.0, 101.0, 101.0), cells)
        assert idx == 0

    def test_returns_index_not_position(self):
        """Returned value should be the DataFrame index, usable with .loc[]."""
        cells = self._make_cells([10.0, 20.0], [10.0, 20.0])
        bbox = (9.5, 9.5, 10.5, 10.5)
        idx = _find_hf_cell_for_bbox(bbox, cells)
        # Verify the returned index can be used to select the right row
        assert cells.loc[idx, "centre_lat"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# download_gcs_embeddings
# ---------------------------------------------------------------------------


class TestDownloadGcsEmbeddings:
    def _make_csv(self, tmp_path):
        csv_path = tmp_path / "gcs_paths.csv"
        pd.DataFrame(
            [
                {"tile_id": "tile_0", "gcs_uri": "gs://bucket/tile_0.tif"},
                {"tile_id": "tile_1", "gcs_uri": "gs://bucket/tile_1.tif"},
            ]
        ).to_csv(csv_path, index=False)
        return csv_path

    def test_calls_gsutil_for_each_tile(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        out_dir = tmp_path / "embeddings"
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            download_gcs_embeddings(str(csv_path), out_dir)
        assert mock_run.call_count == 2

    def test_gsutil_called_with_correct_args(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        out_dir = tmp_path / "embeddings"
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            download_gcs_embeddings(str(csv_path), out_dir)
        first_call_args = mock_run.call_args_list[0][0][0]
        assert first_call_args[0] == "gsutil"
        assert first_call_args[1] == "cp"
        assert "gs://bucket/tile_0.tif" in first_call_args

    def test_skips_existing_files(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        out_dir = tmp_path / "embeddings"
        out_dir.mkdir()
        (out_dir / "tile_0.tif").touch()  # already downloaded
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            download_gcs_embeddings(str(csv_path), out_dir)
        assert mock_run.call_count == 1  # only tile_1

    def test_logs_error_on_gsutil_failure(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        out_dir = tmp_path / "embeddings"
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="permission denied")
            with patch(
                "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.logger"
            ) as mock_logger:
                download_gcs_embeddings(str(csv_path), out_dir)
        assert mock_logger.error.called

    def test_creates_output_dir(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        out_dir = tmp_path / "deep" / "nested" / "embeddings"
        assert not out_dir.exists()
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            download_gcs_embeddings(str(csv_path), out_dir)
        assert out_dir.exists()


# ---------------------------------------------------------------------------
# download_hf_embeddings
# ---------------------------------------------------------------------------


class TestDownloadHfEmbeddings:
    def _make_img(self, tmp_path):
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 32, 32)
        img_path = tmp_path / "img.tif"
        _write_image(img_path, 32, 32, transform)
        return img_path

    def _make_tiles_csv(self, tmp_path, img_path):
        csv_path = tmp_path / "tiles.csv"
        pd.DataFrame(
            [
                {"tile_id": "tile_0", "image_path": str(img_path)},
            ]
        ).to_csv(csv_path, index=False)
        return csv_path

    def _make_mock_ds(self, embedding_dim=64):
        embedding = np.random.rand(embedding_dim).astype(np.float32)
        mock_cells = pd.DataFrame(
            [
                {
                    "centre_lat": 20.5,
                    "centre_lon": 10.5,
                    "embeddings": list(embedding),
                }
            ]
        )
        mock_ds = MagicMock()
        mock_ds.to_pandas.return_value = mock_cells
        return mock_ds

    def test_saves_npy_for_each_tile(self, tmp_path):
        img_path = self._make_img(tmp_path)
        csv_path = self._make_tiles_csv(tmp_path, img_path)
        out_dir = tmp_path / "hf_embeddings"
        mock_ds = self._make_mock_ds()

        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            download_hf_embeddings(str(csv_path), out_dir)

        assert (out_dir / "tile_0.npy").exists()

    def test_saved_embedding_is_float32(self, tmp_path):
        img_path = self._make_img(tmp_path)
        csv_path = self._make_tiles_csv(tmp_path, img_path)
        out_dir = tmp_path / "hf_embeddings"
        mock_ds = self._make_mock_ds(embedding_dim=64)

        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            download_hf_embeddings(str(csv_path), out_dir)

        saved = np.load(out_dir / "tile_0.npy")
        assert saved.dtype == np.float32

    def test_saved_embedding_shape_correct(self, tmp_path):
        img_path = self._make_img(tmp_path)
        csv_path = self._make_tiles_csv(tmp_path, img_path)
        out_dir = tmp_path / "hf_embeddings"
        d = 64
        mock_ds = self._make_mock_ds(embedding_dim=d)

        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            download_hf_embeddings(str(csv_path), out_dir)

        saved = np.load(out_dir / "tile_0.npy")
        assert saved.shape == (d,)

    def test_skips_existing_npy_files(self, tmp_path):
        img_path = self._make_img(tmp_path)
        csv_path = self._make_tiles_csv(tmp_path, img_path)
        out_dir = tmp_path / "hf_embeddings"
        out_dir.mkdir()
        (out_dir / "tile_0.npy").touch()  # already exists
        mock_ds = self._make_mock_ds()

        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            download_hf_embeddings(str(csv_path), out_dir)

        # Dataset was loaded but no .npy was overwritten
        assert (out_dir / "tile_0.npy").stat().st_size == 0  # still the empty touch()

    def test_creates_output_dir(self, tmp_path):
        img_path = self._make_img(tmp_path)
        csv_path = self._make_tiles_csv(tmp_path, img_path)
        out_dir = tmp_path / "deep" / "hf"
        assert not out_dir.exists()
        mock_ds = self._make_mock_ds()

        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            download_hf_embeddings(str(csv_path), out_dir)

        assert out_dir.exists()

    def test_hf_dataset_name_is_major_tom(self, tmp_path):
        img_path = self._make_img(tmp_path)
        csv_path = self._make_tiles_csv(tmp_path, img_path)
        out_dir = tmp_path / "hf_embeddings"
        mock_ds = self._make_mock_ds()

        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            download_hf_embeddings(str(csv_path), out_dir)

        call_args = mock_datasets.load_dataset.call_args
        assert "Major-TOM/Core-AlphaEarth-Embeddings" in call_args[0]


# ---------------------------------------------------------------------------
# _get_tile_bbox — projected CRS branch (lines 54-56)
# ---------------------------------------------------------------------------


class TestGetTileBboxProjected:
    def test_projected_crs_returns_wgs84_degrees(self, tmp_path):
        """Raster in UTM must be reprojected to WGS84 before returning bbox."""
        img_path = tmp_path / "utm.tif"
        _write_image_utm(img_path, 32, 32)
        left, bottom, right, top = _get_tile_bbox(str(img_path))
        # Sao Paulo area: lon ~-46.x, lat ~-23.x
        assert -90.0 < left < 0.0
        assert -90.0 < bottom < 0.0
        assert left < right
        assert bottom < top

    def test_projected_crs_bbox_in_degree_range(self, tmp_path):
        img_path = tmp_path / "utm.tif"
        _write_image_utm(img_path, 32, 32)
        left, bottom, right, top = _get_tile_bbox(str(img_path))
        assert -180.0 <= left <= 180.0
        assert -90.0 <= bottom <= 90.0


# ---------------------------------------------------------------------------
# run() in download_aef_embeddings (lines 219-227)
# ---------------------------------------------------------------------------


class TestRunDownloadAef:
    def _make_gcs_csv(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "gcs.csv"
        pd.DataFrame(
            [
                {"tile_id": "tile_0", "gcs_uri": "gs://bucket/tile_0.tif"},
            ]
        ).to_csv(csv_path, index=False)
        return csv_path

    def test_run_gcs_calls_download_gcs(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
            run,
        )

        csv_path = self._make_gcs_csv(tmp_path)
        out_dir = tmp_path / "embeddings"
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            run(source="gcs", output_dir=out_dir, gcs_paths_csv=str(csv_path))
        assert mock_run.called

    def test_run_gcs_missing_csv_raises(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
            run,
        )

        with pytest.raises(ValueError, match="gcs_paths_csv"):
            run(source="gcs", output_dir=tmp_path / "out")

    def test_run_hf_missing_csv_raises(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
            run,
        )

        with pytest.raises(ValueError, match="tiles_csv"):
            run(source="hf", output_dir=tmp_path / "out")

    def test_run_hf_calls_download_hf(self, tmp_path):
        """run(source='hf') routes to download_hf_embeddings (line 227)."""
        from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
            run,
        )
        import pandas as pd

        # Write a tiles CSV that references a real image
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 16, 16)
        img_path = tmp_path / "tile_0.tif"
        _write_image(img_path, 16, 16, transform)
        tiles_csv = tmp_path / "tiles.csv"
        pd.DataFrame([{"tile_id": "tile_0", "image_path": str(img_path)}]).to_csv(
            tiles_csv, index=False
        )
        out_dir = tmp_path / "hf_out"
        mock_cells = pd.DataFrame(
            [
                {
                    "centre_lat": 20.5,
                    "centre_lon": 10.5,
                    "embeddings": list(np.zeros(64, dtype=np.float32)),
                }
            ]
        )
        mock_ds = MagicMock()
        mock_ds.to_pandas.return_value = mock_cells
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            run(source="hf", output_dir=out_dir, tiles_csv=str(tiles_csv))
        assert mock_datasets.load_dataset.called

    def test_run_sourcecoop_missing_csv_raises(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
            run,
        )

        with pytest.raises(ValueError, match="tiles_csv"):
            run(source="sourcecoop", output_dir=tmp_path / "out")

    def test_run_sourcecoop_calls_download_sourcecoop(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
            run,
        )

        tiles_csv, index_path = (
            TestDownloadSourceCoopEmbeddings()._make_sourcecoop_inputs(tmp_path)
        )
        out_dir = tmp_path / "sourcecoop_out"

        run(
            source="sourcecoop",
            output_dir=out_dir,
            tiles_csv=str(tiles_csv),
            sourcecoop_index_path=str(index_path),
            year=2025,
        )

        assert (out_dir / "tile_0.tif").exists()

    def test_run_unknown_source_raises(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
            run,
        )

        with pytest.raises(ValueError, match="source must be one of"):
            run(source="unknown", output_dir=tmp_path / "out")


class TestDownloadHfEmbeddingsNoCellFound:
    """Cover the 'no HF cell found' warning branch (lines 186-187)."""

    def test_no_cell_found_skips_tile(self, tmp_path):
        transform = from_bounds(10.0, 20.0, 11.0, 21.0, 16, 16)
        img_path = tmp_path / "tile_0.tif"
        _write_image(img_path, 16, 16, transform)
        tiles_csv = tmp_path / "tiles.csv"
        import pandas as pd

        pd.DataFrame([{"tile_id": "tile_0", "image_path": str(img_path)}]).to_csv(
            tiles_csv, index=False
        )
        out_dir = tmp_path / "hf_out"
        # Return an empty DataFrame so _find_hf_cell_for_bbox returns None
        empty_cells = pd.DataFrame(
            {"centre_lat": [], "centre_lon": [], "embeddings": []}
        )
        mock_ds = MagicMock()
        mock_ds.to_pandas.return_value = empty_cells
        with patch(
            "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.datasets"
        ) as mock_datasets:
            mock_datasets.load_dataset.return_value = mock_ds
            download_hf_embeddings(str(tiles_csv), out_dir)
        assert not (out_dir / "tile_0.npy").exists()


# ---------------------------------------------------------------------------
# Source Cooperative STAC/COG mode
# ---------------------------------------------------------------------------


class TestSourceCoopHelpers:
    def test_infer_sourcecoop_year_prefers_explicit_year(self):
        row = pd.Series({"image_path": "/data/tile_20250625_20260106.tif"})
        assert _infer_sourcecoop_year(row, year=2024) == 2024

    def test_infer_sourcecoop_year_uses_first_year_in_image_path(self):
        row = pd.Series({"image_path": "/data/tile_20250625_20260106.tif"})
        assert _infer_sourcecoop_year(row, year=None) == 2025

    def test_infer_sourcecoop_year_uses_csv_year_column(self):
        row = pd.Series({"image_path": "/data/tile.tif", "year": 2022})
        assert _infer_sourcecoop_year(row, year=None) == 2022

    def test_infer_sourcecoop_year_requires_year(self):
        row = pd.Series({"image_path": "/data/tile_without_date.tif"})
        with pytest.raises(ValueError, match="Could not infer AEF year"):
            _infer_sourcecoop_year(row, year=None)

    def test_href_to_public_sourcecoop_url_converts_s3_href(self):
        href = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2025/22S/tile.tiff"
        assert _href_to_public_sourcecoop_url(href) == (
            "https://data.source.coop/tge-labs/aef/v1/annual/2025/22S/tile.tiff"
        )

    def test_href_to_public_sourcecoop_url_leaves_http_href(self):
        href = "https://example.test/aef.tif"
        assert _href_to_public_sourcecoop_url(href) == href

    def test_load_sourcecoop_index_reads_https_url_to_memory(self):
        payload = MagicMock()
        payload.read.return_value = b"parquet-bytes"
        response_cm = MagicMock()
        response_cm.__enter__.return_value = payload
        expected = gpd.GeoDataFrame({"geometry": []}, crs="OGC:CRS84")

        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.urlopen"
            ) as mock_urlopen,
            patch(
                "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.gpd.read_parquet"
            ) as mock_read_parquet,
        ):
            mock_urlopen.return_value = response_cm
            mock_read_parquet.return_value = expected
            result = _load_sourcecoop_index(
                index_path=None,
                index_url="https://data.source.coop/index.parquet",
            )

        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "https://data.source.coop/index.parquet"
        assert request.headers["User-agent"] == "pytorch-smt-tools"
        assert mock_urlopen.call_args.kwargs == {"timeout": 60}
        assert result is expected

    def test_select_sourcecoop_item_filters_by_year_and_intersection(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "datetime": pd.Timestamp("2024-01-01", tz="UTC"),
                    "assets": {"data": {"href": "wrong-year.tif"}},
                    "geometry": box(0, 0, 2, 2),
                },
                {
                    "datetime": pd.Timestamp("2025-01-01", tz="UTC"),
                    "assets": {"data": {"href": "match.tif"}},
                    "geometry": box(0, 0, 2, 2),
                },
                {
                    "datetime": pd.Timestamp("2025-01-01", tz="UTC"),
                    "assets": {"data": {"href": "far.tif"}},
                    "geometry": box(10, 10, 11, 11),
                },
            ],
            crs="OGC:CRS84",
        )
        row = _select_sourcecoop_item(gdf, bbox=(0.25, 0.25, 0.75, 0.75), year=2025)
        assert row["assets"]["data"]["href"] == "match.tif"

    def test_select_sourcecoop_item_returns_largest_overlap(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "datetime": pd.Timestamp("2025-01-01", tz="UTC"),
                    "assets": {"data": {"href": "small.tif"}},
                    "geometry": box(0, 0, 0.5, 0.5),
                },
                {
                    "datetime": pd.Timestamp("2025-01-01", tz="UTC"),
                    "assets": {"data": {"href": "large.tif"}},
                    "geometry": box(0, 0, 2, 2),
                },
            ],
            crs="OGC:CRS84",
        )
        row = _select_sourcecoop_item(gdf, bbox=(0, 0, 1, 1), year=2025)
        assert row["assets"]["data"]["href"] == "large.tif"

    def test_select_sourcecoop_item_returns_none_without_intersection(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "datetime": pd.Timestamp("2025-01-01", tz="UTC"),
                    "assets": {"data": {"href": "far.tif"}},
                    "geometry": box(10, 10, 11, 11),
                }
            ],
            crs="OGC:CRS84",
        )
        assert _select_sourcecoop_item(gdf, bbox=(0, 0, 1, 1), year=2025) is None

    def test_select_sourcecoop_item_returns_none_for_empty_index(self):
        gdf = gpd.GeoDataFrame({"datetime": [], "geometry": []}, crs="OGC:CRS84")
        assert _select_sourcecoop_item(gdf, bbox=(0, 0, 1, 1), year=2025) is None

    def test_select_sourcecoop_item_returns_none_for_missing_year(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "datetime": pd.Timestamp("2024-01-01", tz="UTC"),
                    "assets": {"data": {"href": "old.tif"}},
                    "geometry": box(0, 0, 1, 1),
                }
            ],
            crs="OGC:CRS84",
        )
        assert _select_sourcecoop_item(gdf, bbox=(0, 0, 1, 1), year=2025) is None

    def test_select_sourcecoop_item_reprojects_projected_index(self):
        gdf = gpd.GeoDataFrame(
            [
                {
                    "datetime": pd.Timestamp("2025-01-01", tz="UTC"),
                    "assets": {"data": {"href": "projected.tif"}},
                    "geometry": box(0, 0, 1000, 1000),
                }
            ],
            crs="EPSG:3857",
        )
        row = _select_sourcecoop_item(gdf, bbox=(0, 0, 0.001, 0.001), year=2025)
        assert row["assets"]["data"]["href"] == "projected.tif"

    def test_window_covering_bounds_supports_positive_pixel_height(self, tmp_path):
        path = tmp_path / "positive_y.tif"
        transform = rasterio.Affine(1.0, 0.0, 100.0, 0.0, 1.0, 200.0)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="uint8",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(np.zeros((1, 10, 10), dtype=np.uint8))

        with rasterio.open(path) as src:
            window = _window_covering_bounds(src, (102.2, 203.2, 104.8, 205.8))

        assert window.col_off == 1
        assert window.row_off == 2
        assert window.width == 5
        assert window.height == 5

    def test_write_sourcecoop_crop_sets_vsicurl_env_for_http(self, tmp_path):
        tile = MagicMock()
        tile.bounds = rasterio.coords.BoundingBox(0.0, 0.0, 1.0, 1.0)
        tile.crs = EPSG4326
        tile_cm = MagicMock()
        tile_cm.__enter__.return_value = tile

        src = MagicMock()
        src.crs = EPSG4326
        src.transform = from_bounds(0.0, 0.0, 2.0, 2.0, 8, 8)
        src.width = 8
        src.height = 8
        src.profile = {
            "driver": "GTiff",
            "height": 8,
            "width": 8,
            "count": 64,
            "dtype": "int8",
            "crs": EPSG4326,
            "transform": src.transform,
        }
        src.read.return_value = np.zeros((64, 6, 6), dtype=np.int8)
        src.window_transform.return_value = src.transform
        src_cm = MagicMock()
        src_cm.__enter__.return_value = src

        dst = MagicMock()
        dst_cm = MagicMock()
        dst_cm.__enter__.return_value = dst

        out_path = tmp_path / "crop.tif"
        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.rasterio.Env"
            ) as mock_env,
            patch(
                "pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.rasterio.open"
            ) as mock_open,
        ):
            mock_env.return_value.__enter__.return_value = None
            mock_open.side_effect = [tile_cm, src_cm, dst_cm]
            _write_sourcecoop_crop(
                "https://example.test/aef.tiff", "tile.tif", out_path
            )

        mock_env.assert_called_once_with(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
        )
        dst.write.assert_called_once()


class TestDownloadSourceCoopEmbeddings:
    def _make_sourcecoop_inputs(self, tmp_path):
        image_path = tmp_path / "tile_20250625_20260106.tif"
        _write_image(
            image_path,
            16,
            16,
            from_bounds(10.2, 20.2, 10.8, 20.8, 16, 16),
        )

        aef_path = tmp_path / "aef_2025.tif"
        _write_aef_embedding(aef_path, bounds=(10.0, 20.0, 11.0, 21.0))

        index_path = tmp_path / "aef_index.parquet"
        gdf = gpd.GeoDataFrame(
            [
                {
                    "datetime": pd.Timestamp("2025-01-01", tz="UTC"),
                    "assets": {"data": {"href": str(aef_path)}},
                    "geometry": box(10.0, 20.0, 11.0, 21.0),
                }
            ],
            crs="OGC:CRS84",
        )
        gdf.to_parquet(index_path)

        tiles_csv = tmp_path / "tiles.csv"
        pd.DataFrame([{"tile_id": "tile_0", "image_path": str(image_path)}]).to_csv(
            tiles_csv, index=False
        )
        return tiles_csv, index_path

    def test_download_sourcecoop_writes_cropped_geotiff(self, tmp_path):
        tiles_csv, index_path = self._make_sourcecoop_inputs(tmp_path)
        out_dir = tmp_path / "out"

        download_sourcecoop_embeddings(
            str(tiles_csv),
            out_dir,
            index_path=str(index_path),
        )

        out_path = out_dir / "tile_0.tif"
        assert out_path.exists()
        with rasterio.open(out_path) as src:
            assert src.count == 64
            assert src.dtypes == ("int8",) * 64
            assert src.width < 64
            assert src.height < 64
            assert src.nodata == -128

    def test_download_sourcecoop_respects_explicit_year(self, tmp_path):
        tiles_csv, index_path = self._make_sourcecoop_inputs(tmp_path)
        out_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="No Source Cooperative AEF COG"):
            download_sourcecoop_embeddings(
                str(tiles_csv),
                out_dir,
                index_path=str(index_path),
                year=2024,
            )

    def test_download_sourcecoop_skips_existing_output(self, tmp_path):
        tiles_csv, index_path = self._make_sourcecoop_inputs(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out_path = out_dir / "tile_0.tif"
        out_path.write_text("existing", encoding="utf-8")

        download_sourcecoop_embeddings(
            str(tiles_csv),
            out_dir,
            index_path=str(index_path),
        )

        assert out_path.read_text(encoding="utf-8") == "existing"
