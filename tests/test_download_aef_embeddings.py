# -*- coding: utf-8 -*-
"""Tests for scripts/download_aef_embeddings.py."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from unittest.mock import MagicMock, patch

from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
    _get_tile_bbox,
    _find_hf_cell_for_bbox,
    download_gcs_embeddings,
    download_hf_embeddings,
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
