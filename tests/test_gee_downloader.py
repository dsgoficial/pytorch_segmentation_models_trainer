# -*- coding: utf-8 -*-
"""Tests for tools/gee/gee_downloader.py.

All GEE API calls are mocked — no network access or GEE account required.

Skipped automatically when 'earthengine-api' is not installed.  Install it
with:  pip install pytorch_segmentation_models_trainer[gee]
"""

# Skip the entire module when earthengine-api is not installed so that the
# main CI job (which does not install optional deps) reports SKIPPED instead
# of ERROR.
import pytest

pytest.importorskip(
    "ee",
    reason="earthengine-api not installed – run: pip install '.[gee]'",
)

import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping

from pytorch_segmentation_models_trainer.tools.gee.gee_downloader import (
    DYNAMIC_WORLD_COLLECTION,
    ESRI_LULC_COLLECTION,
    MAPBIOMAS_COLLECTIONS,
    _DEFAULT_SCALES,
    _build_http_transport,
    _download_image_direct,
    _download_one_region,
    _ee_geometry_from_geojson,
    _export_image_to_drive,
    _get_dynamic_world_image,
    _get_esri_lulc_image,
    _get_mapbiomas_image,
    _require_ee,
    download_dynamic_world,
    download_esri_lulc,
    download_mapbiomas,
    download_regions,
    initialize_gee,
    load_regions,
    load_regions_from_bbox,
    load_regions_from_vector,
)

MODULE = "pytorch_segmentation_models_trainer.tools.gee.gee_downloader"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector_file(tmp_path: Path, attribute: str = "name") -> Path:
    """Write a tiny GeoJSON with two polygons."""
    gdf = gpd.GeoDataFrame(
        {attribute: ["region_a", "region_b"]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    path = tmp_path / "regions.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


def _make_vector_file_utm(tmp_path: Path) -> Path:
    """Write a GeoJSON in UTM (EPSG:32723) to test reprojection."""
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(315000, 7395000, 316000, 7396000)],
        crs="EPSG:32723",
    )
    path = tmp_path / "utm_regions.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _make_zip_tif_response() -> bytes:
    """Build a minimal zip that contains a fake .tif file."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image.tif", b"FAKE_TIF_CONTENT")
    return buf.getvalue()


def _make_bare_tif_response() -> bytes:
    """Return raw bytes that are NOT a zip (bare GeoTIFF path)."""
    return b"FAKE_RAW_TIF"


def _mock_image():
    img = MagicMock()
    img.select.return_value = img
    img.clip.return_value = img
    img.getDownloadURL.return_value = "https://example.test/download"
    return img


def _mock_collection(image):
    col = MagicMock()
    col.filterBounds.return_value = col
    col.filterDate.return_value = col
    col.select.return_value = col
    col.mode.return_value = image
    col.first.return_value = image
    return col


# ---------------------------------------------------------------------------
# _require_ee
# ---------------------------------------------------------------------------


class TestRequireEe:
    def test_raises_when_ee_is_none(self):
        with patch(f"{MODULE}.ee", None):
            with pytest.raises(ImportError, match="earthengine-api"):
                _require_ee()

    def test_does_not_raise_when_ee_available(self):
        with patch(f"{MODULE}.ee", MagicMock()):
            _require_ee()  # no exception


# ---------------------------------------------------------------------------
# initialize_gee
# ---------------------------------------------------------------------------


class TestInitializeGee:
    def test_uses_service_account_when_email_and_key_provided(self):
        mock_ee = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            initialize_gee(
                project="my-project",
                service_account_email="bot@proj.iam.gserviceaccount.com",
                service_account_key_file="/key.json",
            )
        mock_ee.ServiceAccountCredentials.assert_called_once_with(
            "bot@proj.iam.gserviceaccount.com", "/key.json"
        )
        mock_ee.Initialize.assert_called_once()

    def test_uses_oauth_when_no_service_account(self):
        mock_ee = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            initialize_gee(project="my-project")
        mock_ee.Authenticate.assert_called_once()
        mock_ee.Initialize.assert_called_once()

    def test_service_account_credentials_passed_to_initialize(self):
        mock_ee = MagicMock()
        fake_creds = MagicMock()
        mock_ee.ServiceAccountCredentials.return_value = fake_creds
        with patch(f"{MODULE}.ee", mock_ee):
            initialize_gee(
                project="proj",
                service_account_email="bot@proj.iam.gserviceaccount.com",
                service_account_key_file="/key.json",
            )
        init_kwargs = mock_ee.Initialize.call_args[1]
        assert init_kwargs["credentials"] is fake_creds

    def test_raises_when_ee_not_installed(self):
        with patch(f"{MODULE}.ee", None):
            with pytest.raises(ImportError, match="earthengine-api"):
                initialize_gee()

    def test_oauth_when_only_email_provided(self):
        """Missing key file → falls through to OAuth flow."""
        mock_ee = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            initialize_gee(service_account_email="bot@proj.iam.gserviceaccount.com")
        mock_ee.Authenticate.assert_called_once()

    def test_oauth_when_only_key_file_provided(self):
        """Missing email → falls through to OAuth flow."""
        mock_ee = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            initialize_gee(service_account_key_file="/key.json")
        mock_ee.Authenticate.assert_called_once()


# ---------------------------------------------------------------------------
# load_regions_from_bbox
# ---------------------------------------------------------------------------


class TestLoadRegionsFromBbox:
    def test_returns_one_element(self):
        result = load_regions_from_bbox(-48.0, -23.5, -47.5, -23.0)
        assert len(result) == 1

    def test_name_is_bbox(self):
        result = load_regions_from_bbox(-48.0, -23.5, -47.5, -23.0)
        assert result[0][0] == "bbox"

    def test_geometry_is_dict(self):
        result = load_regions_from_bbox(-48.0, -23.5, -47.5, -23.0)
        assert isinstance(result[0][1], dict)

    def test_geometry_type_is_polygon(self):
        result = load_regions_from_bbox(-48.0, -23.5, -47.5, -23.0)
        assert result[0][1]["type"] == "Polygon"

    def test_geometry_covers_input_bbox(self):
        xmin, ymin, xmax, ymax = -48.0, -23.5, -47.5, -23.0
        result = load_regions_from_bbox(xmin, ymin, xmax, ymax)
        from shapely.geometry import shape

        geom = shape(result[0][1])
        b = geom.bounds
        assert abs(b[0] - xmin) < 1e-9
        assert abs(b[1] - ymin) < 1e-9
        assert abs(b[2] - xmax) < 1e-9
        assert abs(b[3] - ymax) < 1e-9


# ---------------------------------------------------------------------------
# load_regions_from_vector
# ---------------------------------------------------------------------------


class TestLoadRegionsFromVector:
    def test_returns_one_region_per_feature(self, tmp_path):
        path = _make_vector_file(tmp_path)
        result = load_regions_from_vector(str(path), name_attribute="name")
        assert len(result) == 2

    def test_names_from_attribute(self, tmp_path):
        path = _make_vector_file(tmp_path)
        result = load_regions_from_vector(str(path), name_attribute="name")
        names = [r[0] for r in result]
        assert "region_a" in names
        assert "region_b" in names

    def test_fallback_to_index_when_no_attribute(self, tmp_path):
        path = _make_vector_file(tmp_path)
        result = load_regions_from_vector(str(path), name_attribute=None)
        names = [r[0] for r in result]
        assert all(n.startswith("region_") for n in names)

    def test_fallback_to_index_when_attribute_missing(self, tmp_path):
        path = _make_vector_file(tmp_path)
        result = load_regions_from_vector(str(path), name_attribute="nonexistent")
        names = [r[0] for r in result]
        assert all(n.startswith("region_") for n in names)

    def test_geometry_is_geojson_dict(self, tmp_path):
        path = _make_vector_file(tmp_path)
        result = load_regions_from_vector(str(path))
        for _, geom in result:
            assert isinstance(geom, dict)
            assert "type" in geom
            assert "coordinates" in geom

    def test_reprojects_utm_to_wgs84(self, tmp_path):
        path = _make_vector_file_utm(tmp_path)
        result = load_regions_from_vector(str(path))
        from shapely.geometry import shape

        geom = shape(result[0][1])
        b = geom.bounds
        # UTM Zone 23S near São Paulo → WGS84 lon ~-46.x, lat ~-23.x
        assert -90 < b[0] < 0
        assert -90 < b[1] < 0


# ---------------------------------------------------------------------------
# load_regions
# ---------------------------------------------------------------------------


class TestLoadRegions:
    def test_bbox_path(self):
        result = load_regions(bbox=(-48.0, -23.5, -47.5, -23.0))
        assert len(result) == 1
        assert result[0][0] == "bbox"

    def test_vector_path(self, tmp_path):
        path = _make_vector_file(tmp_path)
        result = load_regions(vector_path=str(path), name_attribute="name")
        assert len(result) == 2

    def test_raises_when_neither_provided(self):
        with pytest.raises(ValueError, match="Either 'bbox' or 'vector_path'"):
            load_regions()

    def test_raises_when_both_provided(self, tmp_path):
        path = _make_vector_file(tmp_path)
        with pytest.raises(ValueError, match="Only one of"):
            load_regions(bbox=(-48.0, -23.5, -47.5, -23.0), vector_path=str(path))


# ---------------------------------------------------------------------------
# _ee_geometry_from_geojson
# ---------------------------------------------------------------------------


class TestEeGeometryFromGeojson:
    def test_calls_ee_geometry_with_dict(self):
        mock_ee = MagicMock()
        geojson = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        with patch(f"{MODULE}.ee", mock_ee):
            result = _ee_geometry_from_geojson(geojson)
        mock_ee.Geometry.assert_called_once_with(geojson)
        assert result is mock_ee.Geometry.return_value


# ---------------------------------------------------------------------------
# _get_mapbiomas_image
# ---------------------------------------------------------------------------


class TestGetMapbiomasImage:
    def test_calls_ee_image_with_collection_id(self):
        mock_ee = MagicMock()
        coll_id = MAPBIOMAS_COLLECTIONS["brazil"]
        with patch(f"{MODULE}.ee", mock_ee):
            _get_mapbiomas_image(2022, coll_id)
        mock_ee.Image.assert_called_once_with(coll_id)

    def test_selects_correct_band_for_year(self):
        mock_ee = MagicMock()
        img = MagicMock()
        mock_ee.Image.return_value = img
        coll_id = MAPBIOMAS_COLLECTIONS["brazil"]
        with patch(f"{MODULE}.ee", mock_ee):
            _get_mapbiomas_image(2022, coll_id)
        img.select.assert_called_once_with("classification_2022")

    def test_different_years_produce_different_bands(self):
        mock_ee = MagicMock()
        img = MagicMock()
        mock_ee.Image.return_value = img
        coll_id = MAPBIOMAS_COLLECTIONS["brazil"]
        with patch(f"{MODULE}.ee", mock_ee):
            _get_mapbiomas_image(2020, coll_id)
            _get_mapbiomas_image(2023, coll_id)
        calls = img.select.call_args_list
        assert calls[0] == call("classification_2020")
        assert calls[1] == call("classification_2023")


# ---------------------------------------------------------------------------
# _get_dynamic_world_image
# ---------------------------------------------------------------------------


class TestGetDynamicWorldImage:
    def test_uses_correct_collection(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_dynamic_world_image(region, 2022)
        mock_ee.ImageCollection.assert_called_once_with(DYNAMIC_WORLD_COLLECTION)

    def test_filters_by_bounds(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_dynamic_world_image(region, 2022)
        col.filterBounds.assert_called_once_with(region)

    def test_filters_date_range_for_year(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_dynamic_world_image(region, 2022)
        col.filterDate.assert_called_once_with("2022-01-01", "2023-01-01")

    def test_selects_label_band(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_dynamic_world_image(region, 2022)
        col.select.assert_called_once_with("label")

    def test_applies_mode_reducer(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_dynamic_world_image(region, 2022)
        col.mode.assert_called_once()


# ---------------------------------------------------------------------------
# _get_esri_lulc_image
# ---------------------------------------------------------------------------


class TestGetEsriLulcImage:
    def test_uses_correct_collection(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_esri_lulc_image(region, 2022)
        mock_ee.ImageCollection.assert_called_once_with(ESRI_LULC_COLLECTION)

    def test_filters_by_bounds(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_esri_lulc_image(region, 2022)
        col.filterBounds.assert_called_once_with(region)

    def test_filters_date_within_year(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_esri_lulc_image(region, 2022)
        col.filterDate.assert_called_once_with("2022-01-01", "2022-12-31")

    def test_selects_b1_band(self):
        mock_ee = MagicMock()
        image = _mock_image()
        col = _mock_collection(image)
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_esri_lulc_image(region, 2022)
        # select("b1") is called on the result of .first(), which is the image
        image.select.assert_called_once_with("b1")

    def test_takes_first_image(self):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        region = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            _get_esri_lulc_image(region, 2022)
        col.first.assert_called_once()


# ---------------------------------------------------------------------------
# _download_image_direct
# ---------------------------------------------------------------------------


class TestDownloadImageDirect:
    def _mock_response(self, content: bytes):
        resp = MagicMock()
        resp.content = content
        resp.raise_for_status = MagicMock()
        return resp

    def test_writes_tif_from_zip_response(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()
        zip_bytes = _make_zip_tif_response()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(zip_bytes)
            _download_image_direct(image, region, out, scale=30, crs="EPSG:4326")

        assert out.exists()
        assert out.read_bytes() == b"FAKE_TIF_CONTENT"

    def test_writes_bare_tif_response(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(_make_bare_tif_response())
            _download_image_direct(image, region, out, scale=30, crs="EPSG:4326")

        assert out.exists()
        assert out.read_bytes() == b"FAKE_RAW_TIF"

    def test_calls_get_download_url_with_correct_params(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(_make_bare_tif_response())
            _download_image_direct(image, region, out, scale=10, crs="EPSG:32723")

        image.getDownloadURL.assert_called_once()
        kwargs = image.getDownloadURL.call_args[0][0]
        assert kwargs["scale"] == 10
        assert kwargs["crs"] == "EPSG:32723"
        assert kwargs["region"] is region
        assert kwargs["format"] == "GEO_TIFF"

    def test_raises_on_http_error(self, tmp_path):
        import requests as req_lib

        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()
        resp = MagicMock()
        resp.content = b""
        resp.raise_for_status.side_effect = req_lib.HTTPError("404")

        with patch(f"{MODULE}.requests.get", return_value=resp):
            with pytest.raises(req_lib.HTTPError):
                _download_image_direct(image, region, out, scale=30, crs="EPSG:4326")

    def test_raises_when_zip_has_no_tif(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        # Build zip with only a .txt file
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "no tif here")
        empty_zip = buf.getvalue()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(empty_zip)
            with pytest.raises(ValueError, match="No .tif file"):
                _download_image_direct(image, region, out, scale=30, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# _export_image_to_drive
# ---------------------------------------------------------------------------


class TestExportImageToDrive:
    def _make_mock_task(self, state: str = "COMPLETED", active_count: int = 2):
        task = MagicMock()
        call_count = {"n": 0}

        def _active():
            if call_count["n"] < active_count:
                call_count["n"] += 1
                return True
            return False

        task.active.side_effect = _active
        task.status.return_value = {"state": state}
        return task

    def test_starts_export_task(self):
        mock_ee = MagicMock()
        task = self._make_mock_task()
        mock_ee.batch.Export.image.toDrive.return_value = task
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.ee", mock_ee), patch(f"{MODULE}.time.sleep"):
            _export_image_to_drive(image, region, "test", "MyFolder", 30, "EPSG:4326")

        task.start.assert_called_once()

    def test_polls_until_inactive(self):
        mock_ee = MagicMock()
        task = self._make_mock_task(active_count=3)
        mock_ee.batch.Export.image.toDrive.return_value = task
        image = _mock_image()
        region = MagicMock()

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}.time.sleep") as mock_sleep,
        ):
            _export_image_to_drive(image, region, "test", "Folder", 30, "EPSG:4326")

        assert mock_sleep.call_count == 3

    def test_returns_completed_state(self):
        mock_ee = MagicMock()
        task = self._make_mock_task(state="COMPLETED", active_count=0)
        mock_ee.batch.Export.image.toDrive.return_value = task
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.ee", mock_ee), patch(f"{MODULE}.time.sleep"):
            result = _export_image_to_drive(
                image, region, "test", "Folder", 30, "EPSG:4326"
            )

        assert result == "COMPLETED"

    def test_raises_on_failed_task(self):
        mock_ee = MagicMock()
        task = self._make_mock_task(state="FAILED", active_count=0)
        task.status.return_value = {"state": "FAILED", "error_message": "oom"}
        mock_ee.batch.Export.image.toDrive.return_value = task
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.ee", mock_ee), patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="FAILED"):
                _export_image_to_drive(image, region, "test", "Folder", 30, "EPSG:4326")

    def test_passes_poll_interval_to_sleep(self):
        mock_ee = MagicMock()
        task = self._make_mock_task(active_count=1)
        mock_ee.batch.Export.image.toDrive.return_value = task
        image = _mock_image()
        region = MagicMock()

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}.time.sleep") as mock_sleep,
        ):
            _export_image_to_drive(
                image, region, "test", "Folder", 30, "EPSG:4326", poll_interval=5
            )

        mock_sleep.assert_called_with(5)

    def test_clips_image_to_region(self):
        mock_ee = MagicMock()
        task = self._make_mock_task(active_count=0)
        mock_ee.batch.Export.image.toDrive.return_value = task
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.ee", mock_ee), patch(f"{MODULE}.time.sleep"):
            _export_image_to_drive(image, region, "test", "Folder", 30, "EPSG:4326")

        image.clip.assert_called_once_with(region)


# ---------------------------------------------------------------------------
# download_mapbiomas
# ---------------------------------------------------------------------------


class TestDownloadMapbiomas:
    def _patch_ee_and_download(self, tmp_path, download_mode="direct", **kwargs):
        mock_ee = MagicMock()
        image = _mock_image()
        mock_ee.Image.return_value = image
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._download_image_direct") as mock_direct,
            patch(f"{MODULE}._export_image_to_drive") as mock_drive,
        ):
            download_mapbiomas(
                name="region",
                region_geojson=geojson,
                output_path=out,
                year=2022,
                download_mode=download_mode,
                **kwargs,
            )
            return mock_direct, mock_drive, image

    def test_calls_direct_download_by_default(self, tmp_path):
        mock_direct, _, _ = self._patch_ee_and_download(tmp_path)
        mock_direct.assert_called_once()

    def test_calls_drive_export_when_mode_is_export_drive(self, tmp_path):
        _, mock_drive, _ = self._patch_ee_and_download(
            tmp_path, download_mode="export-drive"
        )
        mock_drive.assert_called_once()

    def test_uses_brazil_collection_by_default(self, tmp_path):
        mock_ee = MagicMock()
        image = _mock_image()
        mock_ee.Image.return_value = image
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._download_image_direct"),
        ):
            download_mapbiomas(
                name="r", region_geojson=geojson, output_path=out, year=2022
            )

        mock_ee.Image.assert_called_once_with(MAPBIOMAS_COLLECTIONS["brazil"])

    def test_uses_custom_collection_key(self, tmp_path):
        mock_ee = MagicMock()
        image = _mock_image()
        mock_ee.Image.return_value = image
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._download_image_direct"),
        ):
            download_mapbiomas(
                name="r",
                region_geojson=geojson,
                output_path=out,
                year=2022,
                collection="amazon",
            )

        mock_ee.Image.assert_called_once_with(MAPBIOMAS_COLLECTIONS["amazon"])

    def test_uses_explicit_collection_id_override(self, tmp_path):
        mock_ee = MagicMock()
        image = _mock_image()
        mock_ee.Image.return_value = image
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"
        custom_id = "projects/my-org/custom_collection"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._download_image_direct"),
        ):
            download_mapbiomas(
                name="r",
                region_geojson=geojson,
                output_path=out,
                year=2022,
                collection_id=custom_id,
            )

        mock_ee.Image.assert_called_once_with(custom_id)

    def test_raises_for_unknown_collection_key(self, tmp_path):
        mock_ee = MagicMock()
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with patch(f"{MODULE}.ee", mock_ee):
            with pytest.raises(KeyError, match="Unknown MapBiomas collection"):
                download_mapbiomas(
                    name="r",
                    region_geojson=geojson,
                    output_path=out,
                    year=2022,
                    collection="narnia",
                )

    def test_raises_when_ee_not_installed(self, tmp_path):
        with patch(f"{MODULE}.ee", None):
            with pytest.raises(ImportError, match="earthengine-api"):
                download_mapbiomas(
                    name="r",
                    region_geojson=mapping(box(0, 0, 1, 1)),
                    output_path=tmp_path / "out.tif",
                    year=2022,
                )


# ---------------------------------------------------------------------------
# download_dynamic_world
# ---------------------------------------------------------------------------


class TestDownloadDynamicWorld:
    def test_calls_direct_download_by_default(self, tmp_path):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._download_image_direct") as mock_direct,
        ):
            download_dynamic_world(
                name="r", region_geojson=geojson, output_path=out, year=2022
            )
        mock_direct.assert_called_once()

    def test_calls_drive_export_when_requested(self, tmp_path):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._export_image_to_drive") as mock_drive,
        ):
            download_dynamic_world(
                name="r",
                region_geojson=geojson,
                output_path=out,
                year=2022,
                download_mode="export-drive",
            )
        mock_drive.assert_called_once()

    def test_raises_when_ee_not_installed(self, tmp_path):
        with patch(f"{MODULE}.ee", None):
            with pytest.raises(ImportError, match="earthengine-api"):
                download_dynamic_world(
                    name="r",
                    region_geojson=mapping(box(0, 0, 1, 1)),
                    output_path=tmp_path / "out.tif",
                    year=2022,
                )


# ---------------------------------------------------------------------------
# download_esri_lulc
# ---------------------------------------------------------------------------


class TestDownloadEsriLulc:
    def test_calls_direct_download_by_default(self, tmp_path):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._download_image_direct") as mock_direct,
        ):
            download_esri_lulc(
                name="r", region_geojson=geojson, output_path=out, year=2022
            )
        mock_direct.assert_called_once()

    def test_calls_drive_export_when_requested(self, tmp_path):
        mock_ee = MagicMock()
        col = _mock_collection(_mock_image())
        mock_ee.ImageCollection.return_value = col
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "region.tif"

        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._export_image_to_drive") as mock_drive,
        ):
            download_esri_lulc(
                name="r",
                region_geojson=geojson,
                output_path=out,
                year=2022,
                download_mode="export-drive",
            )
        mock_drive.assert_called_once()

    def test_raises_when_ee_not_installed(self, tmp_path):
        with patch(f"{MODULE}.ee", None):
            with pytest.raises(ImportError, match="earthengine-api"):
                download_esri_lulc(
                    name="r",
                    region_geojson=mapping(box(0, 0, 1, 1)),
                    output_path=tmp_path / "out.tif",
                    year=2022,
                )


# ---------------------------------------------------------------------------
# _download_one_region
# ---------------------------------------------------------------------------


class TestDownloadOneRegion:
    def _run(self, tmp_path, name="muni_01", source="mapbiomas", **overrides):
        geojson = mapping(box(0, 0, 1, 1))
        kwargs = dict(
            name=name,
            region_geojson=geojson,
            source=source,
            output_dir=tmp_path,
            year=2022,
            download_mode="direct",
            scale=30,
            crs="EPSG:4326",
            drive_folder="GEE_Downloads",
            mapbiomas_collection="brazil",
            mapbiomas_collection_id=None,
        )
        kwargs.update(overrides)
        return _download_one_region(**kwargs)

    def test_skips_if_output_exists(self, tmp_path):
        (tmp_path / "muni_01.tif").touch()
        result_name, success, err = self._run(tmp_path)
        assert success is True
        assert err is None

    def test_returns_true_on_success(self, tmp_path):
        with patch(f"{MODULE}.download_mapbiomas") as mock_fn:
            mock_fn.return_value = None
            result_name, success, err = self._run(tmp_path)
        assert result_name == "muni_01"
        assert success is True
        assert err is None

    def test_returns_false_on_exception(self, tmp_path):
        with patch(f"{MODULE}.download_mapbiomas", side_effect=RuntimeError("boom")):
            result_name, success, err = self._run(tmp_path)
        assert success is False
        assert "boom" in err

    def test_passes_mapbiomas_collection_kwargs(self, tmp_path):
        with patch(f"{MODULE}.download_mapbiomas") as mock_fn:
            self._run(
                tmp_path, mapbiomas_collection="amazon", mapbiomas_collection_id=None
            )
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["collection"] == "amazon"

    def test_does_not_pass_mapbiomas_kwargs_for_other_sources(self, tmp_path):
        with patch(f"{MODULE}.download_dynamic_world") as mock_fn:
            self._run(tmp_path, source="dynamic-world")
        call_kwargs = mock_fn.call_args[1]
        assert "collection" not in call_kwargs

    def test_output_path_uses_name_as_filename(self, tmp_path):
        captured = {}

        def fake_download(**kwargs):
            captured["path"] = kwargs["output_path"]

        with patch(f"{MODULE}.download_mapbiomas", side_effect=fake_download):
            self._run(tmp_path, name="my_region")

        assert captured["path"].name == "my_region.tif"

    def test_esri_lulc_dispatched_correctly(self, tmp_path):
        with patch(f"{MODULE}.download_esri_lulc") as mock_fn:
            self._run(tmp_path, source="esri-lulc")
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# download_regions
# ---------------------------------------------------------------------------


class TestDownloadRegions:
    def _regions(self, n=2):
        return [(f"region_{i}", mapping(box(i, i, i + 1, i + 1))) for i in range(n)]

    def test_raises_for_invalid_source(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown source"):
            download_regions(
                regions=self._regions(),
                output_dir=tmp_path,
                source="invalid",
                year=2022,
            )

    def test_returns_dict_with_all_region_names(self, tmp_path):
        with patch(f"{MODULE}._download_one_region") as mock_fn:
            mock_fn.side_effect = lambda name, *a, **kw: (name, True, None)
            result = download_regions(
                regions=self._regions(3),
                output_dir=tmp_path,
                source="mapbiomas",
                year=2022,
            )
        assert set(result.keys()) == {"region_0", "region_1", "region_2"}

    def test_all_successes_recorded(self, tmp_path):
        with patch(f"{MODULE}._download_one_region") as mock_fn:
            mock_fn.side_effect = lambda name, *a, **kw: (name, True, None)
            result = download_regions(
                regions=self._regions(2),
                output_dir=tmp_path,
                source="dynamic-world",
                year=2022,
            )
        assert all(result.values())

    def test_failure_recorded_in_result(self, tmp_path):
        def _side_effect(name, *a, **kw):
            if name == "region_1":
                return name, False, "error"
            return name, True, None

        with patch(f"{MODULE}._download_one_region", side_effect=_side_effect):
            result = download_regions(
                regions=self._regions(2),
                output_dir=tmp_path,
                source="esri-lulc",
                year=2022,
            )
        assert result["region_0"] is True
        assert result["region_1"] is False

    def test_creates_output_dir_if_missing(self, tmp_path):
        out_dir = tmp_path / "deep" / "nested"
        assert not out_dir.exists()
        with patch(f"{MODULE}._download_one_region") as mock_fn:
            mock_fn.side_effect = lambda name, *a, **kw: (name, True, None)
            download_regions(
                regions=self._regions(1),
                output_dir=out_dir,
                source="mapbiomas",
                year=2022,
            )
        assert out_dir.exists()

    def test_uses_default_scale_when_not_specified(self, tmp_path):
        captured = {}

        def _side_effect(
            name, geojson, source, output_dir, year, download_mode, scale, *a, **kw
        ):
            captured["scale"] = scale
            return name, True, None

        with patch(f"{MODULE}._download_one_region", side_effect=_side_effect):
            download_regions(
                regions=self._regions(1),
                output_dir=tmp_path,
                source="mapbiomas",
                year=2022,
            )

        assert captured["scale"] == _DEFAULT_SCALES["mapbiomas"]

    def test_uses_default_scale_for_dynamic_world(self, tmp_path):
        captured = {}

        def _side_effect(
            name, geojson, source, output_dir, year, download_mode, scale, *a, **kw
        ):
            captured["scale"] = scale
            return name, True, None

        with patch(f"{MODULE}._download_one_region", side_effect=_side_effect):
            download_regions(
                regions=self._regions(1),
                output_dir=tmp_path,
                source="dynamic-world",
                year=2022,
            )

        assert captured["scale"] == _DEFAULT_SCALES["dynamic-world"]

    def test_explicit_scale_overrides_default(self, tmp_path):
        captured = {}

        def _side_effect(
            name, geojson, source, output_dir, year, download_mode, scale, *a, **kw
        ):
            captured["scale"] = scale
            return name, True, None

        with patch(f"{MODULE}._download_one_region", side_effect=_side_effect):
            download_regions(
                regions=self._regions(1),
                output_dir=tmp_path,
                source="mapbiomas",
                year=2022,
                scale=100,
            )

        assert captured["scale"] == 100

    def test_max_workers_respected(self, tmp_path):
        """ThreadPoolExecutor is created with the given max_workers."""
        with (
            patch(f"{MODULE}._download_one_region") as mock_fn,
            patch(
                f"{MODULE}.ThreadPoolExecutor", wraps=ThreadPoolExecutor
            ) as mock_pool,
        ):
            mock_fn.side_effect = lambda name, *a, **kw: (name, True, None)
            download_regions(
                regions=self._regions(4),
                output_dir=tmp_path,
                source="esri-lulc",
                year=2022,
                max_workers=2,
            )

        mock_pool.assert_called_once_with(max_workers=2)

    def test_mapbiomas_collection_forwarded(self, tmp_path):
        captured = {}

        def _side_effect(
            name,
            geojson,
            source,
            output_dir,
            year,
            download_mode,
            scale,
            crs,
            drive_folder,
            mapbiomas_collection,
            mapbiomas_collection_id,
            proxy_url,
            ca_bundle,
        ):
            captured["collection"] = mapbiomas_collection
            return name, True, None

        with patch(f"{MODULE}._download_one_region", side_effect=_side_effect):
            download_regions(
                regions=self._regions(1),
                output_dir=tmp_path,
                source="mapbiomas",
                year=2022,
                mapbiomas_collection="amazon",
            )

        assert captured["collection"] == "amazon"


# ---------------------------------------------------------------------------
# _build_http_transport
# ---------------------------------------------------------------------------


class TestBuildHttpTransport:
    def _mock_httplib2(self):
        mock = MagicMock()
        mock.socks.PROXY_TYPE_HTTP = 3
        return mock

    def test_creates_http_instance(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://proxy.corp:8080")
        mock_h2.Http.assert_called_once()

    def test_passes_proxy_host(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://proxy.corp:8080")
        proxy_info_kwargs = mock_h2.ProxyInfo.call_args[1]
        assert proxy_info_kwargs["proxy_host"] == "proxy.corp"

    def test_passes_proxy_port(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://proxy.corp:9999")
        proxy_info_kwargs = mock_h2.ProxyInfo.call_args[1]
        assert proxy_info_kwargs["proxy_port"] == 9999

    def test_defaults_port_to_8080_when_absent(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://proxy.corp")
        proxy_info_kwargs = mock_h2.ProxyInfo.call_args[1]
        assert proxy_info_kwargs["proxy_port"] == 8080

    def test_parses_credentials_from_url(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://alice:secret@proxy.corp:8080")
        proxy_info_kwargs = mock_h2.ProxyInfo.call_args[1]
        assert proxy_info_kwargs["proxy_user"] == "alice"
        assert proxy_info_kwargs["proxy_pass"] == "secret"

    def test_credentials_are_none_when_absent(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://proxy.corp:8080")
        proxy_info_kwargs = mock_h2.ProxyInfo.call_args[1]
        assert proxy_info_kwargs["proxy_user"] is None
        assert proxy_info_kwargs["proxy_pass"] is None

    def test_passes_ca_bundle_to_http(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://proxy.corp:8080", ca_bundle="/certs/ca.crt")
        http_kwargs = mock_h2.Http.call_args[1]
        assert http_kwargs["ca_certs"] == "/certs/ca.crt"

    def test_ca_bundle_is_none_when_not_provided(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            _build_http_transport("http://proxy.corp:8080")
        http_kwargs = mock_h2.Http.call_args[1]
        assert http_kwargs["ca_certs"] is None

    def test_returns_http_instance(self):
        mock_h2 = self._mock_httplib2()
        with patch(f"{MODULE}.httplib2", mock_h2):
            result = _build_http_transport("http://proxy.corp:8080")
        assert result is mock_h2.Http.return_value


# ---------------------------------------------------------------------------
# initialize_gee — proxy extension
# ---------------------------------------------------------------------------


class TestInitializeGeeProxy:
    def test_builds_transport_when_proxy_url_given(self):
        mock_ee = MagicMock()
        mock_transport = MagicMock()
        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(
                f"{MODULE}._build_http_transport", return_value=mock_transport
            ) as mock_build,
        ):
            initialize_gee(project="p", proxy_url="http://proxy:8080")
        mock_build.assert_called_once_with("http://proxy:8080", None)

    def test_passes_transport_to_initialize(self):
        mock_ee = MagicMock()
        mock_transport = MagicMock()
        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._build_http_transport", return_value=mock_transport),
        ):
            initialize_gee(project="p", proxy_url="http://proxy:8080")
        init_kwargs = mock_ee.Initialize.call_args[1]
        assert init_kwargs["http_transport"] is mock_transport

    def test_no_transport_when_no_proxy(self):
        mock_ee = MagicMock()
        with patch(f"{MODULE}.ee", mock_ee):
            initialize_gee(project="p")
        init_kwargs = mock_ee.Initialize.call_args[1]
        assert "http_transport" not in init_kwargs

    def test_passes_ca_bundle_to_build_transport(self):
        mock_ee = MagicMock()
        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._build_http_transport") as mock_build,
        ):
            initialize_gee(
                project="p",
                proxy_url="http://proxy:8080",
                ca_bundle="/certs/ca.crt",
            )
        mock_build.assert_called_once_with("http://proxy:8080", "/certs/ca.crt")

    def test_transport_used_with_service_account(self):
        mock_ee = MagicMock()
        mock_transport = MagicMock()
        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._build_http_transport", return_value=mock_transport),
        ):
            initialize_gee(
                project="p",
                service_account_email="bot@proj.iam.gserviceaccount.com",
                service_account_key_file="/key.json",
                proxy_url="http://proxy:8080",
            )
        init_kwargs = mock_ee.Initialize.call_args[1]
        assert init_kwargs["http_transport"] is mock_transport


# ---------------------------------------------------------------------------
# _download_image_direct — proxy extension
# ---------------------------------------------------------------------------


class TestDownloadImageDirectProxy:
    def _mock_response(self, content: bytes):
        resp = MagicMock()
        resp.content = content
        resp.raise_for_status = MagicMock()
        return resp

    def test_passes_proxies_dict_when_proxy_url_set(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(b"data")
            _download_image_direct(
                image, region, out, 30, "EPSG:4326", proxy_url="http://proxy:8080"
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["proxies"] == {
            "http": "http://proxy:8080",
            "https": "http://proxy:8080",
        }

    def test_no_proxies_when_proxy_url_is_none(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(b"data")
            _download_image_direct(image, region, out, 30, "EPSG:4326")

        call_kwargs = mock_get.call_args[1]
        assert "proxies" not in call_kwargs

    def test_passes_ca_bundle_as_verify(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(b"data")
            _download_image_direct(
                image,
                region,
                out,
                30,
                "EPSG:4326",
                ca_bundle="/certs/ca.crt",
            )

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["verify"] == "/certs/ca.crt"

    def test_no_verify_kwarg_when_ca_bundle_is_none(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(b"data")
            _download_image_direct(image, region, out, 30, "EPSG:4326")

        call_kwargs = mock_get.call_args[1]
        assert "verify" not in call_kwargs

    def test_proxy_and_ca_bundle_together(self, tmp_path):
        out = tmp_path / "out.tif"
        image = _mock_image()
        region = MagicMock()

        with patch(f"{MODULE}.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(b"data")
            _download_image_direct(
                image,
                region,
                out,
                30,
                "EPSG:4326",
                proxy_url="http://proxy:8080",
                ca_bundle="/certs/ca.crt",
            )

        call_kwargs = mock_get.call_args[1]
        assert "proxies" in call_kwargs
        assert call_kwargs["verify"] == "/certs/ca.crt"


# ---------------------------------------------------------------------------
# download_mapbiomas/dynamic_world/esri_lulc — proxy forwarding
# ---------------------------------------------------------------------------


class TestDownloadSourcesProxyForwarding:
    """Verify proxy_url and ca_bundle reach _download_image_direct in direct mode."""

    def _assert_proxy_forwarded(self, tmp_path, fn, extra_kwargs=None):
        mock_ee = MagicMock()
        mock_ee.Image.return_value = _mock_image()
        mock_ee.ImageCollection.return_value = _mock_collection(_mock_image())
        geojson = mapping(box(0, 0, 1, 1))
        out = tmp_path / "r.tif"
        kwargs = dict(
            name="r",
            region_geojson=geojson,
            output_path=out,
            year=2022,
            proxy_url="http://proxy:8080",
            ca_bundle="/certs/ca.crt",
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        with (
            patch(f"{MODULE}.ee", mock_ee),
            patch(f"{MODULE}._download_image_direct") as mock_direct,
        ):
            fn(**kwargs)
        call_kwargs = mock_direct.call_args[1]
        assert call_kwargs.get("proxy_url") == "http://proxy:8080"
        assert call_kwargs.get("ca_bundle") == "/certs/ca.crt"

    def test_mapbiomas_forwards_proxy(self, tmp_path):
        self._assert_proxy_forwarded(tmp_path, download_mapbiomas)

    def test_dynamic_world_forwards_proxy(self, tmp_path):
        self._assert_proxy_forwarded(tmp_path, download_dynamic_world)

    def test_esri_lulc_forwards_proxy(self, tmp_path):
        self._assert_proxy_forwarded(tmp_path, download_esri_lulc)


# ---------------------------------------------------------------------------
# _download_one_region — proxy forwarding
# ---------------------------------------------------------------------------


class TestDownloadOneRegionProxy:
    def test_forwards_proxy_to_download_fn(self, tmp_path):
        captured = {}

        def fake_download(**kwargs):
            captured.update(kwargs)

        geojson = mapping(box(0, 0, 1, 1))
        with patch(f"{MODULE}.download_mapbiomas", side_effect=fake_download):
            _download_one_region(
                name="r",
                region_geojson=geojson,
                source="mapbiomas",
                output_dir=tmp_path,
                year=2022,
                download_mode="direct",
                scale=30,
                crs="EPSG:4326",
                drive_folder="GEE_Downloads",
                mapbiomas_collection="brazil",
                mapbiomas_collection_id=None,
                proxy_url="http://proxy:8080",
                ca_bundle="/certs/ca.crt",
            )

        assert captured.get("proxy_url") == "http://proxy:8080"
        assert captured.get("ca_bundle") == "/certs/ca.crt"


# ---------------------------------------------------------------------------
# download_regions — proxy forwarding
# ---------------------------------------------------------------------------


class TestDownloadRegionsProxy:
    def test_forwards_proxy_to_worker(self, tmp_path):
        captured = {}

        def _side_effect(
            name,
            geojson,
            source,
            output_dir,
            year,
            download_mode,
            scale,
            crs,
            drive_folder,
            mapbiomas_collection,
            mapbiomas_collection_id,
            proxy_url,
            ca_bundle,
        ):
            captured["proxy_url"] = proxy_url
            captured["ca_bundle"] = ca_bundle
            return name, True, None

        regions = [("r0", mapping(box(0, 0, 1, 1)))]
        with patch(f"{MODULE}._download_one_region", side_effect=_side_effect):
            download_regions(
                regions=regions,
                output_dir=tmp_path,
                source="mapbiomas",
                year=2022,
                proxy_url="http://proxy:8080",
                ca_bundle="/certs/ca.crt",
            )

        assert captured["proxy_url"] == "http://proxy:8080"
        assert captured["ca_bundle"] == "/certs/ca.crt"

    def test_no_proxy_by_default(self, tmp_path):
        captured = {}

        def _side_effect(
            name,
            geojson,
            source,
            output_dir,
            year,
            download_mode,
            scale,
            crs,
            drive_folder,
            mapbiomas_collection,
            mapbiomas_collection_id,
            proxy_url,
            ca_bundle,
        ):
            captured["proxy_url"] = proxy_url
            captured["ca_bundle"] = ca_bundle
            return name, True, None

        regions = [("r0", mapping(box(0, 0, 1, 1)))]
        with patch(f"{MODULE}._download_one_region", side_effect=_side_effect):
            download_regions(
                regions=regions,
                output_dir=tmp_path,
                source="mapbiomas",
                year=2022,
            )

        assert captured["proxy_url"] is None
        assert captured["ca_bundle"] is None
