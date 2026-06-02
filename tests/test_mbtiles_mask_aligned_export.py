# -*- coding: utf-8 -*-
"""Tests for mask-aligned MBTiles image export tooling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from omegaconf import MissingMandatoryValue, OmegaConf

from pytorch_segmentation_models_trainer.config_definitions import (
    tools_config_def,
)
from pytorch_segmentation_models_trainer.tools import cli as cli_module
from pytorch_segmentation_models_trainer.tools.mbtiles import alignment
from pytorch_segmentation_models_trainer.tools.mbtiles import (
    export_mask_aligned_images as exporter,
)

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")


def _write_rgb_raster(path: Path, width=64, height=64, dtype="uint8"):
    """Write deterministic RGB source raster."""
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((height, width))
    data = np.stack(
        [
            xx.astype(dtype),
            yy.astype(dtype),
            ((xx + yy) % 255).astype(dtype),
        ]
    )
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": dtype,
        "crs": "EPSG:3857",
        "transform": from_origin(0, 64, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return data


def _write_mask(path: Path, width=64, height=64, crs="EPSG:3857"):
    """Write deterministic single-band mask raster."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[16:48, 16:48] = 2
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": from_origin(0, 64, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)
    return mask


def test_export_full_mask_writes_outputs_and_manifest(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "masks" / "mask_a.tif"
    out_dir = tmp_path / "exports"
    _write_rgb_raster(source_path)
    _write_mask(mask_path)

    result = exporter.export_mask_aligned_images(
        mbtiles_path=source_path,
        mask_paths=[mask_path],
        output_dir=out_dir,
        full_mask=True,
        write_sidecar_png=True,
    )

    assert result.count == 1
    assert result.manifest_path == out_dir / "manifest.csv"
    assert result.manifest_path.exists()
    assert len(result.image_paths) == 1
    assert len(result.mask_paths) == 1
    assert len(result.preview_paths) == 1

    with (
        rasterio.open(mask_path) as mask_src,
        rasterio.open(result.image_paths[0]) as img_src,
    ):
        assert img_src.width == mask_src.width
        assert img_src.height == mask_src.height
        assert img_src.crs == mask_src.crs
        assert img_src.transform.almost_equals(mask_src.transform)
        assert img_src.count == 3
        assert img_src.dtypes == ("uint8", "uint8", "uint8")

    with rasterio.open(result.mask_paths[0]) as exported_mask:
        with rasterio.open(mask_path) as mask_src:
            assert exported_mask.read(1).dtype == np.uint8
            assert exported_mask.transform.almost_equals(mask_src.transform)

    manifest = pd.read_csv(result.manifest_path)
    assert list(manifest.columns) == [
        "image_path",
        "mask_path",
        "preview_path",
        "source_mask_path",
        "row_off",
        "col_off",
        "width",
        "height",
        "crs",
        "transform",
    ]
    assert Path(manifest.loc[0, "preview_path"]).exists()


def test_export_patch_mode_generates_expected_windows(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "masks" / "mask_a.tif"
    out_dir = tmp_path / "exports"
    _write_rgb_raster(source_path, width=32, height=32)
    _write_mask(mask_path, width=32, height=32)

    result = exporter.export_mask_aligned_images(
        mbtiles_path=source_path,
        mask_paths=[mask_path],
        output_dir=out_dir,
        patch_size=16,
        stride=16,
        write_sidecar_png=False,
    )

    assert result.count == 4
    assert len(result.preview_paths) == 0
    manifest = pd.read_csv(result.manifest_path)
    assert set(manifest["row_off"]) == {0, 16}
    assert set(manifest["col_off"]) == {0, 16}
    assert set(manifest["width"]) == {16}
    assert set(manifest["height"]) == {16}

    with rasterio.open(result.image_paths[0]) as img_src:
        assert img_src.width == 16
        assert img_src.height == 16


def test_export_skip_empty_masks_filters_zero_windows(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "masks" / "empty.tif"
    out_dir = tmp_path / "exports"
    _write_rgb_raster(source_path, width=16, height=16)
    _write_mask(mask_path, width=16, height=16)
    with rasterio.open(mask_path, "r+") as dst:
        dst.write(np.zeros((16, 16), dtype=np.uint8), 1)

    result = exporter.export_mask_aligned_images(
        mbtiles_path=source_path,
        mask_paths=[mask_path],
        output_dir=out_dir,
        full_mask=True,
        skip_empty_masks=True,
    )

    assert result.count == 0
    assert result.manifest_path.exists()
    assert pd.read_csv(result.manifest_path).empty


def test_export_selected_bands_and_native_dtype(tmp_path):
    source_path = tmp_path / "source_float.tif"
    mask_path = tmp_path / "masks" / "mask_a.tif"
    out_dir = tmp_path / "exports"
    _write_rgb_raster(source_path, dtype="float32")
    _write_mask(mask_path)

    result = exporter.export_mask_aligned_images(
        mbtiles_path=source_path,
        mask_paths=[mask_path],
        output_dir=out_dir,
        full_mask=True,
        selected_bands=[1, 2],
        image_dtype="native",
        write_sidecar_png=True,
    )

    with rasterio.open(result.image_paths[0]) as src:
        assert src.count == 2
        assert src.dtypes == ("float32", "float32")


def test_helper_validation_errors(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "mask.tif"
    _write_rgb_raster(source_path)
    _write_mask(mask_path)

    assert alignment.normalize_selected_bands(None, 3) is None
    assert alignment.normalize_selected_bands([1, 3], 3) == [1, 3]
    assert alignment.resolve_resampling("nearest").name == "nearest"

    with pytest.raises(ValueError, match="Unknown resampling"):
        alignment.resolve_resampling("invalid")
    with pytest.raises(ValueError, match="selected_bands"):
        alignment.normalize_selected_bands([0], 3)
    with pytest.raises(ValueError, match="Either mask_paths or mask_dir"):
        exporter.export_mask_aligned_images(
            mbtiles_path=source_path,
            output_dir=tmp_path / "out",
            full_mask=True,
        )
    with pytest.raises(ValueError, match="No mask files found"):
        exporter.export_mask_aligned_images(
            mbtiles_path=source_path,
            mask_dir=tmp_path / "empty",
            output_dir=tmp_path / "out",
            full_mask=True,
        )
    with pytest.raises(ValueError, match="positive integers"):
        exporter.export_mask_aligned_images(
            mbtiles_path=source_path,
            mask_paths=[mask_path],
            output_dir=tmp_path / "out",
            patch_size=0,
            stride=1,
        )


def test_preview_rgb_handles_channel_variants():
    one_channel = np.ones((1, 4, 4), dtype=np.float32)
    two_channels = np.ones((2, 4, 4), dtype=np.uint8)

    assert exporter._to_preview_rgb(one_channel).shape == (4, 4, 3)
    assert exporter._to_preview_rgb(two_channels).shape == (4, 4, 3)

    with pytest.raises(ValueError, match="Expected image array"):
        exporter._to_preview_rgb(np.ones((4, 4), dtype=np.uint8))


def test_export_requires_full_mask_or_patch_size(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "mask.tif"
    _write_rgb_raster(source_path)
    _write_mask(mask_path)

    with pytest.raises(ValueError, match="full_mask=True or patch_size"):
        exporter.export_mask_aligned_images(
            mbtiles_path=source_path,
            mask_paths=[mask_path],
            output_dir=tmp_path / "exports",
        )


def test_config_dataclass_exposes_export_defaults():
    cfg = OmegaConf.structured(tools_config_def.MBTilesMaskAlignedExportConfig)
    assert "export_mask_aligned_images" in cfg._target_
    assert cfg.output_format == "tif"
    assert cfg.write_sidecar_png is True
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.mbtiles_path


def test_cli_export_mask_aligned_images(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_dir = tmp_path / "masks"
    mask_path = mask_dir / "mask_a.tif"
    out_dir = tmp_path / "exports"
    _write_rgb_raster(source_path)
    _write_mask(mask_path)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "export-mbtiles-mask-aligned",
            "--mbtiles-path",
            str(source_path),
            "--mask-dir",
            str(mask_dir),
            "--output-dir",
            str(out_dir),
            "--full-mask",
            "--no-sidecar-png",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Exported 1 mask-aligned image" in result.output
    assert (out_dir / "manifest.csv").exists()
