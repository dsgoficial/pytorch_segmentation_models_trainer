# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    CSVWindowedSegmentationDataset,
)
from tests.utils import BasicTestCase

try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")


def _write_tif(path: Path, width: int, height: int, bands: int = 3, dtype="uint8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    if dtype == "uint8":
        data = rng.integers(0, 255, (bands, height, width), dtype=np.uint8)
    elif dtype == "uint16":
        data = rng.integers(0, 65535, (bands, height, width), dtype=np.uint16)
    else:
        data = rng.random((bands, height, width)).astype(np.float32)

    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return data


class TestCSVWindowedSegmentationDataset(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.img_path = self.tmp / "large_image.tif"
        self.mask_path = self.tmp / "large_mask.tif"

        # Create a 512x512 image and mask
        self.img_data = _write_tif(self.img_path, 512, 512, bands=3)
        self.mask_data = _write_tif(self.mask_path, 512, 512, bands=1)

        # Create CSV with 2 patches
        # Patch 1: (0, 0, 256)
        # Patch 2: (100, 100, 256)
        self.csv_path = self.tmp / "patches.csv"
        df = pd.DataFrame(
            {
                "image": [str(self.img_path), str(self.img_path)],
                "mask": [str(self.mask_path), str(self.mask_path)],
                "row_off": [0, 100],
                "col_off": [0, 100],
                "patch_size": [256, 256],
            }
        )
        df.to_csv(self.csv_path, index=False)

    def test_dataset_loading(self):
        ds = CSVWindowedSegmentationDataset(input_csv_path=self.csv_path)
        assert len(ds) == 2

        # Check first patch
        item = ds[0]
        assert item["image"].shape == (3, 256, 256)
        assert item["mask"].shape == (256, 256)

        # Check values
        # Default n_classes=2 binarizes mask (>0 -> 1)
        expected_img = self.img_data[:, 0:256, 0:256].astype(np.float32) / 255.0
        np.testing.assert_allclose(item["image"].numpy(), expected_img, atol=1e-5)

        expected_mask = (self.mask_data[0, 0:256, 0:256] > 0).astype(np.int64)
        np.testing.assert_array_equal(item["mask"].numpy(), expected_mask)

        # Check second patch
        item = ds[1]
        assert item["image"].shape == (3, 256, 256)
        expected_img = self.img_data[:, 100:356, 100:356].astype(np.float32) / 255.0
        np.testing.assert_allclose(item["image"].numpy(), expected_img, atol=1e-5)

    def test_custom_keys(self):
        # Create CSV with custom keys
        csv_path = self.tmp / "custom_patches.csv"
        pd.DataFrame(
            {
                "img": [str(self.img_path)],
                "msk": [str(self.mask_path)],
                "r": [0],
                "c": [0],
                "sz": [128],
            }
        ).to_csv(csv_path, index=False)

        ds = CSVWindowedSegmentationDataset(
            input_csv_path=csv_path,
            image_key="img",
            mask_key="msk",
            row_off_key="r",
            col_off_key="c",
            patch_size_key="sz",
        )
        assert len(ds) == 1
        item = ds[0]
        assert item["image"].shape == (3, 128, 128)

    def test_missing_column_error(self):
        csv_path = self.tmp / "bad_patches.csv"
        pd.DataFrame(
            {
                "image": [str(self.img_path)],
                "mask": [str(self.mask_path)],
                # missing row_off
                "col_off": [0],
                "patch_size": [256],
            }
        ).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="coluna 'row_off' é obrigatória"):
            CSVWindowedSegmentationDataset(input_csv_path=csv_path)
