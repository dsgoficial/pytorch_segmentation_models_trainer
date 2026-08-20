# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pytorch_segmentation_models_trainer.dataset_loader.image_dataset import (
    CSVWindowedImageDataset,
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


class TestCSVWindowedImageDataset(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.img_path = self.tmp / "large_image.tif"
        self.img_data = _write_tif(self.img_path, 512, 512, bands=3)

        self.csv_path = self.tmp / "patches.csv"
        df = pd.DataFrame(
            {
                "image": [str(self.img_path), str(self.img_path)],
                "row_off": [0, 100],
                "col_off": [0, 100],
                "patch_size": [256, 256],
            }
        )
        df.to_csv(self.csv_path, index=False)

    def test_dataset_loading(self):
        ds = CSVWindowedImageDataset(input_csv_path=self.csv_path)
        assert len(ds) == 2

        # Check first patch
        item = ds[0]
        assert "image" in item
        assert "path" in item
        assert item["image"].shape == (256, 256, 3)

        # In ImageDataset (without transform), image remains (H, W, C) numpy array from AbstractDataset?
        # Wait, let's check ImageDataset.__getitem__
        # ImageDataset calls self.load_image which returns (H, W, C).
        # CSVWindowedImageDataset.load_image also returns (H, W, C).
        # So item["image"] should be (256, 256, 3) numpy array if no transform.

        assert isinstance(item["image"], np.ndarray)
        assert item["image"].shape == (256, 256, 3)

        expected_img = np.transpose(self.img_data[:, 0:256, 0:256], (1, 2, 0))
        np.testing.assert_array_equal(item["image"], expected_img)

    def test_custom_keys(self):
        csv_path = self.tmp / "custom_patches.csv"
        pd.DataFrame(
            {
                "img_url": [str(self.img_path)],
                "r_offset": [10],
                "c_offset": [20],
                "size": [64],
            }
        ).to_csv(csv_path, index=False)

        ds = CSVWindowedImageDataset(
            input_csv_path=csv_path,
            image_key="img_url",
            row_off_key="r_offset",
            col_off_key="c_offset",
            patch_size_key="size",
        )
        assert len(ds) == 1
        item = ds[0]
        assert item["image"].shape == (64, 64, 3)

    def test_missing_column_error(self):
        csv_path = self.tmp / "bad_patches.csv"
        pd.DataFrame(
            {"image": [str(self.img_path)], "col_off": [0], "patch_size": [256]}
        ).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="coluna 'row_off' é obrigatória"):
            CSVWindowedImageDataset(input_csv_path=csv_path)

    def test_invalid_image_dtype_raises(self):
        with pytest.raises(ValueError, match="image_dtype"):
            CSVWindowedImageDataset(
                input_csv_path=self.csv_path,
                image_dtype="int32",
            )

    def test_native_image_dtype_preserves_dtype(self):
        ds = CSVWindowedImageDataset(
            input_csv_path=self.csv_path,
            image_dtype="native",
        )

        item = ds[0]

        assert item["image"].dtype == np.uint8
