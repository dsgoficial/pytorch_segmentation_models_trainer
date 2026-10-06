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


class TestCSVWindowedMaskClassMapping(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.img_path = self.tmp / "img.tif"
        self.mask_path = self.tmp / "mask.tif"
        _write_tif(self.img_path, 64, 64, bands=3)
        mask = np.tile(np.arange(6, dtype=np.uint8), (64, 11))[:, :64]
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        with rasterio.open(
            self.mask_path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(mask, 1)
        self.mask = mask
        self.csv_path = self.tmp / "patches.csv"
        pd.DataFrame(
            {
                "image": [str(self.img_path)],
                "mask": [str(self.mask_path)],
                "row_off": [0],
                "col_off": [0],
                "patch_size": [64],
            }
        ).to_csv(self.csv_path, index=False)

    def test_mapping_merges_classes(self):
        ds = CSVWindowedSegmentationDataset(
            input_csv_path=self.csv_path, n_classes=5, mask_class_mapping={5: 3}
        )
        out = ds[0]["mask"].numpy()
        expected = np.where(self.mask == 5, 3, self.mask)
        np.testing.assert_array_equal(out, expected)
        assert out.max() == 4

    def test_without_mapping_masks_unchanged(self):
        ds = CSVWindowedSegmentationDataset(input_csv_path=self.csv_path, n_classes=6)
        np.testing.assert_array_equal(ds[0]["mask"].numpy(), self.mask)

    def test_mapping_applied_before_binarization(self):
        ds = CSVWindowedSegmentationDataset(
            input_csv_path=self.csv_path,
            n_classes=2,
            mask_class_mapping={1: 0, 2: 0, 3: 0, 4: 0},
        )
        out = ds[0]["mask"].numpy()
        np.testing.assert_array_equal(out, (self.mask == 5).astype(np.int64))

    def test_mapping_applied_once_on_window(self):
        # Non-idempotent mapping: applying it twice would send 1 -> 3.
        ds = CSVWindowedSegmentationDataset(
            input_csv_path=self.csv_path, n_classes=6, mask_class_mapping={1: 2, 2: 3}
        )
        out = ds[0]["mask"].numpy()
        expected = np.where(self.mask == 1, 2, np.where(self.mask == 2, 3, self.mask))
        np.testing.assert_array_equal(out, expected)

    def test_invalid_mapping_raises_at_init(self):
        with pytest.raises(ValueError):
            CSVWindowedSegmentationDataset(
                input_csv_path=self.csv_path, n_classes=5, mask_class_mapping={5: 300}
            )

    def test_config_dataclass_has_mapping_field(self):
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.config_definitions.dataset_config import (
            CSVWindowedDatasetConfig,
        )

        cfg = OmegaConf.merge(
            OmegaConf.structured(CSVWindowedDatasetConfig),
            {"mask_class_mapping": {5: 3}},
        )
        assert cfg.mask_class_mapping == {5: 3}

    def test_hydra_yaml_instantiation_with_mapping(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(f"""
            _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset
            input_csv_path: {self.csv_path}
            n_classes: 5
            mask_class_mapping:
              5: 3
            """)
        ds = instantiate(cfg)
        assert ds[0]["mask"].max() == 4

    def test_accepts_unknown_kwargs(self):
        ds = CSVWindowedSegmentationDataset(
            input_csv_path=self.csv_path, n_classes=6, unused_option=1
        )
        assert len(ds) == 1

    def test_structured_config_instantiation(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.config_definitions.dataset_config import (
            CSVWindowedDatasetConfig,
        )

        rel_csv = self.tmp / "relative_patches.csv"
        pd.DataFrame(
            {
                "image": [self.img_path.name],
                "mask": [self.mask_path.name],
                "row_off": [0],
                "col_off": [0],
                "patch_size": [64],
            }
        ).to_csv(rel_csv, index=False)
        cfg = OmegaConf.merge(
            OmegaConf.structured(CSVWindowedDatasetConfig),
            {
                "input_csv_path": str(rel_csv),
                "root_dir": str(self.tmp),
                "n_classes": 5,
                "mask_class_mapping": {5: 3},
            },
        )
        root = OmegaConf.create({"hyperparameters": {"batch_size": 2}, "ds": cfg})
        ds = instantiate(root.ds)
        assert int(np.max(np.asarray(ds[0]["mask"]))) == 4


class TestCSVWindowedReadRetry(BasicTestCase):
    """Bounded retry: unreadable windows skip to the next row, image and mask stay paired."""

    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.img_path = self.tmp / "img.tif"
        self.mask_path = self.tmp / "mask.tif"
        _write_tif(self.img_path, 64, 64, bands=3)
        self.mask = np.tile(np.arange(6, dtype=np.uint8), (64, 11))[:, :64]
        with rasterio.open(
            self.mask_path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 1, 1, 64, 64),
        ) as dst:
            dst.write(self.mask, 1)

    def _csv(self, rows):
        path = self.tmp / "rows.csv"
        pd.DataFrame(
            rows, columns=["image", "mask", "row_off", "col_off", "patch_size"]
        ).to_csv(path, index=False)
        return path

    def _good(self, off=0):
        return [str(self.img_path), str(self.mask_path), off, off, 32]

    def _missing(self):
        return [str(self.tmp / "nope.tif"), str(self.tmp / "nope_mask.tif"), 0, 0, 32]

    @pytest.mark.timeout(30)
    def test_all_rows_missing_raises_instead_of_hanging(self):
        ds = CSVWindowedSegmentationDataset(
            input_csv_path=self._csv([self._missing(), self._missing()]), n_classes=6
        )
        with pytest.raises(RuntimeError, match="nope"):
            ds[0]

    def test_unreadable_row_falls_back_to_next_row(self):
        ds = CSVWindowedSegmentationDataset(
            input_csv_path=self._csv([self._missing(), self._good(off=16)]), n_classes=6
        )
        item = ds[0]
        np.testing.assert_array_equal(item["mask"].numpy(), self.mask[16:48, 16:48])

    def test_image_and_mask_come_from_same_row(self):
        # Row 0: image readable, mask missing -> must not pair row-0 image with row-1 mask.
        rows = [
            [str(self.img_path), str(self.tmp / "nope_mask.tif"), 0, 0, 32],
            self._good(off=16),
        ]
        ds = CSVWindowedSegmentationDataset(input_csv_path=self._csv(rows), n_classes=6)
        item = ds[0]
        with rasterio.open(self.img_path) as src:
            expected_img = src.read(window=rasterio.windows.Window(16, 16, 32, 32))
        np.testing.assert_allclose(
            item["image"].numpy(), expected_img.astype(np.float32) / 255.0, atol=1e-5
        )
        np.testing.assert_array_equal(item["mask"].numpy(), self.mask[16:48, 16:48])

    def test_max_read_retries_limits_attempts(self):
        rows = [self._missing(), self._missing(), self._good()]
        with pytest.raises(RuntimeError):
            CSVWindowedSegmentationDataset(
                input_csv_path=self._csv(rows), n_classes=6, max_read_retries=1
            )[0]
        ds = CSVWindowedSegmentationDataset(
            input_csv_path=self._csv(rows), n_classes=6, max_read_retries=2
        )
        assert ds[0]["mask"].shape == (32, 32)

    def test_invalid_max_read_retries_raises(self):
        with pytest.raises(ValueError):
            CSVWindowedSegmentationDataset(
                input_csv_path=self._csv([self._good()]), max_read_retries=-1
            )

    def test_config_dataclass_has_max_read_retries(self):
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.config_definitions.dataset_config import (
            CSVWindowedDatasetConfig,
        )

        assert OmegaConf.structured(CSVWindowedDatasetConfig).max_read_retries == 10
