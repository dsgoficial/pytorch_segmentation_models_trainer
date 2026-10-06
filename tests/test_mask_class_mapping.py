# -*- coding: utf-8 -*-
"""Tests for the on-the-fly mask class remapping helpers."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.dataset_loader.mask_class_mapping import (
    apply_mask_class_lut,
    build_mask_class_lut,
)


class TestBuildMaskClassLut:
    def test_none_returns_none(self):
        assert build_mask_class_lut(None) is None

    def test_empty_mapping_returns_none(self):
        assert build_mask_class_lut({}) is None

    def test_identity_outside_mapping(self):
        lut = build_mask_class_lut({5: 3})
        assert lut.dtype == np.uint8
        assert lut.shape == (256,)
        assert lut[5] == 3
        expected = np.arange(256, dtype=np.uint8)
        expected[5] = 3
        np.testing.assert_array_equal(lut, expected)

    def test_accepts_dictconfig_with_int_keys(self):
        cfg = OmegaConf.create({"mapping": {5: 3, 4: 255}})
        lut = build_mask_class_lut(cfg.mapping)
        assert lut[5] == 3
        assert lut[4] == 255

    def test_accepts_numeric_string_keys(self):
        lut = build_mask_class_lut({"5": "3"})
        assert lut[5] == 3

    def test_mapping_is_not_chained(self):
        # 1 -> 2 and 2 -> 3 must not turn 1 into 3.
        lut = build_mask_class_lut({1: 2, 2: 3})
        assert lut[1] == 2
        assert lut[2] == 3

    @pytest.mark.parametrize(
        "mapping",
        [{-1: 0}, {256: 0}, {0: -1}, {0: 256}, {"a": 0}, {0: 1.5}, {True: 0}],
    )
    def test_invalid_values_raise(self, mapping):
        with pytest.raises(ValueError):
            build_mask_class_lut(mapping)

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError):
            build_mask_class_lut([(5, 3)])


class TestApplyMaskClassLut:
    def test_none_lut_returns_same_array(self):
        mask = np.array([[0, 5]], dtype=np.uint8)
        assert apply_mask_class_lut(mask, None) is mask

    def test_remaps_values_and_keeps_shape_dtype(self):
        mask = np.array([[0, 1, 5], [5, 255, 3]], dtype=np.uint8)
        out = apply_mask_class_lut(mask, build_mask_class_lut({5: 3}))
        np.testing.assert_array_equal(out, [[0, 1, 3], [3, 255, 3]])
        assert out.dtype == np.uint8
        assert out.shape == mask.shape

    def test_can_send_class_to_ignore_index(self):
        mask = np.array([0, 4, 5], dtype=np.uint8)
        out = apply_mask_class_lut(mask, build_mask_class_lut({4: 255}))
        np.testing.assert_array_equal(out, [0, 255, 5])

    def test_does_not_modify_input(self):
        mask = np.array([5, 5], dtype=np.uint8)
        apply_mask_class_lut(mask, build_mask_class_lut({5: 3}))
        np.testing.assert_array_equal(mask, [5, 5])

    def test_non_uint8_mask_is_cast(self):
        mask = np.array([0, 5], dtype=np.int64)
        out = apply_mask_class_lut(mask, build_mask_class_lut({5: 3}))
        np.testing.assert_array_equal(out, [0, 3])
        assert out.dtype == np.uint8

    def test_out_of_range_mask_values_raise(self):
        mask = np.array([0, 300], dtype=np.int64)
        with pytest.raises(ValueError):
            apply_mask_class_lut(mask, build_mask_class_lut({5: 3}))


class TestSegmentationDatasetMaskClassMapping:
    """``mask_class_mapping`` lives in ``SegmentationDataset`` and is inherited."""

    @pytest.fixture()
    def tile_pair(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds

        mask = np.tile(np.arange(6, dtype=np.uint8), (16, 3))[:, :16]
        for sub, data in (
            ("images", np.zeros((3, 16, 16), dtype=np.uint8)),
            ("masks", mask[np.newaxis]),
        ):
            (tmp_path / sub).mkdir()
            with rasterio.open(
                tmp_path / sub / "a.tif",
                "w",
                driver="GTiff",
                height=16,
                width=16,
                count=data.shape[0],
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_bounds(0, 0, 1, 1, 16, 16),
            ) as dst:
                dst.write(data)
        return tmp_path, mask

    def test_segmentation_dataset_from_csv(self, tile_pair, tmp_path):
        import pandas as pd
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            SegmentationDataset,
        )

        root, mask = tile_pair
        csv = tmp_path / "tiles.csv"
        pd.DataFrame(
            {"image": [str(root / "images/a.tif")], "mask": [str(root / "masks/a.tif")]}
        ).to_csv(csv, index=False)
        ds = SegmentationDataset(
            input_csv_path=csv,
            n_classes=5,
            use_rasterio=True,
            mask_class_mapping={5: 3},
        )
        np.testing.assert_array_equal(
            ds[0]["mask"].numpy(), np.where(mask == 5, 3, mask)
        )
        assert ds.mask_class_lut is not None

    def test_segmentation_dataset_without_mapping_unchanged(self, tile_pair, tmp_path):
        import pandas as pd
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            SegmentationDataset,
        )

        root, mask = tile_pair
        csv = tmp_path / "tiles.csv"
        pd.DataFrame(
            {"image": [str(root / "images/a.tif")], "mask": [str(root / "masks/a.tif")]}
        ).to_csv(csv, index=False)
        ds = SegmentationDataset(input_csv_path=csv, n_classes=6, use_rasterio=True)
        np.testing.assert_array_equal(ds[0]["mask"].numpy(), mask)
        assert ds.mask_class_lut is None

    def test_mapping_applied_before_binarization(self, tile_pair, tmp_path):
        import pandas as pd
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            SegmentationDataset,
        )

        root, mask = tile_pair
        csv = tmp_path / "tiles.csv"
        pd.DataFrame(
            {"image": [str(root / "images/a.tif")], "mask": [str(root / "masks/a.tif")]}
        ).to_csv(csv, index=False)
        ds = SegmentationDataset(
            input_csv_path=csv,
            n_classes=2,
            use_rasterio=True,
            mask_class_mapping={1: 0, 2: 0, 3: 0, 4: 0},
        )
        np.testing.assert_array_equal(
            ds[0]["mask"].numpy(), (mask == 5).astype(np.int64)
        )

    def test_segmentation_dataset_from_folder(self, tile_pair):
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            SegmentationDatasetFromFolder,
        )

        root, mask = tile_pair
        ds = SegmentationDatasetFromFolder(
            image_folder=root / "images",
            mask_folder=root / "masks",
            n_classes=5,
            use_rasterio=True,
            mask_class_mapping={5: 3},
        )
        np.testing.assert_array_equal(
            ds[0]["mask"].numpy(), np.where(mask == 5, 3, mask)
        )

    def test_invalid_mapping_raises(self, tile_pair, tmp_path):
        import pandas as pd
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            SegmentationDataset,
        )

        root, _ = tile_pair
        csv = tmp_path / "tiles.csv"
        pd.DataFrame(
            {"image": [str(root / "images/a.tif")], "mask": [str(root / "masks/a.tif")]}
        ).to_csv(csv, index=False)
        with pytest.raises(ValueError):
            SegmentationDataset(input_csv_path=csv, mask_class_mapping={5: 999})

    def test_dataset_config_has_mapping_field(self):
        from pytorch_segmentation_models_trainer.config_definitions.dataset_config import (
            DatasetConfig,
        )

        assert OmegaConf.structured(DatasetConfig).mask_class_mapping is None
