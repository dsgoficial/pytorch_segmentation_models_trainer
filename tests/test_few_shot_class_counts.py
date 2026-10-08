# -*- coding: utf-8 -*-
"""Tests for the per-class pixel counting tool (LDAM counts of ClassTrans)."""

import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_origin

from pytorch_segmentation_models_trainer.tools.few_shot.class_counts import (
    count_class_pixels,
    count_class_pixels_from_config,
)


def _write(path, array):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(array.astype(np.uint8), 1)


@pytest.fixture
def index(tmp_path):
    m = np.zeros((4, 4), dtype=np.uint8)
    m[0] = 3
    m[1] = 5
    m[2, :2] = 255
    _write(tmp_path / "a.tif", m)
    rows = [
        {"mask_path": "a.tif", "row_off": 0, "col_off": 0, "width": 4, "height": 4},
        {"mask_path": "a.tif", "row_off": 0, "col_off": 0, "width": 4, "height": 2},
    ]
    csv = tmp_path / "idx.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return tmp_path, csv


def test_counts_per_class_with_mapping_and_ignore(index):
    d, _ = index
    records = [
        {"mask_path": d / "a.tif", "row_off": 0, "col_off": 0, "width": 4, "height": 4}
    ]
    assert count_class_pixels(records, 6) == [6, 0, 0, 4, 0, 4]
    assert count_class_pixels(records, 5, mask_class_mapping={5: 3}) == [6, 0, 0, 8, 0]


def test_values_outside_range_are_ignored(index):
    d, _ = index
    records = [
        {"mask_path": d / "a.tif", "row_off": 0, "col_off": 0, "width": 4, "height": 4}
    ]
    assert count_class_pixels(records, 4) == [6, 0, 0, 4]


def test_from_config_writes_json(index, tmp_path):
    d, csv = index
    out = tmp_path / "o" / "counts.json"
    cfg = OmegaConf.create(
        {
            "paths": {"masks": str(d)},
            "class_counts": {
                "window_index_cache": str(csv),
                "mask_base_path": "${paths.masks}",
                "num_classes": 5,
                "mask_class_mapping": {5: 3},
                "output_json": str(out),
            },
        }
    )
    assert count_class_pixels_from_config(cfg) == str(out)
    data = json.loads(out.read_text())
    assert data["counts"] == [6, 0, 0, 8 + 8, 0]  # 2nd window: rows 0-1
    assert data["num_classes"] == 5


def test_main_dispatch(monkeypatch):
    from pytorch_segmentation_models_trainer import main as main_module
    from pytorch_segmentation_models_trainer.tools.few_shot import class_counts

    monkeypatch.setattr(class_counts, "count_class_pixels_from_config", lambda c: "ok")
    assert (
        main_module.main.__wrapped__(OmegaConf.create({"mode": "count-class-pixels"}))
        == "ok"
    )
