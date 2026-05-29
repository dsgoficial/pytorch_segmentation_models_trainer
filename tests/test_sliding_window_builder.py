"""Tests for tools/dataset_builder/sliding_window_builder.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds


def make_tiff(path: Path, data: np.ndarray, crs: str = "EPSG:4326") -> None:
    """Helper: write a minimal GeoTIFF."""
    h = data.shape[-2] if data.ndim > 2 else data.shape[0]
    w = data.shape[-1] if data.ndim > 2 else data.shape[1]
    transform = from_bounds(0, 0, 1, 1, w, h)
    bands = 1 if data.ndim == 2 else data.shape[0]
    profile = dict(
        driver="GTiff",
        dtype=data.dtype,
        width=w,
        height=h,
        count=bands,
        crs=crs,
        transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


from pytorch_segmentation_models_trainer.tools.dataset_builder.sliding_window_builder import (
    build_sliding_window_dataset,
    compute_sliding_windows,
)


@pytest.fixture()
def csv_pair(tmp_path: Path):
    """Create a pair of image/mask TIFFs and a CSV pointing to them."""
    img_dir = tmp_path / "src" / "images"
    msk_dir = tmp_path / "src" / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)

    img = img_dir / "tile.tif"
    msk = msk_dir / "tile.tif"

    make_tiff(img, np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8))
    make_tiff(msk, np.random.randint(0, 5, (1, 64, 64), dtype=np.uint8))

    csv_path = tmp_path / "pairs.csv"
    pd.DataFrame({"image": [str(img)], "mask": [str(msk)]}).to_csv(
        csv_path, index=False
    )

    return csv_path, tmp_path / "output"


# ---- compute_sliding_windows ----


def test_compute_sliding_windows_no_overlap() -> None:
    """No-overlap windows should cover the image without repetition."""
    windows = compute_sliding_windows(64, 64, 32, 0.0)
    # 2×2 grid → 4 windows
    assert len(windows) == 4
    for r0, r1, c0, c1 in windows:
        assert r1 - r0 == 32
        assert c1 - c0 == 32


def test_compute_sliding_windows_with_overlap() -> None:
    """Overlap should produce more windows than no overlap."""
    wins_no = compute_sliding_windows(64, 64, 32, 0.0)
    wins_ov = compute_sliding_windows(64, 64, 32, 0.5)
    assert len(wins_ov) > len(wins_no)


def test_compute_sliding_windows_edge_adjustment() -> None:
    """All windows should have exactly window_size rows and columns."""
    for r0, r1, c0, c1 in compute_sliding_windows(70, 70, 32, 0.0):
        assert r1 - r0 == 32
        assert c1 - c0 == 32
        assert r1 <= 70
        assert c1 <= 70


# ---- build_sliding_window_dataset ----


def test_build_sliding_window_dataset_creates_patches(csv_pair) -> None:
    """Patch TIFFs should be created in the output directory."""
    csv_path, out_dir = csv_pair

    df = build_sliding_window_dataset(
        input_csv=csv_path,
        output_dir=out_dir,
        window_size=32,
        overlap=0.0,
        progress=False,
    )

    assert len(df) > 0
    for img_path in df["image"]:
        assert Path(img_path).exists()


def test_build_sliding_window_dataset_output_csv(csv_pair) -> None:
    """An output CSV should be saved to output_dir/{input_csv.stem}.csv."""
    csv_path, out_dir = csv_pair

    build_sliding_window_dataset(
        input_csv=csv_path,
        output_dir=out_dir,
        window_size=32,
        progress=False,
    )

    expected_csv = out_dir / f"{csv_path.stem}.csv"
    assert expected_csv.exists()

    df = pd.read_csv(expected_csv)
    assert "image" in df.columns
    assert "mask" in df.columns


def test_build_sliding_window_dataset_class_remap(csv_pair) -> None:
    """class_remap should apply the mapping to mask patches."""
    csv_path, out_dir = csv_pair

    df = build_sliding_window_dataset(
        input_csv=csv_path,
        output_dir=out_dir,
        window_size=32,
        class_remap={1: 99, 2: 99},
        progress=False,
    )

    # None of the masks should contain value 1 or 2
    for mask_path in df["mask"]:
        with rasterio.open(mask_path) as f:
            data = f.read(1)
        assert 1 not in data
        assert 2 not in data


def test_build_sliding_window_dataset_blacklist_skips(tmp_path: Path) -> None:
    """Rows whose image path matches the blacklist should be skipped."""
    img_dir = tmp_path / "good_dir" / "images"
    bl_dir = tmp_path / "skip_me" / "images"
    msk_dir = tmp_path / "good_dir" / "masks"
    bl_msk_dir = tmp_path / "skip_me" / "masks"
    for d in [img_dir, bl_dir, msk_dir, bl_msk_dir]:
        d.mkdir(parents=True)

    good_img = img_dir / "good.tif"
    bad_img = bl_dir / "bad.tif"
    good_msk = msk_dir / "good.tif"
    bad_msk = bl_msk_dir / "bad.tif"

    for p in [good_img, bad_img]:
        make_tiff(p, np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8))
    for p in [good_msk, bad_msk]:
        make_tiff(p, np.zeros((1, 32, 32), dtype=np.uint8))

    csv_path = tmp_path / "pairs.csv"
    pd.DataFrame(
        {"image": [str(good_img), str(bad_img)], "mask": [str(good_msk), str(bad_msk)]}
    ).to_csv(csv_path, index=False)

    df = build_sliding_window_dataset(
        input_csv=csv_path,
        output_dir=tmp_path / "out",
        window_size=16,
        blacklist=["skip_me"],
        progress=False,
    )

    for img_path in df["image"]:
        assert "skip_me" not in img_path


def test_build_sliding_window_dataset_preserves_mi(tmp_path: Path) -> None:
    """The mi column should be preserved in the output DataFrame."""
    img_dir = tmp_path / "src" / "images"
    msk_dir = tmp_path / "src" / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)

    img = img_dir / "tile.tif"
    msk = msk_dir / "tile.tif"
    make_tiff(img, np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8))
    make_tiff(msk, np.zeros((1, 32, 32), dtype=np.uint8))

    csv_path = tmp_path / "pairs.csv"
    pd.DataFrame({"image": [str(img)], "mask": [str(msk)], "mi": ["MI001"]}).to_csv(
        csv_path, index=False
    )

    df = build_sliding_window_dataset(
        input_csv=csv_path,
        output_dir=tmp_path / "out",
        window_size=16,
        progress=False,
    )

    assert "mi" in df.columns
    assert df["mi"].iloc[0] == "MI001"


def test_build_sliding_window_dataset_georeferences_patches(csv_pair) -> None:
    """Each patch should have a different transform than the source image."""
    csv_path, out_dir = csv_pair

    # Get source image transform
    src_csv = pd.read_csv(csv_path)
    with rasterio.open(src_csv["image"].iloc[0]) as f:
        src_transform = f.transform

    df = build_sliding_window_dataset(
        input_csv=csv_path,
        output_dir=out_dir,
        window_size=32,
        overlap=0.0,
        progress=False,
    )

    # At least some patches should have a different origin than the source
    patch_transforms = []
    for img_path in df["image"]:
        with rasterio.open(img_path) as f:
            patch_transforms.append(f.transform)

    different = any(t != src_transform for t in patch_transforms)
    assert (
        different
    ), "All patches have the same transform as the source — georeferencing may be wrong"
