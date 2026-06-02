# flake8: noqa
"""Tests for tools/dataset_builder/band_combiner.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


from pytorch_segmentation_models_trainer.tools.dataset_builder.band_combiner import (
    combine_all,
    combine_sources_to_tiff,
    find_file_groups,
)


def test_find_file_groups_returns_common_names(tmp_path: Path) -> None:
    """Groups present in all directories should be returned."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    for name in ["file1.tif", "file2.tif"]:
        make_tiff(dir_a / name, np.zeros((4, 4), dtype=np.uint8))
        make_tiff(dir_b / name, np.zeros((4, 4), dtype=np.uint8))

    groups = find_file_groups([dir_a, dir_b], glob_pattern="*.tif")

    names = [g[0] for g in groups]
    assert "file1" in names
    assert "file2" in names
    assert len(groups) == 2


def test_find_file_groups_excludes_missing_in_any_dir(tmp_path: Path) -> None:
    """Groups not present in all directories should be excluded."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    make_tiff(dir_a / "common.tif", np.zeros((4, 4), dtype=np.uint8))
    make_tiff(dir_a / "only_a.tif", np.zeros((4, 4), dtype=np.uint8))
    make_tiff(dir_b / "common.tif", np.zeros((4, 4), dtype=np.uint8))
    make_tiff(dir_b / "only_b.tif", np.zeros((4, 4), dtype=np.uint8))

    groups = find_file_groups([dir_a, dir_b], glob_pattern="*.tif")

    names = [g[0] for g in groups]
    assert names == ["common"]


def test_find_file_groups_with_name_pattern(tmp_path: Path) -> None:
    """Name pattern with {name} capture should extract group key."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    for d in [dir_a, dir_b]:
        make_tiff(d / "MI_tile01.tif", np.zeros((4, 4), dtype=np.uint8))
        make_tiff(d / "MI_tile02.tif", np.zeros((4, 4), dtype=np.uint8))

    groups = find_file_groups(
        [dir_a, dir_b],
        glob_pattern="*.tif",
        name_pattern="MI_{name}.tif",
    )

    names = [g[0] for g in groups]
    assert "tile01" in names
    assert "tile02" in names


def test_find_file_groups_empty_dirs_returns_empty(tmp_path: Path) -> None:
    """Empty source directories should return an empty list."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    groups = find_file_groups([dir_a, dir_b], glob_pattern="*.tif")

    assert groups == []


def test_combine_sources_to_tiff_concatenates_bands(tmp_path: Path) -> None:
    """Output should have bands from all sources concatenated."""
    src1 = tmp_path / "src1.tif"
    src2 = tmp_path / "src2.tif"
    dst = tmp_path / "combined.tif"

    make_tiff(src1, np.ones((2, 4, 4), dtype=np.uint8))
    make_tiff(src2, np.ones((3, 4, 4), dtype=np.uint8) * 2)

    _, success, err = combine_sources_to_tiff([src1, src2], dst)

    assert success is True
    with rasterio.open(dst) as f:
        assert f.count == 5


def test_combine_sources_to_tiff_skip_alpha_4bands(tmp_path: Path) -> None:
    """When skip_alpha=True and source has 4 bands, the last band is dropped."""
    src = tmp_path / "src.tif"
    dst = tmp_path / "out.tif"

    make_tiff(src, np.ones((4, 4, 4), dtype=np.uint8))

    combine_sources_to_tiff([src], dst, skip_alpha=True)

    with rasterio.open(dst) as f:
        assert f.count == 3


def test_combine_sources_to_tiff_no_skip_alpha_flag(tmp_path: Path) -> None:
    """When skip_alpha=False, all bands should be included."""
    src = tmp_path / "src.tif"
    dst = tmp_path / "out.tif"

    make_tiff(src, np.ones((4, 4, 4), dtype=np.uint8))

    combine_sources_to_tiff([src], dst, skip_alpha=False)

    with rasterio.open(dst) as f:
        assert f.count == 4


def test_combine_sources_to_tiff_shape_mismatch_error(tmp_path: Path) -> None:
    """Mismatched spatial shapes should return success=False."""
    src1 = tmp_path / "src1.tif"
    src2 = tmp_path / "src2.tif"
    dst = tmp_path / "combined.tif"

    make_tiff(src1, np.ones((4, 4), dtype=np.uint8))
    make_tiff(src2, np.ones((8, 8), dtype=np.uint8))

    _, success, err = combine_sources_to_tiff([src1, src2], dst)

    assert success is False
    assert err is not None


def test_combine_sources_to_tiff_no_overwrite_skips_existing(tmp_path: Path) -> None:
    """When overwrite=False, existing output should not be replaced."""
    src = tmp_path / "src.tif"
    dst = tmp_path / "out.tif"

    make_tiff(src, np.ones((2, 4, 4), dtype=np.uint8))
    # Write a sentinel file
    dst.write_bytes(b"sentinel")

    _, success, _ = combine_sources_to_tiff([src], dst, overwrite=False)

    assert success is True
    assert dst.read_bytes() == b"sentinel"  # file was not touched


def test_combine_sources_to_tiff_overwrite_replaces(tmp_path: Path) -> None:
    """When overwrite=True, existing output should be replaced."""
    src = tmp_path / "src.tif"
    dst = tmp_path / "out.tif"

    make_tiff(src, np.ones((2, 4, 4), dtype=np.uint8))
    dst.write_bytes(b"sentinel")

    _, success, _ = combine_sources_to_tiff([src], dst, overwrite=True)

    assert success is True
    assert dst.read_bytes() != b"sentinel"


def test_combine_sources_to_tiff_uses_explicit_band_selection(tmp_path: Path) -> None:
    """Explicit band selections should be honored for each source."""
    src1 = tmp_path / "src1.tif"
    src2 = tmp_path / "src2.tif"
    dst = tmp_path / "selected.tif"

    make_tiff(
        src1,
        np.stack(
            [np.full((4, 4), 1, dtype=np.uint8), np.full((4, 4), 2, dtype=np.uint8)]
        ),
    )
    make_tiff(
        src2,
        np.stack(
            [np.full((4, 4), 3, dtype=np.uint8), np.full((4, 4), 4, dtype=np.uint8)]
        ),
    )

    _, success, err = combine_sources_to_tiff(
        [src1, src2],
        dst,
        bands_per_source=[[2], [1]],
        skip_alpha=False,
    )

    assert success is True
    assert err is None
    with rasterio.open(dst) as f:
        assert f.count == 2
        assert f.read(1).max() == 2
        assert f.read(2).max() == 3


def test_combine_all_respects_name_pattern_and_no_groups(tmp_path: Path) -> None:
    """Name pattern filtering should only combine matching filenames."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    out_dir = tmp_path / "out"
    dir_a.mkdir()
    dir_b.mkdir()

    make_tiff(dir_a / "MI_001.tif", np.ones((2, 4, 4), dtype=np.uint8))
    make_tiff(dir_b / "MI_001.tif", np.ones((3, 4, 4), dtype=np.uint8))
    make_tiff(dir_a / "skip.tif", np.ones((2, 4, 4), dtype=np.uint8))

    results = combine_all(
        source_dirs=[dir_a, dir_b],
        output_dir=out_dir,
        glob_pattern="*.tif",
        name_pattern="MI_{name}.tif",
        progress=False,
    )

    assert [p.stem for p in results] == ["001"]
    assert (out_dir / "001.tif").exists()


def test_combine_all_produces_correct_outputs(tmp_path: Path) -> None:
    """combine_all should write one output per matching group."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    out_dir = tmp_path / "out"
    dir_a.mkdir()
    dir_b.mkdir()

    for name in ["tile01.tif", "tile02.tif"]:
        make_tiff(dir_a / name, np.ones((2, 4, 4), dtype=np.uint8))
        make_tiff(dir_b / name, np.ones((3, 4, 4), dtype=np.uint8))

    results = combine_all(
        source_dirs=[dir_a, dir_b],
        output_dir=out_dir,
        glob_pattern="*.tif",
        progress=False,
    )

    assert len(results) == 2
    for p in results:
        assert p.exists()
        with rasterio.open(p) as f:
            assert f.count == 5  # 2+3 bands


def test_combine_all_skips_missing_groups(tmp_path: Path) -> None:
    """Groups missing in any directory should not appear in the output."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    out_dir = tmp_path / "out"
    dir_a.mkdir()
    dir_b.mkdir()

    make_tiff(dir_a / "common.tif", np.ones((2, 4, 4), dtype=np.uint8))
    make_tiff(dir_a / "only_a.tif", np.ones((2, 4, 4), dtype=np.uint8))
    make_tiff(dir_b / "common.tif", np.ones((2, 4, 4), dtype=np.uint8))

    results = combine_all(
        source_dirs=[dir_a, dir_b],
        output_dir=out_dir,
        glob_pattern="*.tif",
        progress=False,
    )

    names = [p.stem for p in results]
    assert names == ["common"]
    assert not (out_dir / "only_a.tif").exists()


def test_find_file_groups_empty_source_list() -> None:
    """An empty source_dirs list should return an empty list immediately."""
    assert find_file_groups([]) == []


def test_combine_sources_to_tiff_nonexistent_file_returns_error(
    tmp_path: Path,
) -> None:
    """A non-existent source path should trigger the exception handler."""
    _, success, err = combine_sources_to_tiff(
        [tmp_path / "nonexistent.tif"], tmp_path / "out.tif", overwrite=True
    )
    assert success is False
    assert err is not None


def test_combine_all_returns_empty_when_no_matching_groups(tmp_path: Path) -> None:
    """combine_all should return [] immediately when dirs share no common files."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    make_tiff(dir_a / "only_a.tif", np.zeros((4, 4), dtype=np.uint8))
    make_tiff(dir_b / "only_b.tif", np.zeros((4, 4), dtype=np.uint8))

    results = combine_all(
        source_dirs=[dir_a, dir_b],
        output_dir=tmp_path / "out",
        glob_pattern="*.tif",
        progress=False,
    )
    assert results == []


def test_combine_all_with_progress_bar(tmp_path: Path) -> None:
    """combine_all with progress=True should still produce correct output."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    make_tiff(dir_a / "tile.tif", np.ones((2, 4, 4), dtype=np.uint8))
    make_tiff(dir_b / "tile.tif", np.ones((2, 4, 4), dtype=np.uint8))

    results = combine_all(
        source_dirs=[dir_a, dir_b],
        output_dir=tmp_path / "out",
        glob_pattern="*.tif",
        progress=True,
    )
    assert len(results) == 1
