# -*- coding: utf-8 -*-
"""Tests for tools/sampling/embedding_extractor.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
    build_tile_id,
    load_embeddings_from_parquet,
    save_embeddings_to_parquet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding_df(n: int = 10, dim: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    return pd.DataFrame(
        {
            "tile_id": [f"img_{i}_0_0" for i in range(n)],
            "image_path": [f"/data/img_{i}.tif" for i in range(n)],
            "mask_path": [f"/data/mask_{i}.tif" for i in range(n)],
            "row_off": [0] * n,
            "col_off": [0] * n,
            "patch_size": [256] * n,
            "embedding": list(embs),
            "dino_model": ["dinov2-base"] * n,
        }
    )


# ---------------------------------------------------------------------------
# build_tile_id
# ---------------------------------------------------------------------------


def test_build_tile_id_basic():
    tid = build_tile_id("/data/img.tif", 128, 256)
    assert tid == "/data/img.tif_128_256"


def test_build_tile_id_zero_offsets():
    tid = build_tile_id("/some/path.tif", 0, 0)
    assert tid == "/some/path.tif_0_0"


# ---------------------------------------------------------------------------
# save / load parquet round-trip
# ---------------------------------------------------------------------------


def test_save_load_parquet_roundtrip(tmp_path):
    df = _make_embedding_df(n=5, dim=8)
    parquet_path = tmp_path / "embeddings.parquet"
    save_embeddings_to_parquet(df, str(parquet_path))
    assert parquet_path.exists()

    loaded = load_embeddings_from_parquet(str(parquet_path))
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 5
    assert "tile_id" in loaded.columns
    assert "embedding" in loaded.columns


def test_save_parquet_creates_file(tmp_path):
    df = _make_embedding_df(n=3, dim=4)
    path = str(tmp_path / "out.parquet")
    save_embeddings_to_parquet(df, path)
    assert Path(path).exists()


def test_load_parquet_returns_correct_shape(tmp_path):
    df = _make_embedding_df(n=8, dim=16)
    path = str(tmp_path / "emb.parquet")
    save_embeddings_to_parquet(df, path)
    loaded = load_embeddings_from_parquet(path)
    assert len(loaded) == 8
    emb_array = np.stack(loaded["embedding"].values)
    assert emb_array.shape == (8, 16)


def test_load_parquet_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_embeddings_from_parquet(str(tmp_path / "missing.parquet"))


def test_save_parquet_overwrites(tmp_path):
    df1 = _make_embedding_df(n=4, dim=4)
    df2 = _make_embedding_df(n=2, dim=4)
    path = str(tmp_path / "out.parquet")
    save_embeddings_to_parquet(df1, path)
    save_embeddings_to_parquet(df2, path)
    loaded = load_embeddings_from_parquet(path)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# extract_embeddings (mocked DINOv2)
# ---------------------------------------------------------------------------


def test_patch_dataset_getitem(tmp_path):
    """Directly test _PatchDataset.__getitem__ to cover lines executed in workers."""
    from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
        _PatchDataset,
    )

    img_path = str(tmp_path / "img.tif")
    import rasterio
    from rasterio.transform import from_bounds

    data = np.ones((3, 64, 64), dtype=np.uint8) * 100
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 64, 64),
    ) as dst:
        dst.write(data)

    dataset = _PatchDataset([img_path], [0], [0], [32])
    item = dataset[0]
    from PIL import Image as _PILImage

    assert isinstance(item, _PILImage.Image)
    assert item.size == (32, 32)


def test_patch_dataset_float_image_rescaled(tmp_path):
    """Float32 images should be rescaled to uint8 (covers branch at line 79)."""
    from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
        _PatchDataset,
    )

    img_path = str(tmp_path / "float.tif")
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.default_rng(42).random((3, 32, 32)).astype(np.float32)
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 32, 32),
    ) as dst:
        dst.write(data)

    dataset = _PatchDataset([img_path], [0], [0], [32])
    item = dataset[0]
    from PIL import Image as _PILImage

    assert isinstance(item, _PILImage.Image)
    assert np.array(item).dtype == np.uint8


def test_patch_dataset_single_band_replicated(tmp_path):
    """Single-band image should be replicated to 3 channels."""
    from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
        _PatchDataset,
    )

    img_path = str(tmp_path / "gray.tif")
    import rasterio
    from rasterio.transform import from_bounds

    data = np.ones((1, 32, 32), dtype=np.uint8) * 128
    with rasterio.open(
        img_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 32, 32),
    ) as dst:
        dst.write(data)

    dataset = _PatchDataset([img_path], [0], [0], [32])
    item = dataset[0]
    from PIL import Image as _PILImage

    assert isinstance(item, _PILImage.Image)
    assert item.mode == "RGB"


def test_extract_embeddings_returns_dataframe(tmp_path):
    from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
        extract_embeddings,
    )

    image_paths = [str(tmp_path / f"img_{i}.tif") for i in range(3)]
    row_offs = [0, 128, 256]
    col_offs = [0, 0, 0]
    patch_sizes = [256, 256, 256]

    # Create dummy image files
    import rasterio
    from rasterio.transform import from_bounds

    for p in image_paths:
        data = np.ones((3, 256, 256), dtype=np.uint8)
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 1, 1, 256, 256),
        ) as dst:
            dst.write(data)

    mock_model = MagicMock()
    mock_processor = MagicMock()

    dim = 768
    mock_outputs = MagicMock()

    import torch

    fake_hidden = torch.zeros(3, 257, dim)
    mock_outputs.last_hidden_state = fake_hidden
    mock_model.return_value = mock_outputs
    mock_model.eval = MagicMock(return_value=None)
    # model.to(device) must return the same mock so return_value stays wired
    mock_model.to.return_value = mock_model

    pixel_values = torch.zeros(3, 3, 224, 224)
    mock_processor.return_value = {"pixel_values": pixel_values}

    with (
        patch(
            "pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor."
            "AutoModel.from_pretrained",
            return_value=mock_model,
        ),
        patch(
            "pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor."
            "AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ),
    ):
        df = extract_embeddings(
            image_paths=image_paths,
            row_offs=row_offs,
            col_offs=col_offs,
            patch_sizes=patch_sizes,
            dino_model="dinov2-base",
            batch_size=8,
            use_gpu=False,
        )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "embedding" in df.columns
    assert "tile_id" in df.columns


def test_extract_embeddings_with_mask_paths(tmp_path):
    """Passing mask_paths stores them in the output DataFrame."""
    from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
        extract_embeddings,
    )

    image_paths = [str(tmp_path / f"img_{i}.tif") for i in range(2)]
    mask_paths = [str(tmp_path / f"msk_{i}.tif") for i in range(2)]
    row_offs = [0, 0]
    col_offs = [0, 0]
    patch_sizes = [256, 256]

    import rasterio
    from rasterio.transform import from_bounds

    for p in image_paths:
        data = np.ones((3, 256, 256), dtype=np.uint8)
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 1, 1, 256, 256),
        ) as dst:
            dst.write(data)

    mock_model = MagicMock()
    mock_processor = MagicMock()
    import torch

    dim = 768
    mock_outputs = MagicMock()
    fake_hidden = torch.zeros(2, 257, dim)
    mock_outputs.last_hidden_state = fake_hidden
    mock_model.return_value = mock_outputs
    mock_model.eval = MagicMock(return_value=None)
    mock_model.to.return_value = mock_model

    pixel_values = torch.zeros(2, 3, 224, 224)
    mock_processor.return_value = {"pixel_values": pixel_values}

    with (
        patch(
            "pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor."
            "AutoModel.from_pretrained",
            return_value=mock_model,
        ),
        patch(
            "pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor."
            "AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ),
    ):
        df = extract_embeddings(
            image_paths=image_paths,
            row_offs=row_offs,
            col_offs=col_offs,
            patch_sizes=patch_sizes,
            mask_paths=mask_paths,
            dino_model="dinov2-base",
            batch_size=8,
            use_gpu=False,
        )

    assert "mask_path" in df.columns
    assert list(df["mask_path"]) == mask_paths
