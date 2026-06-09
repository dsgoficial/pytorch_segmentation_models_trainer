# -*- coding: utf-8 -*-
"""DINOv2 embedding extraction with DataLoader concurrency and Parquet persistence."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import rasterio.windows
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

_DINO_HF_IDS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
}


def build_tile_id(image_path: str, row_off: int, col_off: int) -> str:
    """Build a unique string identifier for a patch.

    Args:
        image_path: file path of the source image.
        row_off: top-left row offset.
        col_off: top-left column offset.

    Returns:
        String key: ``<image_path>_<row_off>_<col_off>``.
    """
    return f"{image_path}_{row_off}_{col_off}"


class _PatchDataset(Dataset):
    """Minimal Dataset that reads image patches from rasterio-readable files.

    Each item is a PIL Image cropped to the requested window, suitable for
    AutoProcessor input.
    """

    def __init__(
        self,
        image_paths: List[str],
        row_offs: List[int],
        col_offs: List[int],
        patch_sizes: List[int],
    ):
        self.image_paths = image_paths
        self.row_offs = row_offs
        self.col_offs = col_offs
        self.patch_sizes = patch_sizes

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        row_off = self.row_offs[idx]
        col_off = self.col_offs[idx]
        ps = self.patch_sizes[idx]
        window = rasterio.windows.Window(col_off, row_off, ps, ps)
        with rasterio.open(path) as src:
            # Read at most 3 bands (RGB); single-band is replicated to 3
            n_bands = min(src.count, 3)
            data = src.read(list(range(1, n_bands + 1)), window=window)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        # (C, H, W) uint8 → PIL Image
        img_array = np.transpose(data, (1, 2, 0))
        if img_array.dtype != np.uint8:
            img_array = (
                (img_array - img_array.min())
                / (img_array.max() - img_array.min() + 1e-8)
                * 255
            ).astype(np.uint8)
        return Image.fromarray(img_array)


def save_embeddings_to_parquet(df: pd.DataFrame, parquet_path: str) -> None:
    """Write an embeddings DataFrame to a Parquet file.

    The 'embedding' column is stored as a list<float32> column.

    Args:
        df: DataFrame with at least 'tile_id' and 'embedding' columns.
        parquet_path: destination file path.
    """
    records = df.copy()
    # Convert embedding lists/arrays to Python lists for Arrow serialisation
    records["embedding"] = records["embedding"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )
    table = pa.Table.from_pandas(records, preserve_index=False)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)


def load_embeddings_from_parquet(parquet_path: str) -> pd.DataFrame:
    """Load an embeddings DataFrame from a Parquet file.

    Args:
        parquet_path: path to the Parquet file.

    Returns:
        DataFrame with 'embedding' column as lists of floats.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    table = pq.read_table(parquet_path)
    return table.to_pandas()


def extract_embeddings(
    image_paths: List[str],
    row_offs: List[int],
    col_offs: List[int],
    patch_sizes: List[int],
    dino_model: str = "dinov2-base",
    batch_size: int = 64,
    use_gpu: bool = True,
    num_workers: int = 4,
    mask_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract DINOv2 CLS-token embeddings for a list of image patches.

    Patches are loaded in parallel via a DataLoader (num_workers workers),
    then processed in batches through the DINOv2 model.

    Args:
        image_paths: list of rasterio-readable image paths.
        row_offs: top-left row offsets per patch.
        col_offs: top-left column offsets per patch.
        patch_sizes: patch height/width per patch.
        dino_model: one of 'dinov2-small', 'dinov2-base', 'dinov2-large'.
        batch_size: inference batch size.
        use_gpu: move model and tensors to CUDA if available.
        num_workers: DataLoader worker count for parallel patch loading.
        mask_paths: optional; stored in the output DataFrame only.

    Returns:
        DataFrame with columns: tile_id, image_path, mask_path, row_off, col_off,
        patch_size, embedding, dino_model.
    """
    hf_id = _DINO_HF_IDS.get(dino_model, dino_model)
    processor = AutoProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    model.eval()

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = _PatchDataset(image_paths, row_offs, col_offs, patch_sizes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=list,  # each item is a PIL Image; collate as list
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for batch_images in loader:
            inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            # CLS token is at position 0 of last_hidden_state
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_tokens.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    tile_ids = [
        build_tile_id(ip, ro, co) for ip, ro, co in zip(image_paths, row_offs, col_offs)
    ]

    rows = {
        "tile_id": tile_ids,
        "image_path": image_paths,
        "row_off": row_offs,
        "col_off": col_offs,
        "patch_size": patch_sizes,
        "embedding": list(embeddings),
        "dino_model": [dino_model] * len(image_paths),
    }
    if mask_paths is not None:
        rows["mask_path"] = mask_paths
    else:
        rows["mask_path"] = [""] * len(image_paths)

    return pd.DataFrame(rows)
