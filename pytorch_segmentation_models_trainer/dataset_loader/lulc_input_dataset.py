# -*- coding: utf-8 -*-
"""Datasets for multi-source LULC concatenated with RGB as model input."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import albumentations as A
import numpy as np
import rasterio
import rasterio.windows
import torch

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AbstractDataset,
)
from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset import (
    MBTilesMaskWindowedDataset,
)
from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
    read_mask_window,
    read_source_aligned_to_mask_window,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One-hot encoding helper
# ---------------------------------------------------------------------------


def _one_hot_chw(
    arr: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """One-hot encode a (H, W) integer class array to (num_classes, H, W) float32.

    Pixels where ``arr == ignore_index`` or ``arr >= num_classes`` or
    ``arr < 0`` produce all-zero vectors (neutral — no class signal).

    Args:
        arr: (H, W) integer array of class indices.
        num_classes: Number of classes C.
        ignore_index: Value treated as "no valid class". Default 255.

    Returns:
        (C, H, W) float32 array with values in {0.0, 1.0}.
    """
    valid = (arr != ignore_index) & (arr >= 0) & (arr < num_classes)
    clipped = np.where(valid, arr, 0).astype(np.int32)
    one_hot = np.eye(num_classes, dtype=np.float32)[clipped]  # (H, W, C)
    one_hot[~valid] = 0.0
    return np.ascontiguousarray(one_hot.transpose(2, 0, 1))  # (C, H, W)


# ---------------------------------------------------------------------------
# LulcInputDataset — tile-based
# ---------------------------------------------------------------------------


class LulcInputDataset(AbstractDataset):
    """Dataset that concatenates RGB with one-hot encoded LULC sources.

    Reads an RGB image and an arbitrary number of single-band LULC
    classification rasters.  Each LULC raster is one-hot encoded and
    concatenated with the RGB channels, producing a
    ``(3 + N * num_classes, H, W)`` input tensor.

    The training target is a hard integer mask (e.g. BAGS cartographic mask).

    The number of LULC sources is fully dynamic: pass any list of CSV column
    names via ``lulc_keys``.  Adding a new source requires only a new column
    in the CSV and a new entry in ``lulc_keys``.

    Geometric augmentations (flip, rotation, …) are applied consistently to
    the image, the training mask, and all LULC rasters via albumentations
    ``additional_targets``.  ``Normalize`` in the pipeline affects only the
    ``image`` target (albumentations never normalises ``mask`` targets), so
    the one-hot channels remain in ``{0, 1}`` after augmentation.

    Args:
        input_csv_path: Path to the CSV.  Must have at least ``image_key``
            and ``mask_key`` columns, plus one column per entry in
            ``lulc_keys``.
        df: Pre-built DataFrame (alternative to ``input_csv_path``).
        lulc_keys: List of CSV column names for LULC raster paths.  Each
            raster must be single-band with integer class indices in
            ``[0, num_classes - 1]``.  Order determines the channel order
            in the output tensor.
        num_classes: Number of LULC classes (same for all sources).
            Default 6 (EDGV 3.0 six-class legend used in experiments E0–E5).
        ignore_lulc_index: Value in LULC rasters meaning "no valid class".
            These pixels produce all-zero one-hot vectors.  Default 255.
        image_key: CSV column for RGB image paths. Default ``"image_path"``.
        mask_key: CSV column for the training target mask. Default ``"mask_path"``.
        root_dir: Root directory prepended to relative CSV paths.
        augmentation_list: Albumentations transform list.  Include geometric
            transforms (HFlip, VFlip, Rotate90) and Normalize / ToTensorV2
            as usual.  Normalize is applied only to the ``image`` target.
        data_loader: DataLoader settings stored on the object.
        n_first_rows_to_read: Optional row limit for CSV reading.
        seed: Optional seed for the augmentation pipeline.
        **kwargs: Reserved for Hydra compatibility.

    Returns:
        Dict with keys:

        - ``"image"``: ``(3 + N * num_classes, H, W)`` float32 tensor —
          RGB channels (normalised if Normalize is in the pipeline) followed
          by the one-hot LULC channels in ``lulc_keys`` order.
        - ``"mask"``: ``(H, W)`` int64 tensor — hard integer class labels.
        - ``"path"``: image file path string.

    Example YAML:

    .. code-block:: yaml

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset.LulcInputDataset
          input_csv_path: /data/splits/train.csv
          image_key: image_path
          mask_key: mask_path
          lulc_keys:
            - mapbiomas_path
            - esri_path
            - dw_path
          num_classes: 6
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.VerticalFlip
              p: 0.5
            - _target_: albumentations.RandomRotate90
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2
          data_loader:
            batch_size: 16
            num_workers: 8
            shuffle: true
            pin_memory: true
    """

    def __init__(
        self,
        input_csv_path: Optional[Union[str, Path]] = None,
        df=None,
        lulc_keys: Optional[List[str]] = None,
        num_classes: int = 6,
        ignore_lulc_index: int = 255,
        image_key: str = "image_path",
        mask_key: str = "mask_path",
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        n_first_rows_to_read: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
            seed=seed,
        )
        self.lulc_keys: List[str] = list(lulc_keys) if lulc_keys else []
        self.num_classes = num_classes
        self.ignore_lulc_index = ignore_lulc_index

        missing = [k for k in self.lulc_keys if k not in self.df.columns]
        if missing:
            raise ValueError(
                f"LulcInputDataset: columns not found in CSV: {missing}. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Rebuild Compose with additional_targets so every geometric transform
        # is applied identically to the image, the training mask, and all LULC
        # rasters.
        if self.transform is not None and self.lulc_keys:
            self.transform = A.Compose(
                self.transform.transforms,
                additional_targets={key: "mask" for key in self.lulc_keys},
            )

    # ------------------------------------------------------------------
    # Loading helpers (overridden by windowed subclass)
    # ------------------------------------------------------------------

    def _load_image_np(self, idx: int, **_) -> np.ndarray:
        """Load the RGB image as (H, W, 3) uint8."""
        return self.load_image(idx, key=self.image_key)

    def _load_mask_np(self, idx: int, **_) -> np.ndarray:
        """Load the training target as (H, W) uint8."""
        return self.load_image(
            idx, key=self.mask_key, is_mask=True, is_binary_mask=False
        )

    def _load_lulc_np(self, idx: int, key: str, **_) -> np.ndarray:
        """Load a single-band LULC raster as (H, W) uint8."""
        with rasterio.open(self.get_path(idx, key=key)) as src:
            return src.read(1).astype(np.uint8)

    # ------------------------------------------------------------------
    # Conversion helpers (shared with windowed subclass)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_chw_float(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert (H, W, 3) image to (3, H, W) float32 in [0, 1]."""
        if isinstance(x, torch.Tensor):
            if x.ndim == 3 and x.shape[0] in (1, 3):
                return x.float()  # ToTensorV2 already produced CHW
            return x.permute(2, 0, 1).float().contiguous()
        arr = np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)
        t = torch.from_numpy(arr)
        if t.max() > 1.0 + 1e-3:
            t = t / 255.0
        return t

    @staticmethod
    def _mask_to_hw_long(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert (H, W) mask to int64 tensor."""
        if isinstance(x, torch.Tensor):
            return x.long()
        return torch.from_numpy(np.ascontiguousarray(x)).long()

    @staticmethod
    def _lulc_to_hw_uint8(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Ensure a LULC target (post-augmentation) is a (H, W) numpy uint8."""
        if isinstance(x, torch.Tensor):
            return x.numpy().astype(np.uint8)
        return np.ascontiguousarray(x).astype(np.uint8)

    # ------------------------------------------------------------------
    # Core sample assembly (shared with windowed subclass via _load_* hooks)
    # ------------------------------------------------------------------

    def _assemble_sample(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        lulc_arrays: Dict[str, np.ndarray],
        idx: int,
    ) -> Dict[str, Any]:
        """Apply augmentations, one-hot encode LULC sources, and build the sample dict.

        Args:
            image_np: (H, W, 3) uint8 RGB array.
            mask_np: (H, W) uint8 training target array.
            lulc_arrays: Dict mapping each lulc_key to its (H, W) uint8 array.
            idx: Sample index (used only to retrieve the image path string).

        Returns:
            Dict with ``"image"``, ``"mask"``, and ``"path"`` keys.
        """
        if self.transform is not None:
            aug_kwargs: Dict[str, Any] = {
                "image": image_np,
                "mask": mask_np,
                **lulc_arrays,
            }
            result = self.transform(**aug_kwargs)
            image_out = result["image"]
            mask_out = result["mask"]
            lulc_out = {key: result[key] for key in self.lulc_keys}
        else:
            image_out = image_np
            mask_out = mask_np
            lulc_out = lulc_arrays

        image_t: torch.Tensor = self._image_to_chw_float(image_out)

        lulc_tensors: List[torch.Tensor] = [
            torch.from_numpy(
                _one_hot_chw(
                    self._lulc_to_hw_uint8(lulc_out[key]),
                    self.num_classes,
                    self.ignore_lulc_index,
                )
            )
            for key in self.lulc_keys
        ]

        combined = (
            torch.cat([image_t, *lulc_tensors], dim=0) if lulc_tensors else image_t
        )

        return {
            "image": combined,
            "mask": self._mask_to_hw_long(mask_out),
            "path": self.get_path(idx, key=self.image_key),
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len
        image_np = self._load_image_np(idx)
        mask_np = self._load_mask_np(idx)
        lulc_arrays = {key: self._load_lulc_np(idx, key=key) for key in self.lulc_keys}
        return self._assemble_sample(image_np, mask_np, lulc_arrays, idx)


# ---------------------------------------------------------------------------
# LulcInputWindowedDataset — on-the-fly windowed reads from full-scene rasters
# ---------------------------------------------------------------------------


class LulcInputWindowedDataset(LulcInputDataset):
    """LulcInputDataset variant that reads fixed-size patches from full-scene rasters.

    Each CSV row defines one patch via pixel offsets into the full-scene
    rasters. Tiles are never pre-cut: the entire scene files (image, mask,
    LULC sources) are stored once and patches are extracted on-the-fly with
    ``rasterio.windows.Window``.  The same raster path may appear in multiple
    rows (one row per patch).

    The ``lulc_keys``, ``num_classes``, ``ignore_lulc_index``, augmentation
    pipeline, and output format are identical to ``LulcInputDataset``.

    Additional CSV columns required:

    - ``row_off_key`` — top-left row offset of the patch (pixels).
    - ``col_off_key`` — top-left column offset of the patch (pixels).
    - ``patch_size_key`` — patch height = width (pixels, square patch).

    Args:
        row_off_key: CSV column for patch row offset. Default ``"row_off"``.
        col_off_key: CSV column for patch column offset. Default ``"col_off"``.
        patch_size_key: CSV column for patch size. Default ``"patch_size"``.
        All other args are forwarded to ``LulcInputDataset``.

    Example YAML:

    .. code-block:: yaml

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset.LulcInputWindowedDataset
          input_csv_path: /data/splits/train.csv
          image_key: image_path
          mask_key: mask_path
          lulc_keys:
            - mapbiomas_path
            - esri_path
            - dw_path
          num_classes: 6
          row_off_key: row_off
          col_off_key: col_off
          patch_size_key: patch_size
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.VerticalFlip
              p: 0.5
            - _target_: albumentations.RandomRotate90
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2
          data_loader:
            batch_size: 8
            num_workers: 8
            shuffle: true
            pin_memory: true
    """

    def __init__(
        self,
        input_csv_path: Optional[Union[str, Path]] = None,
        df=None,
        lulc_keys: Optional[List[str]] = None,
        num_classes: int = 6,
        ignore_lulc_index: int = 255,
        image_key: str = "image_path",
        mask_key: str = "mask_path",
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        n_first_rows_to_read: Optional[int] = None,
        seed: Optional[int] = None,
        row_off_key: str = "row_off",
        col_off_key: str = "col_off",
        patch_size_key: str = "patch_size",
        **kwargs,
    ) -> None:
        super().__init__(
            input_csv_path=input_csv_path,
            df=df,
            lulc_keys=lulc_keys,
            num_classes=num_classes,
            ignore_lulc_index=ignore_lulc_index,
            image_key=image_key,
            mask_key=mask_key,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            n_first_rows_to_read=n_first_rows_to_read,
            seed=seed,
            **kwargs,
        )
        self.row_off_key = row_off_key
        self.col_off_key = col_off_key
        self.patch_size_key = patch_size_key

        for col in [row_off_key, col_off_key, patch_size_key]:
            if col not in self.df.columns:
                raise ValueError(
                    f"LulcInputWindowedDataset: column '{col}' not found in CSV. "
                    f"Available columns: {list(self.df.columns)}"
                )

    # ------------------------------------------------------------------
    # Windowed load helpers (override parent tile-based reads)
    # ------------------------------------------------------------------

    def _window(self, idx: int) -> rasterio.windows.Window:
        row = self.df.iloc[self._index_map[idx]]
        patch = int(row[self.patch_size_key])
        return rasterio.windows.Window(
            col_off=int(row[self.col_off_key]),
            row_off=int(row[self.row_off_key]),
            width=patch,
            height=patch,
        )

    def _load_image_np(self, idx: int, window: rasterio.windows.Window = None, **_) -> np.ndarray:  # type: ignore[override]
        """Load the RGB image patch as (H, W, 3) uint8."""
        if window is None:
            window = self._window(idx)
        with rasterio.open(self.get_path(idx, key=self.image_key)) as src:
            data = src.read(window=window)  # (C, H, W)
        return np.transpose(data, (1, 2, 0)).copy()

    def _load_mask_np(self, idx: int, window: rasterio.windows.Window = None, **_) -> np.ndarray:  # type: ignore[override]
        """Load the training target patch as (H, W) uint8."""
        if window is None:
            window = self._window(idx)
        with rasterio.open(self.get_path(idx, key=self.mask_key)) as src:
            return src.read(1, window=window).astype(np.uint8)

    def _load_lulc_np(self, idx: int, key: str, window: rasterio.windows.Window = None, **_) -> np.ndarray:  # type: ignore[override]
        """Load a single LULC source patch as (H, W) uint8."""
        if window is None:
            window = self._window(idx)
        with rasterio.open(self.get_path(idx, key=key)) as src:
            return src.read(1, window=window).astype(np.uint8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len
        window = self._window(idx)
        image_np = self._load_image_np(idx, window=window)
        mask_np = self._load_mask_np(idx, window=window)
        lulc_arrays = {
            key: self._load_lulc_np(idx, key=key, window=window)
            for key in self.lulc_keys
        }
        return self._assemble_sample(image_np, mask_np, lulc_arrays, idx)


class MBTilesLulcInputMaskWindowedDataset(MBTilesMaskWindowedDataset):
    """MBTiles RGB plus one-hot LULC sources on mask-referenced windows.

    This dataset extends :class:`MBTilesMaskWindowedDataset` for experiments
    where RGB must be read directly from an MBTiles source while external LULC
    rasters are concatenated as one-hot input channels. Each sample window is
    still defined by the cartographic mask grid, so RGB, LULC, and target mask
    are spatially aligned before augmentation.

    Args:
        mbtiles_path: RGB source MBTiles or rasterio-readable raster.
        lulc_paths: Single-band LULC rasters/VRTs with class ids.
        mask_paths: Explicit mask GeoTIFF paths.
        mask_dir: Directory scanned recursively when ``mask_paths`` is omitted.
        window_index_cache: Optional CSV/Parquet with selected windows.
        patch_size: Square patch size used when building a window index.
        num_classes: Number of classes used for LULC one-hot encoding and the
            hard target mask.
        ignore_lulc_index: Value treated as invalid in LULC rasters.
        lulc_resampling: Rasterio resampling method for LULC sources. Keep this
            as ``"nearest"`` for categorical rasters.
        **kwargs: Other arguments accepted by ``MBTilesMaskWindowedDataset``.

    Returns:
        Dict with ``image`` tensor ``(3 + len(lulc_paths) * num_classes, H, W)``
        and ``mask`` tensor ``(H, W)``.

    Example YAML:
        ```yaml
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset.MBTilesLulcInputMaskWindowedDataset
          mbtiles_path: /data/tiles.mbtiles
          mask_dir: /data/masks
          window_index_cache: /data/coreset.csv
          lulc_paths: [/data/mapbiomas.vrt, /data/esri.vrt, /data/dw.vrt]
          num_classes: 6
        ```
    """

    def __init__(
        self,
        mbtiles_path,
        lulc_paths: Sequence[Union[str, Path]],
        mask_paths: Optional[Sequence[str]] = None,
        mask_dir: Optional[str] = None,
        mask_extension: str = ".tif",
        patch_size: int = 512,
        stride: Optional[int] = None,
        selected_bands: Optional[Sequence[int]] = None,
        image_dtype: str = "uint8",
        image_resampling: str = "bilinear",
        num_classes: int = 6,
        ignore_lulc_index: int = 255,
        lulc_resampling: str = "nearest",
        augmentation_list=None,
        data_loader=None,
        return_metadata: bool = True,
        window_index_cache: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            mbtiles_path=mbtiles_path,
            mask_paths=mask_paths,
            mask_dir=mask_dir,
            mask_extension=mask_extension,
            patch_size=patch_size,
            stride=stride,
            selected_bands=selected_bands,
            image_dtype=image_dtype,
            image_resampling=image_resampling,
            n_classes=num_classes,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            return_metadata=return_metadata,
            window_index_cache=window_index_cache,
            **kwargs,
        )
        self.lulc_paths = [Path(p) for p in lulc_paths]
        self.num_classes = num_classes
        self.ignore_lulc_index = ignore_lulc_index
        self.lulc_resampling = lulc_resampling

        if self.transform is not None and self.lulc_paths:
            self.transform = A.Compose(
                self.transform.transforms,
                additional_targets={
                    f"lulc_{idx}": "mask" for idx in range(len(self.lulc_paths))
                },
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one MBTiles RGB + LULC feature patch.

        Args:
            idx: Dataset item index.

        Returns:
            Dict with concatenated image tensor and hard-label mask tensor.

        Raises:
            IndexError: If *idx* is outside the dataset range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for size {len(self)}.")

        record = self.windows[idx]
        mask_path = record["mask_path"]
        window = rasterio.windows.Window(
            record["col_off"],
            record["row_off"],
            record["width"],
            record["height"],
        )

        with rasterio.open(mask_path) as mask_src:
            image_chw = read_source_aligned_to_mask_window(
                source_path=self.mbtiles_path,
                mask_src=mask_src,
                window=window,
                selected_bands=self.selected_bands,
                image_dtype=self.image_dtype,
                image_resampling=self.image_resampling,
            )
            mask = read_mask_window(mask_src, window, n_classes=self.n_classes)
            lulc_arrays = [
                read_source_aligned_to_mask_window(
                    source_path=path,
                    mask_src=mask_src,
                    window=window,
                    selected_bands=[1],
                    image_dtype="uint8",
                    image_resampling=self.lulc_resampling,
                )[0]
                for path in self.lulc_paths
            ]

        image_hwc = np.ascontiguousarray(image_chw.transpose(1, 2, 0))
        lulc_targets = {f"lulc_{i}": arr for i, arr in enumerate(lulc_arrays)}

        if self.transform is not None:
            result = self.transform(image=image_hwc, mask=mask, **lulc_targets)
            image_out = result["image"]
            mask_out = result["mask"]
            lulc_out = {key: result[key] for key in lulc_targets}
        else:
            image_out = image_hwc
            mask_out = mask
            lulc_out = lulc_targets

        image_t = LulcInputDataset._image_to_chw_float(image_out)
        lulc_tensors = [
            torch.from_numpy(
                _one_hot_chw(
                    LulcInputDataset._lulc_to_hw_uint8(lulc_out[key]),
                    self.num_classes,
                    self.ignore_lulc_index,
                )
            )
            for key in sorted(lulc_out)
        ]
        combined = torch.cat([image_t, *lulc_tensors], dim=0)

        sample: Dict[str, Any] = {
            "image": combined,
            "mask": LulcInputDataset._mask_to_hw_long(mask_out),
            "path": str(mask_path),
        }
        if self.return_metadata:
            sample["metadata"] = {
                "mask_path": str(mask_path),
                "row_off": int(record["row_off"]),
                "col_off": int(record["col_off"]),
            }
        return sample
