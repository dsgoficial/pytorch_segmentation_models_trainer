# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""

import bisect
import gc
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    _DTYPE_NORMALIZATION,
    _VALID_IMAGE_DTYPES,
    load_augmentation_object,
)


class RasterPatchDataset(Dataset):
    """Dataset de segmentação com janela deslizante determinística sobre imagens raster.

    Varre recursivamente dois diretórios (imagens e máscaras), emparelha os arquivos
    por caminho relativo, e expõe cada patch possível como um item independente.
    O mapeamento de índice global → (imagem, posição da janela) é feito em O(log N)
    via busca binária sobre uma lista de limites cumulativos, sem pré-carregar nenhuma
    imagem em memória — apenas os metadados (altura, largura) são lidos no ``__init__``.

    A leitura efetiva dos pixels é feita sob demanda em ``__getitem__`` usando
    ``rasterio`` windowed read, tornando o dataset adequado para imagens grandes
    (geotiffs de centenas de megabytes).

    **Comportamento para arquivos sem correspondência:**

    - *Imagem sem máscara*: emite ``UserWarning`` e exclui a imagem do dataset.
    - *Máscara sem imagem*: o ``mask_dir`` é varrido independentemente após o loop
      principal; cada máscara sem imagem correspondente em ``image_dir`` emite
      ``UserWarning`` e é ignorada.

    Exemplo de estrutura de diretórios esperada::

        images_root/
            area_a/
                scene_001.tif
                scene_002.tif
            area_b/
                scene_003.tif

        masks_root/
            area_a/
                scene_001.tif   # mesmo caminho relativo
                scene_002.tif
            area_b/
                scene_003.tif

    Args:
        image_dir (Union[str, Path]): Diretório raiz das imagens.
        mask_dir (Union[str, Path]): Diretório raiz das máscaras.
        extension (str): Extensão dos arquivos de imagem (ex: ``".tif"``).
            O ponto inicial é opcional e normalizado internamente.
        patch_size (int): Tamanho do patch em pixels (``patch_size × patch_size``).
        stride (int): Passo da janela deslizante em pixels. Valores menores que
            ``patch_size`` geram patches sobrepostos.
        mask_extension (Optional[str]): Extensão dos arquivos de máscara.
            Se ``None``, usa o mesmo valor de ``extension``.
        augmentation_list: Lista de augmentações do Albumentations (ou ``A.Compose``).
            ``None`` desabilita augmentação. Quando presente, a imagem é passada no
            formato ``(H, W, C)`` e a máscara no formato ``(H, W)``, seguindo a
            convenção do Albumentations.
        data_loader: Configuração do DataLoader (armazenada como atributo;
            consumida pelo ``Model`` do Lightning para instanciar o ``DataLoader``).
        selected_bands (Optional[List[int]]): Índices das bandas a carregar (base 1,
            convenção rasterio). ``None`` carrega todas as bandas disponíveis.
        image_dtype (str): Tipo de dado para leitura da imagem.
            Valores aceitos: ``"uint8"`` (padrão), ``"uint16"``, ``"float32"``,
            ``"native"`` (preserva o dtype original sem cast).
            Quando não há augmentação, ``"uint8"`` e ``"uint16"`` são normalizados
            para ``[0, 1]`` automaticamente (divisão por 255 ou 65535).
            Nota: ao usar ``augmentation_list``, a normalização fica a cargo do
            Albumentations (ex: ``A.Normalize``, ``A.ToFloat``). O ``image_dtype``
            afeta apenas o cast do array antes de entrar no pipeline.
        n_classes (int): Número de classes de segmentação. Quando ``2`` (padrão),
            as máscaras são binarizadas automaticamente: qualquer valor ``> 0``
            vira ``1``. Para segmentação multi-classe, passe o número real de classes
            e os valores de pixel da máscara devem corresponder diretamente aos índices
            de classe.
        reset_augmentation_function (bool): Quando ``True``, faz deepcopy do pipeline
            de augmentação antes de cada chamada e o deleta em seguida, evitando o
            acúmulo de cache interno do Albumentations em treinos longos que pode levar
            a crashes por falta de memória. Ativa coleta de lixo a cada 100 itens.
            ``False`` (padrão) reutiliza o mesmo objeto — mais rápido, recomendado
            para uso normal.

    Raises:
        ValueError: Se ``image_dtype`` não for um dos valores aceitos.
        ValueError: Se ``selected_bands`` contiver valores não positivos.
        ValueError: Se nenhum par imagem/máscara válido for encontrado.

    Example::

        from pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset import (
            RasterPatchDataset,
        )
        from torch.utils.data import DataLoader

        ds = RasterPatchDataset(
            image_dir="/data/images",
            mask_dir="/data/masks",
            extension=".tif",
            patch_size=256,
            stride=128,
        )
        print(f"Total de patches: {len(ds)}")   # patches de todas as imagens

        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
        for batch in loader:
            imgs  = batch["image"]   # (16, C, 256, 256)
            masks = batch["mask"]    # (16, 256, 256)
            break
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        extension: str = ".tif",
        patch_size: int = 256,
        stride: int = 128,
        mask_extension: Optional[str] = None,
        augmentation_list=None,
        data_loader=None,
        selected_bands: Optional[List[int]] = None,
        image_dtype: str = "uint8",
        n_classes: int = 2,
        reset_augmentation_function: bool = False,
    ) -> None:
        super().__init__()

        if image_dtype not in _VALID_IMAGE_DTYPES:
            raise ValueError(
                f"image_dtype '{image_dtype}' inválido. "
                f"Valores aceitos: {sorted(_VALID_IMAGE_DTYPES)}"
            )
        if selected_bands is not None:
            if not all(isinstance(b, int) and b > 0 for b in selected_bands):
                raise ValueError(
                    "selected_bands deve conter apenas inteiros positivos (base 1)"
                )

        self.patch_size = patch_size
        self.stride = stride
        self.image_dtype = image_dtype
        self.selected_bands = selected_bands
        self.n_classes = n_classes
        self.reset_augmentation_function = reset_augmentation_function
        self.data_loader = data_loader

        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        image_extension = self._normalize_extension(extension)
        mask_ext = (
            image_extension
            if mask_extension is None
            else self._normalize_extension(mask_extension)
        )
        self.image_extension = image_extension
        self.mask_extension = mask_ext

        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list)
        )

        # ── Discover image/mask pairs and compute patch counts ────────────────
        image_paths = sorted(image_dir.rglob(f"*{image_extension}"))

        self._image_info: List[Dict[str, Any]] = []
        self._cumulative: List[int] = [0]  # cumulative patch counts per image
        self._total_patches: int = 0
        matched_mask_paths: set = set()

        for img_path in image_paths:
            mask_path = mask_dir / img_path.relative_to(image_dir).with_suffix(mask_ext)
            if not mask_path.exists():
                warnings.warn(
                    f"Máscara não encontrada para {img_path}, imagem ignorada.",
                    stacklevel=2,
                )
                continue

            with rasterio.open(img_path) as src:
                height = src.height
                width = src.width

            patches_per_col = ((height - patch_size) // stride) + 1
            patches_per_row = ((width - patch_size) // stride) + 1

            if patches_per_col <= 0 or patches_per_row <= 0:
                warnings.warn(
                    f"Imagem {img_path} ({height}×{width}) menor que "
                    f"patch_size={patch_size}; ignorada.",
                    stacklevel=2,
                )
                continue

            n_patches = patches_per_row * patches_per_col
            self._image_info.append(
                {
                    "img_path": str(img_path),
                    "mask_path": str(mask_path),
                    "height": height,
                    "width": width,
                    "patches_per_row": patches_per_row,
                    "patches_per_col": patches_per_col,
                }
            )
            matched_mask_paths.add(mask_path)
            self._total_patches += n_patches
            self._cumulative.append(self._total_patches)

        if not self._image_info:
            raise ValueError(
                f"Nenhum par imagem/máscara válido encontrado em "
                f"'{image_dir}' / '{mask_dir}' com extensão '{image_extension}'. "
                "Verifique os diretórios, a extensão e o tamanho de patch."
            )

        # ── Warn about masks that have no corresponding image ─────────────────
        # Scan mask_dir independently so that orphan mask files (misconfigured
        # directories, deleted images, wrong extension) are detected early rather
        # than silently ignored.
        all_mask_paths = set(mask_dir.rglob(f"*{mask_ext}"))
        for orphan in sorted(all_mask_paths - matched_mask_paths):
            warnings.warn(
                f"Máscara {orphan} não possui imagem correspondente em "
                f"'{image_dir}'; ignorada.",
                stacklevel=2,
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def image_info(self) -> List[Dict[str, Any]]:
        """Lista de metadados por imagem (paths, dimensões, contagem de patches)."""
        return self._image_info

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Retorna o número total de patches em todas as imagens."""
        return self._total_patches

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retorna o patch de imagem e máscara correspondente ao índice global.

        Args:
            idx (int): Índice global do patch no dataset.

        Returns:
            Dict com chaves:
                - ``"image"`` (torch.Tensor float32, shape ``(C, H, W)``):
                  patch de imagem normalizado para ``[0, 1]`` quando não há
                  transform e ``image_dtype`` não é ``"native"`` ou ``"float32"``.
                - ``"mask"`` (torch.Tensor int64, shape ``(H, W)``): patch de
                  máscara. Quando ``n_classes == 2`` a máscara é binarizada
                  (``> 0 → 1``). Quando há transform, o tipo depende do pipeline
                  Albumentations configurado.

        Raises:
            IndexError: Se ``idx`` estiver fora do intervalo ``[0, len(self))``.
        """
        if idx < 0 or idx >= self._total_patches:
            raise IndexError(
                f"Índice {idx} fora dos limites do dataset (tamanho={self._total_patches})."
            )

        # 1. Find which image this global idx belongs to (O(log N))
        img_idx = bisect.bisect_right(self._cumulative, idx) - 1
        info = self._image_info[img_idx]

        # 2. Convert global idx to local patch idx within that image
        local_idx = idx - self._cumulative[img_idx]

        # 3. Convert local patch idx to grid (row, col) coordinates
        grid_row = local_idx // info["patches_per_row"]
        grid_col = local_idx % info["patches_per_row"]

        row_off = grid_row * self.stride
        col_off = grid_col * self.stride

        # 4. Windowed read — never loads the full image
        window = Window(col_off, row_off, self.patch_size, self.patch_size)
        try:
            patch_img = self._read_image(info["img_path"], window)
            patch_mask = self._read_mask(info["mask_path"], window)
        except (rasterio.errors.RasterioIOError, Exception) as e:
            logger.error(
                f"Error reading window {window} from {info['img_path']} or {info['mask_path']}: {e}"
            )
            # Try another index
            new_idx = (idx + 1) % self._total_patches
            return self.__getitem__(new_idx)

        # 5. Build output
        if self.transform is None:
            return self._to_tensors_no_transform(patch_img, patch_mask)

        # Reutiliza o mesmo objeto de transform (caminho padrão, mais rápido)
        if not self.reset_augmentation_function:
            return self.transform(image=patch_img, mask=patch_mask)

        # Deepcopy do transform para evitar acúmulo de cache do Albumentations.
        # Ver nota em SegmentationDataset sobre memory leaks em treinos longos.
        transform_func = deepcopy(self.transform)
        output = transform_func(image=patch_img, mask=patch_mask)
        output_dict = {
            "image": (
                output["image"].clone().detach()
                if torch.is_tensor(output["image"])
                else output["image"].copy()
            ),
            "mask": (
                output["mask"].clone().detach()
                if torch.is_tensor(output["mask"])
                else output["mask"].copy()
            ),
        }
        del output
        del transform_func
        if idx % 100 == 0:
            gc.collect()
        return output_dict

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _read_image(self, path: str, window: Window) -> np.ndarray:
        """Lê patch de imagem como array ``(H, W, C)`` para Albumentations.

        Segue a mesma lógica de ``SegmentationDataset._load_image_with_rasterio``:
        transpõe ``(C, H, W) → (H, W, C)``, libera o buffer intermediário e faz
        o cast via ``np.array(..., dtype=)``.
        """
        with rasterio.open(path) as src:
            data = (
                src.read(window=window)
                if self.selected_bands is None
                else src.read(self.selected_bands, window=window)
            )
            image = np.transpose(data, (1, 2, 0)).copy()
            del data
        if self.image_dtype == "native":
            return image
        return np.array(image, dtype=np.dtype(self.image_dtype))

    def _read_mask(self, path: str, window: Window) -> np.ndarray:
        """Lê patch de máscara como array ``(H, W)`` uint8.

        Quando ``n_classes == 2``, aplica binarização ``(> 0)`` para alinhar
        com o comportamento de ``SegmentationDataset`` que passa
        ``is_binary_mask=(n_classes == 2)`` ao carregar máscaras.
        """
        with rasterio.open(path) as src:
            data = src.read(1, window=window)
        mask = data.astype(np.uint8)
        if self.n_classes == 2:
            mask = (mask > 0).astype(np.uint8)
        return mask

    def _to_tensors_no_transform(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Converte arrays numpy para tensores PyTorch sem augmentação.

        A imagem é convertida para float32 e normalizada para ``[0, 1]`` quando
        ``image_dtype`` é ``"uint8"`` ou ``"uint16"``. A máscara é mantida
        como int64.
        """
        tensor_img = torch.from_numpy(image).float()
        tensor_img = tensor_img.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        norm_factor = _DTYPE_NORMALIZATION.get(self.image_dtype)
        if norm_factor is not None:
            tensor_img = tensor_img / norm_factor
        tensor_mask = torch.from_numpy(mask).long()
        return {"image": tensor_img, "mask": tensor_mask}

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"images={len(self._image_info)}, "
            f"patches={self._total_patches}, "
            f"patch_size={self.patch_size}, "
            f"stride={self.stride})"
        )

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_extension(ext: str) -> str:
        """Garante que a extensão comece com ponto (ex: ``'tif'`` → ``'.tif'``)."""
        return ext if ext.startswith(".") else f".{ext}"
