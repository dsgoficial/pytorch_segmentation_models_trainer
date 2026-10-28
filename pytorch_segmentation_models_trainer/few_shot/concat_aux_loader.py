# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-26
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
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

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch


class ConcatAuxLoader:
    """Concatenates 2+ sub-loaders' embeddings along the channel dim, for
    a method that wants a SINGLE fused input instead of a separate gate
    per product (e.g. ``q: linear_aef_gate`` with ``aef_dim`` set to the
    sum of the sub-loaders' own channel counts) -- early fusion, as
    opposed to ``GateEnsemble``'s late fusion (2 separate gates, each with
    its own head, averaged after their own ``q``).

    Motivated by the companion doctoral project's "fusão única"
    ablation (2026-10-26): isolates whether combining 2 embedding
    products helps because of the mixture-of-experts structure (2
    independently-fit gates) or just because more information reaches a
    single gate -- without this, "AEF+OlmoEarth average beats either
    alone" is also consistent with the weaker claim "more channels help
    a single gate", not necessarily evidence for the MoE architecture
    itself.

    Same ``load_batch(mask_paths, device, size)`` interface as
    :class:`~pytorch_segmentation_models_trainer.few_shot.aef_cache.AefCacheLoader`,
    so it is a drop-in value for any single ``gfss.aux_cache_loaders``
    entry a method's ``aux_names`` declares -- pure I/O/concatenation,
    no knowledge of how the method uses the fused tensor.

    Args:
        loaders: 2+ sub-loaders (already-constructed objects, or Hydra
            sub-configs with their own ``_target_`` -- resolved by hand
            here, same pattern as ``HiSplit._resolve_q_strategy``, since
            a caller instantiating this with ``_recursive_=False`` passes
            nested ``_target_`` blocks un-instantiated).

    Example YAML::

        gfss:
          aux_cache_loaders:
            aef:
              _target_: pytorch_segmentation_models_trainer.few_shot.concat_aux_loader.ConcatAuxLoader
              loaders:
                - /path/to/aef_cache          # bare string -> AefCacheLoader
                - _target_: pytorch_segmentation_models_trainer.few_shot.olmoearth_cache.OlmoEarthCacheLoader
                  cache_dir: /path/to/olmoearth_cache
                  n_bands: 128
        gfss:
          method:
            aef_dim: 192   # 64 (AEF) + 128 (OlmoEarth)
    """

    def __init__(self, loaders: Sequence[Any]) -> None:
        if len(loaders) < 2:
            raise ValueError(
                f"ConcatAuxLoader needs at least 2 sub-loaders, got {len(loaders)}."
            )
        self.loaders = [self._resolve_loader(l) for l in loaders]

    @staticmethod
    def _resolve_loader(loader: Any) -> Any:
        """An already-built loader (plain Python, has ``load_batch``), a
        bare cache-directory string (the default ``AefCacheLoader``, same
        convention as ``GFSSModel.__init__``'s own ``aux_cache_loaders``
        resolution), or a Hydra sub-config with its own ``_target_``."""
        if hasattr(loader, "load_batch"):
            return loader
        if isinstance(loader, str):
            from pytorch_segmentation_models_trainer.few_shot.aef_cache import (
                AefCacheLoader,
            )

            return AefCacheLoader(loader)
        from hydra.utils import instantiate

        return instantiate(loader, _recursive_=False)

    def load_batch(
        self,
        mask_paths: List[str],
        device: torch.device,
        size: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """``(B, Σn_bands, *size)`` tensor on ``device`` -- each
        sub-loader's own channels, concatenated in ``loaders`` order --
        or ``None`` if ``mask_paths`` is empty (propagates whatever the
        first sub-loader does for that case, same convention as every
        other loader in this module)."""
        if not mask_paths:
            return None
        parts = [loader.load_batch(mask_paths, device, size) for loader in self.loaders]
        return torch.cat(parts, dim=1)
