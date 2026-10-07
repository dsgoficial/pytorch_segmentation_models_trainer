# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-07
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

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
    GFSSCheckpointConfig,
)
from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.metrics import GFSSMetrics
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils.checkpoint_loading import (
    load_pretrained_weights,
    resolve_checkpoint_from_runner,
)

logger = logging.getLogger(__name__)


class GFSSModel(Model):
    """LightningModule for generalized few-shot semantic segmentation.

    Builds the base model from ``cfg.model``, loads it from
    ``cfg.gfss.base_checkpoint`` (strict), freezes it as a
    :class:`FrozenLinearHeadSegmenter` and delegates the adaptation to the
    :class:`BaseGFSSMethod` in ``cfg.gfss.method``, which works on decoder
    features. Runs through the regular ``train()`` entry point (and hence
    the ExperimentsRunner):

    * ``train_dataset`` is the **support** set and ``test_dataset`` the
      **query** set; ``val_dataset`` is optional;
    * ``trainer.fit`` = adaptation: the method is initialised from the
      whole support set (``on_fit_start``), then ``support_loss`` is
      minimised for ``pl_trainer.max_steps`` steps (methods without a loss
      need ``max_steps: 0``);
    * transductive methods are adapted again on every test batch inside
      ``test_step`` (requires ``pl_trainer.inference_mode: false``);
    * metrics: :class:`GFSSMetrics` logged as ``test/...`` / ``val/...``.

    Do not add a ``ModelCheckpoint`` callback: the adapted state lives in
    memory and ``train()`` would reload it from a checkpoint before testing.

    Example YAML::

        pl_model:
          _target_: pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel
        model:                       # base model, same as in its training
          _target_: segmentation_models_pytorch.UPerNet
          encoder_name: resnet50
          encoder_weights: null
          classes: 5
        gfss:
          hierarchy: {3: [3, 5]}
          base_checkpoint:
            from_runner: outputs/baselines/r2_pampa   # run with the same seed
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly
        pl_trainer:
          max_steps: 0
          enable_checkpointing: false
    """

    def __init__(self, cfg, inference_mode: bool = False) -> None:
        super().__init__(cfg, inference_mode=inference_mode)
        gfss = cfg.gfss
        names = gfss.get("class_names", None)
        if (
            names is None
            and "class_definitions" in cfg
            and "names" in cfg.class_definitions
        ):
            names = list(cfg.class_definitions.names)
        ignore = gfss.get("ignore_index", 255)
        self.val_gfss_metrics = GFSSMetrics(
            self.hierarchy, class_names=names, ignore_index=ignore, prefix="val/"
        )
        self.test_gfss_metrics = GFSSMetrics(
            self.hierarchy, class_names=names, ignore_index=ignore, prefix="test/"
        )
        self.register_buffer("_gfss_initialised", torch.tensor(False))
        self._support: Optional[Tuple[Tensor, Tensor]] = None

    # ------------------------------------------------------------------
    # Model / method construction
    # ------------------------------------------------------------------

    def _resolve_checkpoint(self) -> GFSSCheckpointConfig:
        node = self.cfg.gfss.get("base_checkpoint", None)
        ckpt = OmegaConf.merge(
            OmegaConf.structured(GFSSCheckpointConfig),
            OmegaConf.to_container(node, resolve=True) if node is not None else {},
        )
        if bool(ckpt.path) == bool(ckpt.from_runner):
            raise ValueError(
                "gfss.base_checkpoint needs exactly one of 'path' and 'from_runner'."
            )
        if ckpt.from_runner:
            seed = self.cfg.get("seed", None)
            if seed is None:
                raise ValueError(
                    "gfss.base_checkpoint.from_runner needs 'seed' in the config "
                    "(injected by the ExperimentsRunner)."
                )
            ckpt.path = resolve_checkpoint_from_runner(ckpt.from_runner, int(seed))
        return ckpt

    def get_model(self) -> FrozenLinearHeadSegmenter:
        """Base model + checkpoint, frozen; also builds hierarchy and method."""
        base = instantiate(self.cfg.model, _recursive_=False)
        ckpt = self._resolve_checkpoint()
        load_pretrained_weights(
            base, ckpt.path, ckpt.source_format, strict_loading=ckpt.strict_loading
        )
        segmenter = FrozenLinearHeadSegmenter(base)
        weight, bias = segmenter.weight, segmenter.bias
        self.hierarchy = ClassHierarchy(
            self.cfg.gfss.hierarchy, num_base_classes=weight.shape[0]
        )
        self.method = instantiate(self.cfg.gfss.method, _recursive_=False)
        self.method.setup(
            self.hierarchy,
            weight,
            bias,
            not_novel_index=self.cfg.gfss.get("not_novel_index", 254),
        )
        logger.info(
            "GFSSModel: base %s from %s, %s, method %s",
            type(base).__name__,
            ckpt.path,
            self.hierarchy,
            type(self.method).__name__,
        )
        return segmenter

    def get_loss_function(self):
        """GFSS losses live in the method (``support_loss``)."""
        return None

    def get_optimizer(self):
        params = [p for p in self.method.parameters() if p.requires_grad]
        return instantiate(self.cfg.optimizer, params=params, _recursive_=False)

    def configure_optimizers(self):
        """No optimizer when the method has no trainable parameters."""
        if not any(p.requires_grad for p in self.method.parameters()):
            return None
        return super().configure_optimizers()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _features_and_masks(
        self, images: Tensor, masks: Tensor
    ) -> Tuple[Tensor, Tensor]:
        feats = self.model.features(images)
        small = F.interpolate(
            masks.unsqueeze(1).float(), size=feats.shape[-2:], mode="nearest"
        )
        return feats, small.squeeze(1).long()

    def _get_support(self) -> Tuple[Tensor, Tensor]:
        """Features and downsampled masks of the whole support set (cached)."""
        if self._support is None:
            loader = DataLoader(
                self.train_ds, batch_size=self.cfg.hyperparameters.batch_size
            )
            feats, masks = [], []
            for batch in loader:
                images, m = self._unpack_batch(batch)
                f, sm = self._features_and_masks(
                    images.to(self.device), m.to(self.device)
                )
                feats.append(f)
                masks.append(sm)
            self._support = (torch.cat(feats), torch.cat(masks))
        return self._support

    def _ensure_initialised(self) -> None:
        if not bool(self._gfss_initialised):
            self.method.init_from_support(*self._get_support())
            self._gfss_initialised.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        """Logits over all classes at the input resolution."""
        return self.model.upsample(self.method(self.model.features(x)), x.shape[-2:])

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        self._ensure_initialised()

    def training_step(self, batch, batch_idx):
        images, masks = self._unpack_batch(batch)
        feats, small = self._features_and_masks(images, masks)
        loss = self.method.support_loss(feats, small)
        if loss is None:
            return None
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch, metrics: GFSSMetrics) -> None:
        self._ensure_initialised()
        images, masks = self._unpack_batch(batch)
        feats = self.model.features(images)
        if self.method.transductive:
            with torch.enable_grad():
                self.method.adapt_to_query(*self._get_support(), feats)
        logits = self.model.upsample(self.method(feats), images.shape[-2:])
        base_pred = self.model.base_logits(images).argmax(1)
        metrics.update(logits.argmax(1), masks.long(), base_pred)

    def val_dataloader(self):
        """Validation is optional in GFSS (no ``val_dataset`` -> no loop)."""
        return [] if self.val_ds is None else super().val_dataloader()

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, self.val_gfss_metrics)

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, self.test_gfss_metrics)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_gfss_metrics.compute())
        self.val_gfss_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_gfss_metrics.compute())
        self.test_gfss_metrics.reset()
