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

import copy
import hashlib
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
    GFSSCheckpointConfig,
)
from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.metrics import GFSSMetrics
from pytorch_segmentation_models_trainer.few_shot.uncertainty import (
    GFSSUncertaintyMetrics,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.tools.tta.tta import ROT0, apply_tta
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
    * metrics: :class:`GFSSMetrics` logged as ``test/...`` / ``val/...``
      (``gfss.boundary_width > 0`` adds the boundary-band IoU,
      ``test/boundary/...``).

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
        bw = int(gfss.get("boundary_width", 0) or 0)
        self.val_gfss_metrics = GFSSMetrics(
            self.hierarchy,
            class_names=names,
            ignore_index=ignore,
            prefix="val/",
            boundary_width=bw,
        )
        self.test_gfss_metrics = GFSSMetrics(
            self.hierarchy,
            class_names=names,
            ignore_index=ignore,
            prefix="test/",
            boundary_width=bw,
        )
        names_u = self.method.uncertainty_names()
        unc_eval = dict(gfss.get("uncertainty_eval", None) or {})
        self.val_unc_metrics = (
            GFSSUncertaintyMetrics(names_u, prefix="val/unc/", **unc_eval)
            if names_u
            else None
        )
        self.test_unc_metrics = (
            GFSSUncertaintyMetrics(names_u, prefix="test/unc/", **unc_eval)
            if names_u
            else None
        )
        # ModuleList + keys: variant names may contain "." (e.g. widen_0.1),
        # which ModuleDict rejects.
        self._variant_keys = [
            (stage, v) for stage in ("val", "test") for v in self.method.variant_names()
        ]
        self.variant_metrics = torch.nn.ModuleList(
            GFSSMetrics(
                self.hierarchy,
                class_names=names,
                ignore_index=ignore,
                prefix=f"{stage}/var/{v}/",
                boundary_width=bw,
            )
            for stage, v in self._variant_keys
        )
        self.register_buffer("_gfss_initialised", torch.tensor(False))
        self._support: Optional[Tuple[Tensor, Tensor]] = None
        self._test_feat_cache: Dict[Any, Tensor] = {}
        self._test_feat_cache_path: Optional[str] = None
        self._test_feat_cache_dirty: bool = False

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
        self._base_checkpoint_path = ckpt.path
        load_pretrained_weights(
            base, ckpt.path, ckpt.source_format, strict_loading=ckpt.strict_loading
        )
        segmenter = FrozenLinearHeadSegmenter(base)
        weight, bias = segmenter.weight, segmenter.bias
        backbone = dict(self.cfg.gfss.get("backbone", None) or {})
        trainable = backbone.get("trainable", "none")
        # Fine-tuning baselines: a frozen copy of the base model is the
        # reference for the KD snapshot and for base predictions (locality).
        self.base_reference = None
        if trainable != "none":
            self.base_reference = FrozenLinearHeadSegmenter(copy.deepcopy(base))
            segmenter.enable_training(trainable, backbone.get("lora", None))
        self.hierarchy = ClassHierarchy(
            self.cfg.gfss.hierarchy, num_base_classes=weight.shape[0]
        )
        self.method = instantiate(self.cfg.gfss.method, _recursive_=False)
        self.method.setup(
            self.hierarchy,
            weight,
            bias,
            not_novel_index=self.cfg.gfss.get("not_novel_index", 254),
            base_output="evidential" if segmenter.evidential else "softmax",
        )
        logger.info(
            "GFSSModel: base %s from %s, %s, method %s",
            type(base).__name__,
            ckpt.path,
            self.hierarchy,
            type(self.method).__name__,
        )
        return segmenter

    def _reference(self) -> FrozenLinearHeadSegmenter:
        """The unmodified base model (frozen copy when the backbone trains)."""
        return self.base_reference if self.base_reference is not None else self.model

    def get_loss_function(self):
        """GFSS losses live in the method (``support_loss``)."""
        return None

    def _trainable_parameters(self):
        return [
            p
            for module in (self.method, self.model)
            for p in module.parameters()
            if p.requires_grad
        ]

    def get_optimizer(self):
        """Optimizer over the method's parameters and, for the fine-tuning
        baselines, the unfrozen backbone parameters."""
        return instantiate(
            self.cfg.optimizer, params=self._trainable_parameters(), _recursive_=False
        )

    def configure_optimizers(self):
        """No optimizer when nothing is trainable."""
        if not self._trainable_parameters():
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

    # ------------------------------------------------------------------
    # Eval-time backbone feature cache
    # ------------------------------------------------------------------
    #
    # Every run of a K-sweep (shots x draws x seeds) that shares a seed and
    # biome evaluates the *same* frozen backbone on the *same* val/test
    # tiles under the *same* TTA augmentations -- only the method's head
    # differs. _test_feat_cache (in-memory, keyed by (stage, batch_idx,
    # aug_name)) avoids recomputing a batch's features twice within one
    # epoch (e.g. the identity pass done directly in _eval_step and again
    # as TTA's "rot0" iteration). When `gfss.test_feature_cache_dir` is
    # set, the same dict is also persisted to/loaded from disk, so later
    # runs of other K/draw/seed combinations that hash to the same
    # (checkpoint, dataset, augmentations, batch_size) key skip the
    # backbone pass entirely instead of just deduplicating in-process --
    # each run is its own subprocess (see ExperimentsRunner._run_single),
    # so only disk persistence makes that cross-run reuse possible.

    @staticmethod
    def _stat_token(path: Optional[str]) -> str:
        if not path:
            return "none"
        try:
            real = os.path.realpath(path)
            return f"{real}:{os.stat(real).st_mtime_ns}"
        except OSError:
            return str(path)

    def _feature_cache_path(
        self, csv_path: Optional[str], augmentations: Optional[list]
    ) -> Optional[str]:
        """Deterministic cache file path for (checkpoint, dataset CSV, TTA
        augmentations, batch size); ``None`` when caching is disabled
        (``gfss.test_feature_cache_dir`` unset), there's no dataset, or the
        backbone is trainable (fine-tuning baselines, ``base_reference is
        not None``) -- there, ``self.model.features`` depends on *this
        run's own* fine-tuning (fit on that run's specific support episode),
        not just the checkpoint path, so a different run sharing the same
        checkpoint+csv+augmentations key would have genuinely different
        features; persisting/reusing them across runs would silently
        corrupt their eval metrics.
        """
        cache_dir = self.cfg.gfss.get("test_feature_cache_dir", None)
        if not cache_dir or not csv_path or self.base_reference is not None:
            return None
        key = "|".join(
            [
                self._stat_token(getattr(self, "_base_checkpoint_path", None)),
                self._stat_token(csv_path),
                ",".join(augmentations or [ROT0]),
                str(int(self.cfg.hyperparameters.batch_size)),
            ]
        )
        digest = hashlib.sha1(key.encode()).hexdigest()
        return os.path.join(cache_dir, f"{digest}.pt")

    def _reset_feature_cache(
        self, csv_path: Optional[str], augmentations: Optional[list]
    ) -> None:
        """Start this epoch's in-memory feature cache, seeded from disk
        when a cache file for this exact key already exists."""
        self._test_feat_cache = {}
        self._test_feat_cache_dirty = False
        self._test_feat_cache_path = self._feature_cache_path(csv_path, augmentations)
        if self._test_feat_cache_path and os.path.exists(self._test_feat_cache_path):
            try:
                self._test_feat_cache = torch.load(
                    self._test_feat_cache_path, map_location="cpu", weights_only=True
                )
            except Exception:
                logger.warning(
                    "GFSSModel: could not load feature cache %s, recomputing.",
                    self._test_feat_cache_path,
                )
                self._test_feat_cache = {}

    def _flush_feature_cache(self) -> None:
        """Persist the in-memory cache when new entries were added (a hit
        on every batch -- the common case for runs 2..N of a seed --
        leaves it unchanged, so this is a no-op then)."""
        if not self._test_feat_cache_path or not self._test_feat_cache_dirty:
            return
        os.makedirs(os.path.dirname(self._test_feat_cache_path), exist_ok=True)
        tmp = f"{self._test_feat_cache_path}.tmp{os.getpid()}"
        torch.save(self._test_feat_cache, tmp)
        os.replace(tmp, self._test_feat_cache_path)
        self._test_feat_cache_dirty = False

    def _backbone_features(self, x: Tensor, key) -> Tensor:
        cached = self._test_feat_cache.get(key)
        if cached is not None:
            return cached.to(x.device)
        feats = self.model.features(x)
        self._test_feat_cache[key] = feats.detach().to("cpu")
        self._test_feat_cache_dirty = True
        return feats

    def on_validation_start(self) -> None:
        csv_path = (
            self.cfg.val_dataset.get("window_index_cache", None)
            if self.val_ds is not None
            else None
        )
        self._reset_feature_cache(csv_path, None)

    def on_test_start(self) -> None:
        self._reset_feature_cache(
            self.cfg.test_dataset.get("window_index_cache", None),
            self._get_tta_augmentations(),
        )

    def on_validation_end(self) -> None:
        self._flush_feature_cache()

    def on_test_end(self) -> None:
        self._flush_feature_cache()

    def _frozen_backbone_fast_path(self) -> bool:
        """Whether ``training_step`` can reuse ``_get_support``'s cached
        backbone features instead of recomputing them from raw images.

        Only safe when the backbone is frozen (``base_reference is None``,
        so the per-step features are identical to the cached support
        features) and the method doesn't need a snapshot of ``ref`` on raw
        images for KD (``requires_snapshot``, used by the fine-tuning
        baselines, which always train the backbone anyway).
        """
        return self.base_reference is None and not getattr(
            self.method, "requires_snapshot", False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Logits over all classes at the input resolution."""
        return self.model.upsample(self.method(self.model.features(x)), x.shape[-2:])

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        self._ensure_initialised()

    def train_dataloader(self):
        """Backed by cached support features when the backbone is frozen.

        The support set's augmentation is deterministic (normalize only), so
        its backbone features are identical on every epoch; with a frozen
        backbone (``q: linear/trans/edl/leak`` etc.) the base ``training_step``
        otherwise re-ran the full backbone forward pass (and the raster
        read it entails) on every one of ``max_steps`` optimizer steps even
        though the support set itself has only a handful of images. Reusing
        ``_get_support()``'s cache here removes that redundant work while
        keeping the exact same batch_size/shuffle/sampler/drop_last schedule.
        """
        if not self._frozen_backbone_fast_path():
            return super().train_dataloader()
        feats, masks = self._get_support()
        dataset = TensorDataset(feats.detach(), masks.detach())
        dl_cfg = self.cfg.train_dataset.data_loader
        sampler = self._make_weighted_sampler(dl_cfg)
        shuffle = dl_cfg.shuffle if sampler is None else False
        return DataLoader(
            dataset,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=(dl_cfg.drop_last if "drop_last" in dl_cfg else True),
            generator=self._make_dataloader_generator(),
        )

    def training_step(self, batch, batch_idx):
        if self._frozen_backbone_fast_path():
            feats, small = self._unpack_batch(batch)
            loss = self.method.support_loss(feats, small)
        else:
            images, masks = self._unpack_batch(batch)
            feats, small = self._features_and_masks(images, masks)
            if getattr(self.method, "requires_snapshot", False):
                ref = self._reference()
                with torch.no_grad():
                    snapshot = ref.linear(ref.features(images), ref.weight, ref.bias)
                loss = self.method.support_loss(feats, small, snapshot_logits=snapshot)
            else:
                loss = self.method.support_loss(feats, small)
        if loss is None:
            return None
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _update_uncertainty(self, feats, images, masks, pred, base_pred, unc_metrics):
        """Evaluate the method's uncertainty maps on the split decisions:
        pixels whose true class is a child (novel or kept mother) and that
        the base model assigns to that child's mother."""
        size = images.shape[-2:]
        maps = {
            k: F.interpolate(
                v.unsqueeze(1), size=size, mode="bilinear", align_corners=True
            )
            .squeeze(1)
            .clamp(0, 1)
            for k, v in self.method.uncertainty(feats).items()
        }
        valid = masks != self.cfg.gfss.get("ignore_index", 255)
        target = masks.clamp(min=0, max=self.hierarchy.num_classes - 1)
        mother_of_target = self.hierarchy.new_to_old.to(masks.device)[target]
        children = torch.tensor(
            [c for m in self.hierarchy.mothers for c in self.hierarchy.children_of(m)],
            device=masks.device,
        )
        region = valid & torch.isin(target, children) & (base_pred == mother_of_target)
        unc_metrics.update(maps, region, pred == masks, valid)

    _MAIN = "__main__"

    def _predict_logits(
        self,
        images: Tensor,
        augmentations,
        batch_idx: Optional[int] = None,
        precomputed_identity_feats: Optional[Tensor] = None,
        stage: str = "test",
    ) -> dict:
        """Logits of the method (key ``__main__``), of each decoding variant
        and of the unmodified base model (``__base__``), averaged over the TTA
        ``augmentations`` (``None`` = single pass) with the framework's
        ``apply_tta`` (mean of the de-augmented logits).

        Two backbone passes are avoided whenever possible (see
        ``_backbone_features``/``_test_feat_cache`` for the eval-time
        feature cache this draws on):

        * the identity (``rot0``) pass reuses ``precomputed_identity_feats``
          (``_eval_step`` already computed it for uncertainty/transductive
          adaptation) instead of recomputing ``features(images)``;
        * ``__base__`` reuses ``feats`` instead of calling
          ``ref.base_logits(x)`` (which recomputes ``features(x)`` from
          scratch) whenever the backbone is frozen (``base_reference is
          None``, every GFSS method except the fine-tuning baselines) —
          ``ref is self.model`` there, so it's the exact same computation,
          not an approximation.
        """
        ref = self._reference()
        frozen = self.base_reference is None
        aug_sequence = list(augmentations) if augmentations else [ROT0]
        state = {"i": 0}

        def predict(x: Tensor) -> dict:
            i = state["i"]
            aug_name = aug_sequence[i] if i < len(aug_sequence) else ROT0
            state["i"] += 1
            if aug_name == ROT0 and precomputed_identity_feats is not None:
                feats = precomputed_identity_feats
            else:
                feats = self._backbone_features(x, (stage, batch_idx, aug_name))
            size = x.shape[-2:]
            out = {self._MAIN: self.method.decode_to(feats, size, self.model.upsample)}
            for name, variant in self.method.decode_variants(feats).items():
                out[name] = self.model.upsample(variant, size)
            out["__base__"] = (
                self.model.upsample(
                    self.model.linear(feats, self.model.weight, self.model.bias), size
                )
                if frozen
                else ref.base_logits(x)
            )
            return out

        if not augmentations:
            return predict(images)
        return apply_tta(model_fn=predict, batch=images, augmentations=augmentations)

    def _eval_step(self, batch, batch_idx, metrics: GFSSMetrics, unc_metrics=None) -> None:
        self._ensure_initialised()
        images, masks = self._unpack_batch(batch)
        masks = masks.long()
        stage = metrics.prefix.rstrip("/")
        feats = self._backbone_features(images, (stage, batch_idx, ROT0))
        if self.method.transductive:
            with torch.enable_grad():
                self.method.adapt_to_query(*self._get_support(), feats)
        # TTA (cfg.tta_mode / use_tta, as in the base models) only at test time.
        augmentations = self._get_tta_augmentations() if stage == "test" else None
        logits = self._predict_logits(
            images,
            augmentations,
            batch_idx=batch_idx,
            precomputed_identity_feats=feats,
            stage=stage,
        )
        pred = logits.pop(self._MAIN).argmax(1)
        base_pred = logits.pop("__base__").argmax(1)
        metrics.update(pred, masks, base_pred)
        for name, variant in logits.items():
            idx = self._variant_keys.index((stage, name))
            self.variant_metrics[idx].update(variant.argmax(1), masks, base_pred)
        if unc_metrics is not None:
            # uncertainty maps come from the original (non-augmented) view
            self._update_uncertainty(feats, images, masks, pred, base_pred, unc_metrics)

    def val_dataloader(self):
        """Validation is optional in GFSS (no ``val_dataset`` -> no loop)."""
        return [] if self.val_ds is None else super().val_dataloader()

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, self.val_gfss_metrics, self.val_unc_metrics)

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, self.test_gfss_metrics, self.test_unc_metrics)

    def _log_and_reset(self, *metrics) -> None:
        for m in metrics:
            if m is not None:
                self.log_dict(m.compute())
                m.reset()

    def _stage_variants(self, stage: str):
        return [
            m
            for (st, _), m in zip(self._variant_keys, self.variant_metrics)
            if st == stage
        ]

    def on_validation_epoch_end(self) -> None:
        self._log_and_reset(
            self.val_gfss_metrics, self.val_unc_metrics, *self._stage_variants("val")
        )

    def on_test_epoch_end(self) -> None:
        self._log_and_reset(
            self.test_gfss_metrics, self.test_unc_metrics, *self._stage_variants("test")
        )
