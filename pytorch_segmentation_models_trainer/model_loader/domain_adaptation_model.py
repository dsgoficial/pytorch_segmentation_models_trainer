# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-14
        git sha              : $Format:%H$
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

import logging
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from hydra.utils import instantiate
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss
from pytorch_segmentation_models_trainer.domain_adaptation.base_method import (
    BaseDomainAdaptationMethod,
    DomainAdaptationLossOutput,
)
from pytorch_segmentation_models_trainer.domain_adaptation.feature_hooks import (
    FeatureExtractorHook,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model

logger = logging.getLogger(__name__)


class DomainAdaptationModel(Model):
    """PyTorch Lightning model for unsupervised / semi-supervised domain adaptation.

    Inherits from :class:`Model` (which inherits from ``pl.LightningModule``).
    Follows the same pattern as ``FrameFieldSegmentationPLModel``:
    ``DomainAdaptationModel`` is the LightningModule orchestrator, while
    ``self.method`` (a :class:`BaseDomainAdaptationMethod`) is a plain
    ``nn.Module`` responsible for the DA-specific computation.

    Training loop per step
    ----------------------
    1. Unpack ``batch["source"]`` and ``batch["target"]`` from the
       ``CombinedLoader``.
    2. Forward both through ``self.model`` (optionally capturing intermediate
       feature maps via ``FeatureExtractorHook``).
    3. Compute segmentation loss on the **source** batch only.
    4. Compute DA loss via ``self.method.compute_da_loss(...)``.
    5. Combine: ``total = seg_loss + lambda_da * da_loss``.
    6. Log all scalars.

    Validation loop per step
    ------------------------
    Evaluates on whichever val sets are available (source and/or target).
    The :class:`DomainAdaptationMonitorCallback` provides richer monitoring on
    top of the built-in validation step.

    Warm-start checkpoint
    ---------------------
    If ``cfg.domain_adaptation.pretrained_checkpoint`` is configured, only the
    model weights are loaded (no optimizer state or epoch counter). This is
    distinct from ``resume_from_checkpoint``, which restores the full
    training state.

    Args:
        cfg: Hydra ``DictConfig`` / ``OmegaConf`` config. Expected to contain
            the ``domain_adaptation`` sub-config in addition to the standard
            ``Model`` fields (``model``, ``optimizer``, ``hyperparameters``,
            etc.).
    """

    def __init__(self, cfg):
        # Call pl.LightningModule.__init__ directly to bypass Model.__init__,
        # which unconditionally accesses cfg.train_dataset / cfg.val_dataset for
        # GPU augmentation setup — keys that do not exist in the DA config.
        # We then replicate the Model.__init__ setup we actually need.
        pl.LightningModule.__init__(self)
        self.cfg = cfg

        # Build the segmentation model (and optional warm-start)
        self.model = self.get_model()

        # These are needed by _compute_loss (inherited from Model)
        self.train_ds = None
        self.val_ds = None
        self.loss_function = self.get_loss_function()
        self.use_compound_loss = isinstance(self.loss_function, MultiLoss)
        self.steps_per_epoch = None
        self.gpu_train_transform = None
        self.gpu_val_transform = None

        self.save_hyperparameters(
            ignore=["model", "loss_function", "train_ds", "val_ds"]
        )

        if "metrics" in self.cfg:
            metrics = torchmetrics.MetricCollection(
                [instantiate(i, _recursive_=False) for i in self.cfg.metrics]
            )
            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")

        self.check_if_should_normalize()

        # DA-specific setup
        da_cfg = cfg.domain_adaptation

        # Instantiate the DA method (nn.Module, NOT LightningModule)
        self.method: BaseDomainAdaptationMethod = instantiate(
            da_cfg.method, _recursive_=False
        )

        # Source and target training datasets
        self.source_train_ds = instantiate(da_cfg.source_dataset, _recursive_=False)
        self.target_train_ds = instantiate(da_cfg.target_dataset, _recursive_=False)

        # Validation datasets (optional — None disables that domain's evaluation)
        self.source_val_ds = (
            instantiate(da_cfg.source_val_dataset, _recursive_=False)
            if da_cfg.source_val_dataset is not None
            else None
        )
        self.target_val_ds = (
            instantiate(da_cfg.target_val_dataset, _recursive_=False)
            if da_cfg.target_val_dataset is not None
            else None
        )

        # Feature extractor hook — only active when the method needs features
        self._feature_hook: Optional[FeatureExtractorHook] = None
        if self.method.requires_features:
            feature_layers = list(da_cfg.feature_layers)
            if not feature_layers:
                logger.warning(
                    "DomainAdaptationModel: method.requires_features=True but "
                    "cfg.domain_adaptation.feature_layers is empty. "
                    "source_features and target_features will be empty dicts."
                )
            else:
                self._feature_hook = FeatureExtractorHook(self.model, feature_layers)
                logger.info(
                    "DomainAdaptationModel: FeatureExtractorHook active on layers %s",
                    feature_layers,
                )

    # ------------------------------------------------------------------
    # Model construction (overrides Model.get_model to add warm-start)
    # ------------------------------------------------------------------

    def get_model(self):
        """Build the segmentation model and optionally load pretrained weights.

        Overrides :meth:`Model.get_model`. After constructing the model with
        the parent implementation, checks for an optional
        ``cfg.domain_adaptation.pretrained_checkpoint`` config. When present,
        loads only the model weights (warm-start) without touching optimizer
        state or epoch counters.

        Returns:
            Instantiated and optionally warm-started ``nn.Module``.
        """
        model = super().get_model()

        da_cfg = getattr(self.cfg, "domain_adaptation", None)
        if da_cfg is None:
            return model

        ckpt_cfg = getattr(da_cfg, "pretrained_checkpoint", None)
        if ckpt_cfg is not None:
            self._load_pretrained_weights(model, ckpt_cfg)

        return model

    def _load_pretrained_weights(self, model, ckpt_cfg) -> None:
        """Load model weights from a checkpoint file (warm-start).

        Handles two formats:

        * ``source_format="pytorch_lightning"``: reads ``ckpt["state_dict"]``
          and strips the ``"model."`` prefix added by Lightning.
        * ``source_format="pytorch"``: reads the file directly as a state dict.

        Args:
            model: The ``nn.Module`` to load weights into.
            ckpt_cfg: :class:`PretrainedCheckpointConfig` instance with
                ``path``, ``source_format``, and ``strict_loading`` fields.
        """
        logger.info(
            "Loading pretrained weights from '%s' (format=%s, strict=%s)",
            ckpt_cfg.path,
            ckpt_cfg.source_format,
            ckpt_cfg.strict_loading,
        )
        ckpt = torch.load(ckpt_cfg.path, map_location="cpu")

        if ckpt_cfg.source_format == "pytorch_lightning":
            raw = ckpt["state_dict"]
            state_dict = {
                k.removeprefix("model."): v
                for k, v in raw.items()
                if k.startswith("model.")
            }
        else:
            state_dict = ckpt

        missing, unexpected = model.load_state_dict(
            state_dict, strict=ckpt_cfg.strict_loading
        )
        if missing:
            logger.warning("Pretrained checkpoint: missing keys: %s", missing)
        if unexpected:
            logger.warning("Pretrained checkpoint: unexpected keys: %s", unexpected)
        logger.info("Pretrained weights loaded successfully.")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> CombinedLoader:
        """Return a ``CombinedLoader`` with source and target training datasets.

        Both dataloaders cycle with ``"max_size_cycle"`` so the longer dataset
        drives the epoch length and the shorter one wraps around.

        Returns:
            ``CombinedLoader`` whose batches are dicts with keys
            ``"source"`` and ``"target"``.
        """
        source_loader = self._make_dataloader(
            self.source_train_ds,
            self.cfg.domain_adaptation.source_dataset.data_loader,
            shuffle=True,
        )
        target_loader = self._make_dataloader(
            self.target_train_ds,
            self.cfg.domain_adaptation.target_dataset.data_loader,
            shuffle=True,
        )
        return CombinedLoader(
            {"source": source_loader, "target": target_loader},
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> Union[DataLoader, CombinedLoader, List[DataLoader]]:
        """Return validation dataloader(s).

        Returns a single ``DataLoader`` when only one val dataset is configured,
        or a ``CombinedLoader`` when both are available.

        Returns:
            Dataloader(s) for the available validation domains.
        """
        loaders: Dict[str, DataLoader] = {}

        if self.source_val_ds is not None:
            loaders["source_val"] = self._make_dataloader(
                self.source_val_ds,
                self.cfg.domain_adaptation.source_val_dataset.data_loader
                if getattr(self.cfg.domain_adaptation, "source_val_dataset", None)
                is not None
                and hasattr(
                    self.cfg.domain_adaptation.source_val_dataset, "data_loader"
                )
                else None,
                shuffle=False,
            )

        if self.target_val_ds is not None:
            loaders["target_val"] = self._make_dataloader(
                self.target_val_ds,
                self.cfg.domain_adaptation.target_val_dataset.data_loader
                if getattr(self.cfg.domain_adaptation, "target_val_dataset", None)
                is not None
                and hasattr(
                    self.cfg.domain_adaptation.target_val_dataset, "data_loader"
                )
                else None,
                shuffle=False,
            )

        if not loaders:
            logger.warning(
                "DomainAdaptationModel: no validation datasets configured. "
                "Returning an empty loader. Validation will be skipped."
            )
            return DataLoader([])

        if len(loaders) == 1:
            return next(iter(loaders.values()))

        return CombinedLoader(loaders, mode="sequential")

    def _make_dataloader(
        self, dataset, data_loader_cfg, shuffle: bool = False
    ) -> DataLoader:
        """Build a DataLoader from a dataset and a data_loader config section.

        Args:
            dataset: The ``torch.utils.data.Dataset`` instance.
            data_loader_cfg: Config object with ``batch_size``, ``num_workers``,
                ``pin_memory``, ``drop_last``, ``prefetch_factor``, and
                ``persistent_workers`` fields. May be ``None``, in which case
                safe defaults are used.
            shuffle: Whether to shuffle. Overrides ``data_loader_cfg.shuffle``
                when explicitly provided.

        Returns:
            Configured ``DataLoader``.
        """
        if data_loader_cfg is not None:
            batch_size = getattr(data_loader_cfg, "batch_size", 8)
            num_workers = getattr(data_loader_cfg, "num_workers", 4)
            pin_memory = getattr(data_loader_cfg, "pin_memory", True)
            drop_last = getattr(data_loader_cfg, "drop_last", True)
            prefetch_factor = getattr(data_loader_cfg, "prefetch_factor", 2)
            persistent_workers = getattr(data_loader_cfg, "persistent_workers", False)
        else:
            batch_size = 8
            num_workers = 4
            pin_memory = True
            drop_last = False
            prefetch_factor = 2
            persistent_workers = False

        kwargs: Dict[str, Any] = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor

        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn

        return DataLoader(dataset, **kwargs)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure main optimizer, with optional extra parameter groups.

        If ``self.method.get_extra_parameter_groups()`` returns non-empty
        results (e.g. a discriminator with a different LR), those groups are
        added to the optimizer while the main group covers all remaining
        parameters.

        Returns:
            Tuple of ``([optimizer], scheduler_list)`` as expected by
            PyTorch Lightning.
        """
        extra_groups = self.method.get_extra_parameter_groups()

        if extra_groups:
            # Keep extra-group params out of the main group to avoid duplicate
            # optimization (which PyTorch would reject as an error).
            extra_param_ids = {id(p) for group in extra_groups for p in group["params"]}
            main_params = [p for p in self.parameters() if id(p) not in extra_param_ids]
            all_param_groups = [{"params": main_params}] + extra_groups

            # Hydra's instantiate converts list-of-dicts to OmegaConf, which
            # breaks PyTorch optimizer. Resolve the optimizer config to plain
            # Python types and instantiate manually.
            from omegaconf import OmegaConf
            from hydra.utils import get_class

            optimizer_dict = OmegaConf.to_container(self.cfg.optimizer, resolve=True)
            optimizer_cls = get_class(optimizer_dict.pop("_target_"))
            optimizer = optimizer_cls(all_param_groups, **optimizer_dict)
        else:
            optimizer = self.get_optimizer()

        scheduler_list = self._build_scheduler_list(optimizer)
        return [optimizer], scheduler_list

    def _build_scheduler_list(self, optimizer) -> list:
        """Build scheduler list from config, delegating to parent logic."""
        scheduler_list = []
        if "scheduler_list" not in self.cfg:
            return scheduler_list
        for item in self.cfg.scheduler_list:
            dict_item = dict(item)
            dict_item["scheduler"] = instantiate(
                item.scheduler, optimizer=optimizer, _recursive_=False
            )
            scheduler_list.append(dict_item)
        return scheduler_list

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward_with_features(self, images: torch.Tensor):
        """Run a forward pass and capture intermediate feature maps if needed.

        Args:
            images: Input image tensor, shape ``(B, C, H, W)``.

        Returns:
            Tuple of ``(output, features_dict)`` where ``features_dict`` is
            empty when ``self._feature_hook`` is ``None``.
        """
        if self._feature_hook is not None:
            self._feature_hook.clear()

        output = self.model(images)

        features = (
            self._feature_hook.get_features() if self._feature_hook is not None else {}
        )
        return output, features

    def _get_lambda_da(self) -> float:
        """Return the current DA loss weight.

        Checks for a ``lambda_schedule`` attribute on the method first
        (allowing per-method scheduling). Falls back to ``method.lambda_da``.

        Returns:
            Scalar float to multiply the DA loss by.
        """
        if hasattr(self.method, "lambda_schedule"):
            max_epochs = self.trainer.max_epochs if self.trainer else 1
            return self.method.lambda_schedule.get_lambda(
                self.current_epoch, max_epochs
            )
        return self.method.lambda_da

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute one DA training step.

        Args:
            batch: Dict with keys ``"source"`` and ``"target"``, each being a
                dataset batch dict with at least an ``"image"`` key.
            batch_idx: Index of the current batch.

        Returns:
            Total loss scalar (seg_loss + lambda_da * da_loss).
        """
        source_batch = batch["source"]
        target_batch = batch["target"]

        source_images = source_batch["image"]
        target_images = target_batch["image"]

        # Forward both domains (with optional feature capture)
        source_output, source_features = self._forward_with_features(source_images)
        target_output, target_features = self._forward_with_features(target_images)

        # Segmentation loss — source domain only (labeled)
        source_masks = source_batch["mask"].long()
        seg_loss, individual_losses, extra_info = self._compute_loss(
            source_output, source_masks
        )

        # DA loss
        da_result: DomainAdaptationLossOutput = self.method.compute_da_loss(
            source_batch=source_batch,
            target_batch=target_batch,
            source_output=source_output,
            target_output=target_output,
            source_features=source_features,
            target_features=target_features,
        )

        lambda_da = self._get_lambda_da()
        total_loss = seg_loss + lambda_da * da_result.loss

        # Logging
        self.log(
            "loss/train_seg",
            seg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "loss/train_da",
            da_result.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "loss/train_total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("da/lambda", lambda_da, on_step=True, on_epoch=False)

        for k, v in da_result.log_dict.items():
            self.log(f"da/{k}", v, on_step=True, on_epoch=True, sync_dist=True)

        if individual_losses:
            for name, val in individual_losses.items():
                self.log(
                    f"losses/train_{name}",
                    val,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=False,
                )

        return total_loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Optional[torch.Tensor]:
        """Execute one validation step.

        Handles batches from either the source or target validation dataloader.
        Computes segmentation loss and logs it with the appropriate domain prefix.

        Args:
            batch: Batch dict with ``"image"`` and ``"mask"`` keys.
            batch_idx: Index of the current batch within the dataloader.
            dataloader_idx: Index indicating which dataloader this batch came
                from. Used to derive the domain label for logging.

        Returns:
            Segmentation loss for this batch, or ``None`` if the batch has no
            mask (e.g. unlabeled target data).
        """
        if "mask" not in batch:
            return None

        images = batch["image"]
        masks = batch["mask"].long()
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        output = self.model(images)
        loss, individual_losses, _ = self._compute_loss(output, masks)

        # Derive domain name for metric prefix from dataloader_idx
        domain = self._get_val_domain_name(dataloader_idx)
        self.log(
            f"loss/val_{domain}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if individual_losses:
            for name, val in individual_losses.items():
                self.log(
                    f"losses/val_{domain}_{name}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    add_dataloader_idx=False,
                )

        return loss

    def _get_val_domain_name(self, dataloader_idx: int) -> str:
        """Map a dataloader index to a domain name string for metric logging.

        Args:
            dataloader_idx: Index of the validation dataloader.

        Returns:
            ``"source"`` or ``"target"`` based on which datasets are configured
            and the order they were added to the val dataloader.
        """
        has_source = self.source_val_ds is not None
        has_target = self.target_val_ds is not None

        if has_source and has_target:
            return "source" if dataloader_idx == 0 else "target"
        elif has_source:
            return "source"
        else:
            return "target"

    # ------------------------------------------------------------------
    # Lightning lifecycle hooks → forward to method
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        super().on_fit_start() if hasattr(super(), "on_fit_start") else None
        self.method.on_fit_start(self)

    def on_train_epoch_start(self) -> None:
        self.method.on_train_epoch_start(self, self.current_epoch)

    def on_train_epoch_end(self) -> None:
        self.method.on_train_epoch_end(self, self.current_epoch)

    def on_fit_end(self) -> None:
        if self._feature_hook is not None:
            self._feature_hook.remove()
