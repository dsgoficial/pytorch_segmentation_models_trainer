# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-09
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

import logging
from pathlib import Path
import pytorch_lightning as pl
import rasterio
import torch

from torch.utils.data import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import concurrent.futures
import itertools

from pytorch_segmentation_models_trainer.predict import instantiate_polygonizer
from concurrent.futures import Future
from pytorch_segmentation_models_trainer.tools.polygonization.methods import (
    active_skeletons,
)
from pytorch_segmentation_models_trainer.utils.polygon_utils import (
    coerce_polygons_to_single_geometry,
)
from pytorch_segmentation_models_trainer.utils.tensor_utils import tensor_dict_to_device

logger = logging.getLogger(__name__)


class WarmupCallback(pl.callbacks.Callback):
    def __init__(self, warmup_epochs=2) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.warmed_up = False

    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins - replaces on_init_end"""
        logger.info("WarmupCallback initialization at epoch %d.", trainer.current_epoch)
        if trainer.current_epoch > self.warmup_epochs - 1:
            self.warmed_up = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.warmed_up or trainer.current_epoch < self.warmup_epochs - 1:
            return
        if not self.warmed_up:
            logger.info(
                "Model will warm up for %d epochs. Freezing encoder weights.",
                self.warmup_epochs,
            )
            self.set_component_trainable(pl_module, trainable=False)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.warmed_up:
            return
        if trainer.current_epoch >= self.warmup_epochs - 1:
            logger.info(
                "Model warm up completed in the end of epoch %d. Unfreezing encoder weights.",
                trainer.current_epoch,
            )
            self.set_component_trainable(pl_module, trainable=True)
            self.warmed_up = True

    def set_component_trainable(self, pl_module, trainable=True):
        pl_module.set_encoder_trainable(trainable=trainable)


class FrameFieldOnlyCrossfieldWarmupCallback(pl.callbacks.Callback):
    def __init__(self, warmup_epochs=2) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.warmed_up = False

    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins - replaces on_init_end"""
        logger.info(
            "FrameFieldWarmupCallback initialization at epoch %d.",
            trainer.current_epoch,
        )
        if trainer.current_epoch > self.warmup_epochs - 1:
            self.warmed_up = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.warmed_up or trainer.current_epoch < self.warmup_epochs - 1:
            return
        if not self.warmed_up:
            logger.info(
                "Frame field model will warm up for %d epochs. Freezing all weights but crossfield's.",
                self.warmup_epochs,
            )
            self.set_component_trainable(pl_module, trainable=False)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.warmed_up:
            return
        if trainer.current_epoch >= self.warmup_epochs - 1:
            logger.info(
                "Model warm up completed in the end of epoch %d. Unfreezing weights.",
                trainer.current_epoch,
            )
            self.set_component_trainable(pl_module, trainable=True)
            self.warmed_up = True

    def set_component_trainable(self, pl_module, trainable=True):
        pl_module.set_all_but_crossfield_trainable(trainable=trainable)


class ComputeWeightNormLossesCallback(pl.callbacks.Callback):
    """
    General callback to compute loss normalization weights before training starts.
    Works with ANY model that uses MultiLoss (compound loss) and has a compute_loss_norms method.

    This callback runs during on_fit_start to avoid CUDA initialization issues
    in multiprocessing contexts (DDP). It ensures CUDA operations happen after
    DataLoader worker processes are properly initialized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_norm_is_initializated = False

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:
        """
        Called when fit begins, after DDP processes are spawned.
        This is the correct hook to use for loss normalization to avoid
        CUDA initialization errors in multiprocessing contexts.
        """
        # Skip if already initialized
        if self.loss_norm_is_initializated:
            return

        # Skip if model doesn't have _compute_loss_normalization method
        if (
            hasattr(pl_module, "check_if_should_normalize")
            and not pl_module.check_if_should_normalize()
        ):
            logger.warning(
                f"Model {type(pl_module).__name__} has normalization loss but the training config tells not to normalize."
                "Skipping loss normalization."
            )
            return
        if not hasattr(pl_module.cfg, "loss_params"):
            logger.warning(
                f"Model {type(pl_module).__name__} has single loss and do not need normalization."
                "Skipping loss normalization."
            )
            return

        if hasattr(pl_module.cfg.loss_params, "compound_loss") and not hasattr(
            pl_module.cfg.loss_params.compound_loss, "normalization_params"
        ):
            logger.warning(
                f"Model {type(pl_module).__name__} does not have the appropriate normalization parameter tags."
                "Skipping loss normalization."
            )
            return

        if pl_module.loss_function.norm_updated:
            logger.warning(
                f"Model {type(pl_module).__name__} norm was already updated."
                "Skipping loss normalization."
            )
            return
        # Only compute on rank 0 to avoid redundant computation
        if trainer.global_rank == 0:
            logger.info("Computing loss normalization weights...")
            # Compute the loss norms
            self.compute_loss_normalization(pl_module)

            logger.info("Loss normalization weights computed successfully")

        # Synchronize across all ranks in distributed training
        # if trainer.world_size > 1:
        #     logger.info("Synchronizing loss normalization across all ranks...")
        #     torch.distributed.barrier()
        logger.info("Synchronization complete")

        self.loss_norm_is_initializated = True

    def compute_loss_normalization(self, pl_module):
        """
        NEW: Compute normalization values for compound loss.
        Only called if using MultiLoss.
        """
        from tqdm import tqdm

        # Get dataloader
        dl = DataLoader(
            pl_module.train_ds,
            batch_size=pl_module.cfg.hyperparameters.batch_size,
            shuffle=True,
            num_workers=4,  # MUST be 0 to avoid CUDA init errors in DDP
            drop_last=False,
            pin_memory=False,  # Also disable pin_memory for safety
        )

        # Number of batches for normalization
        if (
            hasattr(pl_module.cfg, "loss_params")
            and hasattr(pl_module.cfg.loss_params, "compound_loss")
            and hasattr(pl_module.cfg.loss_params.compound_loss, "normalization_params")
        ):
            max_samples = (
                pl_module.cfg.loss_params.compound_loss.normalization_params.get(
                    "max_samples", 1000
                )
            )
            max_batches = max_samples // pl_module.cfg.hyperparameters.batch_size
        else:
            max_batches = min(100, len(dl))

        # Reset norms
        pl_module.loss_function.reset_norm()

        # Compute norms
        pl_module.model.eval()
        logger.info("Evaluating loss norms")
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dl, total=max_batches, desc="Computing loss norms")
            ):
                if batch_idx >= max_batches:
                    break

                # Unpack batch
                images, masks = batch.values()

                # Move to device if needed
                device = pl_module.device
                images = images.to(device)
                if isinstance(masks, dict):
                    masks = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in masks.items()
                    }
                else:
                    masks = masks.to(device)

                # Forward pass
                pred = pl_module.model(images)

                # Prepare batch format for MultiLoss
                if isinstance(masks, dict):
                    pred_batch = {"seg": pred}
                    gt_batch = masks
                else:
                    pred_batch = pred
                    gt_batch = masks

                # Update norms
                batch_size = images.shape[0]
                pl_module.loss_function.update_norm(pred_batch, gt_batch, batch_size)

        # Sync across GPUs if distributed
        if pl_module.trainer.world_size > 1 and torch.distributed.is_initialized():
            logger.info("Performing loss sync across devices")
            pl_module.loss_function.sync(pl_module.trainer.world_size)

        pl_module.loss_function.set_norm_updated(True)
        pl_module.model.train()
        logger.info("Loss normalization computed")
        logger.info(f"Loss function: {pl_module.loss_function}")
        for loss in pl_module.loss_function.loss_funcs:
            logger.info(f"Computed normalization factor for {loss}: {loss.norm}")


# Also update the FrameField-specific callback to use the same fix
class FrameFieldComputeWeightNormLossesCallback(pl.callbacks.Callback):
    """
    Callback to compute loss normalization weights for FrameField models before training starts.

    This callback runs during on_fit_start to avoid CUDA initialization issues
    in multiprocessing contexts (DDP). It ensures CUDA operations happen after
    DataLoader worker processes are properly initialized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_norm_is_initializated = False

    def on_fit_start(self, trainer, pl_module) -> None:
        """
        Called when fit begins, after DDP processes are spawned.
        This is the correct hook to use for loss normalization to avoid
        CUDA initialization errors in multiprocessing contexts.
        """
        # Skip if already initialized
        if self.loss_norm_is_initializated:
            return

        # Only compute on rank 0 to avoid redundant computation
        if trainer.global_rank == 0:
            logger.info("Computing loss normalization weights...")

            pl_module.model.train()
            init_dl = pl_module.train_dataloader()

            with torch.no_grad():
                # Calculate number of batches needed for normalization
                loss_norm_batches_min = (
                    pl_module.cfg.loss_params.multiloss.normalization_params.min_samples
                    // (2 * pl_module.cfg.hyperparameters.batch_size)
                    + 1
                )
                loss_norm_batches_max = (
                    pl_module.cfg.loss_params.multiloss.normalization_params.max_samples
                    // (2 * pl_module.cfg.hyperparameters.batch_size)
                    + 1
                )
                loss_norm_batches = max(
                    loss_norm_batches_min, min(loss_norm_batches_max, len(init_dl))
                )

                logger.info(f"Using {loss_norm_batches} batches for loss normalization")

                # Compute the loss norms
                pl_module.compute_loss_norms(init_dl, loss_norm_batches)

            logger.info("Loss normalization weights computed successfully")

        # Synchronize across all ranks in distributed training
        if trainer.world_size > 1:
            logger.info("Synchronizing loss normalization across all ranks...")
            torch.distributed.barrier()
            logger.info("Synchronization complete")

        self.loss_norm_is_initializated = True


class FrameFieldPolygonizerCallback(pl.callbacks.BasePredictionWriter):
    def __init__(self, write_interval="batch") -> None:
        # Add write_interval for Lightning 2.0+ compatibility
        super().__init__(write_interval=write_interval)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        parent_dir_name_list = [Path(path).stem for path in batch["path"]]
        seg_batch, crossfield_batch = outputs
        if seg_batch is None and crossfield_batch is None:
            return
        with concurrent.futures.ThreadPoolExecutor() as pool:
            polygonizer = instantiate_polygonizer(pl_module.cfg)
            futures = []
            try:
                with torch.enable_grad():
                    futures = polygonizer.process(
                        {"seg": seg_batch, "crossfield": crossfield_batch},
                        profile=None,
                        parent_dir_name=parent_dir_name_list,
                        pool=pool,
                        convert_output_to_world_coords=False,
                    )
            except Exception as e:
                logger.error(f"Error in polygonizer: {e}")
                logger.warning(
                    "Skipping polygonizer for batch with error. Check it later."
                )
            if (
                isinstance(futures, list)
                and len(futures) > 0
                and isinstance(futures[0], Future)
            ):
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in polygonizer: {e}")
                        logger.warning(
                            "Skipping polygonizer for batch with error. Check it later."
                        )


class ActiveSkeletonsPolygonizerCallback(pl.callbacks.BasePredictionWriter):
    def __init__(self, write_interval="batch") -> None:
        # Add write_interval for Lightning 2.0+ compatibility
        super().__init__(write_interval=write_interval)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        parent_dir_name_list = [Path(path).stem for path in batch["path"]]
        seg_batch, crossfield_batch = outputs
        if seg_batch is None and crossfield_batch is None:
            return
        polygonizer = instantiate_polygonizer(pl_module.cfg)
        try:
            polys = self._run_polygonize(pl_module, seg_batch, crossfield_batch)
        except Exception as e:
            polys = []
            for idx, (seg, crossfield) in enumerate(zip(seg_batch, crossfield_batch)):
                try:
                    if (
                        torch.sum(seg > pl_module.cfg.polygonizer.config.seg_threshold)
                        == 0
                    ):
                        polys.append([])
                        continue
                    out_contours = self._run_polygonize(
                        pl_module, seg.unsqueeze(0), crossfield.unsqueeze(0)
                    )
                    if len(out_contours) == 0:
                        polys.append([])
                        continue
                    polys.append(out_contours[0])
                except Exception as e1:
                    logger.exception(
                        f"An error occurred while polygonizing the image {parent_dir_name_list[idx]}. Skipping this image."
                    )
                    logger.exception(e1)
                    polys.append([])
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = []
            for idx, polygon_list in enumerate(polys):
                futures.append(
                    pool.submit(
                        polygonizer.data_writer.write_data,
                        coerce_polygons_to_single_geometry(polygon_list),
                        profile={"crs": None},
                        folder_name=parent_dir_name_list[idx],
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error on writing output: {e}")
                    logger.warning(
                        "Skipping writer for batch with error. Check it later."
                    )

    def _run_polygonize(self, pl_module, seg_batch, crossfield_batch):
        with torch.enable_grad():
            polys, probs = active_skeletons.polygonize(
                seg_batch, crossfield_batch, pl_module.cfg.polygonizer.config
            )
        return polys


class ModPolymapperPolygonizerCallback(pl.callbacks.BasePredictionWriter):
    def __init__(
        self, convert_output_to_world_coords=True, write_interval="batch"
    ) -> None:
        # Add write_interval for Lightning 2.0+ compatibility
        super().__init__(write_interval=write_interval)
        self.convert_output_to_world_coords = convert_output_to_world_coords

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        def process_polygonizer(detection, parent_dir_name, profile=None):
            polygonizer = instantiate_polygonizer(pl_module.cfg)
            detection["output_batch_polygons"] = detection.pop("polygonrnn_output")
            polygonizer.process(
                detection,
                profile=profile,
                parent_dir_name=parent_dir_name,
                convert_output_to_world_coords=self.convert_output_to_world_coords,
            )

        parent_dir_name_list = [Path(path).stem for path in batch["path"]]
        profile_list = (
            self.get_profile_list(batch)
            if self.convert_output_to_world_coords
            else len(parent_dir_name_list) * [None]
        )

        if len(outputs) != len(parent_dir_name_list):
            raise ValueError(
                "The number of detections and the number of parent_dir_name_list must be the same"
            )
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as pool:
            for detection, parent_dir_name, profile in itertools.zip_longest(
                outputs, parent_dir_name_list, profile_list
            ):
                future = pool.submit(
                    process_polygonizer,
                    tensor_dict_to_device(detection, "cpu"),
                    parent_dir_name,
                    profile,
                )
                futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            future.result()

    def get_profile_list(self, batch):
        profile_list = []
        for path in batch["path"]:
            with rasterio.open(path) as raster_ds:
                profile_list.append(raster_ds.profile.copy())
        return profile_list


class EMACallback(pl.callbacks.Callback):
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters updated as:
        shadow = decay_eff * shadow + (1 - decay_eff) * param

    where ``decay_eff = min(decay, (step+1) / (step+10))`` implements an
    EMA warmup that lets early updates track the online model closely,
    avoiding contamination from random initialisation weights.
    Ref: Busbridge et al., "EMA of Weights in DL", TMLR 2024.

    During validation the shadow weights are swapped in so that metrics
    reflect the EMA model.  After validation the original (online) weights
    are restored so training continues normally.

    Fixes vs original:
      - EMA updates only when global_step changes (correct behavior with
        accumulate_grad_batches — avoids updating on micro-batches where
        the optimizer has not yet stepped).
      - on_save_checkpoint injects EMA weights into the checkpoint so that
        ModelCheckpoint saves the EMA model (not online weights).  This
        matches the validation metrics that selected the checkpoint.

    The EMA state is persisted inside checkpoints automatically because
    Lightning serialises all callback states via ``state_dict`` /
    ``load_state_dict``.

    Args:
        decay: Target EMA decay factor (default 0.999).
    """

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self._shadow: dict = {}
        self._original: dict = {}
        self._last_global_step: int = -1

    # -- build / update shadow --

    def on_fit_start(self, trainer, pl_module):
        self._shadow = {
            name: param.data.clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }
        self._last_global_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        # Only update EMA when the optimizer actually stepped.
        # With accumulate_grad_batches > 1, on_train_batch_end fires
        # for every micro-batch, but global_step only increments after
        # the optimizer step.  Skip micro-batches where step hasn't changed.
        if step == self._last_global_step:
            return
        self._last_global_step = step

        # EMA warmup: ramp decay from ~0.1 to target over first ~1000 steps
        effective_decay = min(self.decay, (step + 1) / (step + 10))
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self._shadow:
                self._shadow[name].mul_(effective_decay).add_(
                    param.data, alpha=1.0 - effective_decay
                )

    # -- swap in shadow for validation --

    def on_validation_epoch_start(self, trainer, pl_module):
        self._original = {}
        for name, param in pl_module.named_parameters():
            if name in self._shadow:
                self._original[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self._original:
                param.data.copy_(self._original[name])
        self._original = {}

    # -- checkpoint persistence --

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Inject EMA weights into the saved checkpoint.

        ModelCheckpoint runs after validation, when metrics were computed
        using EMA weights.  Without this override the checkpoint would
        contain the online (non-EMA) weights, which is inconsistent with
        the monitored metric that triggered the save.
        """
        for name, shadow_param in self._shadow.items():
            if name in checkpoint["state_dict"]:
                checkpoint["state_dict"][name] = shadow_param.cpu().clone()

    def state_dict(self):
        return {"shadow": self._shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self._shadow = state_dict.get("shadow", {})
        self.decay = state_dict.get("decay", self.decay)


class MixStyleCallback(pl.callbacks.Callback):
    """MixStyle: plug-and-play feature-level domain augmentation.

    Mixes feature statistics (mean/std) between instances in a batch,
    creating implicit domain augmentation at zero parameter cost.
    Improves generalization to unseen domains / temporal shifts.

    Ref: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
         Zhou et al., "MixStyle Neural Networks", IJCV 2024.

    Registers forward hooks on specified encoder stages to apply MixStyle
    after their output.  Only active during training.

    Args:
        p: Probability of applying MixStyle per forward pass (default 0.5).
        alpha: Beta distribution parameter for mixing (default 0.1).
            Smaller values produce mixing closer to the original style.
        stages: List of encoder stage indices to hook (default [0, 1]).
            Early stages carry more domain/style information.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, stages=None) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.stages = stages if stages is not None else [0, 1]
        self._hooks = []
        self._training = False
        self._beta_dist = torch.distributions.Beta(alpha, alpha)

    def _mixstyle_hook(self, module, input, output):
        """Forward hook that applies MixStyle to a stage's output."""
        if not self._training:
            return output
        if isinstance(output, (list, tuple)):
            return output  # multi-output stages — skip
        x = output
        B = x.size(0)
        if B < 2 or torch.rand(1).item() > self.p:
            return output

        eps = 1e-6
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + eps).sqrt()
        x_normed = (x - mu) / sig

        # Shuffle within batch
        perm = torch.randperm(B).to(x.device)
        mu2, sig2 = mu[perm], sig[perm]

        # Mix with Beta distribution
        lmbd = self._beta_dist.sample((B, 1, 1, 1)).to(x.device)
        mu_mix = mu * lmbd + mu2 * (1 - lmbd)
        sig_mix = sig * lmbd + sig2 * (1 - lmbd)

        return x_normed * sig_mix + mu_mix

    def on_fit_start(self, trainer, pl_module):
        encoder = pl_module.model.encoder

        # SMP TimmUniversalEncoder: encoder.model.stages
        stages_module = None
        if hasattr(encoder, "model") and hasattr(encoder.model, "stages"):
            stages_module = encoder.model.stages
        elif hasattr(encoder, "stages"):
            stages_module = encoder.stages

        if stages_module is None:
            logger.warning(
                "MixStyleCallback: could not find encoder stages. "
                "MixStyle will NOT be applied."
            )
            return

        for stage_idx in self.stages:
            if stage_idx < len(stages_module):
                hook = stages_module[stage_idx].register_forward_hook(
                    self._mixstyle_hook
                )
                self._hooks.append(hook)
                logger.info(f"MixStyle: registered hook on encoder stage {stage_idx}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._training = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._training = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self._training = False

    def on_fit_end(self, trainer, pl_module):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
