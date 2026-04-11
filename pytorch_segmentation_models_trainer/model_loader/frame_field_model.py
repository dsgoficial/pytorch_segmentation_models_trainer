# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-19
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
 *   Code inspired by the one in                                           *
 *   https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/     *
 ****
"""
import copy
from collections import OrderedDict
from pytorch_toolbelt.inference.tiles import TileMerger
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_segmentation_models_trainer.custom_losses.loss import mixup_data
from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss
from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
    build_loss_from_config,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from segmentation_models_pytorch.base.initialization import (
    initialize_decoder,
    initialize_head,
)
from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.utils import tensor_utils
from tqdm import tqdm
import kornia as K


class FrameFieldModel(nn.Module):
    def __init__(
        self,
        segmentation_model,
        use_batchnorm: bool = True,
        replace_seg_head: bool = True,
        compute_seg: bool = True,
        compute_crossfield: bool = True,
        seg_params: dict = None,
        module_activation: str = None,
        frame_field_activation: str = None,
    ):
        """FrameField Model - same as before, no changes needed"""
        super().__init__()
        self.crossfield_channels = 4
        self.seg_params = {
            "compute_interior": True,
            "compute_edge": True,
            "compute_vertex": True,
        }
        if seg_params is not None:
            for param, value in seg_params.items():
                if param not in self.seg_params:
                    continue
                if not isinstance(value, bool):
                    raise ValueError(f"Parameter {param} must be boolean!")
                self.seg_params[param] = value
        self.segmentation_model = (
            instantiate(segmentation_model, _recursive_=False)
            if isinstance(segmentation_model, (str, DictConfig))
            else segmentation_model
        )
        self.replace_seg_head = replace_seg_head
        self.compute_seg = compute_seg
        self.compute_crossfield = compute_crossfield
        self.use_batchnorm = use_batchnorm
        self.frame_field_activation = frame_field_activation
        self.module_activation = module_activation
        if hasattr(self.segmentation_model, "decoder"):
            self.backbone_output = self.get_out_channels(
                self.segmentation_model.decoder
                if self.replace_seg_head
                else self.segmentation_model.segmentation_head
            )
        else:
            self.backbone_output = self.get_out_channels(self.segmentation_model)
        self.seg_channels = sum(self.seg_params.values())
        self.upsampling = self.get_upsampling_method()
        self.seg_module = self.get_seg_module()
        self.crossfield_module = self.get_crossfield_module()
        self.initialize()

    def get_seg_module(self) -> torch.nn.Sequential:
        if not self.replace_seg_head:
            return self.segmentation_model.segmentation_head
        return torch.nn.Sequential(
            torch.nn.Conv2d(self.backbone_output, self.backbone_output, 3, padding=1),
            torch.nn.BatchNorm2d(self.backbone_output),
            torch.nn.ELU(),
            torch.nn.Conv2d(self.backbone_output, self.seg_channels, 1),
            self.upsampling,
            torch.nn.Sigmoid(),
        )

    def get_upsampling_method(self) -> torch.nn.Module:
        return (
            list(self.segmentation_model.segmentation_head.children())[1]
            if hasattr(self.segmentation_model, "segmentation_head")
            else torch.nn.Identity()
        )

    def get_crossfield_module(self) -> torch.nn.Sequential:
        if not self.compute_crossfield:
            return None
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                self.backbone_output + self.seg_channels,
                self.backbone_output,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.backbone_output)
            if self.use_batchnorm
            else nn.Identity(),
            torch.nn.ELU()
            if self.module_activation is None
            else instantiate(self.module_activation, _recursive_=False),
            torch.nn.Conv2d(
                self.backbone_output, self.crossfield_channels, kernel_size=1
            ),
            torch.nn.Tanh()
            if self.frame_field_activation is None
            else instantiate(self.frame_field_activation, _recursive_=False),
        )

    def get_output(self, x):
        if isinstance(
            self.segmentation_model,
            (
                smp.Unet,
                smp.UnetPlusPlus,
                smp.FPN,
                smp.PSPNet,
                smp.PAN,
                smp.DeepLabV3,
                smp.DeepLabV3Plus,
            ),
        ):
            encoder_feats = self.segmentation_model.encoder(x)
            decoder_output = self.segmentation_model.decoder(encoder_feats)
        else:
            decoder_output = self.segmentation_model(x)
        return (
            decoder_output
            if not isinstance(decoder_output, OrderedDict)
            else decoder_output["out"]
        )

    def forward(self, x):
        output_dict = OrderedDict()
        decoder_output = self.get_output(x)
        if self.compute_seg:
            segmentation_features = self.seg_module(decoder_output)
            detached_segmentation_features = segmentation_features.clone().detach()
            decoder_output = torch.cat(
                [self.upsampling(decoder_output), detached_segmentation_features], dim=1
            )
            output_dict["seg"] = segmentation_features
        if self.compute_crossfield:
            output_dict["crossfield"] = 2 * self.crossfield_module(decoder_output)
        return output_dict

    def initialize(self):
        if self.replace_seg_head:
            initialize_head(self.seg_module)
        if self.compute_crossfield:
            initialize_decoder(self.crossfield_module)

    @staticmethod
    def get_out_channels(module):
        if hasattr(module, "out_channels"):
            return module.out_channels
        children = list(module.children())
        i = 1
        out_channels = None
        while out_channels is None and i <= len(children):
            last_child = children[-i]
            out_channels = FrameFieldModel.get_out_channels(last_child)
            i += 1
        return out_channels


class FrameFieldSegmentationPLModel(Model):
    def __init__(self, cfg):
        super(FrameFieldSegmentationPLModel, self).__init__(cfg)
        self.loss_norm_is_initializated = False
        self.use_mixup = (
            False if "use_mixup" not in cfg.pl_model else cfg.pl_model.use_mixup
        )
        self.mixup_alpha = (
            self.cfg.pl_model.mixup_alpha if "mixup_alpha" in cfg.pl_model else -1
        )
        if "set_encoder_trainable" in self.cfg.pl_model:
            self.set_encoder_trainable(
                trainable=self.cfg.pl_model.set_encoder_trainable
            )
        if "set_decoder_trainable" in self.cfg.pl_model:
            self.set_decoder_trainable(
                trainable=self.cfg.pl_model.set_decoder_trainable
            )
        if "set_seg_module_trainable" in self.cfg.pl_model:
            self.set_seg_module_trainable(
                trainable=self.cfg.pl_model.set_seg_module_trainable
            )
        if "set_crossfield_trainable" in self.cfg.pl_model:
            self.set_crossfield_trainable(
                trainable=self.cfg.pl_model.set_crossfield_trainable
            )

    def get_loss_function(self) -> MultiLoss:
        return build_loss_from_config(self.cfg)

    def set_encoder_trainable(self, trainable=False):
        if hasattr(self.model.segmentation_model, "encoder"):
            return self.set_model_component_trainable(
                self.model.segmentation_model.encoder, "Encoder", trainable=trainable
            )
        elif hasattr(self.model.segmentation_model, "set_model_trainable"):
            return self.model.segmentation_model.set_model_trainable(
                trainable=trainable
            )
        else:
            raise NotImplementedError("Set model trainable not implemented.")

    def set_decoder_trainable(self, trainable=False):
        if hasattr(self.model.segmentation_model, "decoder"):
            return self.set_model_component_trainable(
                self.model.segmentation_model.decoder, "Decoder", trainable=trainable
            )

    def set_seg_module_trainable(self, trainable=False):
        return self.set_model_component_trainable(
            self.model.seg_module, "Seg Module", trainable=trainable
        )

    def set_crossfield_trainable(self, trainable=False):
        return self.set_model_component_trainable(
            self.crossfield_module, "Crossfield", trainable=trainable
        )

    def set_model_component_trainable(self, component, component_name, trainable=False):
        for child in component.children():
            for param in child.parameters():
                param.requires_grad = trainable
        print(f"{component_name} weights set to trainable={trainable}")
        return

    def set_all_but_crossfield_trainable(self, trainable=False):
        self.set_encoder_trainable(trainable=trainable)
        self.set_decoder_trainable(trainable=trainable)
        self.set_seg_module_trainable(trainable=trainable)

    def compute_iou_metrics(self, y_pred, y_true, step_prefix="train"):
        """Compute IoU metrics at different thresholds"""
        iou_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        for iou_threshold in iou_thresholds:
            iou = metrics.iou(
                y_pred.reshape(y_pred.shape[0], -1),
                y_true.reshape(y_true.shape[0], -1),
                threshold=iou_threshold,
            )
            mean_iou = torch.mean(iou)
            # Log with step prefix for proper grouping
            self.log(
                f"iou/{step_prefix}_IoU_{iou_threshold}",
                mean_iou,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def compute_loss_norms(self, dl, total_batches):
        self.loss_function.reset_norm()
        t = None
        if self.local_rank == 0:
            t = tqdm(total=total_batches, desc="Init loss norms", leave=False)
        batch_i = 0
        while batch_i < total_batches:
            batch = next(iter(dl))
            batch = (
                tensor_utils.batch_to_cuda(batch)
                if self.cfg.device == "cuda"
                else batch
            )
            pred = self.model(batch["image"])
            self.loss_function.update_norm(pred, batch, batch["image"].shape[0])
            if t is not None:
                t.update(1)
            batch_i += 1
        world_size = self.get_world_size()
        if world_size > 1:
            self.loss_function.sync(world_size)

    def get_world_size(self):
        if self.cfg.device == "cpu" or self.cfg.pl_trainer.devices == 0:
            return 1
        elif isinstance(self.cfg.pl_trainer.devices, list):
            return len(self.cfg.pl_trainer.devices)
        elif self.cfg.pl_trainer.devices == -1:
            return torch.cuda.device_count()
        else:
            return self.cfg.pl_trainer.devices

    def training_step(self, batch, batch_idx):
        batch["image"] = (
            batch["image"]
            if self.gpu_train_transform is None
            else self.gpu_train_transform(batch["image"])
        )
        pred = self.model(batch["image"])

        if self.use_mixup:
            (
                batch["mixup_image"],
                batch["mixup_y_a"],
                batch["mixup_y_b"],
                batch["mixup_lam"],
            ) = mixup_data(
                batch["image"],
                batch["gt_polygons_image"][:, 0, ...],
                self.mixup_alpha,
                use_cuda=False if self.cfg.device == "cpu" else True,
            )
            batch["mixup_pred"] = self.model(batch["mixup_image"])

        loss, individual_metrics_dict, extra_dict = self.loss_function(
            pred, batch, epoch=self.current_epoch
        )

        # Log main loss
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log individual losses with proper prefixing
        for key, value in individual_metrics_dict.items():
            self.log(f"losses/train_{key}", value, on_step=True, on_epoch=True)

        # Compute and log IoU metrics
        if "seg" in pred:
            y_pred = pred["seg"][:, 0, ...]
            y_true = batch["gt_polygons_image"][:, 0, ...]
            self.compute_iou_metrics(y_pred, y_true, step_prefix="train")

            # Log torchmetrics if available
            if hasattr(self, "train_metrics"):
                metrics_output = self.train_metrics(y_pred, y_true.long())
                self.log_dict(metrics_output, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image = (
            batch["image"]
            if self.gpu_val_transform is None
            else self.gpu_val_transform(batch["image"])
        )
        pred = self.model(image)
        loss, individual_metrics_dict, extra_dict = self.loss_function(
            pred, batch, epoch=self.current_epoch
        )

        # Log main loss
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log individual losses with proper prefixing
        for key, value in individual_metrics_dict.items():
            self.log(f"losses/val_{key}", value, on_step=False, on_epoch=True)

        # Compute and log IoU metrics
        if "seg" in pred:
            y_pred = pred["seg"][:, 0, ...]
            y_true = batch["gt_polygons_image"][:, 0, ...]
            self.compute_iou_metrics(y_pred, y_true, step_prefix="val")

            # Log torchmetrics if available
            if hasattr(self, "val_metrics"):
                metrics_output = self.val_metrics(y_pred, y_true.long())
                self.log_dict(metrics_output, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step — runs once after training via trainer.test()."""
        image = (
            batch["image"]
            if self.gpu_test_transform is None
            else self.gpu_test_transform(batch["image"])
        )
        pred = self.model(image)
        loss, individual_metrics_dict, extra_dict = self.loss_function(
            pred, batch, epoch=self.current_epoch
        )

        # Log main loss
        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log individual losses with proper prefixing
        for key, value in individual_metrics_dict.items():
            self.log(f"losses/test_{key}", value, on_step=False, on_epoch=True)

        # Compute and log IoU metrics
        if "seg" in pred:
            y_pred = pred["seg"][:, 0, ...]
            y_true = batch["gt_polygons_image"][:, 0, ...]
            self.compute_iou_metrics(y_pred, y_true, step_prefix="test")

            # Log torchmetrics if available
            if hasattr(self, "test_metrics"):
                metrics_output = self.test_metrics(y_pred, y_true.long())
                self.log_dict(metrics_output, on_step=False, on_epoch=True)

        return loss

    # Removed training_epoch_end and validation_epoch_end
    # Lightning 2.0+ handles aggregation automatically

    def _process_test(self, batch):
        with torch.no_grad():
            batch_predictions = self.model(batch["image"])
        seg_batch, crossfield_batch = batch_predictions.values()
        return seg_batch, crossfield_batch

    def _process_test_like_inference_processor(self, batch):
        ids = torch.unique_consecutive(batch["tile_image_idx"])
        with torch.no_grad():
            batch_predictions = self.model(batch["tiles"])
        seg_batch, crossfield_batch = batch_predictions.values()
        seg_batch = torch.cat(
            [
                self._integrate_tiles(
                    seg_batch[batch["tile_image_idx"] == tile_id],
                    copy.deepcopy(batch["tiler_object_list"][idx]),
                    batch["original_shape"][idx],
                    pad_if_needed=True,
                )
                for idx, tile_id in enumerate(ids)
            ],
            dim=0,
        )
        crossfield_batch = torch.cat(
            [
                self._integrate_tiles(
                    crossfield_batch[batch["tile_image_idx"] == tile_id],
                    copy.deepcopy(batch["tiler_object_list"][idx]),
                    batch["original_shape"][idx],
                    pad_if_needed=True,
                )
                for idx, tile_id in enumerate(ids)
            ],
            dim=0,
        )
        return seg_batch, crossfield_batch

    def _integrate_tiles(
        self, tensor_tiles: torch.Tensor, tiler, original_shape, pad_if_needed=True
    ):
        merger = TileMerger(
            tiler.target_shape,
            tensor_tiles.shape[1],
            tiler.weight,
            device=tensor_tiles.device,
        )
        merger.integrate_batch(tensor_tiles, tiler.crops)
        merged = merger.merge()
        if not pad_if_needed:
            return merged.unsqueeze(0)
        return K.center_crop(merged.unsqueeze(0), size=original_shape)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        seg_batch, crossfield_batch = (
            self._process_test_like_inference_processor(batch)
            if (
                hasattr(self.cfg, "use_inference_processor")
                and self.cfg.use_inference_processor
            )
            else self._process_test(batch)
        )
        return seg_batch, crossfield_batch
