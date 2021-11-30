# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-11-16
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

from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.init
from hydra.utils import instantiate
from shapely.geometry import Polygon
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.utils import (
    object_detection_utils,
    polygonrnn_utils,
)
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MAP


class GenericPolyMapperPLModel(pl.LightningModule):
    def __init__(self, cfg, grid_size=28, perform_evaluation=True):
        super(GenericPolyMapperPLModel, self).__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.grid_size = grid_size
        self.perform_evaluation = perform_evaluation
        self.object_detection_train_ds = instantiate(
            self.cfg.train_dataset.object_detection, _recursive_=False
        )
        self.object_detection_val_ds = instantiate(
            self.cfg.val_dataset.object_detection, _recursive_=False
        )
        self.polygonrnn_train_ds = instantiate(
            self.cfg.train_dataset.polygon_rnn, _recursive_=False
        )
        self.polygonrnn_val_ds = instantiate(
            self.cfg.val_dataset.polygon_rnn, _recursive_=False
        )
        self.val_mAP = MAP()

    def get_model(self):
        model = instantiate(self.cfg.model, _recursive_=False)
        return model

    def get_optimizer(self):
        return instantiate(
            self.cfg.optimizer, params=self.parameters(), _recursive_=False
        )

    def configure_optimizers(self):
        # REQUIRED
        optimizer = self.get_optimizer()
        scheduler_list = []
        if "scheduler_list" not in self.cfg:
            return [optimizer], scheduler_list
        for item in self.cfg.scheduler_list:
            dict_item = dict(item)
            dict_item["scheduler"] = instantiate(
                item.scheduler, optimizer=optimizer, _recursive_=False
            )
            scheduler_list.append(dict_item)
        return [optimizer], scheduler_list

    def forward(self, x):
        return self.model(x)

    def get_train_dataloader(self, ds, batch_size):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=self.cfg.train_dataset.data_loader.shuffle,
            num_workers=self.cfg.train_dataset.data_loader.num_workers,
            pin_memory=self.cfg.train_dataset.data_loader.pin_memory
            if "pin_memory" in self.cfg.train_dataset.data_loader
            else True,
            drop_last=self.cfg.train_dataset.data_loader.drop_last
            if "drop_last" in self.cfg.train_dataset.data_loader
            else True,
            prefetch_factor=self.cfg.train_dataset.data_loader.prefetch_factor,
            collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
        )

    def get_val_dataloader(self, ds, batch_size):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=self.cfg.val_dataset.data_loader.shuffle
            if "shuffle" in self.cfg.val_dataset.data_loader
            else False,
            num_workers=self.cfg.val_dataset.data_loader.num_workers,
            pin_memory=self.cfg.val_dataset.data_loader.pin_memory
            if "pin_memory" in self.cfg.val_dataset.data_loader
            else True,
            drop_last=self.cfg.val_dataset.data_loader.drop_last
            if "drop_last" in self.cfg.val_dataset.data_loader
            else True,
            prefetch_factor=self.cfg.val_dataset.data_loader.prefetch_factor,
            collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
        )

    def train_dataloader(self) -> Dict[str, DataLoader]:
        return {
            "object_detection": self.get_train_dataloader(
                self.object_detection_train_ds,
                self.cfg.hyperparameters.object_detection_batch_size,
            ),
            "polygon_rnn": self.get_train_dataloader(
                self.polygonrnn_train_ds,
                self.cfg.hyperparameters.polygon_rnn_batch_size,
            ),
        }

    def val_dataloader(self) -> CombinedLoader:
        loader_dict = {
            "object_detection": self.get_val_dataloader(
                self.object_detection_val_ds,
                self.cfg.hyperparameters.object_detection_batch_size,
            ),
            "polygon_rnn": self.get_val_dataloader(
                self.polygonrnn_val_ds, self.cfg.hyperparameters.polygon_rnn_batch_size
            ),
        }
        combined_loaders = CombinedLoader(loader_dict, "max_size_cycle")
        return combined_loaders

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def _build_tensorboard_logs(self, outputs, step_type="train"):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_loss": {step_type: avg_loss}}
        if len(outputs) == 0:
            return tensorboard_logs
        for key in outputs[0]["log"].keys():
            tensorboard_logs.update(
                {
                    f"avg_{key}": {
                        step_type: torch.cat(
                            [
                                x["log"][key].unsqueeze(0).float()
                                if x["log"][key].shape == torch.Size([])
                                else x["log"][key].float()
                                for x in outputs
                            ]
                        ).mean()
                    }
                }
            )
        return tensorboard_logs

    def training_step(self, batch, batch_idx):
        obj_det_images, obj_det_targets, _ = batch["object_detection"]
        polygon_rnn_batch = batch["polygon_rnn"]
        loss_dict, acc = self.model(obj_det_images, obj_det_targets, polygon_rnn_batch)
        detached_loss_dict = {key: loss.detach() for key, loss in loss_dict.items()}
        detached_loss_dict.update({"acc": acc.detach()})
        loss = sum(loss for loss in loss_dict.values())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train_acc", acc, on_step=True, prog_bar=True, logger=True, sync_dist=False
        )
        return {"loss": loss, "log": detached_loss_dict}

    def validation_step(self, batch, batch_idx):
        obj_det_images, obj_det_targets, _ = batch["object_detection"]
        polygon_rnn_batch = batch["polygon_rnn"]
        self.model.train()
        with torch.no_grad():
            loss_dict, acc = self.model(
                obj_det_images, obj_det_targets, polygon_rnn_batch
            )
        loss = sum(loss for loss in loss_dict.values())
        detached_loss_dict = {key: loss.detach() for key, loss in loss_dict.items()}
        detached_loss_dict.update({"acc": acc.detach()})
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return_dict = {"loss": loss, "log": detached_loss_dict}
        if self.perform_evaluation:
            self.model.eval()
            outputs = self.model(obj_det_images)
            metrics_dict_item = self.evaluate_output(batch, outputs)
            return_dict["log"].update(metrics_dict_item)
        return return_dict

    def evaluate_output(
        self, batch, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[float, torch.Tensor]]:
        return_dict = dict()
        box_iou, mAP = self._evaluate_obj_det(outputs, batch)
        return_dict.update(mAP)
        batch_polis, intersection, union = self._compute_polygonrnn_metrics(
            outputs, batch["polygon_rnn"]
        )

        return_dict.update(
            {
                "polis": batch_polis,
                "intersection": intersection,
                "union": union,
                "box_iou": box_iou,
            }
        )
        return return_dict

    def _evaluate_obj_det(self, outputs, batch):
        obj_det_images, obj_det_targets, _ = batch["object_detection"]
        box_iou = torch.stack(
            [
                object_detection_utils.evaluate_box_iou(t, o)
                for t, o in zip(obj_det_targets, outputs)
            ]
        )
        mAP = self.val_mAP(outputs, obj_det_targets)

        return box_iou, mAP

    def _compute_polygonrnn_metrics(self, outputs, polygon_rnn_batch):
        outputs_dict = polygonrnn_utils.target_list_to_dict(outputs)
        gt_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            polygon_rnn_batch["ta"],
            polygon_rnn_batch["scale_h"],
            polygon_rnn_batch["scale_w"],
            polygon_rnn_batch["min_col"],
            polygon_rnn_batch["min_row"],
        )
        predicted_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            outputs_dict["polygonrnn_output"],
            outputs_dict["scale_h"],
            outputs_dict["scale_w"],
            outputs_dict["min_col"],
            outputs_dict["min_row"],
        )
        batch_polis = torch.from_numpy(
            metrics.batch_polis(predicted_polygon_list, gt_polygon_list)
        )

        def iou(x):
            return metrics.polygon_iou(x[0], x[1])

        output_tensor_iou = torch.tensor(
            list(map(iou, zip(predicted_polygon_list, gt_polygon_list)))
        )
        intersection = (
            output_tensor_iou[:, 1] if len(output_tensor_iou) > 0 else torch.tensor(0.0)
        )
        union = (
            output_tensor_iou[:, 2]
            if len(output_tensor_iou) > 0
            else torch.tensor(sum(Polygon(p).area for p in gt_polygon_list))
        )

        return batch_polis, intersection, union

    def training_epoch_end(self, outputs):
        tensorboard_logs = self._build_tensorboard_logs(outputs)
        self.log_dict(tensorboard_logs, logger=True)

    def validation_epoch_end(self, outputs):
        tensorboard_logs = self._build_tensorboard_logs(outputs, step_type="val")
        self.log_dict(tensorboard_logs, logger=True)
