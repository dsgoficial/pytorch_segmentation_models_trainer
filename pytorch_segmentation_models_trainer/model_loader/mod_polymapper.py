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
import os
from collections import OrderedDict
from logging import log
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.init
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.custom_models.rnn.polygon_rnn import PolygonRNN
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils import (
    object_detection_utils,
    polygonrnn_utils,
    tensor_utils,
)
from torch import nn
from torch.utils.data import DataLoader

current_dir = os.path.dirname(__file__)


class GenericPolyMapperPLModel(Model):
    def __init__(self, cfg, grid_size=28):
        super(GenericPolyMapperPLModel, self).__init__(cfg)
        self.train_ds = instantiate(self.cfg.train_dataset.dataset, _recursive_=True)
        self.val_ds = instantiate(self.cfg.val_dataset.dataset, _recursive_=True)
        self.perform_evaluation = False

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=self.cfg.train_dataset.data_loader.shuffle,
            num_workers=self.cfg.train_dataset.data_loader.num_workers,
            pin_memory=self.cfg.train_dataset.data_loader.pin_memory
            if "pin_memory" in self.cfg.train_dataset.data_loader
            else True,
            drop_last=self.cfg.train_dataset.data_loader.drop_last
            if "drop_last" in self.cfg.train_dataset.data_loader
            else True,
            prefetch_factor=self.cfg.train_dataset.data_loader.prefetch_factor
            if "prefetch_factor" in self.cfg.train_dataset.data_loader
            else 4 * self.hyperparameters.batch_size,
            collate_fn=self.train_ds.collate_fn
            if hasattr(self.train_ds, "collate_fn")
            else self.train_ds.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.hyperparameters.batch_size,
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
            prefetch_factor=self.cfg.val_dataset.data_loader.prefetch_factor
            if "prefetch_factor" in self.cfg.val_dataset.data_loader
            else 4 * self.hyperparameters.batch_size,
            collate_fn=self.val_ds.collate_fn
            if hasattr(self.val_ds, "collate_fn")
            else self.val_ds.val_dataset.collate_fn,
        )

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
                        step_type: torch.stack([x["log"][key] for x in outputs]).mean()
                    }
                }
            )
        if self.perform_evaluation:
            for metric_key in outputs[0]["metrics"].keys():
                tensorboard_logs.update(
                    {
                        f"avg_{metric_key}": {
                            step_type: torch.stack(
                                [x["metrics"][metric_key] for x in outputs]
                            ).mean()
                        }
                    }
                )
        return tensorboard_logs

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict, acc = self.model(images, targets)
        detached_loss_dict = {k: v.detach() for k, v in loss_dict.items()}
        return {
            "loss": sum(loss for loss in loss_dict.values()),
            "log": detached_loss_dict,
            "progress_bar": detached_loss_dict,
        }

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()
        loss_dict, acc = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        self.model.eval()
        outputs = self.model(images)
        return_dict = {
            "loss": loss,
            "log": {k: v.detach() for k, v in loss_dict.items()},
        }
        if self.perform_evaluation:
            metrics_dict_item = self.evaluate_output(batch, outputs)
            return_dict.update(metrics_dict_item)
        return return_dict

    def evaluate_output(self, batch, outputs):
        images, targets = batch
        targets_dict = polygonrnn_utils.target_list_to_dict(targets)
        outputs_dict = polygonrnn_utils.target_list_to_dict(outputs)

        gt_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            targets_dict["ta"],
            targets_dict["scale_h"],
            targets_dict["scale_w"],
            targets_dict["min_col"],
            targets_dict["min_row"],
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
        iou = lambda x: metrics.polygon_iou(x[0], x[1])
        output_tensor_iou = torch.tensor(
            list(map(iou, zip(predicted_polygon_list, gt_polygon_list)))
        )
        intersection = output_tensor_iou[:, 1]
        union = output_tensor_iou[:, 2]

        box_iou = torch.stack(
            [
                object_detection_utils.evaluate_box_iou(t, o)
                for t, o in zip(targets_dict["boxes"], outputs_dict["boxes"])
            ]
        ).mean()

        return {
            "polis": batch_polis,
            "intersection": intersection,
            "union": union,
            "box_iou": box_iou,
        }

    def training_epoch_end(self, outputs):
        tensorboard_logs = self._build_tensorboard_logs(outputs)
        self.log_dict(tensorboard_logs, logger=True)

    def validation_epoch_end(self, outputs):
        tensorboard_logs = self._build_tensorboard_logs(outputs, step_type="val")
        self.log_dict(tensorboard_logs, logger=True)
