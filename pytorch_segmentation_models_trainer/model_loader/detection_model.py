# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-08-16
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

import torch
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils import object_detection_utils
from torch.utils.data import DataLoader


class ObjectDetectionPLModel(Model):
    def __init__(self, cfg):
        super(ObjectDetectionPLModel, self).__init__(cfg)

    def get_loss_function(self):
        return None

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

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        loss_dict = self.model(images, targets)
        return {
            "loss": sum(loss for loss in loss_dict.values()),
            "log": loss_dict,
            "progress_bar": loss_dict,
        }

    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        self.model.train()
        loss_dict = self.model(images, targets)
        self.model.eval()
        outs = self.model(images)
        iou = torch.stack(
            [
                object_detection_utils.evaluate_box_iou(t, o)
                for t, o in zip(targets, outs)
            ]
        ).mean()
        return {
            "val_iou": iou,
            "loss": sum(loss for loss in loss_dict.values()),
            "log": loss_dict,
        }

    def training_epoch_end(self, outputs):
        tensorboard_logs = self._build_tensorboard_logs(outputs)
        self.log_dict(tensorboard_logs, logger=True)

    def _build_tensorboard_logs(self, outputs, step_type="train"):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_loss": {step_type: avg_loss}}
        if len(outputs) == 0:
            return tensorboard_logs
        for loss_key in outputs[0]["log"].keys():
            tensorboard_logs.update(
                {
                    f"avg_{loss_key}": {
                        step_type: torch.stack(
                            [x["log"][loss_key] for x in outputs]
                        ).mean()
                    }
                }
            )
        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        avg_iou = torch.stack([o["val_iou"] for o in outputs]).mean()
        logs = {"val_iou": avg_iou}
        tensorboard_logs = self._build_tensorboard_logs(outputs, step_type="val")
        self.log_dict(tensorboard_logs, logger=True)
        return {"avg_val_iou": avg_iou, "log": logs}


class InstanceSegmentationPLModel(ObjectDetectionPLModel):
    def __init__(self, cfg):
        super(InstanceSegmentationPLModel, self).__init__(cfg)
