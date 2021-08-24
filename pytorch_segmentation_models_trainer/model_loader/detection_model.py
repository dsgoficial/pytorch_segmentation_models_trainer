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

from pytorch_segmentation_models_trainer.model_loader.model import Model
from torch.utils.data import DataLoader
import torch
from torchvision.ops import box_iou


def collate_fn(batch):
    return tuple(zip(*batch))


def _evaluate_iou(target, pred):
    """Evaluate intersection over union (IOU) for target from dataset and output prediction from model."""
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
        )

    def training_step(self, batch, batch_idx):
        # images = batch.pop('images'
        # tensorboard_logs = {'acc': {'train': acc}}
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # return {'loss': loss, 'log': tensorboard_logs}
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # separate losses
        loss_dict = self.model(images, targets)
        # total loss
        losses = sum(loss for loss in loss_dict.values())

        return {"loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    def validation_step(self, batch, batch_idx):
        # loss, acc = self.compute(batch)
        # tensorboard_logs = {'acc': {'val': acc}}
        # self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # return {'val_loss': loss, 'log': tensorboard_logs}
        images, targets, image_ids = batch
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def training_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([
        #     torch.tensor(x['log']['acc']['train']) for x in outputs]).mean()
        # tensorboard_logs = {
        #     'avg_loss': {'train' : avg_loss},
        #     'avg_acc': {'train' : avg_acc},
        # }
        # self.log_dict(tensorboard_logs, logger=True)
        pass

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}


class InstanceSegmentationPLModel(ObjectDetectionPLModel):
    def __init__(self, cfg):
        super(InstanceSegmentationPLModel, self).__init__(cfg)
