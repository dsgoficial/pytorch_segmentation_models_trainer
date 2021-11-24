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
from collections import OrderedDict
from logging import log
import os
from pathlib import Path
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.init
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.custom_models.rnn.polygon_rnn import PolygonRNN
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils, tensor_utils
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

current_dir = os.path.dirname(__file__)


class GenericPolyMapperPLModel(Model):
    def __init__(self, cfg, grid_size=28):
        super(GenericPolyMapperPLModel, self).__init__(cfg)
        self.train_ds = instantiate(self.cfg.train_dataset.dataset, _recursive_=True)
        self.val_ds = instantiate(self.cfg.val_dataset.dataset, _recursive_=True)

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

    def get_tensorboard_logs(self, input_dict, step_type="train"):
        return {k: {step_type: v} for k, v in input_dict.items()}

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict, acc = self.model(images, targets)
        tensorboard_logs = self.get_tensorboard_logs(loss_dict, step_type="train")
        return {
            "loss": sum(loss for loss in loss_dict.values()),
            "log": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()
        loss_dict, acc = self.model(images, targets)
        self.model.eval()
        outputs = self.model(images)
        # batch_polis, intersection, union = self.evaluate_output(batch, outputs)
        # pass

    def evaluate_output(self, batch, outputs):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass
