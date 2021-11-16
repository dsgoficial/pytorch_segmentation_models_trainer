# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-08-02
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
 *   Part of the code is from                                              *
 *   https://github.com/AlexMa011/pytorch-polygon-rnn                      *
 ****
"""
from collections import OrderedDict
from logging import log
import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.init
import torch.utils.model_zoo as model_zoo
import wget
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


class PolygonRNNPLModel(Model):
    def __init__(self, cfg, grid_size=28):
        super(PolygonRNNPLModel, self).__init__(cfg)
        self.val_seq_len = self.cfg.val_dataset.sequence_length
        self.grid_size = grid_size

    def get_model(self):
        return PolygonRNN(
            load_vgg=self.cfg.model.load_vgg if "load_vgg" in self.cfg.model else False
        )

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        result = self.compute(batch)
        loss, acc = self.compute_loss_acc(batch, result)
        tensorboard_logs = {"acc": {"train": acc}}
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_acc", acc, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return {"loss": loss, "log": tensorboard_logs}

    def compute(self, batch):
        output = self.model(batch["image"], batch["x1"], batch["x2"], batch["x3"])
        return output.contiguous().view(-1, self.grid_size * self.grid_size + 3)

    def compute_loss_acc(self, batch, result):
        target = batch["ta"].contiguous().view(-1)
        loss = self.loss_function(result, target)
        result_index = torch.argmax(result, 1)
        correct = (target == result_index).float().sum().item()
        acc = correct * 1.0 / target.shape[0]
        return loss, acc

    def validation_step(self, batch, batch_idx):
        result = self.compute(batch)
        loss, acc = self.compute_loss_acc(batch, result)
        test_output = self.model.test(batch["image"], self.val_seq_len)
        batch_polis, intersection, union = self.evaluate_batch(batch, test_output)
        tensorboard_logs = {"acc": {"val": acc}}
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
            "val_acc", acc, on_step=True, prog_bar=False, logger=True, sync_dist=True
        )
        return {
            "val_loss": loss,
            "log": tensorboard_logs,
            "polis": batch_polis,
            "intersection": intersection,
            "union": union,
        }

    def evaluate_batch(self, batch, result):
        gt_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            batch["ta"],
            batch["scale_h"],
            batch["scale_w"],
            batch["min_col"],
            batch["min_row"],
        )
        predicted_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            result,
            batch["scale_h"],
            batch["scale_w"],
            batch["min_col"],
            batch["min_row"],
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
        return batch_polis, intersection, union

    def training_epoch_end(self, outputs):
        tensorboard_logs = self._get_tensorboard_logs(outputs, "loss", "train")
        self.log_dict(tensorboard_logs, logger=True)

    def _get_tensorboard_logs(self, outputs, loss_name, step):
        avg_loss = torch.stack([x[loss_name] for x in outputs]).mean()
        avg_acc = torch.stack(
            [torch.tensor(x["log"]["acc"][step]) for x in outputs]
        ).mean()
        if step == "train":
            return {"avg_loss": {step: avg_loss}, "avg_acc": {step: avg_acc}}
        intersection = torch.stack([x["intersection"] for x in outputs]).sum()
        union = torch.stack([x["union"] for x in outputs]).sum()
        iou = intersection / union
        polis = torch.stack([x["polis"].float() for x in outputs]).mean()
        tensorboard_logs = {
            "avg_loss": {step: avg_loss},
            "avg_acc": {step: avg_acc},
            "iou": {step: iou},
            "polis": {step: polis},
        }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        tensorboard_logs = self._get_tensorboard_logs(outputs, "val_loss", "val")
        self.log_dict(tensorboard_logs, logger=True)
