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
 *   Part of the code is from                                              *
 *   https://github.com/AlexMa011/pytorch-polygon-rnn                      *
 ****
"""
from collections import OrderedDict
from logging import log
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
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils import tensor_utils
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

class ObjectDetectionPLModel(Model):
    def __init__(self, cfg):
        super(ObjectDetectionPLModel, self).__init__(cfg)
    
    def get_model(self):
        pass
    
    def get_loss_function(self):
        return None
    
    def training_step(self, batch, batch_idx):
        loss, acc = self.compute(batch)
        tensorboard_logs = {'acc': {'train': acc}}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute(batch)
        tensorboard_logs = {'acc': {'val': acc}}
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'val_loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([
            torch.tensor(x['log']['acc']['train']) for x in outputs]).mean()
        tensorboard_logs = {
            'avg_loss': {'train' : avg_loss},
            'avg_acc': {'train' : avg_acc},
        }
        self.log_dict(tensorboard_logs, logger=True)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([
            torch.tensor(x['log']['acc']['val']) for x in outputs]).mean()
        tensorboard_logs = {
            'avg_loss': {'val' : avg_loss},
            'avg_acc': {'val' : avg_acc},
        }
        self.log_dict(tensorboard_logs, logger=True)
