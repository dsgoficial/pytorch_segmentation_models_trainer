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


class NaiveModPolyMapperPLModel(Model):
    def __init__(self, cfg, grid_size=28):
        super(NaiveModPolyMapperPLModel, self).__init__(cfg)

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        pass

    def compute(self, batch):
        pass

    def compute_loss_acc(self, batch, result):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass
