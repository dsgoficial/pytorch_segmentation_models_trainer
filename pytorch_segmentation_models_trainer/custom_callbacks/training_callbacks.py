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
import albumentations as A
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
#precision e recall com problema no pytorch lightning 1.2, 
# retirar e depois ver o que fazer
from torch.utils.data import DataLoader

from typing import List, Any

class WarmupCallback(pl.callbacks.base.Callback):
    def __init__(self, warmup_epochs=2) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.warmed_up = False

    def on_init_end(self, trainer):
        print(f"\nWarmupCallback initialization at epoch {trainer.current_epoch}.\n")
        if trainer.current_epoch > self.warmup_epochs - 1:
            self.warmed_up = True

    def on_train_start(self, trainer, pl_module):
        if not self.warmed_up:
            print(
                f"\nModel will warm up for {self.warmup_epochs} "
                "epochs. Freezing encoder weights.\n"
            )
            pl_module.set_encoder_trainable(trainable=False)

    def on_train_end(self, trainer, pl_module):
        if self.warmed_up:
            return
        if trainer.current_epoch >= self.warmup_epochs - 1:
            print(
                f"\nModel warm up completed in the end of epoch {trainer.current_epoch}. "
                "Unfreezing encoder weights.\n"
            )
            pl_module.set_encoder_trainable(trainable=True)
            self.warmed_up=True
