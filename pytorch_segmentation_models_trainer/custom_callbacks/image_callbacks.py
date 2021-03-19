# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-10
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
import datetime
import io
import os
import math
import logging
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import rank_zero_only

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)


def denormalize_np_array(image: np.ndarray, \
    mean=None, std=None) -> np.ndarray:
    """Denormalizes normalized image. Used in visualization of augmented images.

    Args:
        image (np.ndarray): input image
        mean (list, optional): Mean used in normalization. 
            Defaults to [0.485, 0.456, 0.406].
        std (list, optional): Standard deviation used in normalization. 
            Defaults to [0.229, 0.224, 0.225].
    """
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    return image * np.array(mean)[..., None, None] + np.array(mean)[..., None, None]

class ImageSegmentationResultCallback(pl.callbacks.base.Callback):

    def __init__(self, n_samples: int, output_path: str = None, \
        normalized_input=True, norm_params=None) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.normalized_input = normalized_input
        self.output_path = None if output_path is None else output_path
        self.norm_params = norm_params if norm_params is not None else {}
        self.save_outputs = False

    def generate_visualization(self, fig_title=None, **images):
        n = len(images)
        fig = plt.figure(figsize=(16, 5))
        if fig_title is not None:
            fig.suptitle(fig_title, fontsize=16)
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        return plt, fig
    

    def prepare_image_to_plot(self, image):
        image = image.squeeze(0) if image.shape[0] == 1 else image
        image = denormalize_np_array(image, **self.norm_params) \
            if self.normalized_input else image
        return np.moveaxis(image, 0, -1) \
            if min(image.shape) == image.shape[0] else image
    
    def prepare_mask_to_plot(self, mask):
        return np.squeeze(mask).astype(np.float64)
    
    def log_data_to_tensorboard(self, saved_image, image_path, logger, current_epoch):
        image = Image.open(saved_image)
        data = np.array(image)
        data = np.moveaxis(data, -1, 0)
        data = torch.from_numpy(data)
        logger.experiment.add_image(image_path, data, current_epoch)

    def save_plot_to_disk(self, plot, image_name, current_epoch):
        image_name = Path(image_name).name.split('.')[0]
        report_path = os.path.join(
            self.output_path,
             'report_image_{name}_epoch_{epoch}_{date}.png'.format(
                name=image_name,
                date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                epoch=current_epoch
            )
        )
        plot.savefig(
            report_path,
            format='png'
        )
        return report_path

    def on_sanity_check_end(self, trainer, pl_module):
        self.save_outputs = True
        self.output_path = os.path.join(
            trainer.log_dir,
            'image_logs'
        )
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if not self.save_outputs:
            return
        val_ds = pl_module.val_dataloader().dataset
        device = pl_module.device
        logger = trainer.logger
        for i in range(self.n_samples):
            image, mask = val_ds[i].values()
            image = image.unsqueeze(0)
            image = image.to(device)
            predicted_mask = pl_module(image)
            image = image.to('cpu')
            predicted_mask = predicted_mask.to('cpu')
            plot_title = val_ds.get_path(i)
            plt_result, fig = self.generate_visualization(
                fig_title=plot_title,
                image=self.prepare_image_to_plot(image.numpy()),
                ground_truth_mask=self.prepare_mask_to_plot(mask.numpy()),
                predicted_mask=self.prepare_mask_to_plot(predicted_mask.numpy())
            )
            if self.save_outputs:
                saved_image = self.save_plot_to_disk(
                    plt_result,
                    plot_title,
                    trainer.current_epoch
                )
                self.log_data_to_tensorboard(
                    saved_image,
                    plot_title,
                    logger,
                    trainer.current_epoch
                )
            plt.close(fig)
        return
