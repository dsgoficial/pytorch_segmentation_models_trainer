# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-30
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
 *   Code inspired by the one in                                           *
 *   https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/     *
 ****
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.io

from pytorch_segmentation_models_trainer.tools.visualization.crossfield_plot import (
    get_tensorboard_image_seg_display, get_image_plot_crossfield, plot_crossfield, save_poly_viz
)

def compute_crossfield_c0c2(u, v):
    c0 = np.power(u, 2) * np.power(v, 2)
    c2 = - (np.power(u, 2) + np.power(v, 2))
    crossfield = np.stack([c0.real, c0.imag, c2.real, c2.imag], axis=-1)
    return crossfield

def compute_crossfield_uv(c0c2):
    c0 = c0c2[..., 0] + 1j * c0c2[..., 1]
    c2 = c0c2[..., 2] + 1j * c0c2[..., 3]
    sqrt_c2_squared_minus_4c0 = np.sqrt(np.power(c2, 2) - 4 * c0)
    u_squared = (c2 + sqrt_c2_squared_minus_4c0) / 2
    v_squared = (c2 - sqrt_c2_squared_minus_4c0) / 2
    u = np.sqrt(u_squared)
    v = np.sqrt(v_squared)
    return u, v

if __name__=="__main__":
    image = torch.zeros((2, 3, 512, 512)) + 0.5
    seg = torch.zeros((2, 2, 512, 512))
    seg[:, 0, 100:200, 100:200] = 1
    crossfield = torch.zeros((2, 4, 512, 512))
    u_angle = 0.25
    v_angle = u_angle + np.pi / 2
    u = np.cos(u_angle) + 1j * np.sin(u_angle)
    v = np.cos(v_angle) + 1j * np.sin(v_angle)
    c0 = np.power(u, 2) * np.power(v, 2)
    c2 = - (np.power(u, 2) + np.power(v, 2))

    crossfield[:, 0, :, :] = c0.real
    crossfield[:, 1, :, :] = c0.imag
    crossfield[:, 2, :, :] = c2.real
    crossfield[:, 3, :, :] = c2.imag

    new_crossfield = crossfield.cpu().detach().numpy().transpose(0, 2, 3, 1)
    # fig, ax = plt.subplots()
    # plot_crossfield(ax, new_crossfield[1], crossfield_stride=10, alpha=1.0, width=0.5, add_scale=1)
    # plt.show()

    # image_seg_display = get_tensorboard_image_seg_display(image, seg, crossfield=crossfield, crossfield_stride=100)
    # image_seg_display = image_seg_display.cpu().numpy().transpose(0, 2, 3, 1)
    # skimage.io.imsave("image_seg_display.png", image_seg_display[0])
    # skimage.io.imsave("image_cross_display.png", image_seg_display[1])
    img = np.moveaxis(image[0].numpy(), 0, -1)
    save_poly_viz(img, [], 'teste.png', seg=seg[0], crossfield=new_crossfield[0], dpi=10, crossfield_stride=100)
