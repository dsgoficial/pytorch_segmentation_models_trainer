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

def bilinear_interpolate(im, pos, batch=None):
    # From https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    x = pos[:, 1]
    y = pos[:, 0]
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0_int = torch.clamp(x0, 0, im.shape[-1] - 1)
    x1_int = torch.clamp(x1, 0, im.shape[-1] - 1)
    y0_int = torch.clamp(y0, 0, im.shape[-2] - 1)
    y1_int = torch.clamp(y1, 0, im.shape[-2] - 1)

    if batch is not None:
        Ia = im[batch, :, y0_int, x0_int]
        Ib = im[batch, :, y1_int, x0_int]
        Ic = im[batch, :, y0_int, x1_int]
        Id = im[batch, :, y1_int, x1_int]
    else:
        Ia = im[..., y0_int, x0_int].t()
        Ib = im[..., y1_int, x0_int].t()
        Ic = im[..., y0_int, x1_int].t()
        Id = im[..., y1_int, x1_int].t()

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", init_val=0, fmt=':f'):
        self.name = name
        self.init_val = init_val
        self.fmt = fmt
        self.val = self.avg = self.init_val
        self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.init_val
        self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class RunningDecayingAverage(object):
    """
    Updates average with val*(1 - decay) + avg*decay
    """
    def __init__(self, decay, init_val=0):
        assert 0 < decay < 1
        self.decay = decay
        self.init_val = init_val
        self.val = self.avg = self.init_val

    def reset(self):
        self.val = self.avg = self.init_val

    def update(self, val):
        self.val = val
        self.avg = (1 - self.decay)*val + self.decay*self.avg

    def get_avg(self):
        return self.avg

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
