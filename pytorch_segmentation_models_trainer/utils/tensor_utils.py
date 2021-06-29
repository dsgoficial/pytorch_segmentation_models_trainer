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

import numpy as np
import torch
import kornia
from kornia.filters.kernels import (get_diff_kernel2d,
                                    get_diff_kernel2d_2nd_order,
                                    get_sobel_kernel2d,
                                    get_sobel_kernel2d_2nd_order,
                                    normalize_kernel2d)


class TensorPoly(object):
    def __init__(self, pos, poly_slice, batch, batch_size, is_endpoint=None):
        """
        :param pos:
        :param poly_slice:
        :param batch: Batch index for each node
        :param is_endpoint: One value per node. If true, that node is an endpoint and is thus part of an open polyline
        """
        assert pos.shape[0] == batch.shape[0]
        self.pos = pos
        self.poly_slice = poly_slice
        self.batch = batch
        self.batch_size = batch_size
        self.is_endpoint = is_endpoint
        self.to_padded_index = None  # No pad initially
        self.to_unpadded_poly_slice = None

    @property
    def num_nodes(self):
        return self.pos.shape[0]

    def to(self, device):
        self.pos = self.pos.to(device)
        self.poly_slice = self.poly_slice.to(device)
        self.batch = self.batch.to(device)
        if self.is_endpoint is not None:
            self.is_endpoint = self.is_endpoint.to(device)
        if self.to_padded_index is not None:
            self.to_padded_index = self.to_padded_index.to(device)
        if self.to_unpadded_poly_slice is not None:
            self.to_unpadded_poly_slice = self.to_unpadded_poly_slice.to(device)

def polygons_to_tensorpoly(polygons_batch):
    """
    Parametrizes N polygons into a 1d grid to be used in 1d conv later on:
    - pos (n1+n2+..., 2) concatenation of polygons vertex positions
    - poly_slice (poly_count, 2) polygon vertex slices [start, end)

    :param polygons_batch: Batch of polygons: [[(n1, 2), (n2, 2), ...], ...]
    :return: TensorPoly(pos, poly_slice, batch, batch_size, is_endpoint)
    """
    # TODO: If there are no polygons
    batch_size = len(polygons_batch)
    is_endpoint_list = []
    batch_list = []
    polygon_list = []
    for i, polygons in enumerate(polygons_batch):
        for polygon in polygons:
            if not np.max(np.abs(polygon[0] - polygon[-1])) < 1e-6:
                # Polygon is open
                is_endpoint = np.zeros(polygon.shape[0], dtype=np.bool)
                is_endpoint[0] = True
                is_endpoint[-1] = True
            else:
                # Polygon is closed, remove last redundant point
                polygon = polygon[:-1, :]
                is_endpoint = np.zeros(polygon.shape[0], dtype=np.bool)
            batch = i * np.ones(polygon.shape[0], dtype=np.long)
            is_endpoint_list.append(is_endpoint)
            batch_list.append(batch)
            polygon_list.append(polygon)
    pos = np.concatenate(polygon_list, axis=0)
    is_endpoint = np.concatenate(is_endpoint_list, axis=0)
    batch = np.concatenate(batch_list, axis=0)

    slice_start = 0
    poly_slice = np.empty((len(polygon_list), 2), dtype=np.long)
    for i, polygon in enumerate(polygon_list):
        slice_end = slice_start + polygon.shape[0]
        poly_slice[i][0] = slice_start
        poly_slice[i][1] = slice_end
        slice_start = slice_end
    pos = torch.tensor(pos, dtype=torch.float)
    is_endpoint = torch.tensor(is_endpoint, dtype=torch.bool)
    poly_slice = torch.tensor(poly_slice, dtype=torch.long)
    batch = torch.tensor(batch, dtype=torch.long)
    tensorpoly = TensorPoly(pos=pos, poly_slice=poly_slice, batch=batch, batch_size=batch_size, is_endpoint=is_endpoint)
    return tensorpoly


def _get_to_padded_index(poly_slice, node_count, padding):
    """
    Pad each polygon with a cyclic padding scheme on both sides.
    Increases length of x by (padding[0] + padding[1])*polygon_count values.

    :param poly_slice:
    :param padding:
    :return:
    """
    assert len(poly_slice.shape) == 2, "poly_slice should have shape (poly_count, 2), not {}".format(poly_slice.shape)
    poly_count = poly_slice.shape[0]
    range_tensor = torch.arange(
        node_count,
        device=poly_slice.device
    )
    to_padded_index = torch.empty(
        (node_count + (padding[0] + padding[1])*poly_count, ),
        dtype=torch.long,
        device=poly_slice.device
    )
    to_unpadded_poly_slice = torch.empty_like(poly_slice)
    start = 0
    for poly_i in range(poly_count):
        poly_indices = range_tensor[poly_slice[poly_i, 0]:poly_slice[poly_i, 1]]

        # Repeat poly_indices if necessary when padding exceeds polygon length
        vertex_count = poly_indices.shape[0]
        left_repeats = padding[0] // vertex_count
        left_padding_remaining = padding[0] % vertex_count
        right_repeats = padding[1] // vertex_count
        right_padding_remaining = padding[1] % vertex_count
        total_repeats = left_repeats + right_repeats
        poly_indices = poly_indices.repeat(total_repeats + 1)  # +1 includes the original polygon

        poly_indices = torch.cat(
            [poly_indices[-left_padding_remaining:], poly_indices, poly_indices[:right_padding_remaining]
            ]
        ) if left_padding_remaining else torch.cat(
            [poly_indices, poly_indices[:right_padding_remaining]]
        )

        end = start + poly_indices.shape[0]
        to_padded_index[start:end] = poly_indices
        to_unpadded_poly_slice[poly_i, 0] = start  # Init value
        to_unpadded_poly_slice[poly_i, 1] = end  # Init value
        start = end
    to_unpadded_poly_slice[:, 0] += padding[0]  # Shift all inited values to the right one
    to_unpadded_poly_slice[:, 1] -= padding[1]  # Shift all inited values to the right one
    return to_padded_index, to_unpadded_poly_slice


def tensorpoly_pad(tensorpoly, padding):
    to_padded_index, to_unpadded_poly_slice = _get_to_padded_index(tensorpoly.poly_slice, tensorpoly.num_nodes, padding)
    tensorpoly.to_padded_index = to_padded_index
    tensorpoly.to_unpadded_poly_slice = to_unpadded_poly_slice
    return tensorpoly

class SpatialGradient(torch.nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel or Scharr
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        input = torch.rand(1, 3, 4, 4)
        output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self,
                 mode: str = 'sobel',
                 order: int = 1,
                 normalized: bool = True,
                 coord: str = "xy",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float) -> None:
        super(SpatialGradient, self).__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode
        self.dtype = dtype
        self.kernel: torch.Tensor = get_spatial_gradient_kernel2d(mode, order, coord)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)
        # Pad with "replicate for spatial dims, but with zeros for channel
        self.spatial_pad = [self.kernel.size(1) // 2,
                            self.kernel.size(1) // 2,
                            self.kernel.size(2) // 2,
                            self.kernel.size(2) // 2]
        # # Prepare kernel
        # self.kernel: torch.Tensor = self.kernel.to(device).to(dtype).detach()
        # self.kernel: torch.Tensor = self.kernel.unsqueeze(1).unsqueeze(1)
        # self.kernel: torch.Tensor = self.kernel.flip(-3)
        return
    
    def prepare_kernel(self, device, dtype):
        kernel = self.kernel.to(device).to(dtype).detach()
        kernel = kernel.unsqueeze(1).unsqueeze(1)
        kernel = kernel.flip(-3)
        return kernel

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'order=' + str(self.order) + ', ' + \
            'normalized=' + str(self.normalized) + ', ' + \
            'mode=' + self.mode + ')'

    def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(inp):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(inp)))
        if not len(inp.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(inp.shape))
        # prepare kernel
        b, c, h, w = inp.shape

        # convolve inp tensor with sobel kernel
        out_channels: int = 3 if self.order == 2 else 2
        padded_inp: torch.Tensor = torch.nn.functional.pad(inp.reshape(b * c, 1, h, w),
                                                           self.spatial_pad, 'replicate')[:, :, None]
        kernel = self.prepare_kernel(padded_inp.device, self.dtype)
        return torch.nn.functional.conv3d(padded_inp, kernel, padding=0).view(b, c, out_channels, h, w)

def get_scharr_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [-47., 0., 47.],
        [-162., 0., 162.],
        [-47., 0., 47.],
    ])


def get_scharr_kernel2d(coord: str = "xy") -> torch.Tensor:
    assert coord == "xy" or coord == "ij"
    kernel_x: torch.Tensor = get_scharr_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    if coord == "xy":
        return torch.stack([kernel_x, kernel_y])
    elif coord == "ij":
        return torch.stack([kernel_y, kernel_x])


def get_spatial_gradient_kernel2d(mode: str, order: int, coord: str = "xy") -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff', 'scharr']:
        raise TypeError("mode should be either sobel, diff or scharr. Got {}".format(mode))
    if order not in [1, 2]:
        raise TypeError("order should be either 1 or 2\
                         Got {}".format(order))
    if mode == 'sobel' and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    elif mode == 'scharr' and order == 1:
        kernel = get_scharr_kernel2d(coord)
    else:
        raise NotImplementedError("")
    return kernel

def batch_to_cuda(batch):
    # Send data to computing device:
    for key, item in batch.items():
        if hasattr(item, "cuda"):
            batch[key] = item.cuda(non_blocking=True)
    return batch