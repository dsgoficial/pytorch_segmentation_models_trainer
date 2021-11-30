# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-31
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba -
                                    Cartographic Engineer @ Brazilian Army
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
import logging
import random
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

logging.getLogger("matplotlib").setLevel(level=logging.CRITICAL)


plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def denormalize_np_array(image: np.ndarray, mean=None, std=None) -> np.ndarray:
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

    return image * np.array(std)[..., None, None] + np.array(mean)[..., None, None]


def batch_denormalize_tensor(
    tensor, mean=None, std=None, inplace=False, clip_range=None, output_type=None
):
    """Denormalize a batched tensor image with batched mean and batched standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    Args:
        tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        mean (sequence): Tensor means of size (B, C).
        std (sequence): Tensor standard deviations of size (B, C).
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    mean = (
        mean
        if mean is not None
        else torch.tensor([0.485, 0.456, 0.406]).expand(
            tensor.shape[0], tensor.shape[1]
        )
    )
    std = (
        std
        if std is not None
        else torch.tensor([0.229, 0.224, 0.225]).expand(
            tensor.shape[0], tensor.shape[1]
        )
    )
    output_type = tensor.dtype if output_type is None else output_type
    assert (
        len(tensor.shape) == 4
    ), "tensor should have 4 dims (B, H, W, C) , not {}".format(len(tensor.shape))
    assert (
        len(mean.shape) == len(std.shape) == 2
    ), "mean and std should have 2 dims (B, C) , not {} and {}".format(
        len(mean.shape), len(std.shape)
    )
    assert (
        tensor.shape[1] == mean.shape[1] == std.shape[1]
    ), "tensor, mean and std should have the same number of channels, not {}, {} and {}".format(
        tensor.shape[-1], mean.shape[-1], std.shape[-1]
    )

    if not inplace:
        tensor = tensor.clone()

    mean = mean.to(tensor.dtype)
    std = std.to(tensor.dtype)
    tensor.mul_(std[..., None, None]).add_(mean[..., None, None])
    if clip_range is None:
        return tensor.to(output_type)
    min_pixel_value, max_pixel_value = clip_range
    return torch.clamp(
        max_pixel_value * tensor, min=min_pixel_value, max=max_pixel_value
    ).to(output_type)


def generate_visualization(fig_title=None, fig_size=None, font_size=16, **images):
    n = len(images)
    fig_size = (16, 5) if fig_size is None else fig_size
    fig, axarr = plt.subplots(1, n, figsize=fig_size)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=font_size)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    fig.subplots_adjust(top=0.8)
    return axarr, fig


def visualize_image_with_bboxes(
    image_batch: torch.Tensor,
    batch_boxes: Union[torch.Tensor, List[torch.Tensor]],
    width: int = 4,
):
    return [
        draw_bounding_boxes(img, boxes=boxes, width=width)
        for img, boxes in zip(image_batch, batch_boxes)
    ]


def generate_bbox_visualization(
    obj_det_axis,
    detection_dict: Dict[str, np.ndarray],
    linewidth: int = 2,
    show_scores: bool = False,
    colors: Optional[List] = None,
    boxes_key: str = "boxes",
    labels_key: str = "labels",
    scores_key: str = "scores",
) -> None:
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)] if colors is None else colors
    labels = set(label for label in detection_dict[labels_key])
    color_dict = {label: color for label, color in zip(labels, colors)}
    for idx, (x1, y1, x2, y2) in enumerate(detection_dict[boxes_key]):
        box_h = y2 - y1
        box_w = x2 - x1
        color = color_dict[detection_dict[labels_key][idx]]
        bbox = patches.Rectangle(
            (x1, y1),
            box_w,
            box_h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        obj_det_axis.add_patch(bbox)
        if not show_scores:
            continue
        score = 100 * detection_dict[scores_key][idx]
        obj_det_axis.text(
            x1,
            y1,
            s=f"{score:.2f}%",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )
