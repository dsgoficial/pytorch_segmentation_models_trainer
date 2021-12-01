# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-25
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

from typing import List, Optional
import torch
from copy import deepcopy


def replace_activation(model, old_activation, new_activation):
    for child_name, child in model.named_children():
        if isinstance(child, type(old_activation)):
            setattr(model, child_name, deepcopy(new_activation))
        else:
            replace_activation(child, old_activation, new_activation)


def set_model_components_trainable(
    model: torch.nn.Module,
    trainable: bool = False,
    exception_list: Optional[List[str]] = None,
):
    """
    Sets the trainable status of the model's parameters.
    :param model: The model whose parameters will be set.
    :param trainable: The trainable status.
    :param exception_list: The list of parameters that will not be set.
    """
    exception_list = exception_list if exception_list is not None else []
    for name, param in model.named_parameters():
        if not any(exception in name for exception in exception_list):
            param.requires_grad = trainable
