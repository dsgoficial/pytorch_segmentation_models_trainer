# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-30
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
#%%
import unittest
import numpy as np
import torch

import matplotlib.pyplot as plt

from pytorch_segmentation_models_trainer.utils.math_utils import (
    compute_crossfield_c0c2,
    compute_crossfield_uv,
)


class Test_MathUtils(unittest.TestCase):
    def test_compute_crossfield_c0c2(self) -> None:
        return True
