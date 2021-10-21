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

import unittest
import torch
import numpy as np

from pytorch_segmentation_models_trainer.custom_metrics import metrics


class Test_TestMetrics(unittest.TestCase):
    def test_iou(self) -> None:
        y_pred = torch.tensor([[0, 0], [1, 0.5], [0.2, 0.8], [0.5, 0.5], [1, 1]])
        y_true = torch.tensor([[0, 0], [1, 1], [0, 0], [1, 1], [0, 0]])
        self.assertAlmostEqual(
            metrics.iou(y_pred, y_true, threshold=0.5).mean(), torch.tensor(0.3000)
        )
        self.assertAlmostEqual(
            metrics.iou(y_pred, y_true, threshold=0.9).mean(), torch.tensor(0.5000)
        )

    def test_vector_iou(self) -> None:
        polygon1 = [2, 0, 2, 2, 0, 0, 0, 2]
        polygon2 = [1, 1, 4, 1, 4, 4, 1, 4]
        self.assertAlmostEqual(metrics.polygon_iou(polygon1, polygon2), 0.083333333333)
