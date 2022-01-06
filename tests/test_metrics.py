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
import json
import shapely.wkt
from shapely.geometry import Polygon, Point

from pytorch_segmentation_models_trainer.custom_metrics import metrics


class Test_Metrics(unittest.TestCase):
    def test_iou(self) -> None:
        y_pred = torch.tensor([[0, 0], [1, 0.5], [0.2, 0.8], [0.5, 0.5], [1, 1]])
        y_true = torch.tensor([[0, 0], [1, 1], [0, 0], [1, 1], [0, 0]])
        self.assertAlmostEqual(
            metrics.iou(y_pred, y_true, threshold=0.5).mean(), torch.tensor(0.3000)
        )
        self.assertAlmostEqual(
            metrics.iou(y_pred, y_true, threshold=0.9).mean(), torch.tensor(0.5000)
        )

    def test_polygon_iou(self) -> None:
        polygon1 = [2, 0, 2, 2, 0, 2, 0, 0]
        polygon2 = [1, 1, 4, 1, 4, 4, 1, 4]
        iou, _, __ = metrics.polygon_iou(polygon1, polygon2)
        self.assertAlmostEqual(iou, 0.083333333333)

    def test_polygon_iou_with_invalid_geom(self) -> None:
        polygon1 = [2, 0, 2, 2, 0, 2, 0, 0]
        polygon2 = [0, 0, 2, 2, 2, 0, 0, 2]  # invalid, bow-tie
        iou, _, __ = metrics.polygon_iou(polygon1, polygon2)
        self.assertAlmostEqual(iou, 0.5)

    def test_polis(self) -> None:
        polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        polygon2 = Polygon([(0, 0), (0, 2), (3, 3), (2, 0), (0, 0)])
        self.assertAlmostEqual(metrics.polis(polygon1, polygon1), 0.0)
        self.assertAlmostEqual(metrics.polis(polygon1, polygon2), 0.20466690944067711)
        self.assertAlmostEqual(
            metrics.polis(polygon1, polygon2), metrics.polis(polygon2, polygon1)
        )

    def test_polis_batch(self) -> None:
        batch1 = np.array(
            [
                [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)],
                [(0, 0), (0, 2.5), (2.5, 2.5), (2.5, 0), (0, 0)],
            ]
        )
        batch2 = np.array(
            [
                [(0, 0), (0, 2), (3, 3), (2, 0), (0, 0)],
                [(0, 0), (0, 2.5), (3.5, 3.5), (2.5, 0), (0, 0)],
            ]
        )
        np.testing.assert_array_almost_equal(
            metrics.batch_polis(batch1, batch2), np.array([0.20466691, 0.21010164])
        )

    def test_polygon_accuracy(self) -> None:
        polygon1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        polygon2 = Polygon([(0, 0), (0, 2), (30, 30), (2, 0), (0, 0)])
        self.assertAlmostEqual(
            metrics.polygon_accuracy(polygon1, polygon1), 1.0
        )  # 100% acc, same polygon
        self.assertAlmostEqual(
            metrics.polygon_accuracy(polygon1, polygon2), 0.8620689655172413
        )
        self.assertAlmostEqual(
            metrics.polygon_accuracy(polygon1, polygon2),
            metrics.polygon_accuracy(polygon2, polygon1),
        )

    def test_polygon_mean_max_tangent_angle_errors(self) -> None:
        polygon_gt = Polygon(
            [
                [159, -187],
                [159, -191],
                [161, -191],
                [161, -239],
                [133, -239],
                [133, -261],
                [100, -261],
                [100, -191],
                [116, -191],
                [116, -189],
                [143, -189],
                [143, -187],
                [159, -187],
            ]
        )
        polygon_pred = shapely.wkt.loads(
            "Polygon ((99.02707672120000382 -228.97312927249998893, 98.78768157960000451 -257.01211547849999306, 100.80204772950000347 -258.19812011720000555, 100.90475463869999828 -260.84509277339998334, 104.0643463134999962 -261.93560791020001943, 104.93199920649999513 -261.0681152344000111, 131.06982421879999379 -260.93005371089998334, 132.06973266599999306 -254.93026733400000694, 131.00732421879999379 -249.99267578120000621, 131.06338500979998685 -242.93661499020001315, 132.98735046389998615 -240.01264953610001385, 145.0406341553000118 -239.9593658446999882, 145.9404296875 -239.0595703125, 159.05958557130000486 -238.94041442869999514, 160.94689941409998823 -237.05310058590001177, 159.99050903319999861 -199.00949096680000139, 157.0296783446999882 -194.9703216553000118, 156.9431152344000111 -190.0568847655999889, 154.87965393069998754 -188.12034606930001246, 145.94723510739999028 -188.05276489260000972, 145.04902648929999032 -188.95097351070000968, 142.90028381350001041 -189.09971618649998959, 141.98739624020001315 -190.01260375979998685, 117.84110260010000104 -190.15890502930000139, 116.98544311519999894 -191.01411437990000763, 102.87686920170000349 -191.12321472170000902, 100.03376770020000208 -193.96626281740000763, 99.99893951420000349 -204.00112915040000416, 100.00197601319999308 -204.99798583980000899, 99.02707672120000382 -228.97312927249998893))"
        )
        self.assertAlmostEqual(
            metrics.polygon_mean_max_tangent_angle_errors(polygon_gt, polygon_gt), 0.0
        )
        self.assertAlmostEqual(
            metrics.polygon_mean_max_tangent_angle_errors(polygon_gt, polygon_pred),
            1.0386727876965822,
        )
