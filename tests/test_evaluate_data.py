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

import os
from typing import List, Tuple
import unittest

from shapely.geometry import Polygon


from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.tools.evaluation import evaluate_data, matching
from tests.utils import load_geometry_list_from_geojson

current_dir = os.path.dirname(__file__)
matching_root_dir = os.path.join(current_dir, "testing_data", "data", "matching_data")


def _load_test_data() -> Tuple[List[Polygon], List[Polygon], List[Polygon]]:
    reference_polygons = load_geometry_list_from_geojson(
        os.path.join(matching_root_dir, "reference_polygons.geojson")
    )
    candidate_polygons = load_geometry_list_from_geojson(
        os.path.join(matching_root_dir, "candidate_polygons.geojson")
    )
    expected_reference_matches = load_geometry_list_from_geojson(
        os.path.join(matching_root_dir, "expected_reference_matches.geojson")
    )
    expected_reference_matches.sort(key=lambda x: x.area)
    return reference_polygons, candidate_polygons, expected_reference_matches


class Test_EvaluateData(unittest.TestCase):
    def test_compute_metrics_on_match_list_dict(self) -> None:
        (
            reference_polygons,
            candidate_polygons,
            expected_reference_matches,
        ) = _load_test_data()
        (
            matched_dict_list,
            unmatched_references_list,
            unmatched_targets_list,
        ) = matching.match_polygon_lists_by_iou(reference_polygons, candidate_polygons)
        evaluated_data = evaluate_data.compute_metrics_on_match_list_dict(
            matched_dict_list,
            metric_list=[
                metrics.polygon_iou,
                metrics.polis,
                metrics.hausdorff_distance,
                metrics.frechet_distance,
            ],
        )
        evaluated_data.sort(key=lambda x: x["reference"].area)
        self.assertAlmostEqual(evaluated_data[0]["iou"], 0.8051144144000144)
        self.assertAlmostEqual(evaluated_data[0]["intersection"], 54.76825052012902)
        self.assertAlmostEqual(evaluated_data[0]["union"], 68.02542538124013)
        self.assertAlmostEqual(evaluated_data[0]["polis"], 0.5665716069659223)
        self.assertAlmostEqual(
            evaluated_data[0]["hausdorff_distance"], 1.026199498916631
        )
        self.assertAlmostEqual(
            evaluated_data[0]["frechet_distance"], 1.1838713742401072
        )

        self.assertAlmostEqual(evaluated_data[1]["iou"], 0.7500752058692223)
        self.assertAlmostEqual(evaluated_data[1]["intersection"], 958.5448025063769)
        self.assertAlmostEqual(evaluated_data[1]["union"], 1277.9315927335183)
        self.assertAlmostEqual(evaluated_data[1]["polis"], 1.758035238733818)
        self.assertAlmostEqual(
            evaluated_data[1]["hausdorff_distance"], 5.0103238459222785
        )
        self.assertAlmostEqual(
            evaluated_data[1]["frechet_distance"], 10.658268896825216
        )
