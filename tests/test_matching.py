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
from typing import Any, Callable, List, Tuple
import unittest
from shapely.geometry.base import BaseGeometry
import torch
import geopandas
import numpy as np
from shapely.geometry import Polygon
from parameterized import parameterized

from pytorch_segmentation_models_trainer.tools.evaluation import matching
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


class Test_Matching(unittest.TestCase):
    @parameterized.expand(
        [
            (matching.match_polygon_lists_by_iou,),
            (matching.match_polygon_lists_by_hausdorff_distance,),
            (matching.match_polygon_lists_by_frechet_distance,),
        ]
    )
    def test_match_polygon_lists_method(self, method: Callable) -> None:
        (
            reference_polygons,
            candidate_polygons,
            expected_reference_matches,
        ) = _load_test_data()
        matched_dict_list, unmatched_references_list, unmatched_targets_list = method(
            reference_polygons, candidate_polygons
        )
        self.assertEqual(len(matched_dict_list), len(expected_reference_matches))
        self.assertEqual(len(unmatched_references_list), 3)
        self.assertEqual(len(unmatched_targets_list), 0)
        for idx, matched_dict in enumerate(
            sorted(matched_dict_list, key=lambda x: x["reference"].area)
        ):
            self.assertTrue(
                matched_dict["reference"].equals(expected_reference_matches[idx])
            )
