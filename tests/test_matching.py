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
from typing import Callable, List, Tuple
import unittest
import json
from shapely.geometry import Polygon, Point
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

    def test_per_vertex_error_list(self) -> None:
        gt_polygon = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        pred_polygon = Polygon(
            [(0.1, 0.1), (0.1, 2.1), (30, 30), (2.4, 2.4), (2.1, 0.1), (0.1, 0.1)]
        )
        output_dict_list = matching.per_vertex_error_list(gt_polygon, pred_polygon)
        self.assertEqual(len(output_dict_list), 4)
        expected_json_str = '[{"gt_vertex": "POINT (0 0)", "pred_vertex": "POINT (0.1 0.1)", "distance": 0.14142135623730953}, {"gt_vertex": "POINT (0 2)", "pred_vertex": "POINT (0.1 2.1)", "distance": 0.14142135623730956}, {"gt_vertex": "POINT (2 2)", "pred_vertex": "POINT (2.4 2.4)", "distance": 0.5656854249492379}, {"gt_vertex": "POINT (2 0)", "pred_vertex": "POINT (2.1 0.1)", "distance": 0.14142135623730956}]'
        output_transformed_dict_list = [
            {k: (v.wkt if isinstance(v, Point) else v) for k, v in d.items()}
            for d in output_dict_list
        ]
        self.assertEqual(expected_json_str, json.dumps(output_transformed_dict_list))

    def test_match_polygon_lists_by_criteria_validates_empty_target_and_criteria(self):
        reference = [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]

        with self.assertRaises(AssertionError):
            matching.match_polygon_lists_by_criteria(
                reference,
                reference,
                lambda _target, _ref: 1.0,
                match_criteria="bad",
            )
        with self.assertRaises(ValueError):
            matching.match_polygon_lists_by_criteria(
                reference,
                [Polygon()],
                lambda _target, _ref: 1.0,
            )

    def test_match_polygon_lists_by_criteria_thresholds_and_replacement(self):
        reference = [Polygon([(0, 0), (10, 0), (10, 10), (0, 0)])]
        target_low = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        target_high = Polygon([(0, 0), (9, 0), (9, 9), (0, 0)])
        target_far = Polygon([(20, 20), (21, 20), (21, 21), (20, 20)])

        matched, unmatched_refs, unmatched_targets = (
            matching.match_polygon_lists_by_criteria(
                reference,
                [target_far],
                lambda _target, _ref: 1.0,
                match_criteria="max",
                match_threshold=0.5,
            )
        )
        self.assertEqual(matched, [])
        self.assertEqual(len(unmatched_refs), 1)
        self.assertEqual(len(unmatched_targets), 1)

        matched, unmatched_refs, unmatched_targets = (
            matching.match_polygon_lists_by_criteria(
                reference,
                [target_low],
                lambda _target, _ref: 0.1,
                match_criteria="max",
                match_threshold=0.5,
            )
        )
        self.assertEqual(matched, [])
        self.assertEqual(len(unmatched_refs), 1)
        self.assertEqual(len(unmatched_targets), 1)

        matched, _, _ = matching.match_polygon_lists_by_criteria(
            reference,
            [target_low, target_high],
            lambda target, _ref: target.area,
            match_criteria="max",
            match_threshold=-1,
        )
        self.assertTrue(matched[0]["target"].equals(target_high))

    def test_match_polygon_lists_by_min_criteria_replacement(self):
        reference = [Polygon([(0, 0), (10, 0), (10, 10), (0, 0)])]
        target_far = Polygon([(0, 0), (9, 0), (9, 9), (0, 0)])
        target_near = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])

        matched, unmatched_refs, unmatched_targets = (
            matching.match_polygon_lists_by_criteria(
                reference,
                [target_near],
                lambda _target, _ref: 0.1,
                match_criteria="min",
                match_threshold=0.5,
            )
        )
        self.assertEqual(matched, [])
        self.assertEqual(len(unmatched_refs), 1)
        self.assertEqual(len(unmatched_targets), 1)

        def asymmetric_metric(first, _second):
            return -1.0 if first.equals(reference[0]) else first.area

        matched, _, _ = matching.match_polygon_lists_by_criteria(
            reference,
            [target_far, target_near],
            asymmetric_metric,
            match_criteria="min",
            match_threshold=-1,
        )

        self.assertTrue(matched[0]["target"].equals(target_near))
