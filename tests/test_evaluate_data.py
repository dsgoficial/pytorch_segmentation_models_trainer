# -*- coding: utf-8 -*-
"""
Unit tests for evaluate_data.py
"""

import unittest
from unittest.mock import MagicMock, patch
from shapely.geometry import Polygon, Point

from pytorch_segmentation_models_trainer.tools.evaluation.evaluate_data import (
    compute_metrics_on_match_list_dict,
    compute_vertex_errors_on_match_list_dict,
)


class TestEvaluateData(unittest.TestCase):
    def test_compute_metrics(self):
        # Mock metric function
        def mock_iou(ref, target):
            return (0.8, 80, 100)

        mock_iou.__name__ = "iou_metric"

        match_list = [
            {
                "reference": Polygon([(0, 0), (1, 0), (1, 1)]),
                "target": Polygon([(0, 0), (1, 0), (1, 1)]),
            }
        ]

        results = compute_metrics_on_match_list_dict(match_list, [mock_iou])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["iou"], 0.8)

    def test_compute_vertex_errors(self):
        match_list = [{"reference": Polygon(), "target": Polygon()}]

        # per_vertex_error_list is imported from matching, let's mock it at the module level
        with patch(
            "pytorch_segmentation_models_trainer.tools.evaluation.evaluate_data.per_vertex_error_list"
        ) as mock_error:
            mock_error.return_value = [{"point": Point(0, 0), "error": 0.1}]

            results = compute_vertex_errors_on_match_list_dict(
                match_list, convert_output_to_wkt=True
            )
            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0]["point"], str)
            self.assertIn("POINT", results[0]["point"])

    def test_compute_metrics_non_iou_tuple_uses_metric_name(self):
        def hausdorff_pair(ref, target):
            return (1.0, 2.0)

        match_list = [
            {
                "reference": Polygon([(0, 0), (1, 0), (1, 1)]),
                "target": Polygon([(0, 0), (1, 0), (1, 1)]),
            }
        ]

        results = compute_metrics_on_match_list_dict(match_list, [hausdorff_pair])

        self.assertEqual(results[0]["hausdorff_pair"], (1.0, 2.0))

    def test_compute_vertex_errors_can_keep_point_objects(self):
        match_list = [{"reference": Polygon(), "target": Polygon()}]

        with patch(
            "pytorch_segmentation_models_trainer.tools.evaluation.evaluate_data.per_vertex_error_list"
        ) as mock_error:
            mock_error.return_value = [{"point": Point(0, 0), "error": 0.1}]

            results = compute_vertex_errors_on_match_list_dict(
                match_list, convert_output_to_wkt=False
            )

        self.assertIsInstance(results[0]["point"], Point)
        self.assertEqual(results[0]["id"], 0)


if __name__ == "__main__":
    unittest.main()
