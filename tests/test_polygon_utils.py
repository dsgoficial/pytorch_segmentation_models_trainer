# -*- coding: utf-8 -*-
"""
Unit tests for polygon_utils.py
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
import rasterio
from rasterio.transform import Affine

from pytorch_segmentation_models_trainer.utils.polygon_utils import (
    polygon_remove_holes,
    polygons_remove_holes,
    polygons_to_pixel_coords,
    polygons_to_world_coords,
    coerce_polygons_to_single_geometry,
    build_crossfield,
    polygon_to_mask,
    polygon_to_mask_from_geojson_string,
    polygon_to_mask_from_coords,
    get_polygon_mask_area,
    create_test_polygon,
    _draw_circle,
)


class TestPolygonUtils(unittest.TestCase):

    def setUp(self):
        # Example transform (identity transform for simplicity in some tests)
        self.transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        self.epsg_number = 4326

    def test_polygon_remove_holes(self):
        exterior = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
        hole = [(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]
        polygon_with_hole = Polygon(exterior, [hole])
        result = polygon_remove_holes(polygon_with_hole)
        np.testing.assert_array_equal(result, np.array(exterior))

    def test_polygons_remove_holes(self):
        exterior1 = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
        polygon_with_hole1 = Polygon(
            exterior1, [[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]]
        )
        exterior2 = [(100, 100), (100, 110), (110, 110), (110, 100), (100, 100)]
        polygon_no_hole2 = Polygon(exterior2)
        result = polygons_remove_holes([polygon_with_hole1, polygon_no_hole2])
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], np.array(exterior1))
        np.testing.assert_array_equal(result[1], np.array(exterior2))

    def test_polygons_to_pixel_coords(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        pixel_coords = polygons_to_pixel_coords([polygon], transform)
        self.assertEqual(len(pixel_coords), 1)
        np.testing.assert_array_almost_equal(
            pixel_coords[0], np.array([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        )

    def test_polygons_to_world_coords(self):
        pixel_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        world_coords = polygons_to_world_coords(
            [pixel_polygon], transform, self.epsg_number
        )
        self.assertEqual(len(world_coords), 1)
        self.assertTrue(
            world_coords[0].equals(Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]))
        )

    def test_coerce_polygons_to_single_geometry(self):
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        poly2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)])
        multi_poly = MultiPolygon([poly1, poly2])
        result = coerce_polygons_to_single_geometry([poly1, multi_poly])
        self.assertEqual(len(result), 3)

    @patch("pytorch_segmentation_models_trainer.utils.polygon_utils.Image.new")
    @patch("pytorch_segmentation_models_trainer.utils.polygon_utils.ImageDraw.Draw")
    def test_polygon_to_mask(self, mock_draw_class, mock_image_new):
        mock_image = MagicMock()
        mock_image_new.return_value = mock_image
        mock_image.getchannel.return_value.getdata.return_value = [0, 255, 0, 0]

        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        shape = (2, 2)
        result_mask = polygon_to_mask(polygon, shape, 255)

        self.assertEqual(result_mask.shape, shape)
        np.testing.assert_array_equal(
            result_mask, np.array([[0, 255], [0, 0]], dtype=np.uint8)
        )

    def test_get_polygon_mask_area(self):
        mask = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)
        self.assertEqual(get_polygon_mask_area(mask), 3)

    def test_create_test_polygon(self):
        polygon = create_test_polygon(0, 10, seed=42)
        self.assertIsInstance(polygon, Polygon)
        self.assertGreater(len(polygon.exterior.coords), 3)

    @patch("pytorch_segmentation_models_trainer.utils.polygon_utils.ImageDraw.Draw")
    @patch("pytorch_segmentation_models_trainer.utils.polygon_utils.Image.new")
    def test__draw_circle(self, mock_image_new, mock_draw_class):
        mock_draw = MagicMock()
        mock_draw_class.return_value = mock_draw
        _draw_circle(mock_draw, (10, 20), 5, "red")
        mock_draw.ellipse.assert_called_once_with(
            [5, 15, 15, 25], fill="red", outline=None
        )


if __name__ == "__main__":
    unittest.main()
