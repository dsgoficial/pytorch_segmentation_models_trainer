# -*- coding: utf-8 -*-
"""
Unit tests for polygon_utils.py
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    Point,
    Polygon,
    MultiPolygon,
)
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import torch

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
    build_crossfield,
    compute_contour_measure,
    compute_polygon_contour_measures,
    compute_raster_masks,
    plot_geometries,
    point_project_onto_geometry,
    PolygonPatch,
    PolygonPath,
    project_onto_geometry,
    sample_geometry,
    _draw_polygons,
)
import pytorch_segmentation_models_trainer.utils.polygon_utils as polygon_utils


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

    def test_polygon_to_mask_real_polygon_hole_and_multipolygon(self):
        exterior = [(1, 1), (1, 8), (8, 8), (8, 1), (1, 1)]
        hole = [(3, 3), (3, 5), (5, 5), (5, 3), (3, 3)]
        polygon = Polygon(exterior, [hole])
        mask = polygon_to_mask(polygon, (10, 10), 7)
        self.assertEqual(mask[2, 2], 7)
        self.assertEqual(mask[4, 4], 0)

        multipolygon = MultiPolygon(
            [
                Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)]),
                Polygon(
                    [(6, 6), (6, 9), (9, 9), (9, 6), (6, 6)],
                    holes=[[(7, 7), (7, 8), (8, 8), (7, 7)]],
                ),
            ]
        )
        multi_mask = polygon_to_mask(multipolygon, (10, 10), 9)
        self.assertEqual(multi_mask[2, 2], 9)
        self.assertEqual(multi_mask[6, 6], 9)

    def test_polygon_mask_helpers_from_geojson_and_coords(self):
        coords = [(1, 1), (1, 4), (4, 4), (4, 1), (1, 1)]
        geojson = shapely.geometry.mapping(Polygon(coords))

        mask_from_coords = polygon_to_mask_from_coords(coords, (6, 6), value=5)
        mask_from_geojson = polygon_to_mask_from_geojson_string(
            __import__("json").dumps(geojson), (6, 6), value=5
        )

        self.assertEqual(mask_from_coords.shape, (6, 6))
        np.testing.assert_array_equal(mask_from_coords, mask_from_geojson)

    def test_get_polygon_mask_area(self):
        mask = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)
        self.assertEqual(get_polygon_mask_area(mask), 3)

    def test_create_test_polygon(self):
        polygon = create_test_polygon(0, 10, seed=42)
        self.assertIsInstance(polygon, Polygon)
        self.assertGreater(len(polygon.exterior.coords), 3)

    def test_create_test_polygon_falls_back_when_convex_hull_fails(self):
        with patch("scipy.spatial.ConvexHull", side_effect=RuntimeError("bad hull")):
            polygon = create_test_polygon(0, 10, seed=1)

        self.assertTrue(polygon.equals(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])))

    @patch("pytorch_segmentation_models_trainer.utils.polygon_utils.ImageDraw.Draw")
    @patch("pytorch_segmentation_models_trainer.utils.polygon_utils.Image.new")
    def test__draw_circle(self, mock_image_new, mock_draw_class):
        mock_draw = MagicMock()
        mock_draw_class.return_value = mock_draw
        _draw_circle(mock_draw, (10, 20), 5, "red")
        mock_draw.ellipse.assert_called_once_with(
            [5, 15, 15, 25], fill="red", outline=None
        )

    def test_build_crossfield_and_raster_masks(self):
        polygon = Polygon([(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)])
        raster = build_crossfield([polygon], (12, 12), self.transform, line_width=1)
        self.assertEqual(raster.shape, (12, 12))
        self.assertGreater(raster.max(), 0)

        masks = compute_raster_masks(
            [polygon, MultiPolygon([polygon])],
            (12, 12),
            self.transform,
            line_width=1,
            antialiasing=True,
        )

        self.assertEqual(
            set(masks),
            {
                "polygon_masks",
                "boundary_masks",
                "vertex_masks",
                "distance_masks",
                "size_masks",
            },
        )
        self.assertEqual(masks["polygon_masks"].shape, (12, 12))

    def test_compute_raster_masks_optional_outputs_and_assertions(self):
        polygon = Polygon([(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)])
        masks = compute_raster_masks(
            [polygon],
            (12, 12),
            self.transform,
            fill=False,
            edges=True,
            vertices=False,
            compute_distances=False,
            compute_sizes=False,
            line_width=1,
        )
        self.assertEqual(set(masks), {"polygon_masks"})

        with self.assertRaises(AssertionError):
            build_crossfield(tuple([polygon]), (12, 12), self.transform)
        with self.assertRaises(AssertionError):
            _draw_polygons(tuple([polygon]), (12, 12))
        with self.assertRaises(AssertionError):
            _draw_polygons([[(0, 0), (1, 1)]], (12, 12))

    def test_draw_polygons_handles_interiors(self):
        polygon = Polygon(
            [(1, 1), (1, 8), (8, 8), (8, 1), (1, 1)],
            holes=[[(3, 3), (3, 5), (5, 5), (3, 3)]],
        )
        raster = _draw_polygons([polygon], (10, 10), line_width=1)

        self.assertEqual(raster.shape, (10, 10, 3))
        self.assertGreater(raster[:, :, 1].max(), 0)
        self.assertGreater(raster[:, :, 2].max(), 0)

    def test_sampling_projection_and_contour_metrics(self):
        polygon = Polygon(
            [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
            holes=[[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        sampled_polygon = sample_geometry(polygon, density=1.0)
        sampled_line = sample_geometry(LineString([(0, 0), (2, 0), (2, 2)]), 0.5)
        sampled_collection = sample_geometry(
            GeometryCollection([polygon.exterior]), 1.0
        )

        self.assertIsInstance(sampled_polygon, Polygon)
        self.assertIsInstance(sampled_line, LineString)
        self.assertIsInstance(sampled_collection, GeometryCollection)
        with self.assertRaises(TypeError):
            sample_geometry(Point(0, 0), 1.0)

        target = GeometryCollection([polygon.exterior])
        self.assertEqual(point_project_onto_geometry((1, 2), target), (0.0, 2.0))
        projected_line = project_onto_geometry(LineString([(1, 2), (3, 2)]), target)
        self.assertIsInstance(projected_line, LineString)
        projected_polygon = project_onto_geometry(polygon, target)
        self.assertIsInstance(projected_polygon, Polygon)
        projected_collection = project_onto_geometry(
            GeometryCollection([LineString([(1, 2), (3, 2)])]), target
        )
        self.assertIsInstance(projected_collection, GeometryCollection)
        with self.assertRaises(TypeError):
            project_onto_geometry(Point(0, 0), target)

        measure = compute_contour_measure(polygon, target, 1.0, max_stretch=10.0)
        self.assertGreaterEqual(measure, 0.0)
        with self.assertRaises(ValueError):
            compute_contour_measure(
                polygon,
                GeometryCollection([LineString([(20, 20), (30, 20)])]),
                1.0,
                max_stretch=1.01,
            )
        self.assertEqual(
            compute_polygon_contour_measures([], [polygon], 1.0, 0.0, 10.0)[0].size,
            0,
        )
        with patch.object(
            polygon_utils,
            "compute_contour_measure",
            return_value=torch.tensor(0.25),
        ) as measure_mock:
            measures = compute_polygon_contour_measures(
                [polygon], [polygon], 1.0, 0.0, 10.0
            )
        self.assertEqual(measures, [torch.tensor(0.25)])
        measure_mock.assert_called_once()

    def test_compute_polygon_contour_measures_progressbar_and_type_guards(self):
        polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
        with self.assertRaises(AssertionError):
            compute_polygon_contour_measures(
                tuple([polygon]), [polygon], 1.0, 0.0, 10.0
            )
        with self.assertRaises(AssertionError):
            compute_polygon_contour_measures(
                [polygon], tuple([polygon]), 1.0, 0.0, 10.0
            )
        with self.assertRaises(AssertionError):
            compute_polygon_contour_measures(
                [LineString([(0, 0), (1, 1)])], [polygon], 1.0, 0.0, 10.0
            )
        with self.assertRaises(AssertionError):
            compute_polygon_contour_measures(
                [polygon], [LineString([(0, 0), (1, 1)])], 1.0, 0.0, 10.0
            )

        current_process = MagicMock()
        current_process.name = "ForkPoolWorker-1"
        with patch.object(
            polygon_utils.multiprocess, "current_process", return_value=current_process
        ):
            with patch.object(
                polygon_utils, "tqdm", side_effect=lambda iterable, **_: iterable
            ):
                with patch.object(
                    polygon_utils,
                    "compute_contour_measure",
                    return_value=0.0,
                ):
                    measures = compute_polygon_contour_measures(
                        [polygon], [polygon], 1.0, 0.0, 10.0, progressbar=True
                    )
        self.assertEqual(measures, [0.0])

    def test_project_onto_geometry_uses_pool_for_collections(self):
        collection = GeometryCollection([LineString([(1, 2), (3, 2)])])
        target = GeometryCollection([LineString([(0, 0), (0, 4)])])
        pool = MagicMock()
        pool.map.side_effect = lambda func, geoms: [func(geom) for geom in geoms]

        projected = project_onto_geometry(collection, target, pool=pool)

        self.assertIsInstance(projected, GeometryCollection)
        pool.map.assert_called_once()

    def test_plot_geometries_and_polygon_paths(self):
        fig, ax = plt.subplots()
        polygon = Polygon(
            [(0, 0), (2, 0), (2, 2), (0, 0)],
            holes=[[(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 0.5)]],
        )
        line = LineString([(0, 0), (1, 1)])
        plot_geometries(ax, [polygon, line])
        self.assertGreaterEqual(len(ax.lines), 2)
        self.assertGreaterEqual(len(ax.collections), 1)
        plt.close(fig)

        patch = PolygonPatch(polygon, fc="blue")
        self.assertIsNotNone(patch.get_path())
        self.assertIsNotNone(PolygonPath(shapely.geometry.mapping(polygon)))
        self.assertIsNotNone(
            PolygonPath(
                {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [
                            [(0, 0), (1, 0), (1, 1), (0, 0)],
                        ]
                    ],
                }
            )
        )
        self.assertIsNotNone(PolygonPath(MultiPolygon([polygon])))
        with self.assertRaises(ValueError):
            PolygonPath(LineString([(0, 0), (1, 1)]))
        with self.assertRaises(ValueError):
            PolygonPath({"type": "LineString", "coordinates": [(0, 0), (1, 1)]})
        with self.assertRaises(NotImplementedError):
            plot_geometries(ax, [Point(0, 0)])

    def test_project_onto_geometry_topological_error_branch(self):
        polygon = Polygon(
            [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
            holes=[[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        target = GeometryCollection([polygon.exterior])

        with patch.object(
            polygon_utils,
            "Polygon",
            side_effect=shapely.errors.TopologicalError("bad topology"),
        ):
            with patch("matplotlib.pyplot.show"):
                with self.assertRaises(shapely.errors.TopologicalError):
                    project_onto_geometry(polygon, target)


if __name__ == "__main__":
    unittest.main()
