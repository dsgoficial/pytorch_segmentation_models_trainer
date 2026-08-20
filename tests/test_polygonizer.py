# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-25
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
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import geopandas
import hydra
import rasterio
import torch
from shapely.geometry import Polygon


def geom_almost_equals(this, that, tolerance=0.01):
    """Wrapper around geom_equals_exact that is robust to vertex order and flakiness.

    1. Uses normalize() to canonicalize vertex order.
    2. Uses IoU as a fallback for slight boundary variations in iterative algorithms.
    """
    if len(this) != len(that):
        return False

    # 1. Try exact match with normalization (handles vertex order)
    if this.normalize().geom_equals_exact(that.normalize(), tolerance=tolerance).all():
        return True

    # 2. Fallback to IoU (Intersection over Union) which is more robust to
    # vertex count and slight boundary variations in iterative algorithms like ACM.
    # We use a high threshold (0.98) to ensure they are semantically "the same".
    try:
        intersection_area = this.intersection(that).area
        union_area = this.union(that).area
        # Handle empty geometries by filling NaN with 1.0 (if both empty) or 0.0
        iou = (intersection_area / union_area).fillna(1.0)
        # If any iou is < 0.98, it might be a real difference
        return (iou >= 0.98).all()
    except Exception:
        # If intersection fails, rely on the first check's result
        return False


def sort_gdf(gdf):
    """Sorts a GeoDataFrame by its geometry centroid for consistent comparison."""
    # Use centroid coordinates for a more stable sort than WKT strings
    gdf = gdf.copy()
    gdf["_x"] = gdf.geometry.centroid.x
    gdf["_y"] = gdf.geometry.centroid.y
    sorted_gdf = (
        gdf.sort_values(by=["_x", "_y"])
        .drop(columns=["_x", "_y"])
        .reset_index(drop=True)
    )
    return sorted_gdf


from hydra import compose, initialize
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    VectorFileDataWriter,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    ACMConfig,
    ACMPolygonizerProcessor,
    ASMConfig,
    ASMPolygonizerProcessor,
    TemplatePolygonizerProcessor,
    PolygonRNNPolygonizerProcessor,
    SimplePolConfig,
    SimplePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.utils import frame_field_utils, seed_utils

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")

frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)

device = "cpu"
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
...


class Test_Polygonize(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()
        self.frame_field_ds = self.get_frame_field_ds()
        with rasterio.open(self.frame_field_ds[0]["path"], "r") as raster_ds:
            self.crs = raster_ds.crs
            self.profile = raster_ds.profile
            self.transform = raster_ds.transform

    def get_frame_field_ds(self):
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    "input_csv_path=" + csv_path,
                    "root_dir=" + frame_field_root_dir,
                ],
            )
            frame_field_ds = hydra.utils.instantiate(cfg, _recursive_=False)
        return frame_field_ds

    def test_polygonizer_simple_processor(self) -> None:
        config = SimplePolConfig()
        output_file_path = os.path.join(self.output_dir, "simple_polygonizer.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="simple_polygonizer.geojson",
        )
        processor = SimplePolygonizerProcessor(data_writer=data_writer, config=config)
        processor.process(
            {
                "seg": torch.movedim(
                    self.frame_field_ds[0]["gt_polygons_image"], -1, 0
                ).unsqueeze(0)
            },
            self.profile,
        )
        assert os.path.isfile(output_file_path)
        expected_output_gdf = sort_gdf(
            geopandas.read_file(
                filename=os.path.join(
                    root_dir,
                    "expected_outputs",
                    "polygonize",
                    "simple_polygonizer.geojson",
                )
            )
        )
        output_features_gdf = sort_gdf(geopandas.read_file(filename=output_file_path))
        assert geom_almost_equals(
            expected_output_gdf["geometry"], output_features_gdf["geometry"]
        )

    def test_polygonizer_acm_processor(self) -> None:
        seed_utils.set_training_seed(42)
        config = ACMConfig()
        output_file_path = os.path.join(self.output_dir, "acm_polygonizer.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="acm_polygonizer.geojson",
        )
        processor = ACMPolygonizerProcessor(data_writer=data_writer, config=config)
        processor.process(
            {
                "seg": torch.movedim(
                    self.frame_field_ds[0]["gt_polygons_image"], -1, 0
                ).unsqueeze(0),
                "crossfield": frame_field_utils.compute_crossfield_to_plot(
                    self.frame_field_ds[0]["gt_crossfield_angle"]
                ),
            },
            self.profile,
        )
        assert os.path.isfile(output_file_path)
        expected_output_gdf = sort_gdf(
            geopandas.read_file(
                filename=os.path.join(
                    root_dir,
                    "expected_outputs",
                    "polygonize",
                    "acm_polygonizer.geojson",
                )
            )
        )
        output_features_gdf = sort_gdf(geopandas.read_file(filename=output_file_path))
        assert geom_almost_equals(
            expected_output_gdf["geometry"],
            output_features_gdf["geometry"],
            tolerance=0.05,
        )

    def test_polygonizer_asm_processor(self) -> None:
        seed_utils.set_training_seed(42)
        config = ASMConfig()
        output_file_path = os.path.join(self.output_dir, "asm_polygonizer.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="asm_polygonizer.geojson",
        )
        processor = ASMPolygonizerProcessor(data_writer=data_writer, config=config)
        processor.process(
            {
                "seg": torch.movedim(
                    self.frame_field_ds[0]["gt_polygons_image"], -1, 0
                ).unsqueeze(0),
                "crossfield": frame_field_utils.compute_crossfield_to_plot(
                    self.frame_field_ds[0]["gt_crossfield_angle"]
                ),
            },
            self.profile,
        )
        assert os.path.isfile(output_file_path)
        expected_output_gdf = sort_gdf(
            geopandas.read_file(
                filename=os.path.join(
                    root_dir,
                    "expected_outputs",
                    "polygonize",
                    "asm_polygonizer.geojson",
                )
            )
        )
        output_features_gdf = sort_gdf(geopandas.read_file(filename=output_file_path))
        assert geom_almost_equals(
            expected_output_gdf["geometry"],
            output_features_gdf["geometry"],
            tolerance=0.05,
        )

    def test_template_polygonizer_abstract_post_init_noop_and_post_process_options(
        self,
    ):
        @dataclass
        class ConcreteTemplate(TemplatePolygonizerProcessor):
            config: object = field(default_factory=object)

            def __post_init__(self):
                return super().__post_init__()

        processor = ConcreteTemplate(data_writer=None)
        self.assertIsNone(processor.__post_init__())
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])

        output = processor.post_process(
            [polygon],
            profile=None,
            convert_output_to_world_coords=False,
        )

        self.assertEqual(len(output), 1)

    def test_template_polygonizer_fallback_single_and_batch_paths(self):
        @dataclass
        class ConcreteTemplate(TemplatePolygonizerProcessor):
            config: object = field(default_factory=object)

            def __post_init__(self):
                self.polygonize_method = MagicMock(
                    side_effect=[
                        RuntimeError("batch failed"),
                        ([[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]], None),
                        RuntimeError("single failed"),
                    ]
                )

            def post_process(self, polygons, profile, parent_dir_name=None, **kwargs):
                return polygons

        writer = MagicMock()
        processor = ConcreteTemplate(data_writer=writer)
        inference = {
            "seg": torch.ones((2, 1, 4, 4)),
            "crossfield": torch.ones((2, 2, 4, 4)),
        }

        output = processor.process(
            inference,
            profile=None,
            pool=None,
            parent_dir_name=["ok", "bad"],
            convert_output_to_world_coords=False,
        )

        self.assertEqual(len(output), 1)

    def test_template_polygonizer_pool_returns_futures(self):
        @dataclass
        class ConcreteTemplate(TemplatePolygonizerProcessor):
            config: object = field(default_factory=object)

            def __post_init__(self):
                self.polygonize_method = MagicMock(
                    return_value=(
                        [
                            [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
                            [Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])],
                        ],
                        None,
                    )
                )

        class DummyPool:
            def submit(self, func, *args, **kwargs):
                return (func, args, kwargs)

        processor = ConcreteTemplate(data_writer=None)
        inference = {
            "seg": torch.ones((2, 1, 4, 4)),
            "crossfield": torch.ones((2, 2, 4, 4)),
        }

        futures = processor.process(
            inference,
            profile=[{"crs": None}, {"crs": None}],
            pool=DummyPool(),
            parent_dir_name=["a", "b"],
            convert_output_to_world_coords=False,
        )

        self.assertEqual(len(futures), 2)

    def test_polygon_rnn_polygonizer_processor_process(self):
        processor = PolygonRNNPolygonizerProcessor(data_writer=None)
        processor.polygonize_method = MagicMock(
            return_value=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]
        )

        output = processor.process(
            {"output_batch_polygons": torch.zeros((1, 4, 2))},
            profile=None,
            parent_dir_name="poly_rnn",
            convert_output_to_world_coords=False,
        )

        self.assertEqual(len(output), 1)
