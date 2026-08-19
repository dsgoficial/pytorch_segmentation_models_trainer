# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-04-02
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

import numpy as np
import geopandas as gpd
import rasterio
from parameterized import parameterized
from pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader import (
    AbstractRasterPathListGetter,
    CSVImageReaderProcessor,
    FolderImageReaderProcessor,
    RasterFile,
    MaskOutputTypeEnum,
    SingleImageReaderProcessor,
    save_with_rasterio,
)
from shapely.geometry import box
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    hash_file,
)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
test_list = [
    (
        RasterFile,
        {"file_name": os.path.join(root_dir, "data", "images", "image1.png")},
        (3, 513, 513),
    ),
    (
        RasterFile,
        {
            "file_name": os.path.join(root_dir, "data", "images", "image1.png"),
            "channels_last": True,
        },
        (513, 513, 3),
    ),
    (
        RasterFile,
        {"file_name": os.path.join(root_dir, "data", "labels", "label1.png")},
        (1, 513, 513),
    ),
]
suffix_dict = {"PNG": ".png", "GTiff": ".tif", "JPEG": ".jpg"}

expected_image_list = [
    os.path.join(
        root_dir, "data", "frame_field_data", "images", "Ortoimagem_MI_2970-1-SO", image
    )
    for image in [
        "Ortoimagem_MI_2970-1-SO_966.tif",
        "Ortoimagem_MI_2970-1-SO_967.tif",
        "Ortoimagem_MI_2970-1-SO_970.tif",
        "Ortoimagem_MI_2970-1-SO_973.tif",
        "Ortoimagem_MI_2970-1-SO_995.tif",
        "Ortoimagem_MI_2970-1-SO_996.tif",
        "Ortoimagem_MI_2970-1-SO_997.tif",
        "Ortoimagem_MI_2970-1-SO_998.tif",
        "Ortoimagem_MI_2970-1-SO_1033.tif",
        "Ortoimagem_MI_2970-1-SO_1036.tif",
        "Ortoimagem_MI_2970-1-SO_1039.tif",
        "Ortoimagem_MI_2970-1-SO_1045.tif",
    ]
]
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
...


class Test_RasterReader(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()
        create_folder(os.path.join(self.output_dir, "mask"))
        create_folder(os.path.join(self.output_dir, "boundary_mask"))
        create_folder(os.path.join(self.output_dir, "vertex_mask"))

    @parameterized.expand(test_list)
    def test_instantiate_object(self, obj_class, params, expected_shape) -> None:
        obj = obj_class(**params)
        numpy_array = obj.read_as_numpy_array()
        self.assertEqual(numpy_array.shape, expected_shape)

    @parameterized.expand(["GTiff", "JPEG"])
    def test_export_to(self, output_format):
        file_name = os.path.join(root_dir, "data", "images", "image1.png")
        expected_output = os.path.join(
            root_dir,
            "expected_outputs",
            "raster_reader",
            "image1" + suffix_dict[output_format],
        )
        raster = RasterFile(file_name=file_name)
        output_raster = raster.export_to(self.output_dir, output_format)
        self.assertEqual(hash_file(expected_output), hash_file(output_raster))

    @parameterized.expand(
        [
            (
                SingleImageReaderProcessor(
                    file_name=os.path.join(
                        root_dir,
                        "data",
                        "frame_field_data",
                        "images",
                        "Ortoimagem_MI_2970-1-SO",
                        "Ortoimagem_MI_2970-1-SO_966.tif",
                    )
                ),
                [
                    os.path.join(
                        root_dir,
                        "data",
                        "frame_field_data",
                        "images",
                        "Ortoimagem_MI_2970-1-SO",
                        "Ortoimagem_MI_2970-1-SO_966.tif",
                    )
                ],
            ),
            (
                FolderImageReaderProcessor(
                    folder_name=os.path.join(
                        root_dir, "data", "frame_field_data", "images"
                    )
                ),
                expected_image_list,
            ),
            (
                CSVImageReaderProcessor(
                    input_csv_path=os.path.join(
                        root_dir, "data", "frame_field_data", "dsg_dataset.csv"
                    ),
                    root_dir=os.path.join(root_dir, "data", "frame_field_data"),
                ),
                expected_image_list,
            ),
        ]
    )
    def test_image_reader_processor(self, processor, expected_output):
        output_list = processor.get_images()
        self.assertListEqual(sorted(output_list), sorted(expected_output))

    def test_abstract_raster_path_list_getter_method_body_is_noop(self):
        class ConcreteGetter(AbstractRasterPathListGetter):
            def get_images(self):
                return super().get_images()

        self.assertIsNone(ConcreteGetter().get_images())

    def test_csv_image_reader_processor_reads_first_rows_and_root_default(self):
        csv_path = os.path.join(self.output_dir, "images.csv")
        with open(csv_path, "w") as f:
            f.write("image\nrelative/a.tif\nrelative/b.tif\n")
        processor = CSVImageReaderProcessor(
            input_csv_path=csv_path,
            n_first_rows_to_read=1,
        )

        self.assertEqual(processor.get_images(), ["/relative/a.tif"])

    def test_build_image_bb_annotations_uses_dataset_transform_when_missing(self):
        file_name = os.path.join(root_dir, "data", "images", "image1.png")
        raster = RasterFile(file_name=file_name)
        vector_feats = gpd.GeoSeries([box(0, 0, 10, 10)])

        annotations = raster.build_image_bb_annotations_from_vector_layer(vector_feats)

        self.assertEqual(len(annotations), 1)
        self.assertIn("bbox", annotations[0])

    def test_save_with_rasterio_writes_single_and_multiband_arrays(self):
        profile = {
            "driver": "GTiff",
            "height": 4,
            "width": 4,
            "count": 1,
            "dtype": "uint8",
            "crs": None,
            "transform": rasterio.Affine.identity(),
        }
        single_output = os.path.join(self.output_dir, "single.tif")
        save_with_rasterio(
            single_output,
            profile.copy(),
            [np.ones((4, 4), dtype=np.uint8)],
            [MaskOutputTypeEnum.SINGLE_FILE_MULTIPLE_BAND],
        )
        self.assertTrue(os.path.exists(single_output))

        multi_output = os.path.join(self.output_dir, "multi.tif")
        multi_profile = profile.copy()
        multi_profile["count"] = 2
        save_with_rasterio(
            multi_output,
            multi_profile,
            [np.ones((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)],
            [
                MaskOutputTypeEnum.SINGLE_FILE_MULTIPLE_BAND,
                MaskOutputTypeEnum.MULTIPLE_FILES_SINGLE_BAND,
            ],
        )
        with rasterio.open(multi_output) as ds:
            self.assertEqual(ds.count, 2)

    def test_write_masks_to_disk_writes_channels_last_multiband(self):
        file_name = os.path.join(root_dir, "data", "images", "image1.png")
        raster = RasterFile(file_name=file_name)
        output_dir = os.path.join(self.output_dir, "multi_mask")
        create_folder(output_dir)
        profile = {
            "driver": "GTiff",
            "height": 4,
            "width": 4,
            "count": 3,
            "dtype": "uint8",
            "crs": None,
            "transform": rasterio.Affine.identity(),
            "photometric": "RGB",
        }
        raster_dict = {
            "mask": np.dstack(
                [
                    np.ones((4, 4), dtype=np.uint8),
                    np.zeros((4, 4), dtype=np.uint8),
                    np.full((4, 4), 2, dtype=np.uint8),
                ]
            )
        }

        outputs = raster.write_masks_to_disk(
            raster_dict,
            profile,
            {"mask": output_dir},
            output_filename="channels_last",
            output_extension="tif",
            replicate_input_structure=False,
        )

        with rasterio.open(outputs["mask"]) as ds:
            self.assertEqual(ds.count, 3)
