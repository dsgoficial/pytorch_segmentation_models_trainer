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
import unittest
import warnings

from parameterized import parameterized
from pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader import (
    CSVImageReaderProcessor,
    FolderImageReaderProcessor,
    RasterFile,
    MaskOutputTypeEnum,
    SingleImageReaderProcessor,
)
from pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader import (
    FileGeoDF,
    GeomTypeEnum,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    hash_file,
    remove_folder,
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


class Test_RasterReader(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        create_folder(os.path.join(root_dir, "test_output", "mask"))
        create_folder(os.path.join(root_dir, "test_output", "boundary_mask"))
        create_folder(os.path.join(root_dir, "test_output", "vertex_mask"))

    def tearDown(self):
        remove_folder(self.output_dir)

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
