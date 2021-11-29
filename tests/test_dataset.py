# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
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

import dataclasses
import json
import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import hydra
import numpy as np
from hydra import compose, initialize
import torch
from pytorch_segmentation_models_trainer.config_definitions.coco_dataset_config import (
    AnnotationConfig,
    CategoryConfig,
    CocoDatasetConfig,
    CocoDatasetInfoConfig,
    ImageConfig,
    LicenseConfig,
)
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    InstanceSegmentationDataset,
    NaiveModPolyMapperDataset,
    ObjectDetectionDataset,
    PolygonRNNDataset,
    SegmentationDataset,
    load_augmentation_object,
    ModPolyMapperDataset,
)

from tests.utils import CustomTestCase

current_dir = os.path.dirname(__file__)
frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)
detection_root_dir = os.path.join(current_dir, "testing_data", "data", "detection_data")


class Test_Dataset(CustomTestCase):
    def test_create_instance(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=["input_csv_path=" + self.csv_ds_file],
            )
            ds_from_cfg = hydra.utils.instantiate(cfg, _recursive_=False)
            ds_from_ref = SegmentationDataset(input_csv_path=self.csv_ds_file)
            self.assertEqual(ds_from_cfg.input_csv_path, ds_from_ref.input_csv_path)

    def test_load_image(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=["input_csv_path=" + self.csv_ds_file],
            )
            ds_from_cfg = hydra.utils.instantiate(cfg, _recursive_=False)
            ds_from_ref = SegmentationDataset(input_csv_path=self.csv_ds_file)
            assert np.array_equal(
                ds_from_cfg[0]["image"], ds_from_ref[0]["image"]
            ) and np.array_equal(ds_from_cfg[0]["mask"], ds_from_ref[0]["mask"])

    def test_load_image_relative_path(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=[
                    "input_csv_path=" + self.csv_ds_file_without_root,
                    "+root_dir=" + self.root_dir,
                ],
            )
            ds_from_cfg = hydra.utils.instantiate(cfg, _recursive_=False)
            ds_from_ref = SegmentationDataset(
                input_csv_path=self.csv_ds_file_without_root, root_dir=self.root_dir
            )
            assert np.array_equal(
                ds_from_cfg[0]["image"], ds_from_ref[0]["image"]
            ) and np.array_equal(ds_from_cfg[0]["mask"], ds_from_ref[0]["mask"])

    def test_load_augmentations(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="augmentations.yaml")
            transformer = load_augmentation_object(cfg["augmentation_list"])
            assert isinstance(transformer, A.Compose)

    def test_dataset_size_limit(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=[
                    "input_csv_path=" + self.csv_ds_file_without_root,
                    "+root_dir=" + self.root_dir,
                    "+n_first_rows_to_read=2",
                ],
            )
            ds_from_cfg = hydra.utils.instantiate(cfg, _recursive_=False)
            self.assertEqual(len(ds_from_cfg), 2)

    def test_create_frame_field_dataset_instance(self):
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
        self.assertEqual(len(frame_field_ds), 12)
        self.assertEqual(frame_field_ds[0]["image"].shape, (571, 571, 3))
        self.assertEqual(frame_field_ds[0]["gt_polygons_image"].shape, (571, 571, 3))
        self.assertEqual(frame_field_ds[0]["gt_crossfield_angle"].shape, (1, 571, 571))

    def test_coco_dataset(self):
        coco_ds = CocoDatasetConfig(
            info=CocoDatasetInfoConfig(
                description="COCO 2017 dataset",
                url="http://cocodataset.org",
                version="1.0",
                year=2017,
                contributor="COCO Consortium",
                date_created="2017/09/01",
            ),
            licenses=[
                LicenseConfig(
                    url="http://creativecommons.org/licenses/by/2.0/",
                    id=4,
                    name="Attribution License",
                )
            ],
            images=[
                ImageConfig(
                    id=242287,
                    license=4,
                    coco_url="http://images.cocodataset.org/val2017/000000242287.jpg",
                    flickr_url="http://farm3.staticflickr.com/2626/4072194513_edb6acfb2b_z.jpg",
                    width=426,
                    height=640,
                    file_name="000000242287.jpg",
                    date_captured="2013-11-15 02:41:42",
                ),
                ImageConfig(
                    id=245915,
                    license=4,
                    coco_url="http://images.cocodataset.org/val2017/000000245915.jpg",
                    flickr_url="http://farm1.staticflickr.com/88/211747310_f58a16631e_z.jpg",
                    width=500,
                    height=333,
                    file_name="000000245915.jpg",
                    date_captured="2013-11-18 02:53:27",
                ),
            ],
            categories=[
                CategoryConfig(supercategory="vehicle", id=2, name="bicycle"),
                CategoryConfig(supercategory="animal", id=22, name="elephant"),
            ],
            annotations=[
                AnnotationConfig(
                    id=125686,
                    category_id=2,
                    iscrowd=0,
                    segmentation=[
                        [
                            164.81,
                            417.51,
                            164.81,
                            417.51,
                            164.81,
                            417.51,
                            159.31,
                            409.27,
                            159.31,
                            409.27,
                            155.19,
                            409.27,
                            155.19,
                            410.64,
                            152.45,
                            413.39,
                            144.21,
                            413.39,
                            140.09,
                            413.39,
                            137.34,
                            413.39,
                            134.59,
                            414.76,
                            129.1,
                            414.76,
                            122.23,
                            414.76,
                            104.38,
                            405.15,
                            104.38,
                            405.15,
                            93.39,
                            401.03,
                            93.39,
                            401.03,
                            87.9,
                            399.66,
                            86.52,
                            399.66,
                            85.15,
                            399.66,
                            93.39,
                            391.42,
                            78.28,
                            383.18,
                            72.79,
                            383.18,
                            68.67,
                            401.03,
                            82.4,
                            402.4,
                            104.38,
                            409.27,
                            104.38,
                            421.63,
                            85.15,
                            458.71,
                            79.66,
                            469.7,
                            65.92,
                            454.59,
                            61.8,
                            455.97,
                            59.06,
                            455.97,
                            42.58,
                            471.07,
                            39.83,
                            479.31,
                            39.83,
                            482.06,
                            31.59,
                            497.17,
                            19.23,
                            510.9,
                            19.23,
                            519.14,
                            19.23,
                            539.74,
                            20.6,
                            578.2,
                            26.09,
                            590.56,
                            27.47,
                            593.3,
                            38.45,
                            596.05,
                            41.2,
                            593.3,
                            50.82,
                            589.18,
                            53.56,
                            587.81,
                            59.06,
                            585.06,
                            71.42,
                            579.57,
                            79.66,
                            568.58,
                            82.4,
                            560.34,
                            85.15,
                            535.62,
                            87.9,
                            520.52,
                            90.64,
                            513.65,
                            93.39,
                            490.3,
                            93.39,
                            487.55,
                            93.39,
                            480.69,
                            108.5,
                            499.91,
                            108.5,
                            509.53,
                            115.36,
                            521.89,
                            104.38,
                            521.89,
                            103.0,
                            528.76,
                            108.5,
                            534.25,
                            119.48,
                            530.13,
                            124.98,
                            535.62,
                            126.35,
                            535.62,
                            129.1,
                            543.86,
                            166.18,
                            582.32,
                            171.67,
                            578.2,
                            197.77,
                            598.8,
                            208.76,
                            608.41,
                            249.96,
                            627.64,
                            269.18,
                            623.52,
                            278.8,
                            618.03,
                            281.55,
                            609.79,
                            285.67,
                            601.55,
                            287.04,
                            597.42,
                            291.16,
                            591.93,
                            291.16,
                            590.56,
                            318.63,
                            590.56,
                            318.63,
                            591.93,
                            329.61,
                            589.18,
                            332.36,
                            586.44,
                            333.73,
                            583.69,
                            309.01,
                            563.09,
                            313.13,
                            547.98,
                            313.13,
                            541.12,
                            313.13,
                            526.01,
                            313.13,
                            523.26,
                            313.13,
                            516.39,
                            313.13,
                            501.29,
                            307.64,
                            486.18,
                            307.64,
                            486.18,
                            302.15,
                            483.43,
                            291.16,
                            472.45,
                            278.8,
                            454.59,
                            273.3,
                            454.59,
                            262.32,
                            447.73,
                            241.72,
                            438.11,
                            226.61,
                            425.75,
                            226.61,
                            420.26,
                            210.13,
                            413.39,
                            206.01,
                            413.39,
                            197.77,
                            414.76,
                            167.55,
                            410.64,
                        ]
                    ],
                    image_id=242287,
                    area=42061.80340000001,
                    bbox=[19.23, 383.18, 314.5, 244.46],
                ),
                AnnotationConfig(
                    id=1409619,
                    category_id=22,
                    iscrowd=0,
                    segmentation=[
                        [
                            376.81,
                            238.8,
                            378.19,
                            228.91,
                            382.15,
                            216.06,
                            383.14,
                            210.72,
                            385.9,
                            207.56,
                            386.7,
                            207.16,
                            387.29,
                            201.43,
                            389.27,
                            197.67,
                            396.58,
                            196.09,
                            404.49,
                            198.07,
                            416.16,
                            190.95,
                            434.75,
                            189.76,
                            458.48,
                            190.95,
                            468.96,
                            196.88,
                            473.51,
                            205.58,
                            473.31,
                            210.52,
                            460.65,
                            225.36,
                            447.6,
                            238.41,
                            437.91,
                            239.59,
                            420.91,
                            233.07,
                            413.98,
                            246.71,
                            406.27,
                            245.92,
                            408.45,
                            232.87,
                            409.24,
                            224.17,
                            407.66,
                            221.6,
                            405.28,
                            221.6,
                            402.91,
                            226.54,
                            395.99,
                            228.12,
                            393.02,
                            229.11,
                            388.28,
                            231.88,
                            383.93,
                            236.82,
                            382.74,
                            241.17,
                        ]
                    ],
                    image_id=245915,
                    area=3556.2197000000015,
                    bbox=[376.81, 189.76, 96.7, 56.95],
                ),
                AnnotationConfig(
                    id=1410165,
                    category_id=22,
                    iscrowd=0,
                    segmentation=[
                        [
                            486.34,
                            239.01,
                            477.88,
                            244.78,
                            468.26,
                            245.16,
                            464.41,
                            244.78,
                            458.64,
                            250.16,
                            451.72,
                            249.39,
                            445.56,
                            243.62,
                            448.26,
                            238.62,
                            476.72,
                            206.69,
                            481.34,
                            205.54,
                            485.57,
                            209.0,
                            500.0,
                            205.16,
                            500.0,
                            267.47,
                            500.0,
                            272.86,
                            498.26,
                            276.71,
                            489.03,
                            275.94,
                            488.65,
                            272.47,
                            494.8,
                            251.32,
                            495.95,
                            244.39,
                        ]
                    ],
                    image_id=245915,
                    area=1775.8932499999994,
                    bbox=[445.56, 205.16, 54.44, 71.55],
                ),
            ],
        )
        with open(
            os.path.join(
                current_dir,
                "testing_data",
                "expected_outputs",
                "dataset",
                "coco_format.json",
            ),
            "r",
        ) as f:
            expected_output = json.load(f)
        self.assertDictEqual(dataclasses.asdict(coco_ds), expected_output)

    def test_polygon_rnn_dataset(self):
        csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        polygon_rnn_ds = PolygonRNNDataset(
            input_csv_path=csv_path,
            sequence_length=60,
            root_dir=polygon_rnn_root_dir,
            augmentation_list=[A.Normalize(), ToTensorV2()],
        )
        self.assertEqual(len(polygon_rnn_ds), 587)

        ds_item = polygon_rnn_ds[0]
        self.assertEqual(ds_item["image"].shape, (3, 224, 224))
        self.assertEqual(ds_item["x1"].shape, (787,))
        self.assertEqual(ds_item["x2"].shape, (58, 787))
        self.assertEqual(ds_item["x3"].shape, (58, 787))
        self.assertEqual(
            ds_item["ta"].shape, (58,)
        )  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    def test_object_detection_dataset(self):
        csv_path = os.path.join(detection_root_dir, "geo", "dsg_dataset.csv")
        obj_det_ds = ObjectDetectionDataset(
            input_csv_path=csv_path,
            root_dir=os.path.dirname(csv_path),
            augmentation_list=A.Compose(
                [A.CenterCrop(512, 512), A.Normalize(), ToTensorV2()],
                bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
            ),
        )
        batch = torch.utils.data.DataLoader(
            obj_det_ds, batch_size=4, shuffle=False, collate_fn=obj_det_ds.collate_fn
        )
        for _, batch_item in enumerate(batch):
            batch_images, batch_targets, indexes = batch_item
            self.assertEqual(batch_images.shape, (4, 3, 512, 512))
            self.assertEqual(len(batch_targets), 4)
        self.assertEqual(len(obj_det_ds), 12)
        image, target, index = obj_det_ds[0]
        self.assertEqual(image.shape, (3, 512, 512))
        self.assertEqual(target["boxes"].shape, (1, 4))
        self.assertEqual(target["labels"].shape, (1,))

    def test_instance_segmentation_dataset(self):
        csv_path = os.path.join(detection_root_dir, "geo", "dsg_dataset.csv")
        obj_det_ds = InstanceSegmentationDataset(
            input_csv_path=csv_path,
            root_dir=os.path.dirname(csv_path),
            augmentation_list=A.Compose(
                [A.Normalize(), ToTensorV2()],
                bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
            ),
        )
        self.assertEqual(len(obj_det_ds), 12)
        image, target, index = obj_det_ds[0]
        self.assertEqual(image.shape, (3, 571, 571))
        self.assertEqual(target["boxes"].shape, (2, 4))
        self.assertEqual(target["labels"].shape, (2,))
        self.assertEqual(target["masks"].shape, (1, 571, 571))

    def test_naive_mod_polymapper_dataset(self):
        csv_path = os.path.join(detection_root_dir, "geo", "dsg_dataset.csv")
        poly_csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        ds = NaiveModPolyMapperDataset(
            object_detection_dataset=ObjectDetectionDataset(
                input_csv_path=csv_path,
                root_dir=os.path.dirname(csv_path),
                augmentation_list=A.Compose(
                    [A.CenterCrop(512, 512), A.Normalize(), ToTensorV2()],
                    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
                ),
            ),
            polygon_rnn_dataset=PolygonRNNDataset(
                input_csv_path=poly_csv_path,
                sequence_length=60,
                root_dir=polygon_rnn_root_dir,
                augmentation_list=[A.Normalize(), ToTensorV2()],
            ),
        )
        self.assertEqual(len(ds), 12)
        image, target, index = ds[0]
        self.assertEqual(image.shape, (3, 512, 512))
        self.assertEqual(target["boxes"].shape, (1, 4))
        self.assertEqual(target["labels"].shape, (1,))
        self.assertEqual(len(target["polygon_rnn_data"]), 2)
        for data in target["polygon_rnn_data"]:
            self.assertEqual(data["image"].shape, (3, 224, 224))
            self.assertEqual(data["x1"].shape, (787,))
            self.assertEqual(data["x2"].shape, (58, 787))
            self.assertEqual(data["x3"].shape, (58, 787))
            self.assertEqual(data["ta"].shape, (58,))

    def test_mod_polymapper_dataset(self):
        csv_path = os.path.join(detection_root_dir, "geo", "dsg_dataset.csv")
        poly_csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        ds = ModPolyMapperDataset(
            object_detection_dataset=ObjectDetectionDataset(
                input_csv_path=csv_path,
                root_dir=os.path.dirname(csv_path),
                augmentation_list=A.Compose(
                    [A.CenterCrop(512, 512), A.Normalize(), ToTensorV2()],
                    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
                ),
            ),
            polygon_rnn_dataset=PolygonRNNDataset(
                input_csv_path=poly_csv_path,
                sequence_length=60,
                root_dir=polygon_rnn_root_dir,
                augmentation_list=[A.Normalize(), ToTensorV2()],
            ),
        )
        self.assertEqual(len(ds), 12)
        image, target, index = ds[0]
        self.assertEqual(image.shape, (3, 512, 512))
        self.assertEqual(target["boxes"].shape, (1, 4))
        self.assertEqual(target["labels"].shape, (1,))
        self.assertEqual(target["x1"].shape, (2, 787))
        self.assertEqual(target["x2"].shape, (2, 58, 787))
        self.assertEqual(target["x3"].shape, (2, 58, 787))
        self.assertEqual(target["ta"].shape, (2, 58))
        # test batch build
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            drop_last=True,
            num_workers=1,
            collate_fn=ds.collate_fn,
        )
        images, targets, indexes = next(iter(data_loader))
        self.assertEqual(images.shape, (2, 3, 512, 512))
        self.assertEqual(targets[0]["boxes"].shape, (1, 4))
        self.assertEqual(targets[0]["labels"].shape, (1,))
        self.assertEqual(targets[0]["x1"].shape, (2, 787))
        self.assertEqual(targets[0]["x2"].shape, (2, 58, 787))
        self.assertEqual(targets[0]["x3"].shape, (2, 58, 787))
        self.assertEqual(targets[0]["ta"].shape, (2, 58))
