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

import json
import os
from pathlib import Path

import albumentations as A
import geopandas
import hydra
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from collections import OrderedDict
from hydra import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.custom_models.models import (
    ObjectDetectionModel,
)
from pytorch_segmentation_models_trainer.custom_models.rnn.polygon_rnn import PolygonRNN
from pytorch_segmentation_models_trainer.dataset_loader.dataset import PolygonRNNDataset
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    VectorFileDataWriter,
)
from pytorch_segmentation_models_trainer.tools.inference.export_inference import (
    MultipleRasterExportInferenceStrategy,
    ObjectDetectionExportInferenceStrategy,
    RasterExportInferenceStrategy,
)
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    ObjectDetectionInferenceProcessor,
    PolygonRNNInferenceProcessor,
    SingleImageFromFrameFieldProcessor,
    SingleImageInfereceProcessor,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    ACMConfig,
    ACMPolygonizerProcessor,
    ASMConfig,
    ASMPolygonizerProcessor,
    PolygonRNNConfig,
    PolygonRNNPolygonizerProcessor,
    SimplePolConfig,
    SimplePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
)
from tests.mock_utils import create_dummy_checkpoint
from tests.utils import BasicTestCase
from unittest.mock import Mock

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)

frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)

device = "cpu" if not torch.cuda.is_available() else "cuda"

pretrained_checkpoints_download_links = {
    "frame_field_resnet152_unet_200_epochs": "https://github.com/phborba/pytorch_smt_pretrained_weights/releases/download/v0.1/frame_field_resnet152_unet_200_epochs.ckpt"
}


class Test_Inference(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()
        self.frame_field_ds = self.get_frame_field_ds()
        with rasterio.open(self.frame_field_ds[0]["path"], "r") as raster_ds:
            self.crs = raster_ds.crs
            self.profile = raster_ds.profile
            self.transform = raster_ds.transform
        self.polygonizers_dict = {
            "simple": self.get_simple_polygonizer(),
            "acm": self.get_acm_polygonizer(),
            "asm": self.get_asm_polygonizer(),
            "polygonrnn": self.get_polygonrnn_polygonizer(),
        }
        self.center_crop = A.CenterCrop(512, 512)

    def get_frame_field_ds(self, with_center_crop=False):
        config_name = (
            "frame_field_dataset.yaml"
            if not with_center_crop
            else "frame_field_dataset_with_center_crop.yaml"
        )
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs", version_base=None):
            cfg = compose(
                config_name=config_name,
                overrides=[
                    "input_csv_path=" + csv_path,
                    "root_dir=" + frame_field_root_dir,
                ],
            )
            frame_field_ds = hydra.utils.instantiate(cfg, _recursive_=False)
        return frame_field_ds

    def get_simple_polygonizer(self):
        config = SimplePolConfig()
        self.simple_output_file_path = os.path.join(
            self.output_dir, "simple_polygonizer.geojson"
        )
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="simple_polygonizer.geojson",
        )
        return SimplePolygonizerProcessor(data_writer=data_writer, config=config)

    def get_acm_polygonizer(self):
        config = ACMConfig()
        self.acm_output_file_path = os.path.join(
            self.output_dir, "acm_polygonizer.geojson"
        )
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="acm_polygonizer.geojson",
        )
        return ACMPolygonizerProcessor(data_writer=data_writer, config=config)

    def get_asm_polygonizer(self):
        config = ASMConfig()
        self.asm_output_file_path = os.path.join(
            self.output_dir, "asm_polygonizer.geojson"
        )
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="asm_polygonizer.geojson",
        )
        return ASMPolygonizerProcessor(data_writer=data_writer, config=config)

    def get_polygonrnn_polygonizer(self):
        config = PolygonRNNConfig()
        self.polygonrnn_output_file_path = os.path.join(
            self.output_dir, "polygonrnn_polygonizer.geojson"
        )
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="polygonrnn_polygonizer.geojson",
        )
        return PolygonRNNPolygonizerProcessor(data_writer=data_writer, config=config)

    def get_model_for_eval(self):
        ds_with_center_crop = self.get_frame_field_ds(with_center_crop=True)
        b1, b2, _ = ds_with_center_crop[0]["gt_polygons_image"]
        mock_model = Mock(spec=FrameFieldSegmentationPLModel)
        mock_model.return_value = OrderedDict(
            {
                "seg": torch.stack([b1, b2]).unsqueeze(0),
                "crossfield": ds_with_center_crop[0]["gt_crossfield_angle"].unsqueeze(
                    0
                ),
            }
        )
        mock_model.eval()
        return mock_model

    def get_checkpoint_file(self, file_name):
        checkpoint_folder = create_folder(os.path.join(root_dir, "data", "checkpoints"))
        ckeckpoint_file_path = os.path.join(checkpoint_folder, file_name)
        if not os.path.isfile(ckeckpoint_file_path):
            create_dummy_checkpoint(ckeckpoint_file_path)
        return ckeckpoint_file_path

    def test_create_inference_from_inference_processor(self) -> None:

        output_file_path = os.path.join(self.output_dir, "output.tif")
        inference_processor = SingleImageInfereceProcessor(
            model=smp.Unet(),
            device=device,
            batch_size=1,
            export_strategy=RasterExportInferenceStrategy(
                output_file_path=output_file_path
            ),
        )
        inference_processor.process(image_path=self.frame_field_ds[0]["path"])
        assert os.path.isfile(output_file_path)

    @parameterized.expand([(False,), (True,)])
    def test_create_frame_field_inference_from_inference_processor(
        self, with_polygonizer
    ) -> None:
        inference_processor = SingleImageFromFrameFieldProcessor(
            model=FrameFieldModel(
                segmentation_model=smp.Unet(),
                seg_params={
                    "compute_interior": True,
                    "compute_edge": True,
                    "compute_vertex": False,
                },
            ),
            device=device,
            batch_size=1,
            export_strategy=MultipleRasterExportInferenceStrategy(
                output_folder=self.output_dir, output_basename="output.tif"
            ),
            mask_bands=2,
            polygonizer=self.polygonizers_dict["simple"] if with_polygonizer else None,
        )
        inference_processor.process(image_path=self.frame_field_ds[0]["path"])
        name = Path(self.frame_field_ds[0]["path"]).stem
        assert os.path.isfile(os.path.join(self.output_dir, f"seg_{name}_output.tif"))
        assert os.path.isfile(
            os.path.join(self.output_dir, f"crossfield_{name}_output.tif")
        )

    @parameterized.expand(
        [("simple", False), ("acm", False), ("asm", False), ("simple", True)]
    )
    def test_create_frame_field_inference_from_pretrained_with_polygonize(
        self, polygonizer_key, group_output_by_image_basename
    ) -> None:
        inference_processor = SingleImageFromFrameFieldProcessor(
            model=self.get_model_for_eval(),
            device=device,
            batch_size=1,
            export_strategy=MultipleRasterExportInferenceStrategy(
                output_folder=self.output_dir,
                output_basename=f"{polygonizer_key}_output.tif",
            ),
            mask_bands=2,
            polygonizer=self.polygonizers_dict[polygonizer_key],
            group_output_by_image_basename=group_output_by_image_basename,
        )
        inference_processor.process(
            image_path=self.frame_field_ds[0]["path"], threshold=0.5
        )
        name = Path(self.frame_field_ds[0]["path"]).stem
        # alterar asserts para tratar caso do encapsulate with folder
        assert os.path.isfile(
            os.path.join(self.output_dir, f"seg_{name}_{polygonizer_key}_output.tif")
        )
        assert os.path.isfile(
            os.path.join(
                self.output_dir, f"crossfield_{name}_{polygonizer_key}_output.tif"
            )
        )
        assert (
            os.path.isfile(getattr(self, f"{polygonizer_key}_output_file_path"))
            if not group_output_by_image_basename
            else os.path.isfile(
                self.polygonizers_dict[
                    polygonizer_key
                ].data_writer.get_output_file_path(name)
            )
        )

    def test_create_polygonrnn_inference_from_model_with_polygonize(self) -> None:
        csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        polygon_rnn_ds = PolygonRNNDataset(
            input_csv_path=csv_path,
            sequence_length=60,
            root_dir=polygon_rnn_root_dir,
            augmentation_list=[A.Normalize(), A.pytorch.ToTensorV2()],
            dataset_type="val",
        )
        mock_model = Mock(spec=PolygonRNN)
        mock_model.test.return_value = polygon_rnn_ds[0]["ta"].unsqueeze(0)
        inference_processor = PolygonRNNInferenceProcessor(
            model=mock_model,
            batch_size=2,
            device=device,
            polygonizer=self.polygonizers_dict["polygonrnn"],
        )
        inference_processor.process(
            image_path=polygon_rnn_ds[0]["original_image_path"],
            bboxes=[np.array([0, 0, 224, 224])],
        )
        assert os.path.isfile(
            os.path.join(self.output_dir, f"polygonrnn_polygonizer.geojson")
        )
        gdf = geopandas.read_file(
            os.path.join(self.output_dir, f"polygonrnn_polygonizer.geojson")
        )
        self.assertEqual(len(gdf), 1)

    def test_create_object_detection_inference_from_model(self) -> None:
        csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        polygon_rnn_ds = PolygonRNNDataset(
            input_csv_path=csv_path,
            sequence_length=60,
            root_dir=polygon_rnn_root_dir,
            augmentation_list=[A.Normalize(), A.pytorch.ToTensorV2()],
            dataset_type="val",
        )
        mock_model = Mock(spec=ObjectDetectionModel)
        mock_model.return_value = [
            {
                "bboxes": torch.tensor([[10, 10, 50, 50], [110, 110, 150, 150]]),
                "scores": torch.tensor([0.9, 0.5]),
                "labels": torch.tensor([1, 1]),
            }
        ]
        inference_processor = ObjectDetectionInferenceProcessor(
            model=mock_model,
            device=device,
            batch_size=1,
            model_input_shape=(400, 400),
            step_shape=(200, 200),
            export_strategy=ObjectDetectionExportInferenceStrategy(
                os.path.join(self.output_dir, f"obj_det_result.json")
            ),
        )
        result = inference_processor.process(
            image_path=polygon_rnn_ds[0]["original_image_path"], output_inferences=True
        )
        assert os.path.isfile(os.path.join(self.output_dir, f"obj_det_result.json"))
        with open(os.path.join(self.output_dir, f"obj_det_result.json")) as f:
            result_list = json.load(f)
        self.assertEqual(len(result_list), 18)

    # ------------------------------------------------------------------
    # TTA tests
    # ------------------------------------------------------------------

    @parameterized.expand(
        [
            ("rotations_4", ["rot0", "rot90", "rot180", "rot270"]),
            (
                "d4_group",
                [
                    "rot0",
                    "rot90",
                    "rot180",
                    "rot270",
                    "flip_h",
                    "flip_v",
                    "flip_h_rot90",
                    "flip_v_rot90",
                ],
            ),
            ("two_rots", ["rot0", "rot180"]),
        ]
    )
    def test_tta_produces_output_file(self, _name, tta_augmentations) -> None:
        """TTA-enabled processor must write a valid output file."""
        output_file_path = os.path.join(self.output_dir, f"output_tta_{_name}.tif")
        inference_processor = SingleImageInfereceProcessor(
            model=smp.Unet(),
            device=device,
            batch_size=1,
            export_strategy=RasterExportInferenceStrategy(
                output_file_path=output_file_path
            ),
            use_tta=True,
            tta_augmentations=tta_augmentations,
        )
        inference_processor.process(image_path=self.frame_field_ds[0]["path"])
        assert os.path.isfile(output_file_path)

    def test_tta_output_shape_matches_no_tta(self) -> None:
        """TTA result must have the same spatial shape as a standard run."""
        image_path = self.frame_field_ds[0]["path"]
        model = smp.Unet()

        def _run(use_tta):
            out_path = os.path.join(self.output_dir, f"shape_check_tta_{use_tta}.tif")
            proc = SingleImageInfereceProcessor(
                model=model,
                device=device,
                batch_size=1,
                export_strategy=RasterExportInferenceStrategy(
                    output_file_path=out_path
                ),
                use_tta=use_tta,
                tta_augmentations=["rot0", "rot90", "rot180", "rot270"],
            )
            proc.process(image_path=image_path)
            with rasterio.open(out_path) as ds:
                return ds.shape

        shape_no_tta = _run(False)
        shape_with_tta = _run(True)
        self.assertEqual(shape_no_tta, shape_with_tta)

    def test_tta_disabled_by_default(self) -> None:
        """Processor created without use_tta must not use TTA."""
        proc = SingleImageInfereceProcessor(
            model=smp.Unet(),
            device=device,
            batch_size=1,
            export_strategy=None,
        )
        self.assertFalse(proc.use_tta)

    def test_tta_frame_field_processor_produces_output(self) -> None:
        """Frame-field processor with TTA must write seg and crossfield files."""
        inference_processor = SingleImageFromFrameFieldProcessor(
            model=self.get_model_for_eval(),
            device=device,
            batch_size=1,
            export_strategy=MultipleRasterExportInferenceStrategy(
                output_folder=self.output_dir,
                output_basename="tta_ff_output.tif",
            ),
            mask_bands=2,
            use_tta=True,
            tta_augmentations=["rot0", "rot90", "rot180", "rot270"],
        )
        inference_processor.process(image_path=self.frame_field_ds[0]["path"])
        name = Path(self.frame_field_ds[0]["path"]).stem
        assert os.path.isfile(
            os.path.join(self.output_dir, f"seg_{name}_tta_ff_output.tif")
        )
        assert os.path.isfile(
            os.path.join(self.output_dir, f"crossfield_{name}_tta_ff_output.tif")
        )
