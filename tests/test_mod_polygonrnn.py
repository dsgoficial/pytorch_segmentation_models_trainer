# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-08-16
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
import subprocess
from typing import Dict, Optional
import unittest
from importlib import import_module

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import hydra
import numpy as np
from pytorch_lightning.trainer.trainer import Trainer
import segmentation_models_pytorch as smp
import torch
from hydra.experimental import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.custom_models import models as pytorch_smt_cm
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    ModPolyMapperDataset,
    ObjectDetectionDataset,
    PolygonRNNDataset,
)
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.mod_polymapper import (
    GenericPolyMapperPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.polygon_rnn_model import (
    PolygonRNN,
)
from pytorch_segmentation_models_trainer.custom_models.mod_polymapper.modpolymapper import (
    ModPolyMapper,
)
from pytorch_segmentation_models_trainer.train import train
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils

from tests.utils import BasicTestCase, CustomTestCase, get_config_from_hydra

from unittest.mock import MagicMock, Mock, patch

current_dir = os.path.dirname(__file__)
detection_root_dir = os.path.join(
    current_dir, "testing_data", "data", "detection_data", "geo"
)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)
csv_path = os.path.join(detection_root_dir, "dsg_dataset.csv")
poly_csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
torch.manual_seed(0)


def mock_model_return(
    obj_det_images: torch.Tensor,
    obj_det_targets: Optional[torch.Tensor] = None,
    polygon_rnn_batch: Optional[Dict[str, torch.Tensor]] = None,
    threshold: Optional[float] = None,
):
    print("Mocking model")
    if obj_det_targets is None and polygon_rnn_batch is None:
        batch_size = obj_det_images.shape[0]
        polygon1 = np.array([[100, 100], [100, 204], [204, 204], [204, 100]])
        polygon2 = np.array([[204, 204], [220, 204], [220, 220]])
        _, label_index_array1 = polygonrnn_utils.build_arrays(polygon1, 4, 60)
        _, label_index_array2 = polygonrnn_utils.build_arrays(polygon2, 3, 60)
        batch = np.stack([label_index_array1[2::], label_index_array2[2::]], axis=0)
        batch_tensor = torch.from_numpy(batch).float().to(obj_det_images.device)
        return [
            {
                "boxes": torch.randint(
                    size=(batch_size, 4), high=255, device=obj_det_images.device
                ),
                "labels": torch.ones(
                    batch_size, device=obj_det_images.device, dtype=torch.int64
                ),
                "scores": torch.randn(batch_size, device=obj_det_images.device),
                "min_row": torch.randint(
                    size=(batch_size,), high=255, device=obj_det_images.device
                ),
                "min_col": torch.randint(
                    size=(batch_size,), high=255, device=obj_det_images.device
                ),
                "scale_h": torch.ones(
                    batch_tensor.shape[0], device=obj_det_images.device
                ),
                "scale_w": torch.ones(
                    batch_tensor.shape[0], device=obj_det_images.device
                ),
                "polygonrnn_output": batch_tensor,
            }
            for _ in range(batch_size)
        ]
    return (
        {
            key: torch.randn(1, requires_grad=True)
            for key in [
                "loss_classifier",
                "loss_box_reg",
                "loss_objectness",
                "loss_rpn_box_reg",
                "polygonrnn_loss",
            ]
        },
        torch.tensor(0.0, device=obj_det_images.device),
    )


class Test_ModPolyMapperModel(BasicTestCase):
    def _get_model(
        self, backbone_trainable_layers=3, pretrained=False, **kwargs
    ) -> ModPolyMapper:
        return ModPolyMapper(
            num_classes=2,
            backbone_trainable_layers=backbone_trainable_layers,
            pretrained=pretrained,
            **kwargs,
        )

    def _get_dataloader(self) -> torch.utils.data.DataLoader:
        obj_det_ds = ObjectDetectionDataset(
            input_csv_path=csv_path,
            root_dir=os.path.dirname(csv_path),
            augmentation_list=A.Compose(
                [A.CenterCrop(512, 512), A.Normalize(), ToTensorV2()],
                bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
            ),
        )
        polygon_rnn_dataset = PolygonRNNDataset(
            input_csv_path=poly_csv_path,
            sequence_length=60,
            root_dir=polygon_rnn_root_dir,
            augmentation_list=A.Compose([A.Normalize(), ToTensorV2()]),
        )
        return {
            "object_detection_dataloader": torch.utils.data.DataLoader(
                obj_det_ds,
                batch_size=2,
                shuffle=False,
                drop_last=True,
                num_workers=1,
                collate_fn=obj_det_ds.collate_fn,
            ),
            "polygon_rnn_dataloader": torch.utils.data.DataLoader(
                polygon_rnn_dataset,
                batch_size=2,
                shuffle=False,
                drop_last=True,
                num_workers=1,
            ),
        }

    def test_model_forward(self) -> None:
        model = self._get_model()
        sample = torch.randn([2, 3, 256, 256])
        model.eval()
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0].keys()), 9)

    def test_model_backwards(self) -> None:
        model = self._get_model(backbone_trainable_layers=5, pretrained=True)
        data_loader = self._get_dataloader()
        model.train()
        obj_det_images, obj_det_targets, index = next(
            iter(data_loader["object_detection_dataloader"])
        )
        polygon_rnn_batch = next(iter(data_loader["polygon_rnn_dataloader"]))
        losses, acc = model(obj_det_images, obj_det_targets, polygon_rnn_batch)
        loss = sum(losses.values())
        loss.backward()
        for n, x in model.named_parameters():
            assert x.grad is not None, f"No gradient for {n}"

    @parameterized.expand(
        [
            ({"train_obj_detection_model": False}, True),
            ({"train_obj_detection_model": False, "train_backbone": False}, False),
            ({"train_polygonrnn_model": False}, True),
        ]
    )
    def test_set_model_components_trainable(
        self, trainable_params_dict, exclude_backbone
    ) -> None:
        model = self._get_model(
            backbone_trainable_layers=5, pretrained=False, **trainable_params_dict
        )
        for param_key in trainable_params_dict:
            model_component = getattr(model, param_key.replace("train_", ""))
            trainable_params = sum(
                p.numel() for p in model_component.parameters() if p.requires_grad
            )
            if exclude_backbone:
                trainable_params = trainable_params - sum(
                    p.numel()
                    for p in model.backbone.parameters()
                    if p.requires_grad  # type: ignore
                )
            self.assertEqual(trainable_params, 0)

    def test_raises_exception(self):
        self.assertRaises(
            ValueError,
            self._get_model,
            backbone_trainable_layers=5,
            pretrained=False,
            train_obj_detection_model=False,
            train_polygonrnn_model=False,
        )

    @parameterized.expand(
        [
            (
                "experiment_mod_polymapper.yaml",
                ["+pl_trainer.fast_dev_run=true", "+pl_model.perform_evaluation=true"],
            ),
            ("experiment_mod_polymapper_with_callback.yaml", None),
        ]
    )
    @unittest.skipIf(
        not torch.cuda.is_available(),
        reason="No GPU available, test is too memory expensive to be run on CPU",
    )
    def test_pl_model_train(self, experiment_name, extra_overrides=None) -> None:
        extra_overrides = extra_overrides if extra_overrides is not None else []
        cfg = get_config_from_hydra(
            config_name=experiment_name,
            overrides_list=[
                f"train_dataset.object_detection.input_csv_path={csv_path}",
                f"train_dataset.object_detection.root_dir={detection_root_dir}",
                f"train_dataset.polygon_rnn.input_csv_path={poly_csv_path}",
                f"train_dataset.polygon_rnn.root_dir={polygon_rnn_root_dir}",
                f"val_dataset.object_detection.input_csv_path={csv_path}",
                f"val_dataset.object_detection.root_dir={detection_root_dir}",
                f"val_dataset.polygon_rnn.input_csv_path={poly_csv_path}",
                f"val_dataset.polygon_rnn.root_dir={polygon_rnn_root_dir}",
                "pl_trainer.gpus=1",
            ]
            + extra_overrides,
        )
        trainer = train(cfg)
        return

    # @parameterized.expand(
    #     [
    #         (
    #             "experiment_mod_polymapper_with_callback.yaml",
    #             ["+pl_trainer.fast_dev_run=true", "+pl_model.perform_evaluation=true"],
    #         )
    #     ]
    # )
    # @unittest.skipIf(
    #     torch.cuda.is_available(),
    #     reason="GPU is available, features already tested in other tests.",
    # )
    # def test_pl_model(self, experiment_name, extra_overrides=None) -> None:
    #     extra_overrides = extra_overrides if extra_overrides is not None else []
    #     cfg = get_config_from_hydra(
    #         config_name=experiment_name,
    #         overrides_list=[
    #             f"train_dataset.object_detection.input_csv_path={csv_path}",
    #             f"train_dataset.object_detection.root_dir={detection_root_dir}",
    #             f"train_dataset.polygon_rnn.input_csv_path={poly_csv_path}",
    #             f"train_dataset.polygon_rnn.root_dir={polygon_rnn_root_dir}",
    #             f"val_dataset.object_detection.input_csv_path={csv_path}",
    #             f"val_dataset.object_detection.root_dir={detection_root_dir}",
    #             f"val_dataset.polygon_rnn.input_csv_path={poly_csv_path}",
    #             f"val_dataset.polygon_rnn.root_dir={polygon_rnn_root_dir}",
    #         ]
    #         + extra_overrides,
    #     )
    #     with patch.object(
    #         GenericPolyMapperPLModel, "get_optimizer"
    #     ) as mock_get_optimizer:
    #         dummy_model = torch.nn.Sequential(
    #             torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1)
    #         )
    #         mock_get_optimizer.return_value = torch.optim.AdamW(
    #             dummy_model.parameters()
    #         )
    #         pl_model = GenericPolyMapperPLModel(cfg)
    #         mock_model = MagicMock(spec=ModPolyMapper, side_effect=mock_model_return)
    #         mock_model.train_obj_detection_model = True
    #         mock_model.train_polygonrnn_model = True
    #         pl_model.model = mock_model
    #         trainer = Trainer(**cfg.pl_trainer)
    #         trainer.fit(pl_model)
