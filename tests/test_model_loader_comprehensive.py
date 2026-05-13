# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pytorch_lightning as pl
import numpy as np

from pytorch_segmentation_models_trainer.model_loader.autoencoder_model import (
    AutoencoderModel,
)
from pytorch_segmentation_models_trainer.model_loader.detection_model import (
    ObjectDetectionPLModel,
    InstanceSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.student_model import (
    StudentSegmentationModel,
)
from pytorch_segmentation_models_trainer.model_loader.mod_polymapper import (
    GenericPolyMapperPLModel,
)


class TestModelLoaderComprehensive(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": 2,
                    "kernel_size": 1,
                },
                "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-3},
                "hyperparameters": {
                    "batch_size": 2,
                    "lr": 1e-3,
                    "object_detection_batch_size": 1,
                    "polygon_rnn_batch_size": 1,
                },
                "train_dataset": {
                    "data_loader": {
                        "shuffle": True,
                        "num_workers": 1,
                        "pin_memory": True,
                        "drop_last": True,
                        "prefetch_factor": 2,
                    },
                    "object_detection": {"_target_": "torch.utils.data.Dataset"},
                    "polygon_rnn": {"_target_": "torch.utils.data.Dataset"},
                },
                "val_dataset": {
                    "data_loader": {"num_workers": 1, "prefetch_factor": 2},
                    "object_detection": {"_target_": "torch.utils.data.Dataset"},
                    "polygon_rnn": {"_target_": "torch.utils.data.Dataset"},
                },
                "pl_model": {
                    "grid_size": 28,
                    "val_seq_len": 60,
                    "perform_evaluation": True,
                    "threshold": 0.5,
                },
                "detection_threshold": 0.5,
            }
        )

    def test_autoencoder_model_forward(self):
        with patch.object(AutoencoderModel, "get_model") as mock_get_model:
            with patch.object(AutoencoderModel, "get_loss_function"):
                mock_inner_model = MagicMock(side_effect=lambda x: x)
                mock_get_model.return_value = mock_inner_model

                pl_model = AutoencoderModel(self.cfg)
                x = torch.randn(1, 3, 32, 32)
                out = pl_model(x)
                self.assertTrue(torch.allclose(out, x))

    def test_autoencoder_model_steps(self):
        with patch.object(AutoencoderModel, "get_model") as mock_get_model:
            with patch.object(AutoencoderModel, "get_loss_function") as mock_get_loss:
                mock_get_model.return_value = nn.Identity()
                mock_get_loss.return_value = nn.MSELoss()

                pl_model = AutoencoderModel(self.cfg)
                pl_model.log = MagicMock()

                batch = {"image": torch.randn(2, 3, 8, 8)}

                loss_train = pl_model.training_step(batch, 0)
                self.assertIsInstance(loss_train, torch.Tensor)

                loss_val = pl_model.validation_step(batch, 0)
                self.assertIsInstance(loss_val, torch.Tensor)

    def test_compute_steps_from_image_dir_dataset_samples_per_epoch(self):
        cfg = OmegaConf.create(
            {
                "train_dataset": {
                    "_target_": (
                        "pytorch_segmentation_models_trainer.dataset_loader"
                        ".image_dataset.AutoencoderRandomCropDataset"
                    ),
                    "image_dir": "/data/unlabeled",
                    "samples_per_epoch": 10,
                    "data_loader": {"batch_size": 2},
                },
                "hyperparameters": {"devices": 1, "accumulate_grad_batches": 1},
            }
        )
        model = object.__new__(AutoencoderModel)
        model.cfg = cfg
        model.train_ds = None

        self.assertEqual(model._compute_steps_from_config(), 5)

    def test_detection_model_init(self):
        with patch.object(ObjectDetectionPLModel, "get_model"):
            with patch.object(ObjectDetectionPLModel, "get_loss_function"):
                model = InstanceSegmentationPLModel(self.cfg)
                self.assertIsInstance(model, InstanceSegmentationPLModel)
                self.assertIsInstance(model, ObjectDetectionPLModel)

    def test_student_model_weight_key_dict(self):
        self.cfg.weight_key = "custom_weight"
        with patch.object(StudentSegmentationModel, "get_model") as mock_get_model:
            with patch.object(
                StudentSegmentationModel, "get_loss_function"
            ) as mock_get_loss:
                mock_get_model.return_value = nn.Conv2d(3, 2, 1)
                mock_get_loss.return_value = nn.CrossEntropyLoss(reduction="none")

                model = StudentSegmentationModel(self.cfg)
                model.log = MagicMock()

                batch = {
                    "image": torch.randn(2, 3, 32, 32),
                    "mask": torch.randint(0, 2, (2, 32, 32)),
                    "custom_weight": torch.tensor([0.5, 1.5]),
                }
                model.train()
                loss = model.training_step(batch, 0)
                self.assertIsInstance(loss, torch.Tensor)

    def test_student_model_dict_output_for_metrics(self):
        with patch.object(StudentSegmentationModel, "get_model") as mock_get_model:
            with patch.object(
                StudentSegmentationModel, "get_loss_function"
            ) as mock_get_loss:
                inner_model = MagicMock()
                inner_model.return_value = {"probs": torch.randn(2, 2, 8, 8)}
                mock_get_model.return_value = inner_model

                loss_fn = MagicMock(return_value=torch.tensor(0.5))
                mock_get_loss.return_value = loss_fn

                model = StudentSegmentationModel(self.cfg)
                model.log = MagicMock()
                model.log_dict = MagicMock()
                model.val_metrics = MagicMock()
                model._prepare_preds_for_metrics = MagicMock(
                    return_value=torch.randn(2, 2, 8, 8)
                )

                batch = {
                    "image": torch.randn(2, 3, 8, 8),
                    "mask": torch.randint(0, 2, (2, 8, 8)),
                }
                model.eval()
                model.validation_step(batch, 0)
                model._prepare_preds_for_metrics.assert_called_once()

    def test_student_model_get_loss_reduction_fallback(self):
        with patch.object(StudentSegmentationModel, "get_model"):
            with patch.object(
                StudentSegmentationModel, "get_loss_function"
            ) as mock_get_loss:
                loss_func = MagicMock()
                if hasattr(loss_func, "reduction"):
                    del loss_func.reduction
                mock_get_loss.return_value = loss_func

                model = StudentSegmentationModel(self.cfg)
                red = model._get_loss_reduction()
                self.assertEqual(red, "mean")

    def test_student_model_set_loss_reduction_multiloss(self):
        with patch.object(StudentSegmentationModel, "get_model"):
            with patch.object(
                StudentSegmentationModel, "get_loss_function"
            ) as mock_get_loss:
                l1 = MagicMock(spec=nn.Module)
                l1.reduction = "mean"

                multiloss = MagicMock()
                multiloss.loss_funcs = [l1]
                if hasattr(multiloss, "reduction"):
                    del multiloss.reduction

                mock_get_loss.return_value = multiloss

                model = StudentSegmentationModel(self.cfg)
                model._set_loss_reduction("none")
                self.assertEqual(l1.reduction, "none")

    def test_student_model_compute_loss_weighted_scalar_loss(self):
        with patch.object(StudentSegmentationModel, "get_model"):
            with patch.object(
                StudentSegmentationModel, "get_loss_function"
            ) as mock_get_loss:
                loss_func = nn.CrossEntropyLoss(reduction="mean")
                mock_get_loss.return_value = loss_func

                model = StudentSegmentationModel(self.cfg)
                model.train()

                pred = torch.randn(2, 2, 8, 8)
                mask = torch.randint(0, 2, (2, 8, 8))
                weights = torch.tensor([1.0, 2.0])

                with patch(
                    "pytorch_segmentation_models_trainer.model_loader.model.Model._compute_loss",
                    return_value=(torch.tensor(0.5), {}, {}),
                ):
                    weighted_loss, _, _ = model._compute_loss(
                        pred, mask, weights=weights
                    )
                    self.assertIsInstance(weighted_loss, torch.Tensor)

    def test_student_model_compute_loss_weighted_ndim_fallback(self):
        with patch.object(StudentSegmentationModel, "get_model"):
            with patch.object(
                StudentSegmentationModel, "get_loss_function"
            ) as mock_get_loss:
                mock_get_loss.return_value = nn.CrossEntropyLoss()
                model = StudentSegmentationModel(self.cfg)
                model.train()

                pred = torch.randn(2, 2, 8, 8)
                mask = torch.randint(0, 2, (2, 8, 8))
                weights = torch.tensor([1.0, 2.0])

                with patch(
                    "pytorch_segmentation_models_trainer.model_loader.model.Model._compute_loss",
                    return_value=(torch.tensor([0.5, 0.6]), {}, {}),
                ):
                    weighted_loss, _, _ = model._compute_loss(
                        pred, mask, weights=weights
                    )
                    self.assertIsInstance(weighted_loss, torch.Tensor)

    def test_student_model_compute_loss_val_fallback(self):
        with patch.object(StudentSegmentationModel, "get_model"):
            with patch.object(StudentSegmentationModel, "get_loss_function"):
                model = StudentSegmentationModel(self.cfg)
                model.eval()

                pred = torch.randn(2, 2, 8, 8)
                mask = torch.randint(0, 2, (2, 8, 8))

                with patch(
                    "pytorch_segmentation_models_trainer.model_loader.model.Model._compute_loss",
                    return_value=(torch.tensor(0.5), {}, {}),
                ) as mock_super_loss:
                    model._compute_loss(pred, mask)
                    mock_super_loss.assert_called_once()

    def test_generic_polymapper_init(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mod_polymapper.instantiate"
        ) as mock_inst:
            mock_inst.side_effect = [
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]

            model = GenericPolyMapperPLModel(self.cfg)
            self.assertIsInstance(model, GenericPolyMapperPLModel)
            self.assertEqual(model.grid_size, 28)
            self.assertEqual(model.val_seq_len, 60)

    def test_generic_polymapper_dataloaders(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mod_polymapper.instantiate"
        ) as mock_inst:
            mock_model = MagicMock()
            mock_model.train_obj_detection_model = True
            mock_model.train_polygonrnn_model = True

            ds_mock = MagicMock()
            ds_mock.__len__.return_value = 10

            mock_inst.side_effect = [mock_model, ds_mock, ds_mock, ds_mock, ds_mock]

            model = GenericPolyMapperPLModel(self.cfg)

            train_loaders = model.train_dataloader()
            self.assertIn("object_detection", train_loaders)
            self.assertIn("polygon_rnn", train_loaders)

            val_loaders = model.val_dataloader()
            self.assertIsInstance(
                val_loaders, pl.utilities.combined_loader.CombinedLoader
            )

    def test_generic_polymapper_steps(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mod_polymapper.instantiate"
        ) as mock_inst:
            mock_model = MagicMock()
            mock_model.train_obj_detection_model = True
            mock_model.train_polygonrnn_model = True
            mock_model.return_value = ({"loss1": torch.tensor(0.1)}, torch.tensor(0.9))

            mock_inst.side_effect = [
                mock_model,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]

            model = GenericPolyMapperPLModel(self.cfg)
            model.log = MagicMock()
            model.perform_evaluation = True
            model.threshold = 0.5

            batch = {
                "object_detection": (torch.randn(1, 3, 8, 8), [MagicMock()], []),
                "polygon_rnn": {
                    "ta": torch.zeros(1, 10, 2),
                    "scale_h": 1.0,
                    "scale_w": 1.0,
                    "min_col": 0,
                    "min_row": 0,
                },
            }

            with patch.object(
                model,
                "evaluate_output",
                return_value={
                    "box_iou": torch.tensor([0.5]),
                    "intersection": 0.5,
                    "union": 1.0,
                },
            ):
                val_out = model.validation_step(batch, 0)
                self.assertIn("loss", val_out)
                self.assertIn("box_iou", val_out)

    def test_generic_polymapper_evaluate_output(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mod_polymapper.instantiate"
        ) as mock_inst:
            mock_inst.side_effect = [
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]
            model = GenericPolyMapperPLModel(self.cfg)

            batch = {
                "object_detection": (None, [MagicMock()], None),
                "polygon_rnn": {
                    "ta": torch.zeros(1, 1, 2),
                    "scale_h": 1.0,
                    "scale_w": 1.0,
                    "min_col": 0,
                    "min_row": 0,
                },
            }
            outputs = [MagicMock()]

            with patch.object(
                model, "_evaluate_obj_det", return_value=(torch.tensor([0.5]), 0.5)
            ):
                with patch.object(
                    model,
                    "_compute_polygonrnn_metrics",
                    return_value=(
                        torch.tensor([1.0]),
                        torch.tensor([1.0]),
                        torch.tensor([1.0]),
                    ),
                ):
                    res = model.evaluate_output(batch, outputs)
                    self.assertIn("box_iou", res)

    def test_generic_polymapper_compute_polygonrnn_metrics(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mod_polymapper.instantiate"
        ) as mock_inst:
            mock_inst.side_effect = [
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]
            model = GenericPolyMapperPLModel(self.cfg)

            outputs = [
                {
                    "polygonrnn_output": torch.zeros(1, 1, 2),
                    "scale_h": 1.0,
                    "scale_w": 1.0,
                    "min_col": 0.0,
                    "min_row": 0.0,
                }
            ]
            polygon_rnn_batch = {
                "ta": torch.zeros(1, 1, 2),
                "scale_h": 1.0,
                "scale_w": 1.0,
                "min_col": 0,
                "min_row": 0,
            }

            with patch(
                "pytorch_segmentation_models_trainer.utils.polygonrnn_utils.get_vertex_list_from_batch_tensors",
                return_value=[np.array([[0, 0], [0, 1], [1, 1], [1, 0]])],
            ):
                with patch(
                    "pytorch_segmentation_models_trainer.custom_metrics.metrics.batch_polis",
                    return_value=np.array([0.5]),
                ):
                    # Mock iou to return a simple list of floats
                    with patch(
                        "pytorch_segmentation_models_trainer.custom_metrics.metrics.polygon_iou",
                        return_value=[0.5, 0.5, 1.0],
                    ):
                        polis, inter, union = model._compute_polygonrnn_metrics(
                            outputs, polygon_rnn_batch
                        )
                        self.assertEqual(polis.item(), 0.5)

    def test_generic_polymapper_optimizers_with_scheduler(self):
        self.cfg.scheduler_list = [
            {
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 1,
                }
            }
        ]
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mod_polymapper.instantiate"
        ) as mock_inst:
            mock_model = MagicMock()
            mock_opt = MagicMock()
            mock_sched = MagicMock()

            mock_inst.side_effect = [
                mock_model,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                mock_opt,
                mock_sched,
            ]

            model = GenericPolyMapperPLModel(self.cfg)
            opts, scheds = model.configure_optimizers()
            self.assertEqual(len(scheds), 1)

    def test_generic_polymapper_predict_step(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mod_polymapper.instantiate"
        ) as mock_inst:
            mock_inst.side_effect = [
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]
            model = GenericPolyMapperPLModel(self.cfg)
            model.model = MagicMock(
                return_value=[{"box": torch.tensor([0, 0, 10, 10])}]
            )

            batch = {"image": torch.randn(1, 3, 32, 32)}
            res = model.predict_step(batch, 0)
            self.assertEqual(len(res), 1)


if __name__ == "__main__":
    unittest.main()
