# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.domain_adaptation.callbacks.monitor_callback import (
    DomainAdaptationMonitorCallback,
)


class _DummyDataset(Dataset):
    def __init__(self, size=4, num_classes=2, h=8, w=8, constant_mask=None):
        self.size = size
        self.num_classes = num_classes
        self.h = h
        self.w = w
        self.constant_mask = constant_mask

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.constant_mask is not None:
            mask = torch.full((self.h, self.w), self.constant_mask, dtype=torch.long)
        else:
            mask = torch.randint(0, self.num_classes, (self.h, self.w))
        return {
            "image": torch.randn(3, self.h, self.w),
            "mask": mask,
        }


class _DummyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        # Return something deterministic if we want to test exact IoU
        # For now, just return zeros for class 0 and high values for class 1
        out = torch.zeros((x.shape[0], self.num_classes, x.shape[2], x.shape[3]))
        if self.num_classes > 1:
            out[:, 1, :, :] = 10.0
        return out


class TestDomainAdaptationMonitorCallback(unittest.TestCase):
    def setUp(self):
        self.num_classes = 2
        self.callback = DomainAdaptationMonitorCallback(
            num_classes=self.num_classes,
            log_every_n_epochs=1,
            forgetting_threshold=0.05,
            eval_batch_size=2,
        )
        self.pl_module = MagicMock(spec=pl.LightningModule)
        self.pl_module.device = torch.device("cpu")
        self.pl_module.model = _DummyModel(num_classes=self.num_classes)

        self.trainer = MagicMock(spec=pl.Trainer)
        self.trainer.current_epoch = 0
        self.trainer.logger = MagicMock()

    def test_init(self):
        cb = DomainAdaptationMonitorCallback(num_classes=3, class_names=["a", "b", "c"])
        self.assertEqual(cb.num_classes, 3)
        self.assertEqual(cb.class_names, ["a", "b", "c"])

        cb2 = DomainAdaptationMonitorCallback(num_classes=2)
        self.assertEqual(cb2.class_names, ["Class 0", "Class 1"])

    def test_on_fit_start_no_source_ds(self):
        self.pl_module.source_val_ds = None
        with self.assertLogs(level="WARNING") as cm:
            self.callback.on_fit_start(self.trainer, self.pl_module)
        self.assertIn("source_val_ds is None", cm.output[0])
        self.assertIsNone(self.callback._source_baseline_iou)

    def test_on_fit_start_with_source_ds(self):
        self.pl_module.source_val_ds = _DummyDataset(constant_mask=1)
        self.callback.on_fit_start(self.trainer, self.pl_module)

        # _DummyModel always predicts class 1.
        # Mask is always 1.
        # Intersection for class 1: all pixels. Union for class 1: all pixels. IoU=1.0.
        # Intersection for class 0: 0. Union for class 0: 0. IoU=0.0.
        # Mean IoU = (1.0 + 0.0) / 2 = 0.5.
        self.assertAlmostEqual(self.callback._source_baseline_iou, 0.5)
        self.trainer.logger.experiment.add_scalar.assert_called_once()

    def test_on_fit_start_no_logger(self):
        self.pl_module.source_val_ds = _DummyDataset(constant_mask=1)
        self.trainer.logger = None
        # Should not raise
        self.callback.on_fit_start(self.trainer, self.pl_module)
        self.assertAlmostEqual(self.callback._source_baseline_iou, 0.5)

    def test_on_validation_epoch_end_frequency(self):
        self.callback.log_every_n_epochs = 2
        self.trainer.current_epoch = 1

        # Should return early
        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)
        self.pl_module.log.assert_not_called()

    def test_on_validation_epoch_end_logging(self):
        self.pl_module.source_val_ds = _DummyDataset(constant_mask=1)
        self.pl_module.target_val_ds = _DummyDataset(constant_mask=0)
        self.callback._source_baseline_iou = 0.5

        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)

        calls = [c[1][0] for c in self.pl_module.log.mock_calls]
        self.assertIn("iou/source_val", calls)
        self.assertIn("iou/target_val", calls)
        self.assertIn("iou/source_drop_from_baseline", calls)
        self.assertIn("iou/gap_source_minus_target", calls)

    def test_on_validation_epoch_end_source_only(self):
        self.pl_module.source_val_ds = _DummyDataset(constant_mask=1)
        self.pl_module.target_val_ds = None
        self.callback._source_baseline_iou = 0.5

        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)

        calls = [c[1][0] for c in self.pl_module.log.mock_calls]
        self.assertIn("iou/source_val", calls)
        self.assertNotIn("iou/target_val", calls)
        self.assertNotIn("iou/gap_source_minus_target", calls)

    def test_on_validation_epoch_end_target_only(self):
        self.pl_module.source_val_ds = None
        self.pl_module.target_val_ds = _DummyDataset(constant_mask=0)

        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)

        calls = [c[1][0] for c in self.pl_module.log.mock_calls]
        self.assertNotIn("iou/source_val", calls)
        self.assertIn("iou/target_val", calls)
        self.assertNotIn("iou/gap_source_minus_target", calls)

    def test_on_validation_epoch_end_no_baseline(self):
        self.pl_module.source_val_ds = _DummyDataset(constant_mask=1)
        self.callback._source_baseline_iou = None

        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)

        calls = [c[1][0] for c in self.pl_module.log.mock_calls]
        self.assertIn("iou/source_val", calls)
        self.assertNotIn("iou/source_drop_from_baseline", calls)

    def test_forgetting_warning(self):
        self.pl_module.source_val_ds = _DummyDataset(constant_mask=0)  # IoU will be 0.0
        self.callback._source_baseline_iou = 0.8  # High baseline
        self.callback.forgetting_threshold = 0.1

        with self.assertLogs(level="WARNING") as cm:
            self.callback.on_validation_epoch_end(self.trainer, self.pl_module)

        self.assertTrue(
            any("Possible catastrophic forgetting" in msg for msg in cm.output)
        )

    def test_compute_mean_iou_4d_mask(self):
        class _4DMaskDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {
                    "image": torch.randn(3, 8, 8),
                    "mask": torch.zeros((1, 8, 8), dtype=torch.long),
                }

        ds = _4DMaskDataset()
        iou = self.callback._compute_mean_iou(self.pl_module, ds)
        # Preds are all 1, mask is all 0. IoU should be 0.0.
        self.assertEqual(iou, 0.0)


if __name__ == "__main__":
    unittest.main()
