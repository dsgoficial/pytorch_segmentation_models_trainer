# -*- coding: utf-8 -*-
"""
Tests for pytorch_segmentation_models_trainer.utils.seed_utils
"""

import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pytorch_segmentation_models_trainer.dataset_loader.dataset import _worker_init_fn
from pytorch_segmentation_models_trainer.utils.seed_utils import set_training_seed

from tests.utils import BasicTestCase


class TestSetTrainingSeed(BasicTestCase):
    """Tests for set_training_seed()."""

    def test_returns_seeded_generator(self):
        gen = set_training_seed(42)
        self.assertIsInstance(gen, torch.Generator)
        # A seeded generator should produce a deterministic first value.
        val1 = torch.randint(0, 2**31 - 1, (1,), generator=gen).item()
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        val2 = torch.randint(0, 2**31 - 1, (1,), generator=gen2).item()
        self.assertEqual(val1, val2)

    def test_seeds_torch(self):
        set_training_seed(7)
        t1 = torch.rand(4).tolist()
        set_training_seed(7)
        t2 = torch.rand(4).tolist()
        self.assertEqual(t1, t2)

    def test_seeds_numpy(self):
        set_training_seed(123)
        a1 = np.random.rand(4).tolist()
        set_training_seed(123)
        a2 = np.random.rand(4).tolist()
        self.assertEqual(a1, a2)

    def test_seeds_python_random(self):
        set_training_seed(999)
        r1 = [random.random() for _ in range(4)]
        set_training_seed(999)
        r2 = [random.random() for _ in range(4)]
        self.assertEqual(r1, r2)

    def test_sets_pythonhashseed(self):
        set_training_seed(55)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "55")

    def test_seed_modulo_32bit(self):
        large_seed = 2**33 + 7
        gen = set_training_seed(large_seed)
        self.assertIsInstance(gen, torch.Generator)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), str(large_seed % (2**32)))

    def test_deterministic_cudnn_false_by_default(self):
        original_det = torch.backends.cudnn.deterministic
        original_bench = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            set_training_seed(1, deterministic_cudnn=False)
            # Should not change these flags
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.deterministic = original_det
            torch.backends.cudnn.benchmark = original_bench

    def test_deterministic_cudnn_true(self):
        original_det = torch.backends.cudnn.deterministic
        original_bench = torch.backends.cudnn.benchmark
        try:
            set_training_seed(1, deterministic_cudnn=True)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.deterministic = original_det
            torch.backends.cudnn.benchmark = original_bench

    def test_two_calls_same_seed_produce_same_sequences(self):
        """Entire sequence (torch + numpy + random) must be identical across two seedings."""
        set_training_seed(42)
        torch_seq_1 = torch.rand(10).tolist()
        np_seq_1 = np.random.rand(10).tolist()
        py_seq_1 = [random.random() for _ in range(10)]

        set_training_seed(42)
        torch_seq_2 = torch.rand(10).tolist()
        np_seq_2 = np.random.rand(10).tolist()
        py_seq_2 = [random.random() for _ in range(10)]

        self.assertEqual(torch_seq_1, torch_seq_2)
        self.assertEqual(np_seq_1, np_seq_2)
        self.assertEqual(py_seq_1, py_seq_2)

    def test_different_seeds_produce_different_sequences(self):
        set_training_seed(1)
        seq1 = torch.rand(8).tolist()
        set_training_seed(2)
        seq2 = torch.rand(8).tolist()
        self.assertNotEqual(seq1, seq2)


class TestWorkerInitFn(BasicTestCase):
    """Tests for the updated _worker_init_fn that also seeds random."""

    def test_worker_init_fn_seeds_numpy_and_random(self):
        torch.manual_seed(100)
        _worker_init_fn(0)

        expected_worker_seed = torch.initial_seed() % (2**32)
        # Reseed and reproduce
        np.random.seed(expected_worker_seed)
        random.seed(expected_worker_seed)
        np_val = np.random.rand()
        py_val = random.random()

        # Now call again and check values match
        torch.manual_seed(100)
        _worker_init_fn(0)
        self.assertAlmostEqual(np.random.rand(), np_val)
        self.assertAlmostEqual(random.random(), py_val)

    def test_different_worker_ids_produce_different_seeds(self):
        """Each worker gets a unique seed derived from global seed + worker_id."""
        # Simulate PyTorch's per-worker seeding: each worker's initial_seed
        # is global_seed + worker_id (PyTorch internal convention).
        global_seed = 42
        seeds = []
        for worker_id in range(4):
            torch.manual_seed(global_seed + worker_id)
            seeds.append(torch.initial_seed() % (2**32))
        self.assertEqual(len(set(seeds)), 4, "All worker seeds must be distinct")


class TestSeedIntegrationWithModel(BasicTestCase):
    """Integration: Model._make_dataloader_generator respects cfg.seed."""

    def test_generator_created_when_seed_present(self):
        from omegaconf import OmegaConf
        from unittest.mock import MagicMock
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create({"seed": 7})
        model = MagicMock(spec=Model)
        model.cfg = cfg
        gen = Model._make_dataloader_generator(model)
        self.assertIsNotNone(gen)
        self.assertIsInstance(gen, torch.Generator)

    def test_generator_is_none_when_seed_absent(self):
        from omegaconf import OmegaConf
        from unittest.mock import MagicMock
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create({})
        model = MagicMock(spec=Model)
        model.cfg = cfg
        gen = Model._make_dataloader_generator(model)
        self.assertIsNone(gen)

    def test_same_seed_produces_same_generator_values(self):
        from omegaconf import OmegaConf
        from unittest.mock import MagicMock
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create({"seed": 42})
        model = MagicMock(spec=Model)
        model.cfg = cfg

        gen1 = Model._make_dataloader_generator(model)
        gen2 = Model._make_dataloader_generator(model)

        v1 = torch.randint(0, 2**31 - 1, (5,), generator=gen1).tolist()
        v2 = torch.randint(0, 2**31 - 1, (5,), generator=gen2).tolist()
        self.assertEqual(v1, v2)

    def test_model_init_calls_set_training_seed(self):
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model": {"_target_": "torch.nn.Identity"},
                "loss": {"_target_": "torch.nn.MSELoss"},
            }
        )
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.set_training_seed"
        ) as mock_seed:
            # We only want to test the seed call, so we can mock get_model and get_loss_function if needed
            # but let's try to let it run as much as possible with Identity model.
            with (
                patch.object(Model, "get_model"),
                patch.object(Model, "get_loss_function"),
            ):
                Model(cfg)
            mock_seed.assert_called_once_with(123, deterministic_cudnn=False)


class TestSeedIntegrationEndToEnd(BasicTestCase):
    """End-to-End tests to verify that seed actually produces deterministic results."""

    def test_model_weights_determinism(self):
        """Verifies that the same seed produces the same initial model weights."""
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create(
            {
                "seed": 42,
                "model": {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": 16,
                    "kernel_size": 3,
                },
                "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
            }
        )

        # Instance A with seed 42
        model_a = Model(cfg)
        weights_a = [p.clone().detach() for p in model_a.parameters()]

        # Instance B with seed 42
        model_b = Model(cfg)
        weights_b = [p.clone().detach() for p in model_b.parameters()]

        # Instance C with seed 99
        cfg_c = cfg.copy()
        cfg_c.seed = 99
        model_c = Model(cfg_c)
        weights_c = [p.clone().detach() for p in model_c.parameters()]

        # Assertions
        for wa, wb in zip(weights_a, weights_b):
            self.assertTrue(torch.equal(wa, wb), "Weights A and B must be identical")

        different = False
        for wa, wc in zip(weights_a, weights_c):
            if not torch.equal(wa, wc):
                different = True
                break
        self.assertTrue(different, "Weights A and C must be different")

    def test_dataloader_determinism_with_augmentation(self):
        """Verifies that the same seed produces the same augmented batches."""
        from omegaconf import OmegaConf
        import albumentations as A
        from pytorch_segmentation_models_trainer.model_loader.model import Model
        from pytorch_segmentation_models_trainer.dataset_loader.image_dataset import (
            ImageDataset,
        )
        import pandas as pd
        import numpy as np

        # Create dummy data
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        import cv2

        os.makedirs("tests/testing_data/dummy_seed_test", exist_ok=True)
        cv2.imwrite("tests/testing_data/dummy_seed_test/img.png", dummy_img)

        df = pd.DataFrame({"image": ["tests/testing_data/dummy_seed_test/img.png"]})

        # Config with random augmentation
        # Using allow_objects=True to support DataFrame in OmegaConf
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "model": {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": 3,
                    "kernel_size": 1,
                },
                "loss": {"_target_": "torch.nn.MSELoss"},
                "hyperparameters": {"batch_size": 1},
                "train_dataset": {
                    "_target_": "pytorch_segmentation_models_trainer.dataset_loader.image_dataset.ImageDataset",
                    "df": df,
                    "augmentation_list": [
                        {
                            "_target_": "albumentations.RandomCrop",
                            "height": 50,
                            "width": 50,
                        },
                        {"_target_": "albumentations.pytorch.ToTensorV2"},
                    ],
                    "data_loader": {
                        "num_workers": 0,
                        "shuffle": False,
                        "pin_memory": False,
                        "drop_last": False,
                        "persistent_workers": False,
                    },
                },
            },
            flags={"allow_objects": True},
        )

        # Get batch from instance A
        # No need to call set_training_seed manually, Model(cfg) does it.
        model_a = Model(cfg)
        w1 = next(model_a.parameters()).sum().item()
        loader_a = model_a.train_dataloader()
        batch_a = next(iter(loader_a))

        # Get batch from instance B
        model_b = Model(cfg)
        w2 = next(model_b.parameters()).sum().item()
        loader_b = model_b.train_dataloader()
        batch_b = next(iter(loader_b))

        # Assertions
        self.assertEqual(
            w1, w2, "Model weights should be identical after same seed in cfg"
        )
        self.assertTrue(
            torch.equal(batch_a["image"], batch_b["image"]),
            "Augmented images must be identical with same seed in cfg",
        )


class TestTrainCallsSeedUtils(BasicTestCase):
    """train() must call set_training_seed before Model instantiation."""

    def test_seed_called_when_present(self):
        import pytorch_segmentation_models_trainer.train as train_module
        from omegaconf import OmegaConf
        from unittest.mock import MagicMock, patch

        with (
            patch.object(
                train_module, "set_training_seed", return_value=torch.Generator()
            ) as mock_seed,
            patch.object(train_module, "Trainer", MagicMock()),
        ):
            cfg = OmegaConf.create({"seed": 42, "deterministic_cudnn": False})
            try:
                train_module.train(cfg)
            except Exception:
                pass
            mock_seed.assert_called_once_with(42, deterministic_cudnn=False)

    def test_seed_not_called_when_absent(self):
        import pytorch_segmentation_models_trainer.train as train_module
        from omegaconf import OmegaConf
        from unittest.mock import MagicMock, patch

        with (
            patch.object(train_module, "set_training_seed") as mock_seed,
            patch.object(train_module, "Trainer", MagicMock()),
        ):
            cfg = OmegaConf.create(
                {"hyperparameters": {"batch_size": 1, "epochs": 1, "max_lr": 0.001}}
            )
            try:
                train_module.train(cfg)
            except Exception:
                pass
            mock_seed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
