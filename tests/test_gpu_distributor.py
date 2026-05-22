# -*- coding: utf-8 -*-
"""
Unit tests for gpu_distributor.py
"""

import unittest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
import torch

from pytorch_segmentation_models_trainer.tools.evaluation.gpu_distributor import (
    GPUDistributor,
)


class TestGPUDistributor(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "pipeline_options": {
                    "parallel_inference": {
                        "enabled": True,
                        "strategy": "one_experiment_per_gpu",
                        "gpus": None,
                    }
                }
            }
        )

    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_gpus_no_cuda(self, mock_cuda_avail):
        distributor = GPUDistributor(self.config)
        self.assertEqual(len(distributor.available_gpus), 0)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=0)
    def test_detect_gpus_no_devices(self, mock_count, mock_avail):
        distributor = GPUDistributor(self.config)
        self.assertEqual(distributor.available_gpus, [])

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.set_device")
    def test_detect_gpus_low_memory_and_device_error(
        self, mock_set_device, mock_mem, mock_props, mock_count, mock_avail
    ):
        props = MagicMock()
        props.total_memory = int(1.5 * (1024**3))
        props.name = "Small GPU"
        mock_props.side_effect = [props, props, RuntimeError("bad gpu")]
        mock_mem.return_value = int(1.0 * (1024**3))

        distributor = GPUDistributor(self.config)

        self.assertEqual(distributor.available_gpus, [])

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.set_device")
    def test_detect_gpus_with_cuda(
        self, mock_set_device, mock_mem, mock_props, mock_count, mock_avail
    ):
        # Mock GPU properties
        mock_props_instance = MagicMock()
        mock_props_instance.total_memory = 8 * (1024**3)  # 8GB
        mock_props_instance.name = "Test GPU"
        mock_props.return_value = mock_props_instance

        distributor = GPUDistributor(self.config)
        self.assertEqual(len(distributor.available_gpus), 2)
        self.assertEqual(distributor.available_gpus, [0, 1])

    def test_detect_gpus_manual(self):
        self.config.pipeline_options.parallel_inference.gpus = [0, 2]
        distributor = GPUDistributor(self.config)
        self.assertEqual(distributor.available_gpus, [0, 2])

    def test_assign_experiments_one_per_gpu(self):
        self.config.pipeline_options.parallel_inference.gpus = [0, 1]
        distributor = GPUDistributor(self.config)

        experiments = [
            OmegaConf.create({"name": "exp1"}),
            OmegaConf.create({"name": "exp2"}),
            OmegaConf.create({"name": "exp3"}),
        ]

        assignments = distributor.assign_experiments(experiments)

        self.assertEqual(len(assignments), 2)
        self.assertEqual(len(assignments[0]), 2)  # exp1, exp3
        self.assertEqual(len(assignments[1]), 1)  # exp2
        self.assertEqual(assignments[0][0].name, "exp1")
        self.assertEqual(assignments[0][1].name, "exp3")
        self.assertEqual(assignments[1][0].name, "exp2")

    def test_assign_experiments_sequential(self):
        self.config.pipeline_options.parallel_inference.strategy = "sequential"
        self.config.pipeline_options.parallel_inference.gpus = [0, 1]
        distributor = GPUDistributor(self.config)

        experiments = [
            OmegaConf.create({"name": "exp1"}),
            OmegaConf.create({"name": "exp2"}),
        ]
        assignments = distributor.assign_experiments(experiments)

        self.assertEqual(len(assignments), 1)
        self.assertIn(0, assignments)
        self.assertEqual(len(assignments[0]), 2)

    def test_assign_experiments_disabled_unknown_and_cpu_fallback(self):
        experiments = [OmegaConf.create({"name": "exp1"})]

        self.config.pipeline_options.parallel_inference.enabled = False
        self.config.pipeline_options.parallel_inference.gpus = [2]
        distributor = GPUDistributor(self.config)
        self.assertEqual(distributor.assign_experiments(experiments), {2: experiments})

        self.config.pipeline_options.parallel_inference.enabled = True
        self.config.pipeline_options.parallel_inference.strategy = "unknown"
        distributor = GPUDistributor(self.config)
        self.assertEqual(distributor.assign_experiments(experiments), {2: experiments})

        self.config.pipeline_options.parallel_inference.strategy = (
            "one_experiment_per_gpu"
        )
        self.config.pipeline_options.parallel_inference.gpus = []
        distributor = GPUDistributor(self.config)
        self.assertEqual(distributor.assign_experiments(experiments), {-1: experiments})

    def test_get_device_for_experiment(self):
        distributor = GPUDistributor(self.config)
        distributor.available_gpus = [3]
        assignments = {
            0: [OmegaConf.create({"name": "exp1"})],
            1: [OmegaConf.create({"name": "exp2"})],
        }

        self.assertEqual(distributor.get_device_for_experiment("exp1", assignments), 0)
        self.assertEqual(distributor.get_device_for_experiment("exp2", assignments), 1)
        self.assertEqual(
            distributor.get_device_for_experiment("missing", assignments), 3
        )

    @patch("torch.cuda.is_available", return_value=False)
    def test_check_gpu_availability_cpu_cuda_false_and_out_of_range(self, mock_avail):
        distributor = GPUDistributor(self.config)
        self.assertTrue(distributor.check_gpu_availability(-1))
        self.assertFalse(distributor.check_gpu_availability(0))

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            self.assertFalse(distributor.check_gpu_availability(2))

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.set_device")
    def test_check_gpu_availability_success_and_exception(
        self, mock_set_device, mock_count, mock_avail
    ):
        distributor = GPUDistributor(self.config)
        tensor = MagicMock()
        tensor.cuda.return_value = tensor
        with patch("torch.zeros", return_value=tensor):
            self.assertTrue(distributor.check_gpu_availability(0))

        tensor.cuda.side_effect = RuntimeError("cuda failed")
        with patch("torch.zeros", return_value=tensor):
            self.assertFalse(distributor.check_gpu_availability(0))

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated", return_value=2 * (1024**3))
    @patch("torch.cuda.set_device")
    def test_get_gpu_memory_info(
        self, mock_set, mock_mem, mock_props, mock_count, mock_avail
    ):
        mock_props_instance = MagicMock()
        mock_props_instance.total_memory = 8 * (1024**3)
        mock_props.return_value = mock_props_instance

        distributor = GPUDistributor(self.config)
        # Manually set available gpus to bypass detection for this test
        distributor.available_gpus = [0]

        # Mock check_gpu_availability to return True without calling .cuda()
        with patch.object(GPUDistributor, "check_gpu_availability", return_value=True):
            info = distributor.get_gpu_memory_info(0)
            self.assertEqual(info["total_gb"], 8.0)
            self.assertEqual(info["used_gb"], 2.0)
            self.assertEqual(info["free_gb"], 6.0)

    def test_get_gpu_memory_info_cpu_unavailable_and_exception(self):
        distributor = GPUDistributor(self.config)
        self.assertEqual(
            distributor.get_gpu_memory_info(-1),
            {"total_gb": 0, "used_gb": 0, "free_gb": 0},
        )

        with patch.object(GPUDistributor, "check_gpu_availability", return_value=False):
            self.assertEqual(
                distributor.get_gpu_memory_info(0),
                {"total_gb": 0, "used_gb": 0, "free_gb": 0},
            )

        with (
            patch.object(GPUDistributor, "check_gpu_availability", return_value=True),
            patch("torch.cuda.set_device", side_effect=RuntimeError("bad memory")),
        ):
            self.assertEqual(
                distributor.get_gpu_memory_info(0),
                {"total_gb": 0, "used_gb": 0, "free_gb": 0},
            )

    def test_wait_for_gpu_memory_success_and_timeout(self):
        distributor = GPUDistributor(self.config)

        with (
            patch.object(
                distributor,
                "get_gpu_memory_info",
                return_value={"free_gb": 2.0},
            ),
            patch(
                "time.time",
                side_effect=[0, 1],
            ),
        ):
            self.assertTrue(distributor.wait_for_gpu_memory(0, 1.0, timeout_seconds=10))

        with (
            patch.object(
                distributor,
                "get_gpu_memory_info",
                return_value={"free_gb": 0.1},
            ),
            patch(
                "time.time",
                side_effect=[0, 1, 6],
            ),
            patch("time.sleep"),
        ):
            self.assertFalse(distributor.wait_for_gpu_memory(0, 1.0, timeout_seconds=5))

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_get_optimal_batch_size(self, mock_count, mock_avail):
        distributor = GPUDistributor(self.config)

        # Mock check_gpu_availability and memory info
        with patch.object(GPUDistributor, "check_gpu_availability", return_value=True):
            distributor.get_gpu_memory_info = MagicMock(return_value={"free_gb": 4.0})

            # 4GB * 0.8 = 3.2GB = 3200MB. 3200 / 100 = 32.
            batch_size = distributor.get_optimal_batch_size(
                0, base_batch_size=16, memory_per_sample_mb=100.0
            )
            self.assertEqual(batch_size, 32)

            # Test capping at 4 * base
            batch_size_capped = distributor.get_optimal_batch_size(
                0, base_batch_size=4, memory_per_sample_mb=100.0
            )
            self.assertEqual(batch_size_capped, 16)

        self.assertEqual(distributor.get_optimal_batch_size(-1, base_batch_size=7), 7)
        distributor.get_gpu_memory_info = MagicMock(return_value={"free_gb": 0})
        self.assertEqual(distributor.get_optimal_batch_size(0, base_batch_size=9), 9)


if __name__ == "__main__":
    unittest.main()
