# -*- coding: utf-8 -*-
import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader
from pytorch_segmentation_models_trainer.dataset_loader.ddoq_dataset import (
    DDOQDistilledDataset,
)


def test_ddoq_dataset_basic():
    num_samples = 100
    images = torch.randn(num_samples, 3, 32, 32)
    masks = torch.randint(0, 2, (num_samples, 32, 32))
    original_dataset = TensorDataset(images, masks)

    medoid_indices = [0, 10, 20, 30]
    ddoq_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

    dataset = DDOQDistilledDataset(
        original_dataset=original_dataset,
        medoid_indices=medoid_indices,
        ddoq_weights=ddoq_weights,
    )

    assert len(dataset) == 4
    img, mask, weight = dataset[1]

    assert torch.allclose(img, images[10])
    assert torch.allclose(mask, masks[10])
    assert weight == pytest.approx(0.2)


def test_ddoq_dataset_soft_labels():
    num_samples = 50
    images = torch.randn(num_samples, 3, 32, 32)
    # TensorDataset with only one tensor returns a tuple of length 1
    original_dataset = TensorDataset(images)

    medoid_indices = [5, 15]
    ddoq_weights = torch.tensor([0.4, 0.6])
    soft_labels = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]

    dataset = DDOQDistilledDataset(
        original_dataset=original_dataset,
        medoid_indices=medoid_indices,
        ddoq_weights=ddoq_weights,
        soft_labels=soft_labels,
    )

    assert len(dataset) == 2
    img, mask, weight = dataset[0]

    assert torch.allclose(img, images[5])
    assert torch.allclose(mask, soft_labels[0])
    assert weight == pytest.approx(0.4)
