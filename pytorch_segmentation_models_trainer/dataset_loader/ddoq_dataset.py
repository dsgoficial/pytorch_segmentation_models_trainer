# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


class DDOQDistilledDataset(Dataset):
    """
    Dataset wrapper for the Coreset distilled dataset.
    Returns (image, mask, ddoq_weight).

    The mask can be a hard-label (from original dataset) or a soft-label
    (pre-computed from a teacher model).
    """

    def __init__(
        self,
        original_dataset: Dataset,
        medoid_indices: List[int],
        ddoq_weights: torch.Tensor,
        soft_labels: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Args:
            original_dataset: The full original dataset.
            medoid_indices: Indices of the selected medoids.
            ddoq_weights: Weights for each medoid (Voronoi weights).
            soft_labels: Optional list of soft-labels for each medoid.
                         If None, it uses the mask from the original dataset.
        """
        self.original_dataset = original_dataset
        self.medoid_indices = medoid_indices
        self.ddoq_weights = ddoq_weights
        self.soft_labels = soft_labels

        if len(medoid_indices) != len(ddoq_weights):
            raise ValueError(
                "Number of medoid indices must match the number of weights."
            )

        if soft_labels is not None and len(soft_labels) != len(medoid_indices):
            raise ValueError("Number of soft-labels must match the number of medoids.")

    def __len__(self) -> int:
        return len(self.medoid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        global_idx = self.medoid_indices[idx]
        weight = self.ddoq_weights[idx].item()

        # Get original data
        data = self.original_dataset[global_idx]

        # Handle different dataset return formats
        if isinstance(data, (list, tuple)):
            image = data[0]
            mask = data[1] if len(data) > 1 else None
        elif isinstance(data, dict):
            image = data["image"]
            mask = data.get("mask") or data.get("label")
        else:
            image = data
            mask = None

        # Use soft labels if provided
        if self.soft_labels is not None:
            mask = self.soft_labels[idx]

        return image, mask, weight
