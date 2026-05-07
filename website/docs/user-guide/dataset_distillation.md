---
sidebar_position: 10
title: Dataset Distillation (Coreset)
---

# Dataset Distillation via Coreset of Medoids

The Dataset Distillation pipeline in `pytorch_smt` is inspired by the **Optimal Quantization** step from the paper *"Dataset Distillation as Pushforward Optimal Quantization"*.

Instead of generating synthetic images (which often causes domain shift and edge hallucinations in satellite imagery), we adapt the quantization to extract a **Coreset of Medoids**. This process identifies the most representative real images from an unlabeled pool based on their latent representations.

## Workflow

1. **Latent Extraction**: Pass unlabeled images through a trained `GenericAutoencoder` to obtain fixed-size embedding vectors.
2. **K-Means Clustering**: Partition the latent space into $K$ clusters, where $K$ is the desired size of your coreset.
3. **Medoid Search**: For each cluster centroid, find the real latent vector (and thus the real image) that is closest in L2 distance.
4. **Distillation**: Create a new dataset containing only these representative medoids.

## Configuration

You can configure the distillation process using the `DatasetDistillationConfig` dataclass:

```yaml
dataset_distillation:
  num_clusters: 100            # Total samples to select (budget)
  batch_size: 32               # Extraction batch size
  device: "cuda"               # Use GPU for extraction and distance calculation
  checkpoint_path: "autoencoder.ckpt"
  output_indices_path: "medoid_indices.pt"
```

## Python API

```python
from pytorch_segmentation_models_trainer.dataset_distillation import (
    extract_all_latents,
    find_coreset_medoids,
    create_distilled_dataloader
)

# 1. Extract latents
latents = extract_all_latents(model, unlabeled_loader, device)

# 2. Find medoids (the coreset indices)
medoid_indices = find_coreset_medoids(latents, num_clusters=100, device=device)

# 3. Create a new dataloader
distilled_loader = create_distilled_dataloader(full_dataset, medoid_indices, batch_size=8)
```

## Implementation Details

- **Device Management**: K-Means clustering is performed on the CPU using Scikit-Learn (highly optimized), while the Medoid search (distance matrix calculation) is performed on the GPU using `torch.cdist` for maximum performance on large datasets.
- **Support for SMP & Transformers**: The extraction utility automatically detects if the model uses a Segmentation Models PyTorch (SMP) encoder or a HuggingFace Transformer adapter.
