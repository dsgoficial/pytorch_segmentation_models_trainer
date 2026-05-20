---
sidebar_position: 20
title: Classic ML Pipeline
---

# Classic ML Pipeline

The `classic_ml` subpackage provides a complete GPU-accelerated classic machine
learning segmentation pipeline.  It integrates with the rest of the framework
through Hydra configuration and exposes a scikit-learn-compatible API.

GPU acceleration is **opt-in** and **transparent**: the same configuration file
works on machines with and without an NVIDIA GPU.

---

## Architecture

```
Image (H, W, C)
      │
      ▼
FeatureEngineeringPipeline   ← GaborFilter + Gradient + Multiscale
      │  (H*W, n_features)
      ▼
GPUAcceleratedRandomForest   ← fit / predict_proba
      │  (H*W, n_classes)
      ▼
DenseCRFPostprocessor        ← optional — refine boundaries
      │  (H, W)
      ▼
Label map
```

All components are instantiated via Hydra.  The
`ClassicMLOrchestrator` glues them together.

---

## Installation

For CPU-only inference no extra dependencies are required — `scikit-image` and
`scikit-learn` are already part of the base install.

For GPU acceleration install the optional extras:

```bash
pip install "pytorch-segmentation-models-trainer[gpu-ml]"
```

This adds `cupy`, `cucim`, `cuml`, `pydensecrf`, and `pygco`.

---

## Feature Engineering

Three extractors are provided.  They accept `numpy` or `cupy` arrays and always
return `numpy` arrays.

### GaborFilterExtractor

Applies a Gabor filter bank at multiple frequencies and orientations.

```yaml
- _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GaborFilterExtractor
  frequencies: [0.1, 0.25, 0.4]
  num_orientations: 4
```

Output shape: `(H, W, n_channels × len(frequencies) × num_orientations)`

### GradientExtractor

Computes horizontal gradient, vertical gradient, and magnitude (Sobel).

```yaml
- _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GradientExtractor
```

Output shape: `(H, W, n_channels × 3)`

### MultiscaleExtractor

Applies Gaussian smoothing at multiple sigma values.

```yaml
- _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.MultiscaleExtractor
  sigmas: [1.0, 2.0, 4.0]
```

Output shape: `(H, W, n_channels × len(sigmas))`

### FeatureEngineeringPipeline

Composes extractors and flattens to `(H*W, total_features)`:

```yaml
feature_pipeline:
  _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.FeatureEngineeringPipeline
  extractors:
    - _target_: ...GaborFilterExtractor
      frequencies: [0.1, 0.25, 0.4]
      num_orientations: 4
    - _target_: ...GradientExtractor
    - _target_: ...MultiscaleExtractor
      sigmas: [1.0, 2.0, 4.0]
```

---

## Estimators

All estimators wrap `sklearn` classes and expose `.fit()`, `.predict()`, and
`.predict_proba()`.

### RandomForest

```yaml
classifier:
  _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedRandomForest
  n_estimators: 200
  max_depth: null
  random_state: 42
```

### SVM

```yaml
classifier:
  _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedSVM
  C: 1.0
  kernel: rbf
```

### KMeans

```yaml
classifier:
  _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedKMeans
  n_clusters: 8
  random_state: 42
```

`predict_proba` returns soft assignments based on inverse centroid distances.

### Enabling GPU acceleration

Call `enable_gpu_acceleration()` **once** at startup, **before** creating
estimators:

```python
from pytorch_segmentation_models_trainer.classic_ml.estimators import (
    enable_gpu_acceleration,
)

if enable_gpu_acceleration():
    print("cuml GPU acceleration active")
```

:::caution
`enable_gpu_acceleration()` patches **all of sklearn** globally for the
process via `cuml.accel.install()`.  This includes k-fold splitters and
clustering metrics used elsewhere in the framework.  Only enable it when
the entire workload should run on GPU.
:::

---

## Post-processing

### Dense CRF

Fully-connected Dense CRF via `pydensecrf`:

```yaml
postprocessor:
  _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.DenseCRFPostprocessor
  n_iterations: 5
  bilateral_sxy: 80.0
  bilateral_srgb: 13.0
  bilateral_compat: 10.0
  gaussian_sxy: 3.0
  gaussian_compat: 3.0
```

Requires `pydensecrf` (included in the `[gpu-ml]` extras).

### Graph Cuts

Min-Cut/Max-Flow with image-gradient edge weights via `pygco`:

```yaml
postprocessor:
  _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.GraphCutsPostprocessor
  unary_scale: 10.0
  pairwise_weight: 1.0
```

Requires `pygco` (included in the `[gpu-ml]` extras).

Both classes raise `ImportError` at **instantiation** when their backend is
absent, so misconfiguration is caught early rather than at inference time.

---

## Orchestrator

`ClassicMLOrchestrator` ties the pipeline together.  It is a plain Python class
— **not** a `pl.LightningModule` — because classic ML does not use iterative
backpropagation.

```python
from pytorch_segmentation_models_trainer.classic_ml import ClassicMLOrchestrator
from pytorch_segmentation_models_trainer.classic_ml.feature_engineering import (
    FeatureEngineeringPipeline, GaborFilterExtractor, GradientExtractor,
)
from pytorch_segmentation_models_trainer.classic_ml.estimators import (
    GPUAcceleratedRandomForest,
)

pipeline = FeatureEngineeringPipeline(
    extractors=[GaborFilterExtractor(frequencies=[0.1, 0.25], num_orientations=4),
                GradientExtractor()]
)
clf = GPUAcceleratedRandomForest(n_estimators=100, random_state=42)
orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)

# Training
orch.fit(train_images, train_masks)

# Inference
labels = orch.predict(test_image)
labels, probabilities = orch.predict(test_image, return_probabilities=True)

# Persistence
orch.save("model.pkl")
orch2 = ClassicMLOrchestrator.load("model.pkl")
```

---

## Full YAML example

See [`conf/examples/classic_ml_random_forest.yaml`](../../../conf/examples/classic_ml_random_forest.yaml)
for a complete end-to-end configuration with Random Forest, multi-scale cucim
features, and Dense CRF post-processing.

---

## Tensor utilities

The `utils/tensor_conversion` module provides helpers for converting between
PyTorch, NumPy, and CuPy arrays:

| Function | Description |
|---|---|
| `tensor_to_numpy(tensor)` | GPU/CPU tensor → NumPy (copies to CPU) |
| `numpy_to_tensor(arr, device)` | NumPy → PyTorch tensor |
| `tensor_to_cupy(tensor)` | CUDA tensor → CuPy (zero-copy) |
| `cupy_to_tensor(arr)` | CuPy → CUDA tensor (zero-copy) |
| `ensure_numpy(arr)` | Any array type → NumPy |
