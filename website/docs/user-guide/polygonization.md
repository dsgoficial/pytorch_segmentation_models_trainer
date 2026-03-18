---
sidebar_position: 10
title: "Polygonization: Masks to Vector Polygons"
---

# Polygonization: Masks to Vector Polygons

Polygonization converts a raster segmentation mask — a grid of predicted class probabilities or binary labels — into a set of **vector polygon geometries**. The output can be written as GeoJSON, Shapefile, or directly into a PostGIS database.

Polygonization can be applied:
- **During inference** by attaching a polygonizer to any inference processor.
- **As a post-processing step** on saved prediction rasters.

---

## Architecture Overview

All polygonizers implement the `TemplatePolygonizerProcessor` interface (defined in `pytorch_segmentation_models_trainer/tools/polygonization/polygonizer.py`). Each processor wraps a specific polygonization algorithm and a `data_writer` that handles output.

The core method is `process(inference, profile, ...)`, which:
1. Calls the underlying polygonization algorithm on the `seg` (and optionally `crossfield`) tensors.
2. Projects the resulting pixel-coordinate polygons to world coordinates using the rasterio `profile` (CRS and affine transform).
3. Passes projected polygons to the `data_writer` for persistence.

---

## Available Methods

### Simple Polygonization

**Class:** `SimplePolygonizerProcessor`

Fast contour-based extraction using `skimage.measure`. Contours are converted to Shapely polygons and then optionally simplified with a tolerance parameter.

- Best for: regular, blocky shapes; situations where speed matters more than boundary precision.
- Does **not** require a cross-field — works with a plain `seg` band.

```yaml
polygonizer:
  _target_: pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.SimplePolygonizerProcessor
  config:
    data_level: 0.5         # contour iso-level in [0,1]
    tolerance: 1.0          # Douglas-Peucker simplification tolerance (pixels)
    seg_threshold: 0.5      # binary threshold applied before contouring
    min_area: 10.0          # minimum polygon area in pixels (smaller are discarded)
  data_writer:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter
    output_file_path: /output/polygons.geojson
    driver: GeoJSON
```

---

### Active Contours (ACM)

**Class:** `ACMPolygonizerProcessor`

Energy-minimization method that deforms initial polygon contours toward the true boundary by minimizing a combination of data fidelity, length regularization, and cross-field alignment terms. This produces **smooth, well-aligned boundaries** that closely follow the segmentation mask.

- Best for: curved or organic shapes; masks from frame-field models where the `crossfield` output is available.
- Requires a `crossfield` tensor in addition to `seg`.

Key config parameters:

| Parameter | Default | Description |
|---|---|---|
| `steps` | `500` | Number of optimization iterations |
| `data_level` | `0.5` | Target probability level for the mask |
| `data_coef` | `0.1` | Weight for data fidelity term |
| `length_coef` | `0.4` | Weight for length regularization term |
| `crossfield_coef` | `0.5` | Weight for cross-field alignment term |
| `poly_lr` | `0.01` | Learning rate for contour optimization |
| `tolerance` | `0.5` | Final polygon simplification tolerance |
| `seg_threshold` | `0.5` | Threshold before contour initialization |
| `min_area` | `1` | Minimum polygon area in pixels |
| `device` | `cpu` | Device for optimization (`cpu` or `cuda:0`) |

```yaml
polygonizer:
  _target_: pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.ACMPolygonizerProcessor
  config:
    steps: 500
    data_level: 0.5
    data_coef: 0.1
    length_coef: 0.4
    crossfield_coef: 0.5
    poly_lr: 0.01
    tolerance: 0.5
    seg_threshold: 0.5
    min_area: 1
    device: cpu
  data_writer:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter
    output_file_path: /output/polygons_acm.geojson
    driver: GeoJSON
```

---

### Active Skeletons (ASM)

**Class:** `ASMPolygonizerProcessor`

Skeleton-based energy optimization. The algorithm first skeletonizes the segmentation mask to extract a structural skeleton, then fits polygon contours anchored to that skeleton using energy minimization. The result is particularly accurate for **elongated or complex structures** like roads and building footprints.

- Best for: thin structures, complex topology, building footprint extraction.
- Requires a `crossfield` tensor.

Key config parameters:

| Parameter | Default | Description |
|---|---|---|
| `init_method` | `skeleton` | Initialization: `skeleton` or `marching_squares` |
| `data_level` | `0.5` | Target probability level |
| `lr` | `0.001` | Learning rate for optimization |
| `gamma` | `0.0001` | Momentum decay |
| `tolerance` | `22` | Final simplification tolerance |
| `seg_threshold` | `0.5` | Binarization threshold |
| `min_area` | `12` | Minimum polygon area in pixels |
| `device` | `cpu` | Device for optimization |

```yaml
polygonizer:
  _target_: pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.ASMPolygonizerProcessor
  config:
    init_method: skeleton
    data_level: 0.5
    lr: 0.001
    gamma: 0.0001
    tolerance: 22
    seg_threshold: 0.5
    min_area: 12
    device: cpu
  data_writer:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter
    output_file_path: /output/polygons_asm.geojson
    driver: GeoJSON
```

:::tip ASM vs ACM
ASM (`active_skeletons`) tends to produce sharper corners and is better for rectilinear buildings. ACM (`active_contours`) produces smoother curves and is better for organic shapes. For most remote-sensing building extraction tasks, ASM is the recommended starting point.
:::

---

### Polygon RNN

**Class:** `PolygonRNNPolygonizerProcessor`

A neural-network-based approach using a recurrent network that **sequentially predicts polygon vertices**. Instead of deforming a contour, the model autoregressively generates the vertex sequence. This is the highest-accuracy method when the network has been trained on the target domain.

- Best for: highest accuracy when a trained PolygonRNN model is available.
- Used together with `PolygonRNNInferenceProcessor`.
- Does not require a crossfield; operates on cropped bounding-box images.

```yaml
polygonizer:
  _target_: pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.PolygonRNNPolygonizerProcessor
  config:
    tolerance: 0.0    # vertex simplification tolerance (0 = no simplification)
    grid_size: 28     # internal grid resolution for vertex prediction
    min_area: 10.0    # minimum polygon area in pixels
  data_writer:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter
    output_file_path: /output/polygons_rnn.geojson
    driver: GeoJSON
```

---

## Method Comparison

| Method | Speed | Boundary Quality | Requires crossfield | Use Case |
|---|---|---|---|---|
| Simple | Fast | Low | No | Quick prototyping, regular shapes |
| ACM | Slow | High (smooth) | Yes | Organic shapes, curved boundaries |
| ASM | Slow | High (sharp) | Yes | Buildings, rectilinear structures |
| Polygon RNN | Medium | Highest (learned) | No | High-accuracy with trained model |

---

## Output Writers

The `data_writer` field inside each polygonizer config controls output format.

### GeoJSON File

```yaml
data_writer:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter
  output_file_path: /output/buildings.geojson
  driver: GeoJSON
```

### Shapefile

```yaml
data_writer:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter
  output_file_path: /output/buildings.shp
  driver: ESRI Shapefile
```

### PostGIS Database

```yaml
data_writer:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorDatabaseDataWriter
  user: postgres
  password: secret
  database: geodata
  host: localhost
  port: 5432
  sql: "SELECT id, geom FROM buildings"
  table_name: buildings
  geometry_column: geom
```

---

## Attaching a Polygonizer to the Inference Pipeline

Add the `polygonizer` key to any `predict` config. The polygonizer runs automatically after each image's inference is complete.

```yaml title="configs/predict_with_polygonization.yaml"
checkpoint_path: /checkpoints/framefield_best.ckpt
device: cuda:0

pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.frame_field_model.FrameFieldSegmentationPLModel

hyperparameters:
  batch_size: 4

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /data/test_images/
  image_extension: tif
  recursive: true

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageFromFrameFieldProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]

polygonizer:
  _target_: pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.ASMPolygonizerProcessor
  config:
    init_method: skeleton
    data_level: 0.5
    lr: 0.001
    tolerance: 22
    seg_threshold: 0.5
    min_area: 12
    device: cpu
  data_writer:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter
    output_file_path: /output/buildings.geojson
    driver: GeoJSON

inference_threshold: 0.5
```

---

## Programmatic Usage

You can instantiate and call polygonizers directly in Python:

```python
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    ASMPolygonizerProcessor,
    ASMConfig,
)
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    VectorFileDataWriter,
)

polygonizer = ASMPolygonizerProcessor(
    config=ASMConfig(
        seg_threshold=0.5,
        min_area=12,
        tolerance=22,
        device="cpu",
    ),
    data_writer=VectorFileDataWriter(
        output_file_path="/output/polygons.geojson",
        driver="GeoJSON",
    ),
)

# inference is a dict with "seg" and "crossfield" tensors (batch of 1)
# profile is a rasterio profile dict with CRS and transform
polygons = polygonizer.process(inference, profile)
```

:::note World Coordinates
When a rasterio profile with a valid CRS is provided, output polygons are automatically projected from pixel space to world coordinates (the CRS from the input raster). If the input image is not georeferenced, polygons are in pixel space with a simple identity transform.
:::
