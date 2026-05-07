---
sidebar_position: 5
title: GPU K-Means Clustering
---

# GPU K-Means Clustering

The framework provides a high-performance **Mini-Batch K-Means** implementation powered by PyTorch, specifically designed for clustering large volumes of embeddings or features on GPU.

## Key Features

- **GPU Acceleration**: Utilizes PyTorch's vectorized operations (`torch.cdist`) for fast distance calculations.
- **Scalability**: Mini-Batch approach allows processing millions of points without exhausting VRAM.
- **K-Means++ Initialization**: Intelligent centroid initialization for faster convergence and better cluster quality.
- **Spatial Integration**: Built-in support for GeoPandas and PostGIS, allowing you to cluster spatial features and export them directly to GIS formats.

## Basic Usage

The `KMeansClusteringTool` is the main entry point for running clustering pipelines.

```python
import torch
from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_exporter import KMeansClusteringTool
from shapely.geometry import Point

# 1. Prepare your data
ids = ["loc_1", "loc_2", "loc_3"]
embeddings = torch.randn(3, 128)  # 3 samples, 128-dimensional features
geometries = [Point(0, 0), Point(1, 1), Point(2, 2)]

# 2. Initialize the tool
tool = KMeansClusteringTool(
    n_clusters=3,
    batch_size=1024,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 3. Run clustering
gdf = tool.run(ids, embeddings, geometries)

# 4. Export results
tool.export_to_parquet(gdf, "clusters.parquet")
```

## Advanced Exporting

### PostGIS Export

You can export your clustered data directly to a PostGIS-enabled PostgreSQL database:

```python
tool.export_to_postgis(
    gdf,
    table_name="embedding_clusters",
    engine_url="postgresql://user:password@localhost:5432/mydatabase"
)
```

## Performance Tips

- **Batch Size**: For GPU execution, larger batch sizes (e.g., 1024, 2048) usually improve performance but increase VRAM usage.
- **Data Types**: The implementation uses `float32` by default for the best balance between precision and performance on GPU.
- **Convergence**: If your clusters are not converging, try increasing `max_iter` or using a smaller `tol`.

## Implementation Details

The core logic is implemented in `MiniBatchKMeans`, which mimics the scikit-learn API but runs entirely on PyTorch tensors. Centroids are updated incrementally using the formula:

$$
\eta = \frac{1}{\text{counts}_j}
$$

$$
c_j = (1 - \eta)c_j + \eta \bar{x}_{batch}
$$

This ensures that the centroids converge to the global mean of the assigned points over time.
