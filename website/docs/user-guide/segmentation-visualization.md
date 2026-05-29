---
sidebar_position: 12
title: "Segmentation Visualization"
---

# Segmentation Visualization

Tools for colorizing segmentation masks and building comparison grids that put ground truth alongside one or more prediction sets.

## Overview

The `segmentation_vis` module provides:

- `colorize_mask` — convert an integer class mask to an RGB image.
- `prepare_image_for_display` — normalize a raster array for matplotlib.
- `create_segmentation_grid` — build a multi-column comparison figure.

All tools are accessible programmatically or via the `visualize-predictions` CLI command.

---

## `visualize-predictions` CLI

```bash
pytorch-smt-tools visualize-predictions \
  --records-csv results.csv \
  --gt-dir /data/gt_masks \
  --pred-dir /data/model_a/masks \
  --pred-label "Model A" \
  --pred-dir /data/model_b/masks \
  --pred-label "Model B" \
  --output grid.png \
  --sort-by mean_iou \
  --n-samples 5 \
  --mode best \
  --color-map '{"0":[0,0,0],"1":[255,0,0],"2":[0,255,0]}'
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--records-csv` | (required) | CSV with `tile_id`, `mi`, and the sort-by column. |
| `--gt-dir` | (required) | Root directory with ground-truth masks. |
| `--pred-dir` | (required) | Prediction directory.  Repeat for each experiment. |
| `--pred-label` | (required) | Display label for each `--pred-dir`. |
| `--output` | (required) | Output figure path (e.g. `grid.png`). |
| `--sort-by` | `mean_iou` | Column used for sorting. |
| `--n-samples` | 5 | Number of rows in the grid. |
| `--mode` | `best` | `best`, `worst`, or `random`. |
| `--image-dir` | None | Source image directory (adds an image column). |
| `--dpi` | 150 | Figure DPI. |
| `--color-map` | auto | JSON string `{"class_id":[R,G,B],...}` or path to a JSON file. |

### Directory layout

The GT and prediction directories should follow this layout:

```
gt_dir/
  {mi}/
    {tile_id}.tif
  {tile_id}.tif    ← flat fallback
```

---

## Python API

### `colorize_mask`

```python
import numpy as np
from pytorch_segmentation_models_trainer.tools.visualization.segmentation_vis import (
    colorize_mask,
)

color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}
mask = np.array([[0, 1], [2, 1]], dtype=np.uint8)

rgb = colorize_mask(mask, color_map, nodata_value=255)
# rgb.shape == (2, 2, 3)
```

### `prepare_image_for_display`

```python
import numpy as np
from pytorch_segmentation_models_trainer.tools.visualization.segmentation_vis import (
    prepare_image_for_display,
)

# Accepts (C, H, W) or (H, W, C) or (H, W)
img = np.random.randint(0, 255, (4, 64, 64), dtype=np.uint8)
display = prepare_image_for_display(img)
# display.shape == (64, 64, 3), dtype float32, values in [0, 1]
```

### `create_segmentation_grid`

```python
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_segmentation_models_trainer.tools.visualization.segmentation_vis import (
    create_segmentation_grid,
)

records = pd.read_csv("results.csv")  # needs: tile_id, mi, mean_iou
color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}

fig = create_segmentation_grid(
    records=records,
    gt_dir=Path("gt_masks/"),
    pred_dirs=[Path("model_a/"), Path("model_b/")],
    pred_labels=["Model A", "Model B"],
    color_map=color_map,
    class_labels={0: "background", 1: "building", 2: "road"},
    output_path=Path("grid.png"),
    sort_by="mean_iou",
    n_samples=5,
    mode="best",   # or "worst" or "random"
    dpi=150,
)
plt.close(fig)
```
