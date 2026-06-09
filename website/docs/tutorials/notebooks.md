---
title: Notebook Tutorials
sidebar_position: 1
---

This project now includes notebook-oriented tutorials that demonstrate the framework with real datasets and end-to-end workflows.

## Available notebooks

- [Quickstart](../../../notebooks/00_quickstart.ipynb)
- [Potsdam windowed segmentation](../../../notebooks/01_potsdam_windowed_segmentation.ipynb)
- [Inference and export](../../../notebooks/02_inference_and_export.ipynb)

## Recommended tutorial path

1. Start with the quickstart notebook to verify the environment.
2. Use the Potsdam notebook to build a small sliding-window segmentation dataset from a real remote-sensing dataset.
3. Finish with the inference notebook to inspect predictions and exports.

## Potsdam workflow

The Potsdam notebook is designed around a small subset of the dataset so it stays lightweight enough for local experimentation:

1. Mount or download the Google Drive folder locally.
2. Export a CSV with at least `image` and `mask` columns.
3. Run `build_sliding_window_dataset` to create patch-level training data.
4. Train a small segmentation model with a short schedule.
5. Run tiled inference on a full image to validate the result.
6. Re-run with a single tile if you want a quick smoke test.

## Tutorial config

- [Potsdam windowed tutorial YAML](../../../pytorch_segmentation_models_trainer/conf/examples/potsdam_windowed_tutorial.yaml)

## Exact commands

```bash
uv run python -m pytorch_segmentation_models_trainer.main \
  --config-dir pytorch_segmentation_models_trainer/conf/examples \
  --config-name potsdam_windowed_tutorial
```

If you want to rebuild the dataset first, use the notebook cell that calls `InferenceCSVBuilder` and `build_sliding_window_dataset`.

## Related docs

- [Dataset Builder](../user-guide/dataset-builder.md)
- [Segmentation Datasets](../user-guide/dataset-segmentation.md)
- [Inference Guide](../user-guide/inference.md)
- [Quick Start](../getting-started/quickstart.md)
