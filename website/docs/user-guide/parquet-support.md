---
sidebar_position: 10
title: Parquet Support & Caching
---

# Parquet Support & Caching

The framework supports [Apache Parquet](https://parquet.apache.org/) for dataset metadata, offering significantly faster loading times and lower memory consumption compared to standard CSV files.

## Automatic CSV Caching

By default, when you provide a `.csv` file in `input_csv_path`, the framework will:
1. Check if a `.cache.parquet` file exists in the same directory.
2. Compare the modification time of the CSV and the Parquet cache.
3. If the Parquet file is newer, it loads it directly (instant loading).
4. If it doesn't exist or is outdated, it reads the CSV and updates the cache automatically.

This behavior is transparent and happens inside the `AbstractDataset` class.

## Native Parquet Support

You can also provide a `.parquet` file directly in your configuration:

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: data/train_metadata.parquet
  root_dir: data/images
```

## CLI Conversion Tool

If you want to convert your CSV files to Parquet manually to avoid the first-run overhead or for better data management, use the `csv-to-parquet` tool:

```bash
# Convert a single file
csv-to-parquet metadata.csv

# Convert all CSVs in a directory recursively
csv-to-parquet datasets/ -r

# Specify a different output path
csv-to-parquet metadata.csv -o processed_metadata.parquet
```

## Performance Benefits

- **Faster Initialization**: Reading a Parquet file with 1M rows is often 10x-50x faster than parsing a CSV.
- **Lower Memory**: Parquet files are loaded more efficiently by Pandas using memory mapping.
- **Strong Typing**: Column types are preserved, avoiding "string vs float" inference issues common with CSVs.
