#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin shim — use 'pytorch-smt-tools generate-training-csv' instead.

This file exists for backward compatibility with direct script invocations.
All logic lives in pytorch_segmentation_models_trainer.tools.soft_labels.generate_training_csv.
"""

from pytorch_segmentation_models_trainer.tools.soft_labels.generate_training_csv import (  # noqa: F401
    run,
    split_dataframe,
)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate CSV splits. Use 'pytorch-smt-tools generate-training-csv' instead."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--image-extension", default=".tif")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        manifest_path=args.manifest,
        output_dir=Path(args.output_dir),
        image_dir=args.image_dir,
        image_extension=args.image_extension,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
