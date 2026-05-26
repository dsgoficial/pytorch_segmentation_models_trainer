#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin shim — use 'pytorch-smt-tools download-aef-embeddings' instead.

This file exists for backward compatibility with direct script invocations.
All logic lives in pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.
"""

from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (  # noqa: F401
    download_gcs_embeddings,
    download_hf_embeddings,
    run,
    _find_hf_cell_for_bbox,
    _get_tile_bbox,
)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Download AEF embeddings. Use 'pytorch-smt-tools download-aef-embeddings' instead."
    )
    parser.add_argument("--source", choices=["gcs", "hf"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gcs-paths-csv", default=None)
    parser.add_argument("--tiles-csv", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    run(
        source=args.source,
        output_dir=Path(args.output_dir),
        gcs_paths_csv=args.gcs_paths_csv,
        tiles_csv=args.tiles_csv,
        max_workers=args.max_workers,
    )
