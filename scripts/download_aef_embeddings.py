#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin shim — use 'pytorch-smt-tools download-aef-embeddings' instead.

This file exists for backward compatibility with direct script invocations.
All logic lives in pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings.
"""

from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (  # noqa: F401
    download_gcs_embeddings,
    download_hf_embeddings,
    download_sourcecoop_embeddings,
    run,
    _href_to_public_sourcecoop_url,
    _infer_sourcecoop_year,
    _select_sourcecoop_item,
    _find_hf_cell_for_bbox,
    _get_tile_bbox,
)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Download AEF embeddings. Use 'pytorch-smt-tools download-aef-embeddings' instead."
    )
    parser.add_argument("--source", choices=["gcs", "hf", "sourcecoop"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gcs-paths-csv", default=None)
    parser.add_argument("--tiles-csv", default=None)
    parser.add_argument("--sourcecoop-index-path", default=None)
    parser.add_argument("--sourcecoop-index-url", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    from pytorch_segmentation_models_trainer.tools.soft_labels.download_aef_embeddings import (
        SOURCECOOP_AEF_INDEX_URL,
    )

    run(
        source=args.source,
        output_dir=Path(args.output_dir),
        gcs_paths_csv=args.gcs_paths_csv,
        tiles_csv=args.tiles_csv,
        sourcecoop_index_path=args.sourcecoop_index_path,
        sourcecoop_index_url=args.sourcecoop_index_url or SOURCECOOP_AEF_INDEX_URL,
        year=args.year,
        max_workers=args.max_workers,
    )
