#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin shim — use 'pytorch-smt-tools build-soft-labels' instead.

This file exists for backward compatibility with direct script invocations.
All logic lives in pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels.
"""

from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (  # noqa: F401
    aggregate_aef_to_image_grid,
    compute_p_soft,
    compute_w_conf,
    compute_w_embed,
    expand_manifest_to_patches,
    generate_patch_rows,
    load_aef_embedding,
    process_tile,
    run,
    _compute_hf_class_centroids,
    _reproject_lulc_to_image_grid,
    _write_geotiff,
)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Build P_soft and W_conf rasters. Use 'pytorch-smt-tools build-soft-labels' instead."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--aef-embeddings-dir", default=None)
    parser.add_argument("--aef-source", choices=["gcs", "hf"], default="gcs")
    parser.add_argument(
        "--aef-resampling",
        choices=["auto", "aggregate", "nearest", "none"],
        default="auto",
        help="AEF raster alignment strategy for GCS embeddings.",
    )
    parser.add_argument(
        "--save-aef-aligned",
        action="store_true",
        default=False,
        help="Write aligned/resampled GCS AEF embeddings to output_dir/aef_aligned/.",
    )
    parser.add_argument(
        "--aef-aligned-dtype",
        choices=["int8", "float32"],
        default="int8",
        help="Output dtype for --save-aef-aligned.",
    )
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--no-border",
        action="store_true",
        default=False,
        help="Omit border-distance component (original paper formula).",
    )
    args = parser.parse_args()

    run(
        input_csv=args.input_csv,
        output_dir=Path(args.output_dir),
        num_classes=args.num_classes,
        alpha=args.alpha,
        max_workers=args.max_workers,
        patch_size=args.patch_size,
        stride=args.stride,
        aef_embeddings_dir=args.aef_embeddings_dir,
        aef_source=args.aef_source,
        aef_resampling=args.aef_resampling,
        save_aligned_aef=args.save_aef_aligned,
        aligned_aef_dtype=args.aef_aligned_dtype,
        beta=args.beta,
        use_border=not args.no_border,
    )
