# -*- coding: utf-8 -*-
"""
pytorch-smt-tools — generic CLI tooling entry point.

New subcommands should be added as ``@cli.command(...)`` functions here, or
imported from dedicated modules and registered with ``cli.add_command(...)``.
"""

import click


@click.group()
def cli():
    """Command-line utilities for pytorch_segmentation_models_trainer."""
    pass


@cli.command("compute-stats")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--dataset-key",
    default="train_dataset",
    show_default=True,
    help="Top-level YAML key for the dataset used to compute statistics.",
)
@click.option(
    "--batch-size",
    default=64,
    show_default=True,
    type=int,
    help="Batch size for the DataLoader during stats computation.",
)
@click.option(
    "--num-workers",
    default=4,
    show_default=True,
    type=int,
    help="Number of DataLoader worker processes.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would change without modifying the file.",
)
@click.option(
    "--skip-callbacks",
    is_flag=True,
    default=False,
    help="Do not update image callbacks with norm_params.",
)
def compute_stats_cmd(
    yaml_path, dataset_key, batch_size, num_workers, dry_run, skip_callbacks
):
    """Compute per-channel mean and std from YAML_PATH and write results back.

    Reads the dataset at DATASET_KEY, iterates through all samples, computes
    per-channel mean and standard deviation, then writes an
    albumentations.Normalize entry into every dataset's augmentation_list and
    (unless --skip-callbacks) updates norm_params in image callbacks.
    """
    from pathlib import Path
    from pytorch_segmentation_models_trainer.tools.compute_dataset_stats import (
        process_yaml,
    )

    process_yaml(
        yaml_path=Path(yaml_path),
        dataset_key=dataset_key,
        batch_size=batch_size,
        num_workers=num_workers,
        dry_run=dry_run,
        skip_callbacks=skip_callbacks,
    )


@cli.command("export-tb-images")
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--list-tags",
    "list_tags",
    is_flag=True,
    default=False,
    help="List available image tags found in event files and exit.",
)
@click.option(
    "--tags",
    default=None,
    help="Comma-separated image tags to export. Omit to export all tags.",
)
@click.option(
    "--steps",
    default=None,
    help=(
        "Steps to export. Accepts integers, ranges, and combinations: "
        "'5', '0-20', '0,10,20', '0-5,10'."
    ),
)
@click.option(
    "--output",
    "-o",
    default="tb_exports",
    show_default=True,
    help="Output directory for exported PNG files.",
)
def export_tb_images_cmd(log_dir, list_tags, tags, steps, output):
    """Export images from TensorBoard event files.

    LOG_DIR is searched recursively for TFRecord event files.
    Images are saved as PNG files named <tag>_step<step>.png.

    Examples:

    \b
        # List available tags
        pytorch-smt-tools export-tb-images runs/exp1 --list-tags

    \b
        # Export one tag, all steps
        pytorch-smt-tools export-tb-images runs/exp1 --tags tile_001_idx0 -o exports/

    \b
        # Export two tags, epochs 10-20
        pytorch-smt-tools export-tb-images runs/exp1 \\
            --tags "tile_001_idx0,tile_005_idx4" --steps 10-20 -o exports/
    """
    from pathlib import Path
    from pytorch_segmentation_models_trainer.tools.export_tb_images import (
        export_images,
        list_image_tags,
    )

    log_path = Path(log_dir)

    if list_tags:
        available = sorted(list_image_tags(log_path))
        if not available:
            click.echo("No image tags found in event files.")
            return
        click.echo(f"Found {len(available)} image tag(s):")
        for t in available:
            click.echo(f"  {t}")
        return

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    count = export_images(
        log_dir=log_path,
        output_dir=Path(output),
        tags=tag_list,
        steps=steps,
    )
    click.echo(f"Exported {count} image(s) to '{output}'.")


@cli.command("export-mbtiles-mask-aligned")
@click.option(
    "--mbtiles-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to MBTiles imagery, or any raster readable by rasterio.",
)
@click.option(
    "--mask-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory scanned recursively for mask rasters.",
)
@click.option(
    "--mask-path",
    "mask_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Explicit mask path. Can be repeated.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory receiving images/, masks/, previews/, and manifest.csv.",
)
@click.option(
    "--mask-extension",
    default=".tif",
    show_default=True,
    help="Mask extension used with --mask-dir.",
)
@click.option(
    "--patch-size",
    default=None,
    type=int,
    help="Patch size in mask pixels. Omit when using --full-mask.",
)
@click.option(
    "--stride",
    default=None,
    type=int,
    help="Patch stride in mask pixels. Defaults to --patch-size.",
)
@click.option(
    "--full-mask",
    is_flag=True,
    default=False,
    help="Export one aligned image for each complete mask raster.",
)
@click.option(
    "--selected-bands",
    default=None,
    help="Comma-separated 1-based source bands to export, e.g. '1,2,3'.",
)
@click.option(
    "--image-dtype",
    default="uint8",
    show_default=True,
    help="Output image dtype, or 'native'.",
)
@click.option(
    "--image-resampling",
    default="bilinear",
    show_default=True,
    type=click.Choice(["nearest", "bilinear", "cubic", "average"]),
    help="Resampling method for source imagery.",
)
@click.option(
    "--skip-empty-masks",
    is_flag=True,
    default=False,
    help="Skip windows where the mask has no non-zero pixels.",
)
@click.option(
    "--sidecar-png/--no-sidecar-png",
    "write_sidecar_png",
    default=True,
    show_default=True,
    help="Write RGB PNG previews with mask overlay.",
)
def export_mbtiles_mask_aligned_cmd(
    mbtiles_path,
    mask_dir,
    mask_paths,
    output_dir,
    mask_extension,
    patch_size,
    stride,
    full_mask,
    selected_bands,
    image_dtype,
    image_resampling,
    skip_empty_masks,
    write_sidecar_png,
):
    """Export source imagery aligned to GeoTIFF mask grids for QA."""
    from pathlib import Path

    from pytorch_segmentation_models_trainer.tools.mbtiles.export_mask_aligned_images import (
        export_mask_aligned_images,
    )

    bands = (
        [int(part.strip()) for part in selected_bands.split(",")]
        if selected_bands
        else None
    )
    try:
        result = export_mask_aligned_images(
            mbtiles_path=Path(mbtiles_path),
            mask_dir=Path(mask_dir) if mask_dir else None,
            mask_paths=[Path(p) for p in mask_paths] if mask_paths else None,
            output_dir=Path(output_dir),
            mask_extension=mask_extension,
            patch_size=patch_size,
            stride=stride,
            full_mask=full_mask,
            selected_bands=bands,
            image_dtype=image_dtype,
            image_resampling=image_resampling,
            skip_empty_masks=skip_empty_masks,
            write_sidecar_png=write_sidecar_png,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    suffix = "" if result.count == 1 else "s"
    click.echo(f"Exported {result.count} mask-aligned image{suffix} to '{output_dir}'.")
    click.echo(f"Manifest written to '{result.manifest_path}'.")


@cli.command("ddoq-vae")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--k", default=None, type=int, help="Override number of clusters.")
@click.option(
    "--checkpoint",
    "checkpoint_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Override trained VAE checkpoint path.",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Override output directory.",
)
@click.option(
    "--format",
    "distilled_image_format",
    default=None,
    type=click.Choice(["auto", "tif", "png", "jpg", "pt"], case_sensitive=False),
    help="Override distilled image format.",
)
def ddoq_vae_cmd(yaml_path, k, checkpoint_path, output_dir, distilled_image_format):
    """Run VAE-backed DDOQ image distillation from YAML_PATH.

    The config writes ``embeddings.parquet`` with all input embeddings and
    ``distilled_images.parquet`` with one row per decoded cluster center.
    """
    from pytorch_segmentation_models_trainer.tools.dataset_distillation import (
        vae_ddoq_distillation,
    )

    result = vae_ddoq_distillation.run_vae_ddoq_from_config_file(
        yaml_path=yaml_path,
        k=k,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        distilled_image_format=distilled_image_format,
    )
    click.echo(f"Wrote embeddings parquet: {result.embeddings_parquet_path}")
    click.echo(
        f"Wrote distilled images parquet: {result.distilled_images_parquet_path}"
    )
    click.echo(f"Wrote {len(result.distilled_image_paths)} distilled image(s).")


@cli.command("build-soft-labels")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Root directory for output p_soft/ and w_conf/ sub-directories.",
)
@click.option(
    "--num-classes",
    default=4,
    show_default=True,
    type=int,
    help="Number of land-cover classes (0-indexed).",
)
@click.option(
    "--alpha",
    default=0.6,
    show_default=True,
    type=float,
    help="Entropy/border blend weight for W_conf (0=border-only, 1=entropy-only).",
)
@click.option(
    "--max-workers",
    default=4,
    show_default=True,
    type=int,
    help="Number of parallel worker processes.",
)
@click.option(
    "--patch-size",
    default=None,
    type=int,
    help="When set, expand the manifest into patch rows for SoftLabelWindowedDataset.",
)
@click.option(
    "--stride",
    default=None,
    type=int,
    help="Sliding-window stride in pixels (default: same as --patch-size).",
)
@click.option(
    "--aef-embeddings-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory with pre-downloaded AEF embeddings.",
)
@click.option(
    "--aef-source",
    default="gcs",
    show_default=True,
    type=click.Choice(["gcs", "hf"], case_sensitive=False),
    help="AEF embedding source.",
)
@click.option(
    "--aef-resampling",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "aggregate", "nearest", "none"], case_sensitive=False),
    help=(
        "AEF raster alignment strategy for GCS embeddings. 'auto' aggregates "
        "for downsampling and uses nearest-neighbor for upsampling."
    ),
)
@click.option(
    "--beta",
    default=0.0,
    show_default=True,
    type=float,
    help="AEF embedding blend weight (alpha + beta must be <= 1.0 unless --no-border).",
)
@click.option(
    "--no-border",
    "use_border",
    is_flag=True,
    default=True,
    flag_value=False,
    help="Omit the border-distance component (replicates original paper formula).",
)
def build_soft_labels_cmd(
    input_csv,
    output_dir,
    num_classes,
    alpha,
    max_workers,
    patch_size,
    stride,
    aef_embeddings_dir,
    aef_source,
    aef_resampling,
    beta,
    use_border,
):
    """Build P_soft and W_conf rasters from multiple LULC sources in INPUT_CSV.

    INPUT_CSV must have columns: tile_id, image_path, source_name, lulc_path, weight.
    """
    from pathlib import Path
    from pytorch_segmentation_models_trainer.tools.soft_labels import build_soft_labels

    manifest_path = build_soft_labels.run(
        input_csv=input_csv,
        output_dir=Path(output_dir),
        num_classes=num_classes,
        alpha=alpha,
        max_workers=max_workers,
        patch_size=patch_size,
        stride=stride,
        aef_embeddings_dir=aef_embeddings_dir,
        aef_source=aef_source,
        aef_resampling=aef_resampling,
        beta=beta,
        use_border=use_border,
    )
    click.echo(f"Manifest written to '{manifest_path}'.")


@cli.command("download-aef-embeddings")
@click.option(
    "--source",
    required=True,
    type=click.Choice(["gcs", "hf", "sourcecoop"], case_sensitive=False),
    help=(
        "Embedding source: 'gcs' for GeoTIFFs from GCS, 'hf' for HuggingFace, "
        "or 'sourcecoop' for cropped COGs from Source Cooperative."
    ),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where downloaded embedding files will be written.",
)
@click.option(
    "--gcs-paths-csv",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV with tile_id and gcs_uri (required for --source gcs).",
)
@click.option(
    "--tiles-csv",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV with tile_id and image_path (required for --source hf/sourcecoop).",
)
@click.option(
    "--sourcecoop-index-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional local Source Cooperative STAC GeoParquet index.",
)
@click.option(
    "--sourcecoop-index-url",
    default=None,
    help="Remote Source Cooperative STAC GeoParquet index URL override.",
)
@click.option(
    "--year",
    default=None,
    type=int,
    help="Annual AEF product year override for --source sourcecoop.",
)
@click.option(
    "--max-workers",
    default=4,
    show_default=True,
    type=int,
    help="Reserved for future parallel GCS downloads.",
)
def download_aef_embeddings_cmd(
    source,
    output_dir,
    gcs_paths_csv,
    tiles_csv,
    sourcecoop_index_path,
    sourcecoop_index_url,
    year,
    max_workers,
):
    """Download AlphaEarth Foundation embeddings for a set of tiles.

    Use --source gcs for per-pixel GeoTIFFs from Google Cloud Storage,
    --source hf for patch-level vectors from HuggingFace, or --source
    sourcecoop for cropped per-pixel GeoTIFFs from Source Cooperative COGs.
    """
    from pathlib import Path
    from pytorch_segmentation_models_trainer.tools.soft_labels import (
        download_aef_embeddings,
    )

    try:
        download_aef_embeddings.run(
            source=source,
            output_dir=Path(output_dir),
            gcs_paths_csv=gcs_paths_csv,
            tiles_csv=tiles_csv,
            sourcecoop_index_path=sourcecoop_index_path,
            sourcecoop_index_url=(
                sourcecoop_index_url or download_aef_embeddings.SOURCECOOP_AEF_INDEX_URL
            ),
            year=year,
            max_workers=max_workers,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    click.echo("Download complete.")


@cli.command("generate-training-csv")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where train.csv, val.csv, test.csv are written.",
)
@click.option(
    "--image-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory containing <tile_id>.tif images. "
    "Ignored when manifest already has image_path column.",
)
@click.option(
    "--image-extension",
    default=".tif",
    show_default=True,
    help="File extension for images in --image-dir.",
)
@click.option(
    "--train-ratio",
    default=0.70,
    show_default=True,
    type=float,
    help="Fraction of rows for training.",
)
@click.option(
    "--val-ratio",
    default=0.15,
    show_default=True,
    type=float,
    help="Fraction of rows for validation.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
def generate_training_csv_cmd(
    manifest_path,
    output_dir,
    image_dir,
    image_extension,
    train_ratio,
    val_ratio,
    seed,
):
    """Generate train/val/test CSV splits from MANIFEST_PATH.

    MANIFEST_PATH is the CSV produced by build-soft-labels (or any CSV with
    tile_id, p_soft_path, and optionally w_conf_path).
    """
    from pathlib import Path
    from pytorch_segmentation_models_trainer.tools.soft_labels import (
        generate_training_csv,
    )

    try:
        paths = generate_training_csv.run(
            manifest_path=manifest_path,
            output_dir=Path(output_dir),
            image_dir=image_dir,
            image_extension=image_extension,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    for name, path in paths.items():
        click.echo(f"  {name}: {path}")


@cli.command("scan-mask-colors")
@click.argument("mask_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--bands",
    default="1,2,3",
    show_default=True,
    help=(
        "Comma-separated 1-based band indices to use as R, G, B.  "
        "Repeat a band to duplicate it, e.g. '1,1,1' for single-band masks."
    ),
)
@click.option(
    "--window-size",
    default=1024,
    show_default=True,
    type=int,
    help="Tile height and width in pixels for each windowed read.",
)
@click.option(
    "--workers",
    default=None,
    type=int,
    help="Thread pool size.  Defaults to min(n_tiles, cpu_count).",
)
@click.option(
    "--no-progress",
    is_flag=True,
    default=False,
    help="Suppress the tqdm progress bar.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional path to write the JSON report.  Always printed to stdout as well.",
)
def scan_mask_colors_cmd(mask_path, bands, window_size, workers, no_progress, output):
    """Scan MASK_PATH for unique RGB tuples and print a color_map JSON.

    Reads the raster in parallel windows using a ThreadPoolExecutor and
    reports every unique (R, G, B) colour together with auto-assigned class
    indices ready to paste into the ``color_map`` field of
    ``MBTilesPolygonDataset``.

    (0, 0, 0) is treated as background and receives class 0; all other
    colours are assigned 1, 2, 3 … in sorted order.  Edit the indices before
    pasting into your training YAML.

    \b
    Examples:

        # Scan a mask MBTiles
        pytorch-smt-tools scan-mask-colors masks.mbtiles

        # Single-band mask — duplicate the band for RGB encoding
        pytorch-smt-tools scan-mask-colors mask.tif --bands 1,1,1

        # Save JSON to a file as well
        pytorch-smt-tools scan-mask-colors masks.mbtiles -o colors.json
    """
    import json as _json

    from pytorch_segmentation_models_trainer.tools.mbtiles.scan_unique_colors import (
        scan_and_report,
    )

    try:
        parsed_bands = [int(b.strip()) for b in bands.split(",")]
    except ValueError:
        raise click.UsageError(
            f"--bands must be comma-separated integers, got '{bands}'."
        )

    try:
        report = scan_and_report(
            raster_path=mask_path,
            bands=parsed_bands,
            window_size=window_size,
            workers=workers,
            progress=not no_progress,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    json_str = _json.dumps(report, indent=2)
    click.echo(json_str)

    if output:
        from pathlib import Path as _Path

        _Path(output).parent.mkdir(parents=True, exist_ok=True)
        _Path(output).write_text(json_str)
        click.echo(f"Report written to '{output}'.", err=True)


@cli.command("combine-bands")
@click.option(
    "--source-dir",
    "source_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Source directory to scan for rasters.  Repeat for each directory.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where combined GeoTIFFs will be written.",
)
@click.option(
    "--glob",
    "glob_pattern",
    default="**/*.vrt",
    show_default=True,
    help="Glob pattern used to find files inside each source directory.",
)
@click.option(
    "--name-pattern",
    default=None,
    help=(
        "Optional filename pattern to extract group key, e.g. 'MI_{name}.tif'."
        "  Must contain the literal '{name}'."
    ),
)
@click.option(
    "--skip-alpha/--no-skip-alpha",
    "skip_alpha",
    default=True,
    show_default=True,
    help="Skip last band when source has >3 bands (assumed alpha).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output files.",
)
@click.option(
    "--workers",
    default=4,
    show_default=True,
    type=int,
    help="Number of worker threads.",
)
def combine_bands_cmd(
    source_dirs,
    output_dir,
    glob_pattern,
    name_pattern,
    skip_alpha,
    overwrite,
    workers,
):
    """Combine bands from rasters with matching names across SOURCE_DIRs.

    Finds files matching GLOB in each SOURCE_DIR, groups them by filename stem
    (or by NAME_PATTERN capture), and combines all bands into a single GeoTIFF
    per group.
    """
    from pathlib import Path

    from pytorch_segmentation_models_trainer.tools.dataset_builder.band_combiner import (
        combine_all,
    )

    result = combine_all(
        source_dirs=[Path(d) for d in source_dirs],
        output_dir=Path(output_dir),
        glob_pattern=glob_pattern,
        name_pattern=name_pattern,
        skip_alpha=skip_alpha,
        overwrite=overwrite,
        n_workers=workers,
    )
    click.echo(f"Combined {len(result)} group(s) into '{output_dir}'.")


@cli.command("build-tile-dataset")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
def build_tile_dataset_cmd(yaml_path):
    """Build a tile dataset from raster images + vector masks using YAML_PATH.

    YAML_PATH must contain keys matching ``build_tile_dataset()`` parameters:
    ``image_paths``, ``vector_path``, ``class_attribute``, ``output_dir``,
    and optionally ``vector_layer``, ``tile_width``, ``tile_height``,
    ``overlap_x_percent``, ``overlap_y_percent``, ``min_valid_pixel_ratio``,
    ``skip_empty_tiles``, ``generate_full_size_masks``, ``max_workers``.
    """
    from pathlib import Path

    import yaml

    from pytorch_segmentation_models_trainer.tools.dataset_builder.tile_dataset_builder import (
        build_tile_dataset,
    )

    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    cfg["image_paths"] = [Path(p) for p in cfg["image_paths"]]
    cfg["vector_path"] = Path(cfg["vector_path"])
    cfg["output_dir"] = Path(cfg["output_dir"])

    df = build_tile_dataset(**cfg)
    click.echo(f"Dataset built: {len(df)} tiles saved to '{cfg['output_dir']}'.")


@cli.command("build-sliding-window-dataset")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Root output directory for patches.",
)
@click.option(
    "--window-size",
    default=256,
    show_default=True,
    type=int,
    help="Patch size in pixels (square).",
)
@click.option(
    "--overlap",
    default=0.0,
    show_default=True,
    type=float,
    help="Overlap fraction in [0, 1).",
)
@click.option(
    "--remap",
    default=None,
    help="Class remapping in the format '7:5,6:4'.",
)
@click.option(
    "--blacklist",
    default=None,
    help="Comma-separated directory name segments to skip.",
)
@click.option(
    "--workers",
    default=8,
    show_default=True,
    type=int,
    help="Number of worker threads.",
)
def build_sliding_window_dataset_cmd(
    input_csv,
    output_dir,
    window_size,
    overlap,
    remap,
    blacklist,
    workers,
):
    """Crop image/mask pairs from INPUT_CSV into sliding-window patches."""
    from pathlib import Path

    from pytorch_segmentation_models_trainer.tools.dataset_builder.sliding_window_builder import (
        build_sliding_window_dataset,
    )

    class_remap = None
    if remap:
        try:
            class_remap = {
                int(pair.split(":")[0].strip()): int(pair.split(":")[1].strip())
                for pair in remap.split(",")
            }
        except (ValueError, IndexError) as exc:
            raise click.UsageError(
                f"--remap must be in the format '7:5,6:4', got '{remap}'."
            ) from exc

    blacklist_list = [s.strip() for s in blacklist.split(",")] if blacklist else None

    df = build_sliding_window_dataset(
        input_csv=Path(input_csv),
        output_dir=Path(output_dir),
        window_size=window_size,
        overlap=overlap,
        class_remap=class_remap,
        blacklist=blacklist_list,
        n_workers=workers,
    )
    click.echo(f"Generated {len(df)} patch pair(s) in '{output_dir}'.")


@cli.command("remap-mask-classes")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory tree of mask rasters to remap.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory (mirrors the structure of --input-dir).",
)
@click.option(
    "--mapping",
    required=True,
    help="Pixel value remapping in the format '7:5,6:4'.",
)
@click.option(
    "--workers",
    default=None,
    type=int,
    help="Number of worker threads.  Defaults to cpu_count.",
)
def remap_mask_classes_cmd(input_dir, output_dir, mapping, workers):
    """Remap pixel class values in all TIFFs under INPUT_DIR.

    OUTPUT_DIR mirrors the subdirectory structure of INPUT_DIR.
    """
    from pathlib import Path

    from pytorch_segmentation_models_trainer.tools.raster.tiff_remap import (
        remap_raster_folder,
    )

    try:
        pixel_mapping = {
            int(pair.split(":")[0].strip()): int(pair.split(":")[1].strip())
            for pair in mapping.split(",")
        }
    except (ValueError, IndexError) as exc:
        raise click.UsageError(
            f"--mapping must be in the format '7:5,6:4', got '{mapping}'."
        ) from exc

    n_success, n_errors = remap_raster_folder(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        pixel_mapping=pixel_mapping,
        n_workers=workers,
    )
    click.echo(f"Remapped {n_success} file(s), {n_errors} error(s).")


@cli.command("convert-to-tiff")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory to scan for matching files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory (mirrors structure of --input-dir).",
)
@click.option(
    "--glob",
    "glob_pattern",
    default="**/*.vrt",
    show_default=True,
    help="Glob pattern for files to convert.",
)
@click.option(
    "--compression",
    default="LZW",
    show_default=True,
    type=click.Choice(["LZW", "DEFLATE", "NONE", "JPEG"], case_sensitive=False),
    help="GeoTIFF compression codec.",
)
@click.option(
    "--workers",
    default=4,
    show_default=True,
    type=int,
    help="Number of worker threads.",
)
def convert_to_tiff_cmd(input_dir, output_dir, glob_pattern, compression, workers):
    """Convert VRT (or any rasterio-readable) files to GeoTIFF.

    Files matching GLOB under INPUT_DIR are converted to compressed GeoTIFFs
    under OUTPUT_DIR, preserving subdirectory structure.
    """
    from pathlib import Path

    from pytorch_segmentation_models_trainer.tools.raster.vrt2tif import convert_folder

    n_success, n_errors = convert_folder(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        glob_pattern=glob_pattern,
        compression=compression.upper(),
        n_workers=workers,
    )
    click.echo(f"Converted {n_success} file(s), {n_errors} error(s).")


@cli.command("visualize-predictions")
@click.option(
    "--records-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV with tile_id, mi, and the sort-by column.",
)
@click.option(
    "--gt-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory with ground-truth masks.",
)
@click.option(
    "--pred-dir",
    "pred_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="Prediction directory.  Repeat for each experiment.",
)
@click.option(
    "--pred-label",
    "pred_labels",
    multiple=True,
    help="Label for each --pred-dir.  Must match the count of --pred-dir.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output path for the figure (e.g. grid.png).",
)
@click.option(
    "--sort-by",
    default="mean_iou",
    show_default=True,
    help="Column in --records-csv used for sorting.",
)
@click.option(
    "--n-samples",
    default=5,
    show_default=True,
    type=int,
    help="Number of sample rows in the grid.",
)
@click.option(
    "--mode",
    default="best",
    show_default=True,
    type=click.Choice(["best", "worst", "random"], case_sensitive=False),
    help="Sample selection strategy.",
)
@click.option(
    "--image-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Optional directory with source images (adds an image column).",
)
@click.option(
    "--dpi",
    default=150,
    show_default=True,
    type=int,
    help="Figure DPI.",
)
@click.option(
    "--color-map",
    default=None,
    help=(
        "JSON string or path to JSON file mapping class IDs to RGB colors. "
        'Format: \'{"0":[0,0,0],"1":[255,0,0],...}\'.  '
        "If omitted, a default map is generated from GT raster values."
    ),
)
def visualize_predictions_cmd(
    records_csv,
    gt_dir,
    pred_dirs,
    pred_labels,
    output,
    sort_by,
    n_samples,
    mode,
    image_dir,
    dpi,
    color_map,
):
    """Build a segmentation comparison grid from predictions vs GT.

    Produces an image with one row per sample showing the source image (if
    --image-dir is given), the GT mask, and one column per --pred-dir.
    """
    import json
    from pathlib import Path

    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")

    from pytorch_segmentation_models_trainer.tools.visualization.segmentation_vis import (
        create_segmentation_grid,
    )

    records = pd.read_csv(records_csv)

    # Parse color map
    parsed_color_map = None
    if color_map is not None:
        p = Path(color_map)
        if p.exists():
            with open(p) as fh:
                raw = json.load(fh)
        else:
            raw = json.loads(color_map)
        parsed_color_map = {int(k): tuple(v) for k, v in raw.items()}

    if parsed_color_map is None:
        # Generate a default color map from GT rasters
        import rasterio

        unique_vals: set = set()
        gt_path = Path(gt_dir)
        for tif in list(gt_path.rglob("*.tif"))[:20]:
            with rasterio.open(tif) as src:
                data = src.read(1)
            unique_vals.update(int(v) for v in set(data.ravel()))

        rng = __import__("random").Random(42)
        parsed_color_map = {}
        for cls_id in sorted(unique_vals):
            parsed_color_map[cls_id] = (
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
            )

    labels = (
        list(pred_labels)
        if pred_labels
        else [f"pred_{i}" for i in range(len(pred_dirs))]
    )

    fig = create_segmentation_grid(
        records=records,
        gt_dir=Path(gt_dir),
        pred_dirs=[Path(d) for d in pred_dirs],
        pred_labels=labels,
        color_map=parsed_color_map,
        output_path=Path(output),
        image_dir=Path(image_dir) if image_dir else None,
        sort_by=sort_by,
        n_samples=n_samples,
        mode=mode,
        dpi=dpi,
    )
    click.echo(f"Figure saved to '{output}'.")


def entry():
    """Entry point registered in pyproject.toml."""
    cli()
