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


def entry():
    """Entry point registered in pyproject.toml."""
    cli()
