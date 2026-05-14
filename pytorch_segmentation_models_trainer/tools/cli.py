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


def entry():
    """Entry point registered in pyproject.toml."""
    cli()
