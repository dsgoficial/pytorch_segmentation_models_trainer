# -*- coding: utf-8 -*-
"""Thin shim — use 'pytorch-smt-tools download-gee-lulc' instead."""

if __name__ == "__main__":
    from pytorch_segmentation_models_trainer.tools.cli import cli

    cli(["download-gee-lulc"])
