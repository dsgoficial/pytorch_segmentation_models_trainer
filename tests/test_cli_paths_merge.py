# -*- coding: utf-8 -*-
"""Tests for `_load_config_with_paths` and its use by the SAM/SLICO
correction CLI commands — the `${paths.xxx}` interpolation + sibling
`paths.yaml` auto-merge these non-Hydra `pytorch-smt-tools` commands
otherwise have no way to support (plain `yaml.safe_load` has neither)."""

import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from pytorch_segmentation_models_trainer.tools.cli import (
    _load_config_with_paths,
    cli,
)


class TestLoadConfigWithPaths:
    def _write(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    def test_no_sibling_paths_yaml_loads_plain(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        self._write(cfg_path, "foo: 1\nbar: baz\n")
        result = _load_config_with_paths(str(cfg_path))
        assert result == {"foo": 1, "bar": "baz"}

    def test_sibling_paths_yaml_resolves_interpolation(self, tmp_path):
        self._write(tmp_path / "paths.yaml", "paths:\n  base: /data/real\n")
        cfg_path = tmp_path / "config.yaml"
        self._write(
            cfg_path,
            "sam_label_correction:\n  coreset_csv: ${paths.base}/coreset.csv\n",
        )
        result = _load_config_with_paths(str(cfg_path))
        assert result["sam_label_correction"]["coreset_csv"] == "/data/real/coreset.csv"

    def test_sibling_paths_yaml_absent_leaves_interpolation_unresolved_raises(
        self, tmp_path
    ):
        cfg_path = tmp_path / "config.yaml"
        self._write(cfg_path, "x: ${paths.base}/y\n")
        import omegaconf

        with __import__("pytest").raises(omegaconf.errors.OmegaConfBaseException):
            _load_config_with_paths(str(cfg_path))

    def test_paths_key_isolated_from_nested_subsection(self, tmp_path):
        """Nested form (`sam_label_correction:` wrapper) keeps the merged-in
        `paths:` key out of the dict handed to the dataclass — flat-form
        configs (no wrapper) are the caller's responsibility not to collide,
        documented behavior, not this function's job to filter."""
        self._write(tmp_path / "paths.yaml", "paths:\n  base: /data/x\n")
        cfg_path = tmp_path / "config.yaml"
        self._write(cfg_path, "sam_label_correction:\n  a: 1\n")
        result = _load_config_with_paths(str(cfg_path))
        assert "paths" in result  # present at top level
        assert (
            "paths" not in result["sam_label_correction"]
        )  # not leaked into subsection


_SAM_MODULE = (
    "pytorch_segmentation_models_trainer.tools.sam_correction.sam_label_corrector"
)
_SLICO_MODULE = (
    "pytorch_segmentation_models_trainer.tools.slico_correction.slico_label_corrector"
)


class TestSamCliUsesPathsMerge:
    def _write(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    @patch(f"{_SAM_MODULE}.SamLabelCorrector")
    @patch(f"{_SAM_MODULE}.SAMLabelCorrectionConfig")
    def test_interpolated_path_reaches_config(
        self, mock_config_cls, mock_corrector_cls, tmp_path
    ):
        mock_corrector_cls.return_value.run.return_value = {
            "n_tiles": 0,
            "elapsed_s": 0,
        }
        mock_config_cls.return_value.sam_checkpoint = "/fake/checkpoint.pth"
        self._write(tmp_path / "paths.yaml", "paths:\n  base: /data/real\n")
        cfg_path = tmp_path / "sam_gc.yaml"
        self._write(
            cfg_path,
            "sam_label_correction:\n"
            "  coreset_csv: ${paths.base}/coreset.csv\n"
            "  sam_checkpoint: /fake/does/not/need/to/exist/for/this/mock\n",
        )
        with patch("pathlib.Path.exists", return_value=True):
            runner = CliRunner()
            result = runner.invoke(cli, ["build-sam-corrected-masks", str(cfg_path)])

        assert result.exit_code == 0, result.output
        called_kwargs = mock_config_cls.call_args.kwargs
        assert called_kwargs["coreset_csv"] == "/data/real/coreset.csv"


class TestSlicoCliUsesPathsMerge:
    def _write(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    @patch(f"{_SLICO_MODULE}.SlicoLabelCorrector")
    @patch(f"{_SLICO_MODULE}.SLICOLabelCorrectionConfig")
    def test_interpolated_path_reaches_config(
        self, mock_config_cls, mock_corrector_cls, tmp_path
    ):
        mock_corrector_cls.return_value.run.return_value = {
            "n_tiles": 0,
            "elapsed_s": 0,
        }
        self._write(tmp_path / "paths.yaml", "paths:\n  base: /data/real\n")
        cfg_path = tmp_path / "slico_gc.yaml"
        self._write(
            cfg_path,
            "slico_label_correction:\n  coreset_csv: ${paths.base}/coreset.csv\n",
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["build-slico-corrected-masks", str(cfg_path)])

        assert result.exit_code == 0, result.output
        called_kwargs = mock_config_cls.call_args.kwargs
        assert called_kwargs["coreset_csv"] == "/data/real/coreset.csv"
