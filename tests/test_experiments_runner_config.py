# -*- coding: utf-8 -*-
"""
Tests for experiments_runner_config dataclasses.
"""

from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.config_definitions.experiments_runner_config import (
    ExperimentsRunnerConfig,
)


class TestExperimentsRunnerConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(ExperimentsRunnerConfig)
        assert cfg.n_runs is None
        assert cfg.seeds is None
        assert cfg.output_base_dir == "outputs/experiments_runner"
        assert cfg.save_summary is True
        assert cfg.summary_metrics == ["val/loss"]
        assert cfg.resume is False

    def test_kfold_is_none_by_default(self):
        cfg = OmegaConf.structured(ExperimentsRunnerConfig)
        assert cfg.kfold is None

    def test_accepts_kfold_block(self):
        from pytorch_segmentation_models_trainer.config_definitions.kfold_config import (
            KFoldConfig,
        )

        runner = ExperimentsRunnerConfig()
        runner.kfold = KFoldConfig()
        cfg = OmegaConf.structured(runner)
        assert cfg.kfold is not None
        assert cfg.kfold.n_splits == 5


class TestKFoldConfig:
    def _import(self):
        from pytorch_segmentation_models_trainer.config_definitions.kfold_config import (
            KFoldConfig,
        )

        return KFoldConfig

    def test_defaults(self):
        KFoldConfig = self._import()
        cfg = OmegaConf.structured(KFoldConfig)
        assert cfg.n_splits == 5
        assert cfg.seed == 42
        assert cfg.split_strategy == "by_image"
        assert cfg.group_col == "image"
        assert cfg.buffer_px == 0
        assert cfg.row_col == "row_off"
        assert cfg.col_col == "col_off"
        assert cfg.input_csv_path == ""
        assert cfg.save_fold_csvs_dir == "fold_splits"

    def test_can_override_fields(self):
        KFoldConfig = self._import()
        cfg = OmegaConf.structured(KFoldConfig)
        OmegaConf.update(cfg, "n_splits", 10)
        OmegaConf.update(cfg, "split_strategy", "by_spatial_region")
        OmegaConf.update(cfg, "buffer_px", 256)
        assert cfg.n_splits == 10
        assert cfg.split_strategy == "by_spatial_region"
        assert cfg.buffer_px == 256

    def test_to_container(self):
        KFoldConfig = self._import()
        cfg = OmegaConf.structured(KFoldConfig)
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        assert "n_splits" in container
        assert "split_strategy" in container
