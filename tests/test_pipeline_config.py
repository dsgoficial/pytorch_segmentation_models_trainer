# -*- coding: utf-8 -*-
"""
Tests for pipeline_config dataclasses.
"""

from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.config_definitions.pipeline_config import (
    PipelineConfig,
    PipelineGpu,
    PipelineResources,
    PipelineStep,
)


class TestPipelineStep:
    def test_defaults(self):
        cfg = OmegaConf.structured(PipelineStep)
        assert cfg.name == ""
        assert cfg.config_dir == ""
        assert cfg.config_name == ""
        assert list(cfg.command) == []
        assert list(cfg.overrides) == []
        assert cfg.overwrite is None
        assert cfg.enabled is True
        assert list(cfg.depends_on) == []
        assert cfg.gpus == 0
        assert cfg.vram_gb is None

    def test_can_set_fields(self):
        step = PipelineStep(
            name="sam_gc",
            config_dir="conf/examples",
            config_name="sam_label_correction",
            overrides=["seed=42"],
            overwrite=True,
            enabled=False,
        )
        cfg = OmegaConf.structured(step)
        assert cfg.name == "sam_gc"
        assert cfg.config_dir == "conf/examples"
        assert cfg.config_name == "sam_label_correction"
        assert list(cfg.overrides) == ["seed=42"]
        assert cfg.overwrite is True
        assert cfg.enabled is False

    def test_can_set_command(self):
        step = PipelineStep(
            name="slico_gc",
            command=["pytorch-smt-tools", "build-slico-corrected-masks", "x.yaml"],
        )
        cfg = OmegaConf.structured(step)
        assert list(cfg.command) == [
            "pytorch-smt-tools",
            "build-slico-corrected-masks",
            "x.yaml",
        ]

    def test_can_set_depends_on_and_gpu_fields(self):
        step = PipelineStep(
            name="unet_slico_gc",
            config_dir="conf",
            config_name="x",
            depends_on=["slico_gc"],
            gpus=2,
            vram_gb=12.0,
        )
        cfg = OmegaConf.structured(step)
        assert list(cfg.depends_on) == ["slico_gc"]
        assert cfg.gpus == 2
        assert cfg.vram_gb == 12.0


class TestPipelineGpu:
    def test_defaults(self):
        cfg = OmegaConf.structured(PipelineGpu)
        assert cfg.id == 0
        assert cfg.vram_gb == 0.0

    def test_can_set_fields(self):
        gpu = PipelineGpu(id=1, vram_gb=24.0)
        cfg = OmegaConf.structured(gpu)
        assert cfg.id == 1
        assert cfg.vram_gb == 24.0


class TestPipelineResources:
    def test_defaults(self):
        cfg = OmegaConf.structured(PipelineResources)
        assert list(cfg.gpus) == []
        assert cfg.max_parallel_cpu_steps == 1

    def test_accepts_gpu_list(self):
        res = PipelineResources(
            gpus=[PipelineGpu(id=0, vram_gb=24.0), PipelineGpu(id=1, vram_gb=24.0)],
            max_parallel_cpu_steps=4,
        )
        cfg = OmegaConf.structured(res)
        assert len(cfg.gpus) == 2
        assert cfg.gpus[1].id == 1
        assert cfg.max_parallel_cpu_steps == 4


class TestPipelineConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(PipelineConfig)
        assert list(cfg.steps) == []
        assert cfg.output_base_dir == "outputs/pipeline"
        assert cfg.resume is True
        assert cfg.overwrite is False
        assert cfg.on_error == "stop"
        assert cfg.preflight is True
        assert cfg.pytorch_smt_bin == "pytorch-smt"
        assert cfg.resources is None

    def test_accepts_steps_list(self):
        pipeline = PipelineConfig(
            steps=[
                PipelineStep(name="a", config_dir="conf", config_name="a_config"),
                PipelineStep(name="b", config_dir="conf", config_name="b_config"),
            ]
        )
        cfg = OmegaConf.structured(pipeline)
        assert len(cfg.steps) == 2
        assert cfg.steps[0].name == "a"
        assert cfg.steps[1].name == "b"
