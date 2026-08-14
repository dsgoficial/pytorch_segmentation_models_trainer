# conftest.py — resolve conflito fbgemm.dll vs GDAL no Windows
# Deve ser processado pelo pytest ANTES de importar qualquer test module.
import os
import sys

if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(_torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import pytorch_segmentation_models_trainer  # noqa: F401, E402

import tempfile
import pytest
from unittest.mock import MagicMock
from omegaconf import OmegaConf
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


@pytest.fixture
def mock_trainer():
    """MagicMock of pl.Trainer with fit() as a no-op."""
    trainer = MagicMock(spec=pl.Trainer)
    trainer.fit.return_value = None
    return trainer


@pytest.fixture
def temp_output_dir():
    """Temporary directory with automatic cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_simple_model():
    """MagicMock of smp.Unet."""
    return MagicMock(spec=smp.Unet)


@pytest.fixture
def basic_cfg():
    """Minimal OmegaConf config without disk I/O."""
    return OmegaConf.create(
        {
            "model": {
                "_target_": "segmentation_models_pytorch.Unet",
                "encoder_name": "resnet18",
                "in_channels": 3,
                "classes": 1,
            },
            "loss": {
                "_target_": "torch.nn.BCEWithLogitsLoss",
            },
            "hyperparameters": {
                "batch_size": 4,
                "devices": 1,
                "accelerator": "cpu",
            },
        }
    )
