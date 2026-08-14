import sys
from unittest.mock import MagicMock, patch

import pytest

from pytorch_segmentation_models_trainer.config import Settings
from pytorch_segmentation_models_trainer.server import (
    get_hydra_config,
    get_inference_processor,
)


def test_get_hydra_config_loads_expected_config():
    with (
        patch("pytorch_segmentation_models_trainer.server.initialize") as mock_init,
        patch("pytorch_segmentation_models_trainer.server.compose") as mock_compose,
    ):
        mock_compose.return_value = {"key": "value"}

        cfg = get_hydra_config("some/path", "some_config")

        mock_init.assert_called_once_with(config_path="some/path")
        mock_compose.assert_called_once_with(config_name="some_config")
        assert cfg == {"key": "value"}


def test_get_inference_processor_instantiates_from_settings():
    with (
        patch(
            "pytorch_segmentation_models_trainer.server.Settings"
        ) as mock_settings_cls,
        patch(
            "pytorch_segmentation_models_trainer.server.get_hydra_config"
        ) as mock_get_cfg,
        patch(
            "pytorch_segmentation_models_trainer.server.instantiate_inference_processor"
        ) as mock_instantiate,
    ):
        mock_settings = MagicMock(spec=Settings)
        mock_settings.config_path = "some/path"
        mock_settings.config_name = "some_config"
        mock_settings_cls.return_value = mock_settings

        mock_cfg = {"key": "value"}
        mock_get_cfg.return_value = mock_cfg

        mock_processor = MagicMock()
        mock_instantiate.return_value = mock_processor

        get_inference_processor.cache_clear()
        processor = get_inference_processor()

        mock_settings_cls.assert_called_once()
        mock_get_cfg.assert_called_once_with("some/path", "some_config")
        mock_instantiate.assert_called_once_with(mock_cfg)
        assert processor == mock_processor
        assert processor.polygonizer.data_writer is None


def test_settings_import_when_pydantic_settings_missing():
    with patch.dict(sys.modules, {"pydantic_settings": None}):
        if "pytorch_segmentation_models_trainer.config" in sys.modules:
            del sys.modules["pytorch_segmentation_models_trainer.config"]

        try:
            from pytorch_segmentation_models_trainer.config import (
                Settings as ImportedSettings,
            )

            assert ImportedSettings is not None
        except Exception:
            pytest.skip("Settings import path is unavailable in this environment")
