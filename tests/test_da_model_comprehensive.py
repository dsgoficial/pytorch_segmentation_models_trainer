# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model import (
    DomainAdaptationModel,
)
from pytorch_segmentation_models_trainer.domain_adaptation.base_method import (
    BaseDomainAdaptationMethod,
    DomainAdaptationLossOutput,
)


class _DummyDAMethod(BaseDomainAdaptationMethod):
    def __init__(self):
        super().__init__(lambda_da=1.0)
        # Define as property or normal attribute that exists
        self.requires_features = True

    def forward(self, *args, **kwargs):
        return torch.tensor(0.1)

    def compute_da_loss(self, *args, **kwargs):
        return DomainAdaptationLossOutput(
            loss=torch.tensor(0.1), log_dict={"dummy_loss": 0.1}
        )


class TestDAModelComprehensive(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": 2,
                    "kernel_size": 1,
                },
                "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-3},
                "hyperparameters": {
                    "batch_size": 2,
                    "lr": 1e-3,
                },
                "domain_adaptation": {
                    "method": {
                        "_target_": "pytorch_segmentation_models_trainer.domain_adaptation.methods.dann.DANNMethod",
                        "lambda_da": 1.0,
                        "classifier": {"in_channels": 64, "hidden_channels": 64},
                    },
                    "feature_layers": ["encoder"],
                    "source_dataset": {"_target_": "torch.utils.data.Dataset"},
                    "target_dataset": {"_target_": "torch.utils.data.Dataset"},
                    "source_val_dataset": {"_target_": "torch.utils.data.Dataset"},
                    "target_val_dataset": {"_target_": "torch.utils.data.Dataset"},
                },
                "train_dataset": {"data_loader": {"num_workers": 0}},
                "val_dataset": {"data_loader": {"num_workers": 0}},
            }
        )

    def test_da_model_init_and_steps(self):
        # PATCH the Model.get_model to return our custom mock_model
        # because the original code calls self.get_model() which calls Model.get_model()
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.Model.get_model"
        ) as mock_get_model:
            with patch(
                "pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model.instantiate"
            ) as mock_inst:
                # 1. Setup Model
                mock_model = MagicMock(spec=nn.Module)
                mock_model.encoder = nn.Conv2d(3, 64, 1)
                mock_model.return_value = torch.randn(2, 2, 8, 8)
                mock_get_model.return_value = mock_model

                # 2. Setup DA Method
                mock_da_method = _DummyDAMethod()

                # 3-7. instantiate will be called for: method, source_ds, target_ds, val_source_ds, val_target_ds
                mock_inst.side_effect = [
                    mock_da_method,
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ]

                model = DomainAdaptationModel(self.cfg)
                model.log = MagicMock()
                model.log_dict = MagicMock()

                batch = {
                    "source": {
                        "image": torch.randn(2, 3, 8, 8),
                        "mask": torch.randint(0, 2, (2, 8, 8)),
                    },
                    "target": {
                        "image": torch.randn(2, 3, 8, 8),
                        "mask": torch.randint(0, 2, (2, 8, 8)),
                    },
                }

                loss = model.training_step(batch, 0)
                self.assertIsInstance(loss, torch.Tensor)

                model.validation_step(batch["source"], 0, dataloader_idx=0)
                model.validation_step(batch["target"], 0, dataloader_idx=1)


if __name__ == "__main__":
    unittest.main()
