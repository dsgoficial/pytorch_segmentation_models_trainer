# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import unittest
from pytorch_segmentation_models_trainer.utils.model_utils import (
    replace_activation,
    set_model_components_trainable,
)


class TestModelUtils(unittest.TestCase):
    def test_replace_activation(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.relu = nn.ReLU()
                self.layer2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU())

        model = SimpleModel()
        # Verify initial activations
        self.assertIsInstance(model.relu, nn.ReLU)
        self.assertIsInstance(model.layer2[1], nn.ReLU)

        new_activation = nn.LeakyReLU()
        replace_activation(model, nn.ReLU(), new_activation)

        self.assertIsInstance(model.relu, nn.LeakyReLU)
        self.assertIsInstance(model.layer2[1], nn.LeakyReLU)

    def test_set_model_components_trainable(self):
        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 5)

        model = MultiLayerModel()

        # Set all to false
        set_model_components_trainable(model, trainable=False)
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

        # Set all to true
        set_model_components_trainable(model, trainable=True)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

        # Set with exception
        set_model_components_trainable(
            model, trainable=False, exception_list=["layer1"]
        )
        self.assertTrue(model.layer1.weight.requires_grad)
        self.assertTrue(model.layer1.bias.requires_grad)
        self.assertFalse(model.layer2.weight.requires_grad)
        self.assertFalse(model.layer2.bias.requires_grad)


if __name__ == "__main__":
    unittest.main()
