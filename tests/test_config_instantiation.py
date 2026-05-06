# -*- coding: utf-8 -*-
import unittest
from typing import Any
from pytorch_segmentation_models_trainer.config_definitions import (
    coco_dataset_config,
    dataset_config,
    domain_adaptation_config,
    edl_config,
    evaluation_config,
    fine_tuning_config,
    inference_config,
    loss_config_definition,
    mc_dropout_config,
    predict_config,
    train_config,
    tools_config_def,
)


class TestConfigInstantiation(unittest.TestCase):
    def test_coco_dataset_config(self):
        config = coco_dataset_config.CocoDatasetConfig()
        self.assertEqual(config.info.year, "???")

    def test_dataset_config(self):
        config = dataset_config.DatasetConfig()
        self.assertEqual(config.input_csv_path, "???")

    def test_domain_adaptation_config(self):
        source = domain_adaptation_config.DomainAdaptationDatasetConfig()
        method = domain_adaptation_config.DomainAdaptationMethodConfig()
        config = domain_adaptation_config.DomainAdaptationConfig(
            source_dataset=source, method=method
        )
        self.assertEqual(config.method._target_, "???")

    def test_edl_config(self):
        config = edl_config.EvidentialWrapperConfig(model={"_target_": "some_model"})
        self.assertEqual(config.freeze_encoder, False)

    def test_evaluation_config(self):
        config = evaluation_config.ExperimentConfig(
            name="test", predict_config="p", checkpoint_path="c", output_folder="o"
        )
        self.assertEqual(config.name, "test")

    def test_fine_tuning_config(self):
        config = fine_tuning_config.FineTuningConfig()
        self.assertEqual(config.strategy, "full")

    def test_inference_config(self):
        config = inference_config.InferenceDatasetConfig(input_csv_path="csv")
        self.assertEqual(config.input_csv_path, "csv")

    def test_loss_config(self):
        config = loss_config_definition.SegLossConfig(name="seg")
        self.assertEqual(config.name, "seg")

    def test_mc_dropout_config(self):
        config = mc_dropout_config.MCDropoutInferenceProcessorConfig()
        self.assertEqual(config.n_samples, 10)

    def test_predict_config(self):
        config = predict_config.PredictSingleImageConfig()
        self.assertEqual(config.inference_threshold, 0.5)

    def test_train_config(self):
        config = train_config.Hyperparameters()
        self.assertEqual(config.epochs, 10)

    def test_tools_config(self):
        config = tools_config_def.LossParamsConfig()
        self.assertEqual(config.curvature_dissimilarity_threshold, 15)


if __name__ == "__main__":
    unittest.main()
