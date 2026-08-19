# -*- coding: utf-8 -*-
"""
Tests for tools_config_def dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.tools_config_def import (
    SingleImageInfereceProcessorConfig,
    SingleImageFromFrameFieldProcessorConfig,
    MultiClassInferenceProcessorConfig,
    ObjectDetectionInferenceProcessorConfig,
    PolygonRNNInferenceProcessorConfig,
    SingleImageReaderProcessorConfig,
    FolderImageReaderProcessorConfig,
    CSVImageReaderProcessorConfig,
    VectorFileDataWriterConfig,
    RasterDataWriterConfig,
    VectorDatabaseDataWriterConfig,
    MBTilesMulticlassMaskBuilderConfig,
    RasterExportInferenceStrategyConfig,
    MultipleRasterExportInferenceStrategyConfig,
    VectorFileExportInferenceStrategyConfig,
    VectorDatabaseExportInferenceStrategyConfig,
    ObjectDetectionExportInferenceStrategyConfig,
    SimplePolygonizerProcessorConfig,
    ACMPolygonizerProcessorConfig,
    ASMPolygonizerProcessorConfig,
    PolygonRNNPolygonizerProcessorConfig,
)


class TestInferenceProcessorConfigs:
    def test_single_image(self):
        cfg = OmegaConf.structured(SingleImageInfereceProcessorConfig)
        assert "SingleImageInfereceProcessor" in cfg._target_
        assert cfg.mask_bands == 1

    def test_frame_field(self):
        cfg = OmegaConf.structured(SingleImageFromFrameFieldProcessorConfig)
        assert "SingleImageFromFrameFieldProcessor" in cfg._target_

    def test_multiclass(self):
        cfg = OmegaConf.structured(MultiClassInferenceProcessorConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_object_detection(self):
        cfg = OmegaConf.structured(ObjectDetectionInferenceProcessorConfig)
        assert "ObjectDetectionInferenceProcessor" in cfg._target_

    def test_polygon_rnn(self):
        cfg = OmegaConf.structured(PolygonRNNInferenceProcessorConfig)
        assert cfg.sequence_length == 60


class TestReaderConfigs:
    def test_single_image_reader(self):
        cfg = OmegaConf.structured(SingleImageReaderProcessorConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.file_name

    def test_folder_image_reader(self):
        cfg = OmegaConf.structured(FolderImageReaderProcessorConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.folder_name
        assert cfg.image_extension == "tif"

    def test_csv_image_reader(self):
        cfg = OmegaConf.structured(CSVImageReaderProcessorConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.input_csv_path


class TestWriterConfigs:
    def test_vector_file_writer(self):
        cfg = OmegaConf.structured(VectorFileDataWriterConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.output_file_folder
        assert cfg.driver == "GeoJSON"

    def test_raster_writer(self):
        cfg = OmegaConf.structured(RasterDataWriterConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.output_file_path

    def test_vector_db_writer(self):
        cfg = OmegaConf.structured(VectorDatabaseDataWriterConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.user
        assert cfg.port == 5432


class TestMBTilesMulticlassMaskBuilderConfig:
    def test_defaults_and_hydra_registration(self):
        cfg = OmegaConf.structured(MBTilesMulticlassMaskBuilderConfig)
        assert cfg.frame_id_attribute == "rect_id"
        assert cfg.class_attribute == "tipo"
        assert cfg.output_subdir == "masks"
        assert cfg.background_value == 255

        from hydra.core.config_store import ConfigStore

        cs = ConfigStore.instance()
        node = cs.load("mbtiles_multiclass_mask/default.yaml")
        assert "build_mbtiles_multiclass_masks" in node.node["_target_"]


class TestExportStrategyConfigs:
    def test_raster_export(self):
        cfg = OmegaConf.structured(RasterExportInferenceStrategyConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.output_file_path

    def test_multiple_raster_export(self):
        cfg = OmegaConf.structured(MultipleRasterExportInferenceStrategyConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.output_folder

    def test_vector_file_export(self):
        cfg = OmegaConf.structured(VectorFileExportInferenceStrategyConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.output_file_path

    def test_vector_db_export(self):
        cfg = OmegaConf.structured(VectorDatabaseExportInferenceStrategyConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.user

    def test_obj_detection_export(self):
        cfg = OmegaConf.structured(ObjectDetectionExportInferenceStrategyConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.output_file_path


class TestPolygonizerConfigs:
    def test_simple_polygonizer(self):
        cfg = OmegaConf.structured(SimplePolygonizerProcessorConfig)
        assert cfg.config.data_level == 0.5
        assert cfg.config.min_area == 10

    def test_acm_polygonizer(self):
        cfg = OmegaConf.structured(ACMPolygonizerProcessorConfig)
        assert cfg.config.steps == 500
        assert cfg.config.inner_polylines_params.enable is False

    def test_asm_polygonizer(self):
        cfg = OmegaConf.structured(ASMPolygonizerProcessorConfig)
        assert cfg.config.init_method == "skeleton"
        assert cfg.config.loss_params.coefs.data == [1.0, 0.1, 0.0, 0.0]

    def test_polygon_rnn_polygonizer(self):
        cfg = OmegaConf.structured(PolygonRNNPolygonizerProcessorConfig)
        assert cfg.config.grid_size == 28
