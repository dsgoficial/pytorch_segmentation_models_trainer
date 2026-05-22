# -*- coding: utf-8 -*-
import runpy
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer import predict_from_batch as module


class _SeriesSwifter:
    def __init__(self, series):
        self._series = series

    def apply(self, func):
        return self._series.apply(func)


def _install_swifter_accessor(monkeypatch):
    monkeypatch.setattr(
        pd.Series,
        "swifter",
        property(lambda series: _SeriesSwifter(series)),
        raising=False,
    )


def _base_cfg(**overrides):
    data = {
        "inference_dataset": {
            "input_csv_path": "/data/inference.csv",
            "root_dir": "/data/images",
            "data_loader": {"num_workers": 0, "prefetch_factor": 2},
        },
        "hyperparameters": {"batch_size": 2},
        "pl_model": {"_target_": "tests.MockModel"},
        "checkpoint_path": "/models/model.ckpt",
        "pl_trainer": {"accelerator": "cpu"},
    }
    data.update(overrides)
    return OmegaConf.create(data)


def test_prepare_inference_csv_uses_inference_dataset_direct_path():
    cfg = _base_cfg()

    csv_path, root_dir = module.prepare_inference_csv(cfg)

    assert csv_path == "/data/inference.csv"
    assert root_dir == "/data/images"


def test_prepare_inference_csv_builds_from_folder():
    cfg = OmegaConf.create(
        {
            "inference_dataset": {
                "build_csv_from_folder": {
                    "enabled": True,
                    "images_folder": "/images",
                }
            }
        }
    )

    with patch(
        "pytorch_segmentation_models_trainer.tools.inference."
        "inference_csv_builder.build_inference_csv_from_config",
        return_value="/tmp/built.csv",
    ) as mock_builder:
        csv_path, root_dir = module.prepare_inference_csv(cfg)

    assert csv_path == "/tmp/built.csv"
    assert root_dir == "/images"
    mock_builder.assert_called_once_with(cfg.inference_dataset.build_csv_from_folder)


def test_prepare_inference_csv_uses_legacy_val_dataset():
    cfg = OmegaConf.create({"val_dataset": {"input_csv_path": "/legacy/val.csv"}})

    csv_path, root_dir = module.prepare_inference_csv(cfg)

    assert csv_path == "/legacy/val.csv"
    assert root_dir == "/legacy"


def test_prepare_inference_csv_rejects_missing_dataset_config():
    with pytest.raises(ValueError, match="No valid dataset configuration"):
        module.prepare_inference_csv(OmegaConf.create({}))


def test_instantiate_dataloaders_reads_limited_rows():
    cfg = _base_cfg(
        inference_dataset={
            "input_csv_path": "/data/inference.csv",
            "root_dir": "/data/images",
            "n_first_rows_to_read": 3,
        }
    )
    df = pd.DataFrame({"image": ["a.tif"], "width": [8], "height": [8]})

    with (
        patch.object(module.pd, "read_csv", return_value=df) as mock_read_csv,
        patch.object(
            module, "get_grouped_dataloaders", return_value=["loader"]
        ) as mock_grouped,
    ):
        loaders = module.instantiate_dataloaders(cfg)

    assert loaders == ["loader"]
    mock_read_csv.assert_called_once_with("/data/inference.csv", nrows=3)
    mock_grouped.assert_called_once_with(cfg, df, "/data/images", False)


def test_instantiate_dataloaders_uses_windowed_flag_from_config():
    cfg = _base_cfg(use_inference_processor=True)
    df = pd.DataFrame({"image": ["a.tif"], "width": [8], "height": [8]})

    with (
        patch.object(module.pd, "read_csv", return_value=df),
        patch.object(
            module, "get_grouped_dataloaders", return_value=["loader"]
        ) as mock_grouped,
    ):
        module.instantiate_dataloaders(cfg)

    mock_grouped.assert_called_once_with(cfg, df, "/data/images", True)


def test_get_grouped_dataloaders_sorts_by_area_and_uses_collate_fn():
    cfg = _base_cfg()
    small_ds = MagicMock()
    large_ds = MagicMock()
    large_ds.collate_fn = MagicMock()

    with (
        patch.object(
            module,
            "get_grouped_datasets",
            return_value={(4, 4): small_ds, (8, 8): large_ds},
        ),
        patch.object(module.torch.utils.data, "DataLoader") as mock_dataloader,
    ):
        mock_dataloader.side_effect = lambda ds, **kwargs: (ds, kwargs)

        loaders = module.get_grouped_dataloaders(cfg, MagicMock(), "/root")

    assert loaders[0][0] == (8, 8)
    assert loaders[1][0] == (4, 4)
    assert loaders[0][1][0] is large_ds
    assert loaders[0][1][1]["batch_size"] == 2
    assert loaders[0][1][1]["collate_fn"] is large_ds.collate_fn


def test_get_grouped_dataloaders_uses_val_dataset_loader_config():
    cfg = OmegaConf.create(
        {
            "val_dataset": {"data_loader": {"num_workers": 1, "prefetch_factor": 3}},
            "hyperparameters": {"batch_size": 5},
        }
    )

    with (
        patch.object(module, "get_grouped_datasets", return_value={(2, 2): object()}),
        patch.object(module.torch.utils.data, "DataLoader") as mock_dataloader,
    ):
        module.get_grouped_dataloaders(cfg, MagicMock(), "/root")

    assert mock_dataloader.call_args.kwargs["num_workers"] == 1
    assert mock_dataloader.call_args.kwargs["prefetch_factor"] == 3
    assert mock_dataloader.call_args.kwargs["batch_size"] == 5


def test_get_grouped_dataloaders_uses_default_loader_config():
    cfg = OmegaConf.create({"hyperparameters": {"batch_size": 4}})

    with (
        patch.object(module, "get_grouped_datasets", return_value={(2, 2): object()}),
        patch.object(module.torch.utils.data, "DataLoader") as mock_dataloader,
    ):
        module.get_grouped_dataloaders(cfg, MagicMock(), "/root")

    assert mock_dataloader.call_args.kwargs["num_workers"] == 4
    assert mock_dataloader.call_args.kwargs["prefetch_factor"] == 2
    assert mock_dataloader.call_args.kwargs["batch_size"] == 4


def test_get_grouped_datasets_uses_image_dataset_for_non_windowed():
    cfg = _base_cfg()
    df = pd.DataFrame({"image": ["a.tif"], "width": [8], "height": [8]})

    with patch.object(
        module.ImageDataset, "get_grouped_datasets", return_value={"plain": object()}
    ) as mock_grouped:
        result = module.get_grouped_datasets(cfg, df, "/root", windowed=False)

    assert list(result) == ["plain"]
    assert mock_grouped.call_args.kwargs["group_by_keys"] == ["width", "height"]
    assert mock_grouped.call_args.kwargs["root_dir"] == "/root"


def test_get_grouped_datasets_uses_tiled_dataset_for_windowed():
    cfg = _base_cfg(
        inference_processor={"model_input_shape": [32, 32], "step_shape": [16, 16]}
    )
    df = pd.DataFrame({"image": ["a.tif"], "width": [8], "height": [8]})

    with patch.object(
        module.TiledInferenceImageDataset,
        "get_grouped_datasets",
        return_value={"windowed": object()},
    ) as mock_grouped:
        result = module.get_grouped_datasets(cfg, df, "/root", windowed=True)

    assert list(result) == ["windowed"]
    assert mock_grouped.call_args.kwargs["model_input_shape"] == (32, 32)
    assert mock_grouped.call_args.kwargs["step_shape"] == (16, 16)


def test_get_grouped_datasets_skips_existing_polygon_files(tmp_path, monkeypatch):
    _install_swifter_accessor(monkeypatch)
    output_dir = tmp_path / "polygons"
    existing_dir = output_dir / "done"
    existing_dir.mkdir(parents=True)
    (existing_dir / "output.geojson").write_text("{}", encoding="utf-8")
    df = pd.DataFrame(
        {
            "image": ["done.tif", "missing.tif"],
            "width": [8, 8],
            "height": [8, 8],
        }
    )
    cfg = _base_cfg(
        skip_existing_polygons=True,
        skip_if_folder_or_file_created="file",
        save_not_found_image_list_to_csv=True,
        polygonizer={"data_writer": {"output_file_folder": str(output_dir)}},
    )

    with patch.object(
        module.ImageDataset, "get_grouped_datasets", return_value={"filtered": object()}
    ) as mock_grouped:
        result = module.get_grouped_datasets(cfg, df, "/root", windowed=False)

    filtered_df = mock_grouped.call_args.args[0]
    assert list(result) == ["filtered"]
    assert filtered_df["image"].tolist() == ["missing.tif"]
    assert (output_dir / "not_found_image_list.csv").exists()


def test_get_grouped_datasets_skips_existing_polygon_folders(tmp_path, monkeypatch):
    _install_swifter_accessor(monkeypatch)
    output_dir = tmp_path / "polygons"
    (output_dir / "done").mkdir(parents=True)
    df = pd.DataFrame(
        {
            "image": ["done.tif", "missing.tif"],
            "width": [8, 8],
            "height": [8, 8],
        }
    )
    cfg = _base_cfg(
        skip_existing_polygons=True,
        polygonizer={"data_writer": {"output_file_folder": str(output_dir)}},
    )

    with patch.object(
        module.ImageDataset, "get_grouped_datasets", return_value={"filtered": object()}
    ) as mock_grouped:
        module.get_grouped_datasets(cfg, df, "/root", windowed=False)

    filtered_df = mock_grouped.call_args.args[0]
    assert filtered_df["image"].tolist() == ["missing.tif"]


def test_predict_from_batch_processes_with_inference_processor():
    cfg = _base_cfg(
        inference_processor={"model_input_shape": [32, 32]},
        save_inference=False,
        inference_threshold=0.6,
    )
    df = pd.DataFrame({"image": ["relative.tif", "/abs/image.tif"]})
    processor = MagicMock()
    processor.process.side_effect = [None, RuntimeError("bad image")]

    with (
        patch(
            "pytorch_segmentation_models_trainer.predict."
            "instantiate_inference_processor",
            return_value=processor,
        ),
        patch.object(
            module, "prepare_inference_csv", return_value=("/tmp/list.csv", "/root")
        ),
        patch.object(module.pd, "read_csv", return_value=df),
        patch.object(module, "tqdm", side_effect=lambda iterable, **_kwargs: iterable),
    ):
        module.predict_from_batch(cfg)

    assert processor.process.call_args_list[0].args == ("/root/relative.tif",)
    assert processor.process.call_args_list[0].kwargs == {
        "save_inference_output": False,
        "inference_threshold": 0.6,
    }
    assert processor.process.call_args_list[1].args == ("/abs/image.tif",)


def test_predict_from_batch_legacy_predicts_all_dataloaders():
    cfg = _base_cfg()
    model = MagicMock()
    model_cls = MagicMock()
    model_cls.load_from_checkpoint.return_value = model
    trainer = MagicMock()

    with (
        patch.object(module, "import_module_from_cfg", return_value=model_cls),
        patch.object(
            module, "instantiate_dataloaders", return_value=[((8, 8), "loader")]
        ),
        patch.object(module, "Trainer", return_value=trainer) as mock_trainer,
        patch.object(module, "tqdm", side_effect=lambda iterable, **_kwargs: iterable),
    ):
        module.predict_from_batch(cfg)

    model_cls.load_from_checkpoint.assert_called_once_with(
        "/models/model.ckpt",
        cfg=cfg,
        inference_mode=True,
        weights_only=False,
        strict=False,
    )
    model.eval.assert_called_once_with()
    mock_trainer.assert_called_once_with(accelerator="cpu", callbacks=[])
    trainer.predict.assert_called_once_with(model, "loader")


def test_predict_from_batch_legacy_adds_frame_field_callback():
    class FakeFrameFieldModel:
        def eval(self):
            return None

    cfg = _base_cfg()
    model = FakeFrameFieldModel()
    model_cls = MagicMock()
    model_cls.load_from_checkpoint.return_value = model
    callback = MagicMock()

    with (
        patch.object(module, "FrameFieldSegmentationPLModel", FakeFrameFieldModel),
        patch.object(
            module, "ActiveSkeletonsPolygonizerCallback", return_value=callback
        ),
        patch.object(module, "import_module_from_cfg", return_value=model_cls),
        patch.object(module, "instantiate_dataloaders", return_value=[]),
        patch.object(module, "Trainer") as mock_trainer,
    ):
        module.predict_from_batch(cfg)

    mock_trainer.assert_called_once_with(accelerator="cpu", callbacks=[callback])


def test_predict_from_batch_legacy_continues_when_trainer_fails():
    cfg = _base_cfg()
    model = MagicMock()
    model_cls = MagicMock()
    model_cls.load_from_checkpoint.return_value = model
    trainer = MagicMock()
    trainer.predict.side_effect = RuntimeError("predict failed")

    with (
        patch.object(module, "import_module_from_cfg", return_value=model_cls),
        patch.object(
            module, "instantiate_dataloaders", return_value=[((8, 8), "loader")]
        ),
        patch.object(module, "Trainer", return_value=trainer),
        patch.object(module, "tqdm", side_effect=lambda iterable, **_kwargs: iterable),
        patch.object(module.logger, "exception") as mock_exception,
    ):
        module.predict_from_batch(cfg)

    assert trainer.predict.call_count == 1
    assert mock_exception.call_count == 2


def test_module_main_guard_runs_predict_from_batch(monkeypatch):
    calls = []

    def fake_hydra_main(**_kwargs):
        def decorator(func):
            def wrapper():
                calls.append("predict")
                return None

            return wrapper

        return decorator

    fake_hydra = types.SimpleNamespace(main=fake_hydra_main)
    fake_predict_module = types.ModuleType(
        "pytorch_segmentation_models_trainer.predict"
    )
    fake_processor = MagicMock()
    fake_predict_module.instantiate_inference_processor = MagicMock(
        return_value=fake_processor
    )

    monkeypatch.setitem(sys.modules, "hydra", fake_hydra)
    monkeypatch.setitem(
        sys.modules, "pytorch_segmentation_models_trainer.predict", fake_predict_module
    )
    monkeypatch.delitem(
        sys.modules,
        "pytorch_segmentation_models_trainer.predict_from_batch",
        raising=False,
    )

    runpy.run_module(
        "pytorch_segmentation_models_trainer.predict_from_batch",
        run_name="__main__",
    )

    assert calls == ["predict"]
