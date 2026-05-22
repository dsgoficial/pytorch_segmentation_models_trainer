# -*- coding: utf-8 -*-
import runpy
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer import (
    predict_mod_polymapper_from_batch as module,
)


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


def _cfg(**overrides):
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


def test_instantiate_dataloaders_reads_limited_rows():
    cfg = _cfg(
        inference_dataset={
            "input_csv_path": "/data/inference.csv",
            "root_dir": "/root",
            "n_first_rows_to_read": 2,
            "data_loader": {"num_workers": 1, "prefetch_factor": 3},
        }
    )
    df = pd.DataFrame({"image": ["a.tif"], "width": [8], "height": [8]})
    ds = object()

    with (
        patch.object(module.pd, "read_csv", return_value=df) as mock_read_csv,
        patch.object(
            module.ImageDataset, "get_grouped_datasets", return_value={(8, 8): ds}
        ) as mock_grouped,
        patch.object(
            module.torch.utils.data,
            "DataLoader",
            side_effect=lambda dataset, **kwargs: (dataset, kwargs),
        ) as mock_dataloader,
    ):
        loaders = module.instantiate_dataloaders(cfg)

    mock_read_csv.assert_called_once_with("/data/inference.csv", nrows=2)
    assert mock_grouped.call_args.kwargs["root_dir"] == "/root"
    assert loaders == [(ds, mock_dataloader.call_args.kwargs)]
    assert loaders[0][1]["batch_size"] == 2
    assert loaders[0][1]["num_workers"] == 1
    assert loaders[0][1]["prefetch_factor"] == 3


def test_instantiate_dataloaders_reads_all_rows_when_limit_missing():
    cfg = _cfg()

    with (
        patch.object(
            module.pd, "read_csv", return_value=pd.DataFrame()
        ) as mock_read_csv,
        patch.object(module.ImageDataset, "get_grouped_datasets", return_value={}),
    ):
        loaders = module.instantiate_dataloaders(cfg)

    assert loaders == []
    mock_read_csv.assert_called_once_with("/data/inference.csv")


def test_instantiate_dataloaders_skips_existing_polygon_folders(tmp_path, monkeypatch):
    _install_swifter_accessor(monkeypatch)
    output_dir = tmp_path / "polygons"
    (output_dir / "done").mkdir(parents=True)
    cfg = _cfg(
        skip_existing_polygons=True,
        polygonizer={"data_writer": {"output_file_folder": str(output_dir)}},
    )
    df = pd.DataFrame(
        {
            "image": ["done.tif", "missing.tif"],
            "width": [8, 8],
            "height": [8, 8],
        }
    )

    with (
        patch.object(module.pd, "read_csv", return_value=df),
        patch.object(
            module.ImageDataset, "get_grouped_datasets", return_value={}
        ) as mock_grouped,
    ):
        module.instantiate_dataloaders(cfg)

    filtered_df = mock_grouped.call_args.args[0]
    assert filtered_df["image"].tolist() == ["missing.tif"]


def test_instantiate_dataloaders_skips_existing_polygon_files(tmp_path, monkeypatch):
    _install_swifter_accessor(monkeypatch)
    output_dir = tmp_path / "polygons"
    existing_dir = output_dir / "done"
    existing_dir.mkdir(parents=True)
    (existing_dir / "output.geojson").write_text("{}", encoding="utf-8")
    cfg = _cfg(
        skip_existing_polygons=True,
        skip_if_folder_or_file_created="file",
        save_not_found_image_list_to_csv=True,
        polygonizer={"data_writer": {"output_file_folder": str(output_dir)}},
    )
    df = pd.DataFrame(
        {
            "image": ["done.tif", "missing.tif"],
            "width": [8, 8],
            "height": [8, 8],
        }
    )

    with (
        patch.object(module.pd, "read_csv", return_value=df),
        patch.object(
            module.ImageDataset, "get_grouped_datasets", return_value={}
        ) as mock_grouped,
    ):
        module.instantiate_dataloaders(cfg)

    filtered_df = mock_grouped.call_args.args[0]
    assert filtered_df["image"].tolist() == ["missing.tif"]
    assert (output_dir / "not_found_image_list.csv").exists()


def test_predict_mod_polymapper_from_batch_predicts_each_loader():
    cfg = _cfg(convert_output_to_world_coords=True)
    model = MagicMock()
    model_cls = MagicMock()
    model_cls.load_from_checkpoint.return_value = model
    trainer = MagicMock()
    callback = MagicMock()

    with (
        patch.object(module, "import_module_from_cfg", return_value=model_cls),
        patch.object(
            module, "instantiate_dataloaders", return_value=["loader-a", "loader-b"]
        ),
        patch.object(module, "Trainer", return_value=trainer) as mock_trainer,
        patch.object(
            module,
            "ModPolymapperPolygonizerCallback",
            return_value=callback,
        ) as mock_callback,
        patch.object(module, "tqdm", side_effect=lambda iterable, **_kwargs: iterable),
    ):
        module.predict_mod_polymapper_from_batch(cfg)

    model_cls.load_from_checkpoint.assert_called_once_with(
        "/models/model.ckpt", cfg=cfg
    )
    mock_callback.assert_called_once_with(convert_output_to_world_coords=True)
    mock_trainer.assert_called_once_with(accelerator="cpu", callbacks=[callback])
    model.model.eval.assert_called_once_with()
    assert trainer.predict.call_args_list[0].args == (model, "loader-a")
    assert trainer.predict.call_args_list[1].args == (model, "loader-b")


def test_predict_mod_polymapper_defaults_world_coords_to_false():
    cfg = _cfg()
    model = MagicMock()
    model_cls = MagicMock()
    model_cls.load_from_checkpoint.return_value = model

    with (
        patch.object(module, "import_module_from_cfg", return_value=model_cls),
        patch.object(module, "instantiate_dataloaders", return_value=[]),
        patch.object(module, "Trainer"),
        patch.object(module, "ModPolymapperPolygonizerCallback") as mock_callback,
    ):
        module.predict_mod_polymapper_from_batch(cfg)

    mock_callback.assert_called_once_with(convert_output_to_world_coords=False)


def test_module_main_guard_runs_predict_mod_polymapper(monkeypatch):
    calls = []

    def fake_hydra_main(**_kwargs):
        def decorator(func):
            def wrapper():
                calls.append("predict")
                return None

            return wrapper

        return decorator

    fake_hydra = types.SimpleNamespace(main=fake_hydra_main)
    monkeypatch.setitem(sys.modules, "hydra", fake_hydra)
    monkeypatch.delitem(
        sys.modules,
        "pytorch_segmentation_models_trainer.predict_mod_polymapper_from_batch",
        raising=False,
    )

    runpy.run_module(
        "pytorch_segmentation_models_trainer.predict_mod_polymapper_from_batch",
        run_name="__main__",
    )

    assert calls == ["predict"]
