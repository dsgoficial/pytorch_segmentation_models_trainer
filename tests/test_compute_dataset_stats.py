# -*- coding: utf-8 -*-
import math
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from ruamel.yaml import YAML

from pytorch_segmentation_models_trainer.tools.compute_dataset_stats import (
    _is_image_callback,
    _make_normalize_entry,
    _make_to_tensor_entry,
    _strip_hydra_refs,
    compute_stats,
    update_callbacks,
    update_dataset_augmentation_list,
    update_normalization_parameters,
    process_yaml,
)
from tests.utils import BasicTestCase

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


class _ConstantDataset(torch.utils.data.Dataset):
    """Returns images where channel c has constant value ``c * 0.3 + 0.2``."""

    def __init__(self, n=8, C=3, H=4, W=4):
        self.n = n
        self.C = C
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.zeros(self.C, self.H, self.W)
        for c in range(self.C):
            image[c] = c * 0.3 + 0.2
        return {"image": image, "path": "fake"}


class _BinaryDataset(torch.utils.data.Dataset):
    """Half pixels = 0.3, half = 0.7 per channel → mean=0.5, std=0.2."""

    def __init__(self, n=8, C=3, H=4, W=4):
        self.n = n
        self.C = C
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.zeros(self.C, self.H, self.W)
        for c in range(self.C):
            image[c, : self.H // 2, :] = 0.3
            image[c, self.H // 2 :, :] = 0.7
        return {"image": image, "path": "fake"}


class _EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError


# ---------------------------------------------------------------------------
# _strip_hydra_refs
# ---------------------------------------------------------------------------


class TestStripHydraRefs(unittest.TestCase):
    def test_removes_hydra_ref_values(self):
        d = {"a": "normal", "b": "${some.ref}", "c": 42}
        result = _strip_hydra_refs(d)
        assert "b" not in result
        assert result["a"] == "normal"
        assert result["c"] == 42

    def test_nested_dict(self):
        d = {"outer": {"inner": "${ref}", "keep": 1}}
        result = _strip_hydra_refs(d)
        assert "inner" not in result["outer"]
        assert result["outer"]["keep"] == 1

    def test_list_with_ref(self):
        d = {"lst": ["ok", "${bad}", "also_ok"]}
        result = _strip_hydra_refs(d)
        assert result["lst"] == ["ok", "also_ok"]

    def test_no_refs_unchanged(self):
        d = {"x": 1, "y": "hello", "z": [1, 2, 3]}
        assert _strip_hydra_refs(d) == d

    def test_list_of_dicts_nested(self):
        d = {"items": [{"a": "${ref}", "b": "keep"}]}
        result = _strip_hydra_refs(d)
        assert "a" not in result["items"][0]
        assert result["items"][0]["b"] == "keep"


# ---------------------------------------------------------------------------
# _is_image_callback
# ---------------------------------------------------------------------------


class TestIsImageCallback(unittest.TestCase):
    def test_known_image_callbacks(self):
        targets = [
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback",
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.ImageSegmentationResultCallback",
            "pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks.EvidentialUncertaintyVisualizationCallback",
        ]
        for t in targets:
            assert _is_image_callback(t), f"Expected True for {t}"

    def test_non_image_callbacks(self):
        targets = [
            "pytorch_lightning.callbacks.ModelCheckpoint",
            "pytorch_lightning.callbacks.EarlyStopping",
            "pytorch_lightning.callbacks.LearningRateMonitor",
        ]
        for t in targets:
            assert not _is_image_callback(t), f"Expected False for {t}"


# ---------------------------------------------------------------------------
# _make_normalize_entry / _make_to_tensor_entry
# ---------------------------------------------------------------------------


class TestEntryFactories(unittest.TestCase):
    def test_make_normalize_entry(self):
        entry = _make_normalize_entry([0.5, 0.4, 0.3], [0.1, 0.2, 0.3], 255.0)
        assert entry["_target_"] == "albumentations.Normalize"
        assert entry["mean"] == [0.5, 0.4, 0.3]
        assert entry["std"] == [0.1, 0.2, 0.3]
        assert entry["max_pixel_value"] == 255.0

    def test_make_to_tensor_entry(self):
        entry = _make_to_tensor_entry()
        assert entry["_target_"] == "albumentations.pytorch.ToTensorV2"


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats(BasicTestCase):
    def test_constant_channels_mean_correct(self):
        ds = _ConstantDataset(n=8, C=3)
        mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=False)
        for c in range(3):
            expected = c * 0.3 + 0.2
            assert abs(mean[c] - expected) < 1e-4, f"ch{c}: {mean[c]} != {expected}"

    def test_constant_channels_std_zero(self):
        ds = _ConstantDataset(n=8, C=3)
        _, std = compute_stats(ds, batch_size=4, num_workers=0, progress=False)
        for v in std:
            assert v < 1e-5, f"Expected std≈0, got {v}"

    def test_binary_dataset_mean_and_std(self):
        ds = _BinaryDataset(n=8, C=3)
        mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=False)
        for c in range(3):
            assert abs(mean[c] - 0.5) < 1e-4, f"ch{c} mean {mean[c]}"
            assert abs(std[c] - 0.2) < 1e-4, f"ch{c} std {std[c]}"

    def test_empty_dataset_raises(self):
        ds = _EmptyDataset()
        with self.assertRaises(RuntimeError):
            compute_stats(ds, batch_size=4, num_workers=0, progress=False)

    def test_returns_rounded_floats(self):
        ds = _ConstantDataset(n=4, C=2)
        mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=False)
        for v in mean + std:
            assert isinstance(v, float)
            assert len(str(v).split(".")[-1]) <= 6

    def test_single_sample(self):
        ds = _ConstantDataset(n=1, C=3)
        mean, std = compute_stats(ds, batch_size=1, num_workers=0, progress=False)
        assert len(mean) == 3
        assert len(std) == 3

    def test_progress_false_no_error(self):
        ds = _ConstantDataset(n=4, C=3)
        mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=False)
        assert len(mean) == 3


# ---------------------------------------------------------------------------
# update_dataset_augmentation_list
# ---------------------------------------------------------------------------


class TestUpdateDatasetAugmentationList(unittest.TestCase):
    def _mean_std(self):
        return [0.4, 0.5, 0.6], [0.1, 0.2, 0.1]

    def test_no_augmentation_list_creates_new(self):
        cfg = {"image_dtype": "uint8"}
        mean, std = self._mean_std()
        update_dataset_augmentation_list(cfg, mean, std, 255.0)
        aug = cfg["augmentation_list"]
        assert len(aug) == 2
        assert aug[0]["_target_"] == "albumentations.Normalize"
        assert aug[0]["mean"] == mean
        assert aug[0]["std"] == std
        assert aug[0]["max_pixel_value"] == 255.0
        assert aug[1]["_target_"] == "albumentations.pytorch.ToTensorV2"

    def test_existing_normalize_is_updated(self):
        cfg = {
            "augmentation_list": [
                {
                    "_target_": "albumentations.Normalize",
                    "mean": [0.0],
                    "std": [1.0],
                    "max_pixel_value": 255.0,
                },
                {"_target_": "albumentations.pytorch.ToTensorV2"},
            ]
        }
        mean, std = self._mean_std()
        update_dataset_augmentation_list(cfg, mean, std, 255.0)
        aug = cfg["augmentation_list"]
        assert len(aug) == 2
        assert aug[0]["mean"] == mean
        assert aug[0]["std"] == std

    def test_no_normalize_prepends_it(self):
        cfg = {
            "augmentation_list": [
                {"_target_": "albumentations.HorizontalFlip"},
                {"_target_": "albumentations.pytorch.ToTensorV2"},
            ]
        }
        mean, std = self._mean_std()
        update_dataset_augmentation_list(cfg, mean, std, 255.0)
        aug = cfg["augmentation_list"]
        assert aug[0]["_target_"] == "albumentations.Normalize"
        assert len(aug) == 3

    def test_no_to_tensor_appends_it(self):
        cfg = {
            "augmentation_list": [
                {
                    "_target_": "albumentations.Normalize",
                    "mean": [0.0],
                    "std": [1.0],
                    "max_pixel_value": 255.0,
                },
            ]
        }
        update_dataset_augmentation_list(cfg, *self._mean_std(), 255.0)
        targets = [e["_target_"] for e in cfg["augmentation_list"]]
        assert "albumentations.pytorch.ToTensorV2" in targets

    def test_max_pixel_value_propagated(self):
        cfg = {}
        update_dataset_augmentation_list(cfg, [0.5], [0.1], 65535.0)
        assert cfg["augmentation_list"][0]["max_pixel_value"] == 65535.0


# ---------------------------------------------------------------------------
# update_callbacks
# ---------------------------------------------------------------------------


class TestUpdateCallbacks(unittest.TestCase):
    def _yaml_with_callbacks(self, *targets):
        return {"callbacks": [{"_target_": t, "n_samples": 4} for t in targets]}

    def test_image_callbacks_updated(self):
        target = "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback"
        yaml_data = self._yaml_with_callbacks(target)
        mean, std = [0.4, 0.5, 0.6], [0.1, 0.2, 0.1]
        n = update_callbacks(yaml_data, mean, std)
        assert n == 1
        cb = yaml_data["callbacks"][0]
        assert cb["normalized_input"] is True
        assert cb["norm_params"]["mean"] == mean
        assert cb["norm_params"]["std"] == std

    def test_image_callbacks_updated_with_interpolation(self):
        target = "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback"
        yaml_data = self._yaml_with_callbacks(target)
        mean_ref = "${normalization_parameters.mean}"
        std_ref = "${normalization_parameters.std}"
        n = update_callbacks(yaml_data, mean_ref, std_ref)
        assert n == 1
        cb = yaml_data["callbacks"][0]
        assert cb["norm_params"]["mean"] == mean_ref
        assert cb["norm_params"]["std"] == std_ref

    def test_non_image_callbacks_skipped(self):
        yaml_data = self._yaml_with_callbacks(
            "pytorch_lightning.callbacks.ModelCheckpoint",
            "pytorch_lightning.callbacks.EarlyStopping",
        )
        n = update_callbacks(yaml_data, [0.5], [0.1])
        assert n == 0
        for cb in yaml_data["callbacks"]:
            assert "norm_params" not in cb

    def test_no_callbacks_key(self):
        n = update_callbacks({}, [0.5], [0.1])
        assert n == 0

    def test_mixed_callbacks_only_updates_image(self):
        yaml_data = {
            "callbacks": [
                {"_target_": "pytorch_lightning.callbacks.ModelCheckpoint"},
                {
                    "_target_": "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback"
                },
            ]
        }
        n = update_callbacks(yaml_data, [0.5], [0.1])
        assert n == 1
        assert "norm_params" not in yaml_data["callbacks"][0]
        assert "norm_params" in yaml_data["callbacks"][1]

    def test_multiple_image_callbacks(self):
        yaml_data = {
            "callbacks": [
                {
                    "_target_": "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback"
                },
                {
                    "_target_": "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.ImageSegmentationResultCallback"
                },
            ]
        }
        n = update_callbacks(yaml_data, [0.5], [0.1])
        assert n == 2


# ---------------------------------------------------------------------------
# update_normalization_parameters
# ---------------------------------------------------------------------------


class TestUpdateNormalizationParameters(unittest.TestCase):
    def test_writes_top_level_key(self):
        yaml_data = {}
        update_normalization_parameters(yaml_data, [0.4, 0.5, 0.6], [0.1, 0.2, 0.1])
        assert "normalization_parameters" in yaml_data
        assert yaml_data["normalization_parameters"]["mean"] == [0.4, 0.5, 0.6]
        assert yaml_data["normalization_parameters"]["std"] == [0.1, 0.2, 0.1]

    def test_overwrites_existing_key(self):
        yaml_data = {"normalization_parameters": {"mean": [0.0], "std": [1.0]}}
        update_normalization_parameters(yaml_data, [0.5, 0.5], [0.2, 0.2])
        assert yaml_data["normalization_parameters"]["mean"] == [0.5, 0.5]
        assert yaml_data["normalization_parameters"]["std"] == [0.2, 0.2]

    def test_preserves_other_keys(self):
        yaml_data = {"hyperparameters": {"batch_size": 64}}
        update_normalization_parameters(yaml_data, [0.5], [0.1])
        assert yaml_data["hyperparameters"]["batch_size"] == 64

    def test_single_channel(self):
        yaml_data = {}
        update_normalization_parameters(yaml_data, [0.5], [0.1])
        assert len(yaml_data["normalization_parameters"]["mean"]) == 1
        assert len(yaml_data["normalization_parameters"]["std"]) == 1


# ---------------------------------------------------------------------------
# process_yaml (integration, uses temp file + mock)
# ---------------------------------------------------------------------------

_SAMPLE_YAML = textwrap.dedent("""\
    hyperparameters:
      batch_size: 64

    train_dataset:
      _target_: fake.module.FakeDataset
      image_dir: /fake/train
      image_dtype: uint8
      data_loader:
        shuffle: true
        batch_size: "${hyperparameters.batch_size}"

    val_dataset:
      _target_: fake.module.FakeDataset
      image_dir: /fake/val
      image_dtype: uint8

    callbacks:
      - _target_: pytorch_lightning.callbacks.ModelCheckpoint
        monitor: val/loss
      - _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback
        n_samples: 8
""")


def _load_yaml(path: Path) -> dict:
    ryaml = YAML()
    with open(path) as fh:
        return ryaml.load(fh)


class TestProcessYaml(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.yaml_path = self.tmp / "config.yaml"
        self.yaml_path.write_text(_SAMPLE_YAML)
        self.fake_mean = [0.4, 0.5, 0.6]
        self.fake_std = [0.1, 0.2, 0.1]

    def _mock_dataset(self):
        ds = _ConstantDataset(n=4, C=3)
        return ds

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_dry_run_does_not_modify_file(self, mock_inst):
        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        original = self.yaml_path.read_text()
        process_yaml(self.yaml_path, dry_run=True, num_workers=0, progress=False)
        assert self.yaml_path.read_text() == original

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_saves_normalize_in_all_dataset_keys(self, mock_inst):
        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        process_yaml(self.yaml_path, num_workers=0, progress=False)
        data = _load_yaml(self.yaml_path)
        assert "normalization_parameters" in data
        for key in ("train_dataset", "val_dataset"):
            aug = data[key]["augmentation_list"]
            targets = [e["_target_"] for e in aug]
            assert "albumentations.Normalize" in targets, f"{key} missing Normalize"
            assert (
                "albumentations.pytorch.ToTensorV2" in targets
            ), f"{key} missing ToTensorV2"

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_stats_values_written_correctly(self, mock_inst):
        mock_inst.return_value = _BinaryDataset(n=8, C=3)
        process_yaml(self.yaml_path, num_workers=0, progress=False)
        data = _load_yaml(self.yaml_path)
        # Values stored in normalization_parameters
        norm_params = data["normalization_parameters"]
        for v in norm_params["mean"]:
            assert abs(v - 0.5) < 1e-3
        for v in norm_params["std"]:
            assert abs(v - 0.2) < 1e-3
        # Normalize entry uses interpolation references
        aug = data["train_dataset"]["augmentation_list"]
        norm = next(e for e in aug if "Normalize" in e["_target_"])
        assert norm["mean"] == "${normalization_parameters.mean}"
        assert norm["std"] == "${normalization_parameters.std}"

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_callback_norm_params_updated(self, mock_inst):
        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        process_yaml(self.yaml_path, num_workers=0, progress=False)
        data = _load_yaml(self.yaml_path)
        autoencoder_cb = next(
            cb
            for cb in data["callbacks"]
            if "AutoencoderResultCallback" in cb["_target_"]
        )
        assert autoencoder_cb["normalized_input"] is True
        assert "norm_params" in autoencoder_cb
        assert (
            autoencoder_cb["norm_params"]["mean"] == "${normalization_parameters.mean}"
        )
        assert autoencoder_cb["norm_params"]["std"] == "${normalization_parameters.std}"

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_skip_callbacks_leaves_callbacks_unchanged(self, mock_inst):
        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        process_yaml(self.yaml_path, skip_callbacks=True, num_workers=0, progress=False)
        data = _load_yaml(self.yaml_path)
        for cb in data["callbacks"]:
            assert "norm_params" not in cb

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_missing_dataset_key_raises(self, mock_inst):
        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        with self.assertRaises(KeyError):
            process_yaml(
                self.yaml_path,
                dataset_key="nonexistent_key",
                num_workers=0,
                progress=False,
            )

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_returns_mean_and_std(self, mock_inst):
        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        mean, std = process_yaml(self.yaml_path, num_workers=0, progress=False)
        assert len(mean) == 3
        assert len(std) == 3

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_max_pixel_value_uint8(self, mock_inst):
        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        process_yaml(self.yaml_path, num_workers=0, progress=False)
        data = _load_yaml(self.yaml_path)
        aug = data["train_dataset"]["augmentation_list"]
        norm = next(e for e in aug if "Normalize" in e["_target_"])
        assert norm["max_pixel_value"] == 255.0


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLI(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.yaml_path = self.tmp / "config.yaml"
        self.yaml_path.write_text(_SAMPLE_YAML)

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_cli_compute_stats_dry_run(self, mock_inst):
        from click.testing import CliRunner
        from pytorch_segmentation_models_trainer.tools.cli import cli

        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        runner = CliRunner()
        original = self.yaml_path.read_text()
        result = runner.invoke(
            cli,
            ["compute-stats", str(self.yaml_path), "--dry-run", "--num-workers", "0"],
        )
        assert result.exit_code == 0, result.output
        assert self.yaml_path.read_text() == original

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_cli_compute_stats_saves(self, mock_inst):
        from click.testing import CliRunner
        from pytorch_segmentation_models_trainer.tools.cli import cli

        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["compute-stats", str(self.yaml_path), "--num-workers", "0"],
        )
        assert result.exit_code == 0, result.output
        data = _load_yaml(self.yaml_path)
        assert "augmentation_list" in data["train_dataset"]
        assert "normalization_parameters" in data


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


class _NumpyDataset(torch.utils.data.Dataset):
    """Returns numpy arrays instead of tensors (exercises the numpy fallback path)."""

    def __init__(self, n=4, C=3, H=4, W=4):
        self.n = n
        self.C = C
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import numpy as np

        image = np.full((self.C, self.H, self.W), 0.5, dtype=np.float32)
        return {"image": image, "path": "fake"}


class _2DDataset(torch.utils.data.Dataset):
    """Returns (H, W) tensors (no channel dim) — exercises the ndim==3 unsqueeze path."""

    def __init__(self, n=4, H=4, W=4):
        self.n = n
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"image": torch.full((self.H, self.W), 0.5), "path": "fake"}


class TestCoverageGaps(BasicTestCase):
    def test_import_class_torch(self):
        from pytorch_segmentation_models_trainer.tools.compute_dataset_stats import (
            _import_class,
        )

        cls = _import_class("torch.utils.data.Dataset")
        assert cls is torch.utils.data.Dataset

    def test_instantiate_dataset_for_stats_calls_class(self):
        from pytorch_segmentation_models_trainer.tools.compute_dataset_stats import (
            instantiate_dataset_for_stats,
        )

        sentinel = object()

        with patch(
            "pytorch_segmentation_models_trainer.tools.compute_dataset_stats._import_class"
        ) as mock_import:
            mock_cls = MagicMock(return_value=sentinel)
            mock_import.return_value = mock_cls
            result = instantiate_dataset_for_stats(
                {
                    "_target_": "fake.module.MyDataset",
                    "image_dir": "/fake",
                    "data_loader": {"batch_size": 64},
                    "augmentation_list": [{"_target_": "albumentations.Normalize"}],
                    "corruption_augmentation_list": [],
                    "stride": "${some.ref}",
                }
            )
        assert result is sentinel
        mock_import.assert_called_once_with("fake.module.MyDataset")
        call_kwargs = mock_cls.call_args[1]
        assert "data_loader" not in call_kwargs
        assert "augmentation_list" not in call_kwargs
        assert "corruption_augmentation_list" not in call_kwargs
        assert "stride" not in call_kwargs

    def test_compute_stats_numpy_images(self):
        # DataLoader.default_collate converts numpy arrays to tensors automatically.
        ds = _NumpyDataset(n=4, C=3)
        mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=False)
        assert len(mean) == 3
        for v in mean:
            assert abs(v - 0.5) < 1e-4

    def test_compute_stats_2d_images(self):
        ds = _2DDataset(n=4, H=4, W=4)
        mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=False)
        # (B=4, H=4, W=4) batch gets unsqueeze(0) → (1,4,4,4): C==H==4
        assert len(mean) == 4
        assert len(std) == 4

    def test_compute_stats_with_progress_true(self):
        ds = _ConstantDataset(n=4, C=3)
        mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=True)
        assert len(mean) == 3

    def test_compute_stats_tqdm_import_error(self):
        import sys

        ds = _ConstantDataset(n=4, C=3)
        real_tqdm = sys.modules.get("tqdm")
        try:
            sys.modules["tqdm"] = None  # type: ignore[assignment]
            mean, std = compute_stats(ds, batch_size=4, num_workers=0, progress=True)
            assert len(mean) == 3
        finally:
            if real_tqdm is None:
                sys.modules.pop("tqdm", None)
            else:
                sys.modules["tqdm"] = real_tqdm

    def test_update_callbacks_non_dict_item_skipped(self):
        yaml_data = {
            "callbacks": [
                "not_a_dict",
                {"_target_": "pytorch_lightning.callbacks.ModelCheckpoint"},
            ]
        }
        n = update_callbacks(yaml_data, [0.5], [0.1])
        assert n == 0

    def test_cli_entry_callable(self):
        from pytorch_segmentation_models_trainer.tools.cli import entry
        from click.testing import CliRunner

        runner = CliRunner()
        with patch("pytorch_segmentation_models_trainer.tools.cli.cli") as mock_cli:
            entry()
            mock_cli.assert_called_once()

    @patch(
        "pytorch_segmentation_models_trainer.tools.compute_dataset_stats.instantiate_dataset_for_stats"
    )
    def test_process_yaml_prints_saved_message(self, mock_inst):
        import io, sys

        mock_inst.return_value = _ConstantDataset(n=4, C=3)
        tmp = Path(self.make_temp_dir())
        yaml_path = tmp / "config.yaml"
        yaml_path.write_text(_SAMPLE_YAML)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            process_yaml(yaml_path, num_workers=0, progress=True)
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "Saved" in output


if __name__ == "__main__":
    unittest.main()
