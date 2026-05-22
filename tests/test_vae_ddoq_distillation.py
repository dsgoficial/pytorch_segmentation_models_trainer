# -*- coding: utf-8 -*-
"""Tests for VAE-backed DDOQ image distillation."""

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from click.testing import CliRunner
from torch.utils.data import DataLoader, Dataset

from pytorch_segmentation_models_trainer.tools.dataset_distillation import (
    vae_ddoq_distillation as ddoq_module,
)

vae_ddoq_distillation = ddoq_module
VaeDdoqDistillationPipeline = ddoq_module.VaeDdoqDistillationPipeline
VaeDdoqDistillationResult = ddoq_module.VaeDdoqDistillationResult
_flatten_embedding = ddoq_module._flatten_embedding
_resolve_output_format = ddoq_module._resolve_output_format
write_distilled_images_parquet = ddoq_module.write_distilled_images_parquet
write_embeddings_parquet = ddoq_module.write_embeddings_parquet


class TinyPathImageDataset(Dataset):
    """Small deterministic image dataset exposing file paths in each sample."""

    def __init__(self, root: Path, n_samples: int = 6):
        self.paths = [
            str((root / f"image_{i}.png").resolve()) for i in range(n_samples)
        ]
        self.images = torch.linspace(0.0, 1.0, n_samples * 3 * 8 * 8).reshape(
            n_samples, 3, 8, 8
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "path": self.paths[idx]}


class TinyVAE(nn.Module):
    """VAE-like module with deterministic posterior means and decoder."""

    def encode(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        mu = torch.cat([mean[:, :1], mean[:, 1:2]], dim=1)
        logvar = torch.zeros_like(mu)
        return mu, logvar

    def decode(self, z, output_size):
        if z.ndim == 2:
            z = z[:, :, None, None]
        if z.shape[1] < 3:
            pad = torch.zeros(
                z.shape[0],
                3 - z.shape[1],
                z.shape[2],
                z.shape[3],
                device=z.device,
                dtype=z.dtype,
            )
            z = torch.cat([z, pad], dim=1)
        return torch.nn.functional.interpolate(
            z[:, :3], size=output_size, mode="nearest"
        )


def test_flatten_embedding_contract():
    emb2d = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert _flatten_embedding(emb2d).shape == (2, 3)

    emb = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    flat = _flatten_embedding(emb)
    assert flat.shape == (2, 48)
    assert flat.dtype == torch.float32

    mean = _flatten_embedding(emb, reduction="mean_spatial")
    assert mean.shape == (2, 3)
    seq_mean = _flatten_embedding(torch.ones(2, 4, 3), reduction="mean_spatial")
    assert seq_mean.shape == (2, 3)

    with pytest.raises(ValueError, match="latent_reduction"):
        _flatten_embedding(emb, reduction="bad")
    with pytest.raises(ValueError, match="Unsupported embedding shape"):
        _flatten_embedding(torch.ones(2), reduction="mean_spatial")


def test_batch_helpers_and_image_format_resolution(tmp_path):
    assert ddoq_module._extract_batch_images({"x": torch.ones(1, 3, 2, 2)}).shape == (
        1,
        3,
        2,
        2,
    )
    assert ddoq_module._extract_batch_images(
        (torch.ones(1, 3, 2, 2), "label")
    ).shape == (
        1,
        3,
        2,
        2,
    )
    assert ddoq_module._extract_batch_images(torch.ones(1, 3, 2, 2)).shape == (
        1,
        3,
        2,
        2,
    )
    with pytest.raises(ValueError, match="Could not extract"):
        ddoq_module._extract_batch_images({"image": None})

    generated = ddoq_module._extract_batch_paths({}, batch_size=2, start_idx=3)
    assert generated[0].endswith("sample_3")
    assert ddoq_module._extract_batch_paths({"path": "one.png"}, 1, 0)[0].endswith(
        "one.png"
    )
    assert (
        ddoq_module._resolve_output_format(".jpeg", [str(tmp_path / "sample.tif")])
        == "jpg"
    )


def test_resolve_output_format_auto_from_input_path(tmp_path):
    fmt = _resolve_output_format("auto", [str(tmp_path / "sample.tif")])
    assert fmt == "tif"

    with pytest.raises(ValueError, match="Cannot infer"):
        _resolve_output_format("auto", [])

    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_output_format("bmp", [str(tmp_path / "sample.tif")])


def test_decoded_image_savers_cover_tensor_formats(tmp_path):
    with pytest.raises(ValueError, match="Expected image tensor"):
        ddoq_module._tensor_to_uint8_image(torch.ones(3, 4))

    pt_path = tmp_path / "decoded.pt"
    ddoq_module._save_decoded_image(torch.ones(3, 4, 4), pt_path, "pt")
    assert pt_path.exists()

    gray_path = tmp_path / "gray.png"
    ddoq_module._save_decoded_image(torch.ones(1, 4, 4), gray_path, "png")
    assert gray_path.exists()

    rgba_path = tmp_path / "rgba.png"
    ddoq_module._save_decoded_image(torch.ones(4, 4, 4), rgba_path, "png")
    assert rgba_path.exists()

    with pytest.raises(ValueError, match="Cannot save 5-channel"):
        ddoq_module._save_decoded_image(
            torch.ones(5, 4, 4), tmp_path / "bad.png", "png"
        )
    with pytest.raises(Exception):
        ddoq_module._save_decoded_image(
            torch.ones(5, 4, 4), tmp_path / "bad.tif", "tif"
        )


def test_parquet_writers_preserve_contract(tmp_path):
    embeddings_path = tmp_path / "embeddings.parquet"
    distilled_path = tmp_path / "distilled_images.parquet"

    write_embeddings_parquet(
        image_paths=["/tmp/a.png", "/tmp/b.png"],
        embeddings=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        cluster_ids=torch.tensor([0, 1]),
        output_path=embeddings_path,
    )
    write_distilled_images_parquet(
        distilled_image_paths=["/tmp/d0.png", "/tmp/d1.png"],
        cluster_ids=torch.tensor([0, 1]),
        cluster_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        weights=torch.tensor([0.5, 0.7]),
        output_path=distilled_path,
    )

    embeddings_df = pd.read_parquet(embeddings_path)
    assert list(embeddings_df.columns) == ["image_path", "embedding", "cluster_id"]
    assert embeddings_df["embedding"].iloc[0].tolist() == [1.0, 2.0]
    assert embeddings_df["cluster_id"].tolist() == [0, 1]

    distilled_df = pd.read_parquet(distilled_path)
    assert list(distilled_df.columns) == [
        "distilled_image_path",
        "cluster_id",
        "cluster_embedding",
        "weight",
    ]
    assert distilled_df["cluster_embedding"].iloc[1].tolist() == pytest.approx(
        [0.3, 0.4]
    )
    assert distilled_df["weight"].tolist() == pytest.approx([0.5, 0.7])


def test_pipeline_writes_k_distilled_images_and_all_embeddings(tmp_path):
    dataset = TinyPathImageDataset(tmp_path, n_samples=6)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    pipeline = VaeDdoqDistillationPipeline(
        vae=TinyVAE(),
        dataloader=dataloader,
        output_dir=tmp_path / "ddoq",
        k=2,
        device="cpu",
        output_size=(8, 8),
        distilled_image_format="png",
        kmeans_max_iter=4,
        kmeans_batch_size=4,
        random_seed=123,
    )

    result = pipeline.run()

    embeddings_df = pd.read_parquet(result.embeddings_parquet_path)
    distilled_df = pd.read_parquet(result.distilled_images_parquet_path)

    assert len(embeddings_df) == len(dataset)
    assert len(distilled_df) == 2
    assert set(embeddings_df.columns) == {"image_path", "embedding", "cluster_id"}
    assert set(distilled_df.columns) == {
        "distilled_image_path",
        "cluster_id",
        "cluster_embedding",
        "weight",
    }
    assert all(Path(path).is_absolute() for path in embeddings_df["image_path"])
    assert all(Path(path).exists() for path in distilled_df["distilled_image_path"])
    assert sorted(distilled_df["cluster_id"].tolist()) == [0, 1]
    assert result.embeddings.shape == (6, 2)
    assert result.cluster_centers.shape == (2, 2)
    assert result.cluster_labels.shape == (6,)
    assert result.weights.shape == (2,)


def test_pipeline_rejects_k_larger_than_dataset(tmp_path):
    dataset = TinyPathImageDataset(tmp_path, n_samples=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    pipeline = VaeDdoqDistillationPipeline(
        vae=TinyVAE(),
        dataloader=dataloader,
        output_dir=tmp_path / "ddoq",
        k=3,
        device="cpu",
        output_size=(8, 8),
    )

    with pytest.raises(ValueError, match="k must be <= number of embeddings"):
        pipeline.run()


def test_pipeline_encode_cluster_decode_edge_cases(tmp_path):
    empty_loader = DataLoader([], batch_size=1)
    pipeline = VaeDdoqDistillationPipeline(
        vae=TinyVAE(),
        dataloader=empty_loader,
        output_dir=tmp_path / "empty",
        k=1,
        device="cpu",
    )
    with pytest.raises(ValueError, match="No embeddings"):
        pipeline._encode_all()

    dataset = TinyPathImageDataset(tmp_path, n_samples=2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    pipeline = VaeDdoqDistillationPipeline(
        vae=TinyVAE(),
        dataloader=dataloader,
        output_dir=tmp_path / "z",
        k=1,
        device="cpu",
        latent="z",
        latent_reduction="mean_spatial",
        distilled_image_format="pt",
        output_size=(8, 8),
    )
    embeddings, paths, output_size = pipeline._encode_all()
    assert embeddings.shape == (2, 2)

    inferred_size_pipeline = VaeDdoqDistillationPipeline(
        vae=TinyVAE(),
        dataloader=dataloader,
        output_dir=tmp_path / "inferred",
        k=1,
        device="cpu",
        output_size=None,
    )
    _, _, inferred_size = inferred_size_pipeline._encode_all()
    assert inferred_size == (8, 8)

    pipeline.latent = "bad"
    with pytest.raises(ValueError, match="latent must be"):
        pipeline._encode_all()

    pipeline.k = 0
    with pytest.raises(ValueError, match="k must be positive"):
        pipeline._cluster(embeddings)

    pipeline.k = 1
    centers = torch.zeros(1, 2)
    saved = pipeline._decode_centers(centers, output_size, paths)
    assert saved[0].suffix == ".pt"


def test_config_loaders_and_runners(tmp_path, monkeypatch):
    assert (
        ddoq_module._as_config_node(
            ddoq_module.OmegaConf.create({"dataset_distillation": {"k": 1}})
        ).k
        == 1
    )

    class LightningModel:
        model = TinyVAE()

        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(
        ddoq_module.OmegaConf,
        "load",
        lambda _path: ddoq_module.OmegaConf.create(
            {"pl_model": {"_target_": "fake.Lightning"}}
        ),
    )
    monkeypatch.setattr(ddoq_module.OmegaConf, "resolve", lambda _cfg: None)
    monkeypatch.setattr(ddoq_module, "get_class", lambda _target: LightningModel)
    vae = ddoq_module.load_vae_from_checkpoint("cfg.yaml", "model.ckpt", device="cpu")
    assert isinstance(vae, TinyVAE)

    monkeypatch.setattr(
        ddoq_module, "instantiate", lambda cfg, **kwargs: [torch.ones(3, 2, 2)]
    )
    monkeypatch.setattr(
        ddoq_module.OmegaConf,
        "load",
        lambda _path: ddoq_module.OmegaConf.create(
            {
                "train_dataset": {
                    "_target_": "fake.Dataset",
                    "data_loader": {
                        "batch_size": 2,
                        "num_workers": 0,
                        "pin_memory": False,
                    },
                }
            }
        ),
    )
    loader = ddoq_module.build_dataloader_from_config(
        "dataset.yaml",
        batch_size=None,
        num_workers=None,
        seed=123,
    )
    assert loader.batch_size == 2

    monkeypatch.setattr(
        ddoq_module.OmegaConf,
        "load",
        lambda _path: ddoq_module.OmegaConf.create({}),
    )
    with pytest.raises(KeyError, match="Dataset key"):
        ddoq_module.build_dataloader_from_config("dataset.yaml")

    with pytest.raises(ValueError, match="vae_config_path"):
        ddoq_module.run_vae_ddoq_from_config(ddoq_module.OmegaConf.create({}))
    with pytest.raises(ValueError, match="vae_checkpoint_path"):
        ddoq_module.run_vae_ddoq_from_config(
            ddoq_module.OmegaConf.create({"vae_config_path": "cfg.yaml"})
        )

    calls = {}
    monkeypatch.setattr(
        ddoq_module, "load_vae_from_checkpoint", lambda *args, **kwargs: TinyVAE()
    )
    monkeypatch.setattr(
        ddoq_module, "build_dataloader_from_config", lambda *args, **kwargs: "loader"
    )

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def run(self):
            return "result"

    monkeypatch.setattr(ddoq_module, "VaeDdoqDistillationPipeline", FakePipeline)
    result = ddoq_module.run_vae_ddoq_from_config(
        ddoq_module.OmegaConf.create(
            {
                "dataset_distillation": {
                    "k": 3,
                    "vae_config_path": "cfg.yaml",
                    "vae_checkpoint_path": "ckpt.ckpt",
                    "use_sqrt_heuristic": False,
                }
            }
        )
    )
    assert result == "result"
    assert calls["weight_mode"] == "density"


def test_run_config_file_wraps_plain_config_and_applies_overrides(
    tmp_path, monkeypatch
):
    yaml_path = tmp_path / "ddoq.yaml"
    yaml_path.write_text(
        "k: 2\nvae_config_path: cfg.yaml\nvae_checkpoint_path: old.ckpt\n"
    )
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        return "done"

    monkeypatch.setattr(ddoq_module, "run_vae_ddoq_from_config", fake_run)
    result = ddoq_module.run_vae_ddoq_from_config_file(
        yaml_path,
        k=5,
        checkpoint_path="new.ckpt",
        output_dir=tmp_path / "out",
        distilled_image_format="png",
    )

    assert result == "done"
    assert captured["cfg"].dataset_distillation.k == 5
    assert captured["cfg"].dataset_distillation.num_clusters == 5
    assert captured["cfg"].dataset_distillation.vae_checkpoint_path == "new.ckpt"


def test_ddoq_vae_cli_passes_overrides(tmp_path, monkeypatch):
    yaml_path = tmp_path / "ddoq.yaml"
    checkpoint_path = tmp_path / "vae.ckpt"
    yaml_path.write_text("dataset_distillation:\n  k: 2\n")
    checkpoint_path.write_text("checkpoint")
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return VaeDdoqDistillationResult(
            embeddings=torch.empty(0, 2),
            cluster_centers=torch.empty(0, 2),
            cluster_labels=torch.empty(0, dtype=torch.long),
            weights=torch.empty(0),
            embeddings_parquet_path=tmp_path / "embeddings.parquet",
            distilled_images_parquet_path=tmp_path / "distilled_images.parquet",
            distilled_image_paths=[tmp_path / "cluster_000000.png"],
        )

    monkeypatch.setattr(
        vae_ddoq_distillation, "run_vae_ddoq_from_config_file", fake_run
    )

    from pytorch_segmentation_models_trainer.tools.cli import cli

    result = CliRunner().invoke(
        cli,
        [
            "ddoq-vae",
            str(yaml_path),
            "--k",
            "5",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(tmp_path / "out"),
            "--format",
            "png",
        ],
    )

    assert result.exit_code == 0
    assert calls["yaml_path"] == str(yaml_path)
    assert calls["k"] == 5
    assert calls["checkpoint_path"] == str(checkpoint_path)
    assert calls["output_dir"] == str(tmp_path / "out")
    assert calls["distilled_image_format"] == "png"
