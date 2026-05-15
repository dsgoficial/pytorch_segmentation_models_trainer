# -*- coding: utf-8 -*-
import torch
from hydra.utils import instantiate
from pytorch_segmentation_models_trainer.model_loader.model import Model


class AutoencoderModel(Model):
    """LightningModule for Autoencoder training."""

    def __init__(self, cfg, inference_mode=False):
        """Create the autoencoder LightningModule.

        Args:
            cfg: Hydra training configuration.
            inference_mode: When ``True``, skips dataset/loss setup in the base
                class.

        Returns:
            ``None``.

        Example YAML:
            latent_metrics:
              _target_: pytorch_segmentation_models_trainer.custom_metrics.\
autoencoder_latent_clustering.AutoencoderLatentClusteringMetrics
              n_clusters: 8
              max_samples: 2048
        """
        super().__init__(cfg, inference_mode=inference_mode)
        if not inference_mode and "latent_metrics" in self.cfg:
            self.val_latent_metrics = instantiate(
                self.cfg.latent_metrics, _recursive_=False
            )
            self.test_latent_metrics = instantiate(
                self.cfg.latent_metrics, _recursive_=False
            )

    def _shared_step(self, batch, prefix):
        """
        Overridden step for reconstruction.
        batch: dict with 'image' and 'target' (which are the same image).
        """
        images = batch["image"]
        targets = batch.get("target", images)

        reconstructed = self(images)
        loss = self.loss_function(reconstructed, targets)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        metrics_attr = f"{prefix}_metrics"
        if hasattr(self, metrics_attr) and isinstance(reconstructed, torch.Tensor):
            metrics = getattr(self, metrics_attr)(reconstructed, targets)
            self.log_dict(
                metrics,
                on_step=(prefix == "train"),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        self._update_latent_metrics(images, batch, prefix)
        return loss

    def _update_latent_metrics(self, images, batch, prefix):
        metrics_attr = f"{prefix}_latent_metrics"
        if not hasattr(self, metrics_attr) or not hasattr(self.model, "encode"):
            return

        latent_metrics = getattr(self, metrics_attr)
        labels = self._get_latent_metric_labels(batch, latent_metrics)
        with torch.no_grad():
            latents = self.model.encode(images)
        latent_metrics.update(latents, target_labels=labels)

    def _get_latent_metric_labels(self, batch, latent_metrics):
        label_key = getattr(latent_metrics, "label_key", None)
        if label_key is None or not isinstance(batch, dict) or label_key not in batch:
            return None
        labels = batch[label_key]
        if isinstance(labels, torch.Tensor):
            return labels
        return torch.as_tensor(labels)

    def _log_latent_metrics_epoch_end(self, prefix):
        metrics_attr = f"{prefix}_latent_metrics"
        if not hasattr(self, metrics_attr):
            return
        latent_metrics = getattr(self, metrics_attr)
        if not getattr(latent_metrics, "_embeddings", None):
            return
        metrics = latent_metrics.compute()
        self.log_dict(
            {f"{prefix}/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        latent_metrics.reset()

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Run one test reconstruction step.

        Args:
            batch: Batch dictionary with ``image`` and optional ``target``.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar test loss tensor.
        """
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        """Compute and log accumulated validation latent clustering metrics."""
        self._log_latent_metrics_epoch_end("val")

    def on_test_epoch_end(self):
        """Compute and log accumulated test latent clustering metrics."""
        self._log_latent_metrics_epoch_end("test")

    def forward(self, x):
        return self.model(x)
