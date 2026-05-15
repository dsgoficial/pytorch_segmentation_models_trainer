# -*- coding: utf-8 -*-
import torch
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
            callbacks:
              - _target_: pytorch_segmentation_models_trainer.custom_callbacks.\
AutoencoderLatentClusteringCallback
              n_clusters: 8
              max_samples: 2048
        """
        super().__init__(cfg, inference_mode=inference_mode)

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

        return loss

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

    def forward(self, x):
        return self.model(x)
