# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
from pytorch_segmentation_models_trainer.model_loader.model import Model


class AutoencoderModel(Model):
    """
    LightningModule for Autoencoder training.
    Reuses setup from the base Model class but overrides the training/validation step
    for image reconstruction.
    """

    def _shared_step(self, batch, prefix):
        """
        Overridden step for reconstruction.
        batch: dict with 'image' and 'target' (which are the same image).
        """
        images = batch["image"]
        targets = batch.get("target", images)

        # Forward pass
        reconstructed = self(images)

        # Compute loss
        # Note: Model.get_loss_function() was called in __init__
        loss = self.loss_function(reconstructed, targets)

        # Log loss
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def forward(self, x):
        return self.model(x)
