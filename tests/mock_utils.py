import torch
import os

def create_dummy_checkpoint(path):
    """Creates a dummy PyTorch Lightning checkpoint file."""
    checkpoint = {
        "state_dict": {},
        "pytorch-lightning_version": "2.4.0",
        "epoch": 0,
        "global_step": 0,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
