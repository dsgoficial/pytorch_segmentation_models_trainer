# pytorch_segmentation_models_trainer

[![Python application](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-app.yml/badge.svg)](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-app.yml)
[![Upload Python Package](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-publish.yml)
[![maintainer](https://img.shields.io/badge/maintainer-phborba-blue.svg)](https://github.com/phborba)

Framework based on Pytorch, Pytorch Lightning,  segmentation_models.pytorch and hydra to train semantic segmentation models using yaml config files as follows:

```
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 1

loss:
  _target_: segmentation_models_pytorch.utils.losses.DiceLoss

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4

hyperparameters:
  batch_size: 1
  epochs: 2
  max_lr: 0.1

pl_trainer:
  max_epochs: ${hyperparameters.batch_size}
  gpus: 0

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /path/to/input.csv
  data_loader:
    shuffle: True
    num_workers: 1
    pin_memory: True
    drop_last: True
    prefetch_factor: 1
  augmentation_list:
    - _target_: albumentations.HueSaturationValue
      always_apply: false
      hue_shift_limit: 0.2
      p: 0.5
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
    - _target_: albumentations.RandomCrop
      always_apply: true
      height: 256
      width: 256
      p: 1.0
    - _target_: albumentations.Flip
      always_apply: true
    - _target_: albumentations.Normalize
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /path/to/input.csv
  data_loader:
    shuffle: True
    num_workers: 1
    pin_memory: True
    drop_last: True
    prefetch_factor: 1
  augmentation_list:
    - _target_: albumentations.Resize
      always_apply: true
      height: 256
      width: 256
      p: 1.0
    - _target_: albumentations.Normalize
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true
```

To train a model with configuration path ```/path/to/config/folder``` and name ```test.yaml```:

```
pytorch-segmentation-models-trainer --config-path /path/to/config/folder --config-name test +mode=train
```

The mode can be stored in configuration yaml as well. In this case, do not pass the +mode= argument. If the mode is stored in the yaml and you want to overwrite the value, do not use the + clause, just mode= .

This module suports hydra features such as configuration composition. For further information, please visit https://hydra.cc/docs/intro