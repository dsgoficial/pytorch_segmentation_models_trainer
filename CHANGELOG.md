# Version 0.13.0

- Dataset conversion added. It is possible to convert between some formats of dataset;
- Tversky Loss and Focal Tversky Loss added;
- LabelSmoothingLoss added;
- MixUpAugmentationLoss added;
- KnowledgeDistillationLoss added;
- Mixup augmentation added to Frame Field Model;

# Version 0.12.1

- Bug fixes on mask building;
- Bug fixes on detection model training.
- New mode on build masks;

# Version 0.12.0

- Minor improvements on polygonization methods;
- Inference server added;


# Version 0.11.0

- Gradient Centralization added;


# Version 0.10.0

- Object Detection added;
- Instance Segmentation added;

# Version 0.9.0

- PolygonRNN model added;
- Added the option of choosing the number of images on ImageCallback;
- Added the option of adding created masks to existing csv;
- Added the option of generating bounding boxes in create masks;
- Added the option of converting csv dataset to coco dataset;

# Version 0.8.2

- Fixes on requirements;


# Version 0.8.1

- Minor improvements and bug fixes on polygon building inference;
- Bug fixes on mask builder;
- Performance improvement on mask builder using coco format;

# Version 0.8.0

- Added inference features;
- Improved polygon inference;

# Version 0.7.2

- Changed the versions of pytorch and torchvision.

# Version 0.7.1

- Added MANIFEST.in to include missing yml on pypi packaging.

# Version 0.7.0

- Bug fix on loss sync;
- Custom models from Frame Field implementation (to compare training results);
- New HRNet-OCR-W48 backbone;
- Fixed bugs on new versions of pytorch-lightning;
- Build mask from COCO dataset format;

# Version 0.6.0

- Polygon inference
- Unittests to Polygon inference;
- Bug fixes warmup callback (invalid signature on method);
- FrameFieldResultCallback renamed to FrameFieldOverlayedResultCallback;
- New implementation of FrameFieldResultCallback;
- Invalid mask handling (frame field training mask with only polygon mask and empty vertex and boundary masks);
- Added multiple schedulers option;
- Added IoU 10, 25, 50, 75 and 90;
- Added GPU augmentation using kornia;

# Version 0.5.1

- Bug fixes when inputs are RGBA images;
- Bug fixes on frame field model with models other than U-Net;
- Bug fixes on FrameFieldResultCallback (all black image fixed).

# Version 0.5.0

- Added frame field training image visualization callback.

# Version 0.4.1

Bug fixes on missing entrypoints and mask process execution.

# Version 0.4

## Polygoniztion by Frame Field Learning features

- FrameField dataset
- FrameField Learning
- Polygonization

# Version 0.3.2

Bug fixes on image callback when Pytorch Lightning DDP is used.

# Version 0.3.1

Bug fixes when Pytorch Lightning DDP is used.

# Version 0.3.0

- Custom metric option in the model config;
- pytorch_toolbelt added as required package. This enables usage of the models, losses and metrics in the training;
- Added the option of setting a limit of rows to be read in the csv dataset;
- Added the option of setting a root_dir to the dataset. This root_dir will be concatenated to the entry in the csv dataset before loading the image;
- Bug fixes on image_callback;

# Version 0.2.1

Fixes relative path bug on dataset

# Version 0.2.0

## New custom callbacks:

- ImageSegmentationResultCallback: Callback that logs the results of the training on TensorBoard and on saved files; and
- WarmupCallback: Applies freeze weight on encoder during callback epochs and then unfreezes the weights after the warmup epochs.

## Metrics added to Segmentation Model:

- Accuracy;
- Precision;
- Recall; and
- Jaccard Index (IoU).

# Version 0.1.4

First version of metrics added.

Bug fixes on dataset reading with prefix path.

# Version 0.1.3

Bug fix on entry points and --config-dir syntax.

# Version 0.1.2

Bug fix on Python's version.

# Minor bug fix

Bug fix.

# First Release

Framework based on Pytorch, Pytorch Lightning, segmentation_models.pytorch and hydra to train semantic segmentation models using yaml config files.
