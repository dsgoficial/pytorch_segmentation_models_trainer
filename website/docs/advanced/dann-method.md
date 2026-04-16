---
sidebar_position: 6
title: DANN — Gradient Reversal
---

# DANN — Domain Adversarial Neural Network

`DANNMethod` is the built-in implementation of the **Domain Adversarial Neural Network** strategy from:

> Ganin, Y. & Lempitsky, V. *Unsupervised Domain Adaptation by Backpropagation.* ICML 2015.

It adapts a segmentation network trained on a **labeled source domain** to an **unlabeled target domain** by forcing the encoder to produce features that a domain classifier cannot distinguish — while simultaneously keeping them discriminative for segmentation.

---

## How it works

The training loop adds an adversarial branch to the normal segmentation training:

```
Source images → Encoder → GRL(λ) → DomainClassifier → L_domain
Target images → Encoder → GRL(λ) → DomainClassifier ↗

Source images → Encoder → Decoder → L_seg  (labeled, source only)
```

**Gradient Reversal Layer (GRL):** during the forward pass it acts as identity. During backpropagation it multiplies the incoming gradient by `−λ`. This forces the encoder to produce features that *fool* the domain classifier — i.e., features that look the same regardless of which domain they came from.

**What each part of the U-Net learns:**

| Component | Gradient source | Effect |
|---|---|---|
| Encoder | `L_seg` (normal) + `L_domain` via GRL (reversed) | Learns features that are both discriminative and domain-invariant |
| Decoder | `L_seg` only, source domain only | Learns to decode source-domain features |
| DomainClassifier | `L_domain` (normal) | Learns to distinguish source from target |

**Total loss:**

```
total_loss = L_seg + λ · L_domain
```

`λ` starts at 0 and grows following the DANN schedule, so the encoder has time to stabilize before adversarial pressure is applied.

---

## Quick start

### 1. Find `in_channels` for your encoder

`in_channels` must match the number of output channels of the hooked encoder layer. For SMP models, inspect `model.encoder.out_channels` — use the **last** value:

```python
import segmentation_models_pytorch as smp

model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=2)
print(model.encoder.out_channels)
# (3, 64, 64, 128, 256, 512) → use 512
```

Common values:

| Encoder | `in_channels` |
|---|---|
| `resnet18`, `resnet34` | 512 |
| `resnet50`, `resnet101` | 2048 |
| `efficientnet-b0` | 1280 |
| `efficientnet-b4` | 1792 |
| `mit_b0` | 256 |
| `mit_b2`, `mit_b4` | 512 |
| `timm-resnest50d` | 2048 |

### 2. Write the config

```yaml
domain_adaptation:
  # Layer to hook — must appear in feature_layers AND method.feature_layer
  feature_layers:
    - encoder

  method:
    _target_: pytorch_segmentation_models_trainer.domain_adaptation.methods.dann.DANNMethod
    feature_layer: encoder        # dot-separated path to the hooked layer
    in_channels: 512              # channels at that layer (see table above)
    hidden_size: 1024             # domain classifier MLP hidden width
    discriminator_lr: 1.0e-4     # separate LR for the domain classifier
    lambda_da: 1.0                # fallback if no lambda_schedule is configured

    # Recommended: DANN annealing schedule
    lambda_schedule:
      _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler
      gamma: 10.0
```

### 3. Full working config

Copy and adjust [conf/examples/dann_domain_adaptation.yaml](https://github.com/dsgoficial/pytorch_segmentation_models_trainer/blob/main/pytorch_segmentation_models_trainer/conf/examples/dann_domain_adaptation.yaml), which contains a complete example with a U-Net ResNet-34.

---

## Configuration reference

### `DANNMethod` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `feature_layer` | string | — | Dot-separated name of the encoder layer to hook, e.g. `"encoder"`. Must also appear in `feature_layers`. |
| `in_channels` | int | — | Number of output channels of `feature_layer`. See the table above. |
| `hidden_size` | int | `1024` | Width of each hidden layer in the domain classifier MLP. |
| `discriminator_lr` | float | `1e-4` | Learning rate for the domain classifier. Added as a separate optimizer parameter group. |
| `lambda_da` | float | `1.0` | Global DA loss weight. Used as a constant when `lambda_schedule` is absent. |
| `lambda_schedule` | dict | — | Optional lambda scheduler config (see [Lambda Schedulers](domain-adaptation.md#lambda-schedulers)). |
| `step_mode` | `"epoch"` \| `"batch"` | `"epoch"` | Granularity at which the lambda schedule is applied. `"batch"` updates λ every training step for smoother growth (closer to the original Ganin et al. implementation). See [Lambda update granularity](#lambda-update-granularity) below. |

### Logged metrics

The following scalars are logged under the `da/` prefix every training step:

| Metric | Description |
|---|---|
| `da/domain_loss` | Cross-entropy loss of the domain classifier |
| `da/domain_acc` | Accuracy of the domain classifier (0.5 = fully confused = ideal) |
| `da/lambda` | Current value of λ |
| `loss/train_seg` | Segmentation loss (source domain) |
| `loss/train_da` | DA loss before weighting |
| `loss/train_total` | `seg_loss + λ · da_loss` |

A `domain_acc` consistently above 0.9 means the classifier can still distinguish the domains — the adaptation is not working. A value near 0.5 means the encoder has learned domain-invariant features.

---

## Lambda update granularity

By default (`step_mode: "epoch"`), the GRL coefficient λ is updated once per epoch. Setting `step_mode: "batch"` updates λ at every training batch, producing smoother growth and matching the original Ganin et al. formulation more closely.

```yaml
domain_adaptation:
  method:
    _target_: pytorch_segmentation_models_trainer.domain_adaptation.methods.dann.DANNMethod
    feature_layer: encoder
    in_channels: 512
    lambda_da: 1.0
    step_mode: batch          # update λ every batch instead of every epoch
    lambda_schedule:
      _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler
      gamma: 10.0
```

**When to prefer each mode:**

| Mode | Progress counter | Suitable for |
|---|---|---|
| `"epoch"` (default) | `epoch / max_epochs` | Long training runs (≥ 50 epochs), simpler to reason about |
| `"batch"` | `global_batch / (max_epochs × batches_per_epoch)` | Short training runs or fine-tuning where epoch granularity is too coarse |

In both modes the same `DANNScheduler` formula is used — only the frequency of updates changes.

---

## Choosing `feature_layer`

The GRL is applied to the features captured at `feature_layer`. This determines which encoder layers are affected by the adversarial gradient.

For a U-Net the natural choice is `"encoder"` (the bottleneck), which exposes the deepest, most semantic features and causes the adversarial gradient to flow through the **entire encoder**.

For a model with multiple encoder stages you may prefer a mid-level layer if you want only the deeper layers to be domain-adapted:

```yaml
# Adapt only from layer3 onward (resnet50)
feature_layers:
  - encoder.layer3
method:
  feature_layer: encoder.layer3
  in_channels: 1024
```

To list all available layer names for a given model:

```python
model = smp.Unet(encoder_name="resnet50", ...)
for name, _ in model.named_modules():
    print(name)
```

---

## Warm-starting from a source checkpoint

The most effective workflow is to first train the model on the source domain, then run DANN adaptation from that checkpoint:

```yaml
domain_adaptation:
  pretrained_checkpoint:
    path: /checkpoints/source_model.ckpt
    source_format: pytorch_lightning
    strict_loading: true
```

Starting from a pretrained encoder means the adversarial branch starts with already-useful features rather than adapting from a randomly initialized state.

---

## Monitoring adaptation

Add `DomainAdaptationMonitorCallback` to your callbacks to track forgetting and adaptation progress:

```yaml
callbacks:
  - _target_: pytorch_segmentation_models_trainer.domain_adaptation.callbacks.DomainAdaptationMonitorCallback
    num_classes: 2
    log_every_n_epochs: 1
    forgetting_threshold: 0.05
```

Watch `da/domain_acc` alongside `iou/target_val`:

- `domain_acc` near **0.5** → encoder is producing domain-invariant features ✓
- `iou/target_val` **rising** → adaptation is improving target performance ✓
- `iou/source_val` **stable** → no catastrophic forgetting ✓

---

## Limitations

- **Single feature layer:** `DANNMethod` hooks one layer. If you need multi-scale alignment, implement a custom method (see [Implementing a DA Method](domain-adaptation-implementing-methods.md)) that combines losses from multiple layers.
- **Symmetric adaptation:** DANN does not distinguish which domain is "easier" or "harder". For highly asymmetric domain gaps, methods like ADDA or CycleGAN-based adaptation may perform better.
- **Decoder is not adapted:** Only the encoder receives the adversarial gradient. If the domain gap also affects the decoder-level features (e.g., due to different label distributions), consider adding entropy minimization on the target output alongside DANN.

---

## References

- Ganin, Y. & Lempitsky, V. *Unsupervised Domain Adaptation by Backpropagation.* ICML 2015.
- Vega, P.J.S. et al. *Weakly Supervised Domain Adversarial Neural Network for Deforestation Detection in Tropical Forests.* IEEE JSTARS 2023.
