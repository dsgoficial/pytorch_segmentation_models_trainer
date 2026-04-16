---
sidebar_position: 4
title: Implementing a DA Method
---

# Implementing a Domain Adaptation Method

Adding a new DA method requires creating a single file with a subclass of `BaseDomainAdaptationMethod`. No framework code needs to be modified.

---

## The Contract

```python
from pytorch_segmentation_models_trainer.domain_adaptation.base_method import (
    BaseDomainAdaptationMethod,
    DomainAdaptationLossOutput,
)

class MyMethod(BaseDomainAdaptationMethod):
    # ── Class-level flags ────────────────────────────────────────────────
    requires_features: bool = False   # True → FeatureExtractorHook is activated
    requires_target_labels: bool = False  # True → SSDA (target has labels)

    def __init__(self, my_param=0.5, **kwargs):
        super().__init__(**kwargs)         # passes lambda_da and other base kwargs
        self.my_param = my_param
        # auxiliary networks go here as nn.Module attributes:
        # self.discriminator = nn.Sequential(...)

    # ── REQUIRED ────────────────────────────────────────────────────────
    def compute_da_loss(
        self,
        source_batch,       # dict with "image", "mask"
        target_batch,       # dict with "image" (no "mask" for UDA)
        source_output,      # model logits for source, shape (B, C, H, W)
        target_output,      # model logits for target, shape (B, C, H, W)
        source_features,    # dict of layer_name → tensor, or {} if requires_features=False
        target_features,    # dict of layer_name → tensor, or {} if requires_features=False
        **kwargs,
    ) -> DomainAdaptationLossOutput:
        loss = ...  # your logic
        return DomainAdaptationLossOutput(
            loss=loss,
            log_dict={"my_metric": loss.item()},
        )

    # ── OPTIONAL hooks (default = no-op) ────────────────────────────────
    def on_fit_start(self, pl_module): ...
    def on_train_epoch_start(self, pl_module, epoch): ...
    def on_train_epoch_end(self, pl_module, epoch): ...

    # ── OPTIONAL: override if auxiliary networks need a different LR ────
    def get_extra_parameter_groups(self):
        return [{"params": self.discriminator.parameters(), "lr": 1e-4}]
```

---

## `__init__` Convention

Hydra instantiates your method from the config by calling it with all config keys as keyword arguments:

```yaml
method:
  _target_: my_package.methods.MyMethod
  lambda_da: 1.0
  my_param: 0.5
```

→ `MyMethod(lambda_da=1.0, my_param=0.5)`

Always forward remaining kwargs to `super().__init__(**kwargs)` so that `lambda_da` and any future base-class kwargs are handled correctly.

---

## `DomainAdaptationLossOutput`

Every `compute_da_loss` must return a `DomainAdaptationLossOutput`:

```python
@dataclass
class DomainAdaptationLossOutput:
    loss: Tensor           # DA loss — combined as: total = seg_loss + λ * loss
    log_dict: dict         # logged under "da/" prefix in TensorBoard/W&B
    extra: dict = {}       # not logged automatically — available to callbacks
```

---

## Example 1 — Entropy Minimization (ADVENT)

The simplest possible method. No auxiliary networks, no feature maps needed.

```python
class ADVENTMethod(BaseDomainAdaptationMethod):
    """Entropy minimization on the target domain (Vu et al., 2019)."""

    requires_features = False

    def compute_da_loss(
        self, source_batch, target_batch,
        source_output, target_output,
        source_features, target_features, **kwargs
    ):
        probs = torch.softmax(target_output, dim=1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean()
        return DomainAdaptationLossOutput(
            loss=entropy,
            log_dict={"entropy_loss": entropy.item()},
        )
```

Config:

```yaml
method:
  _target_: my_package.methods.ADVENTMethod
  lambda_da: 0.001
```

---

## Example 2 — DANN with Gradient Reversal

:::tip Built-in available
The framework ships a production-ready `DANNMethod` — you do not need to implement this yourself. See the dedicated [DANN guide](dann-method.md) for usage and configuration.
:::

The example below shows the key concepts so you understand how to build a similar method from scratch, or how to extend `DANNMethod` for a custom variant.

```python
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.domain_adaptation.methods.gradient_reversal import (
    GradientReversalLayer,
)


class MyCustomDANNVariant(BaseDomainAdaptationMethod):
    """DANN variant with custom discriminator architecture."""

    requires_features = True   # needs intermediate feature maps

    def __init__(self, in_channels=512, hidden_size=1024, discriminator_lr=1e-4, **kwargs):
        super().__init__(**kwargs)
        # GradientReversalLayer is provided by the framework — no need to
        # re-implement torch.autograd.Function manually.
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),   # source=0, target=1
        )
        self._discriminator_lr = discriminator_lr

    def compute_da_loss(
        self, source_batch, target_batch,
        source_output, target_output,
        source_features, target_features, **kwargs
    ):
        src_feat = source_features["encoder"]   # matches cfg.feature_layers
        tgt_feat = target_features["encoder"]

        features = torch.cat([src_feat, tgt_feat], dim=0)
        domain_labels = torch.cat([
            torch.zeros(src_feat.shape[0], dtype=torch.long, device=src_feat.device),
            torch.ones(tgt_feat.shape[0], dtype=torch.long, device=tgt_feat.device),
        ])

        loss = nn.CrossEntropyLoss()(
            self.domain_classifier(self.grl(features)),
            domain_labels,
        )
        return DomainAdaptationLossOutput(
            loss=loss,
            log_dict={"dann_loss": loss.item()},
        )

    def on_train_epoch_start(self, pl_module, epoch):
        # Update GRL lambda from schedule each epoch
        if hasattr(self, "lambda_schedule"):
            lam = self.lambda_schedule.get_lambda(epoch, pl_module.trainer.max_epochs)
            self.grl.set_lambda(lam)

    def get_extra_parameter_groups(self):
        return [{"params": list(self.domain_classifier.parameters()),
                 "lr": self._discriminator_lr}]
```

Config:

```yaml
domain_adaptation:
  feature_layers:
    - encoder
  method:
    _target_: my_package.methods.MyCustomDANNVariant
    in_channels: 512
    hidden_size: 1024
    discriminator_lr: 1.0e-4
    lambda_da: 1.0
    lambda_schedule:
      _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler
      gamma: 10.0
```

To use the built-in `DANNMethod` instead, replace the `_target_` with:

```yaml
method:
  _target_: pytorch_segmentation_models_trainer.domain_adaptation.methods.dann.DANNMethod
  feature_layer: encoder
  in_channels: 512
```

See the [DANN guide](dann-method.md) for the full parameter reference.

---

## Example 3 — Pseudo-labels with Confidence Threshold

Self-training approach: generate pseudo-labels on the target domain and use
them as supervision only where the model is confident.

```python
class PseudoLabelMethod(BaseDomainAdaptationMethod):
    """Self-training with confidence-thresholded pseudo-labels."""

    requires_features = False

    def __init__(self, threshold=0.9, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def compute_da_loss(
        self, source_batch, target_batch,
        source_output, target_output,
        source_features, target_features, **kwargs
    ):
        probs = torch.softmax(target_output, dim=1)
        confidence, pseudo_labels = probs.max(dim=1)

        # Only supervise pixels above confidence threshold
        mask = confidence > self.threshold
        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=target_output.device, requires_grad=True)
            return DomainAdaptationLossOutput(
                loss=loss,
                log_dict={"pseudo_label_ratio": 0.0},
            )

        loss = F.cross_entropy(
            target_output[mask.unsqueeze(1).expand_as(target_output)].view(-1, target_output.shape[1]),
            pseudo_labels[mask],
        )
        ratio = mask.float().mean().item()
        return DomainAdaptationLossOutput(
            loss=loss,
            log_dict={"pseudo_label_loss": loss.item(), "pseudo_label_ratio": ratio},
        )
```

---

## Lifecycle Hooks

Use hooks when your method needs to maintain state across epochs:

```python
def on_fit_start(self, pl_module):
    # Called once when trainer.fit() begins.
    # Good for: initialising memory banks, computing dataset statistics.
    pass

def on_train_epoch_start(self, pl_module, epoch):
    # Called at the start of each epoch.
    # Good for: updating threshold schedules, resetting accumulators.
    pass

def on_train_epoch_end(self, pl_module, epoch):
    # Called at the end of each epoch.
    # Good for: recomputing pseudo-labels, logging epoch-level stats.
    pass
```

---

## Using Intermediate Feature Maps

Set `requires_features = True` and list the layers to capture in the config:

```yaml
domain_adaptation:
  feature_layers:
    - encoder.layer3
    - encoder.layer4
```

`DomainAdaptationModel` installs a `FeatureExtractorHook` that captures those layers during forward. The tensors arrive in `source_features` and `target_features` as a dict:

```python
src_feat = source_features["encoder.layer3"]   # Tensor (B, C, H, W)
tgt_feat = target_features["encoder.layer3"]
```

The layer names must match the attribute path on `self.model`. Use
`dict(model.named_modules())` to inspect available names.
