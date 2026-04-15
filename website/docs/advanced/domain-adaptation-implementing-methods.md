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

Uses intermediate feature maps and a discriminator network.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DANNMethod(BaseDomainAdaptationMethod):
    """Domain-Adversarial Neural Network (Ganin et al., 2016)."""

    requires_features = True   # needs intermediate feature maps

    def __init__(self, feature_dim=256, hidden_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.lambda_schedule = None   # set from config if desired

    def compute_da_loss(
        self, source_batch, target_batch,
        source_output, target_output,
        source_features, target_features, **kwargs
    ):
        # Use the feature map captured by FeatureExtractorHook
        src_feat = source_features["encoder.layer3"]   # matches cfg.feature_layers
        tgt_feat = target_features["encoder.layer3"]

        alpha = self.lambda_da   # or use self.lambda_schedule if configured

        # Gradient reversal — forward is identity, backward reverses gradient
        src_rev = GradientReversal.apply(src_feat, alpha)
        tgt_rev = GradientReversal.apply(tgt_feat, alpha)

        src_pred = self.discriminator(src_rev)
        tgt_pred = self.discriminator(tgt_rev)

        labels = torch.cat([
            torch.zeros(len(src_pred), device=src_pred.device),
            torch.ones(len(tgt_pred), device=tgt_pred.device),
        ])
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([src_pred, tgt_pred]).squeeze(), labels
        )
        return DomainAdaptationLossOutput(
            loss=loss,
            log_dict={"dann_loss": loss.item()},
        )

    def get_extra_parameter_groups(self):
        # discriminator can use a higher LR than the encoder
        return [{"params": self.discriminator.parameters(), "lr": 1e-3}]
```

Config:

```yaml
method:
  _target_: my_package.methods.DANNMethod
  lambda_da: 1.0
  feature_dim: 256
  hidden_dim: 1024
  lambda_schedule:
    _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler
    gamma: 10.0

feature_layers:
  - encoder.layer3
```

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
