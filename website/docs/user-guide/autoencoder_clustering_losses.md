---
title: Autoencoder Clustering Losses
sidebar_position: 9
---

# Autoencoder Clustering Losses

Three loss functions designed for **Phase-2 DCEC-style fine-tuning** of
variational autoencoders.  The goal is to preserve reconstruction quality
(PSNR) while encouraging the latent space to form well-separated clusters.

## Background

Training a VAE with MSE alone produces good reconstruction but no latent
structure.  Adding standard KL regularisation (β-VAE) can destroy
reconstruction quality when β is too large.

The DCEC approach solves this with a two-phase strategy:

| Phase | Loss | Goal |
|-------|------|------|
| 1 | MSE (existing `VariationalAutoencoderLoss`) | maximise PSNR |
| 2 | `ClusteringAwareVAELoss` | maintain PSNR + improve clustering |

**References**

- Xie et al., *"Unsupervised Deep Embedding for Clustering Analysis"*, ICML 2016 — DEC soft-assignment loss.
- Guo et al., *"Deep Convolutional Embedded Clustering"*, IJCAI 2017 — DCEC: joint reconstruction + clustering.
- Wen et al., *"A Discriminative Feature Learning Approach for Deep Face Recognition"*, ECCV 2016 — Center loss.

---

## `DECSoftAssignmentLoss`

Minimises the KL divergence between soft cluster assignments **Q** and a
sharpened target distribution **P**.  Pushing **Q** toward **P** forces the
encoder to make confident, unambiguous cluster assignments.

```
Q_ij  = (1 + ‖z_i − c_j‖² / α)^{−(α+1)/2}   (Student-t kernel, α=1 default)
        ─────────────────────────────────────
        Σ_j (1 + ‖z_i − c_j‖² / α)^{−(α+1)/2}

P_ij  = Q_ij² / Σ_i Q_ij     (sharpened, normalised)
        ─────────────────────

L_DEC = KL(P ‖ Q)
```

### YAML

```yaml
loss:
  _target_: ...DECSoftAssignmentLoss
  n_clusters: 8
  embedding_dim: 128
  alpha: 1.0
```

### Warm-start (mandatory)

Initialise cluster centers with K-Means before training:

```python
loss_fn.initialize_centers(kmeans_centroids)  # tensor (K, D)
```

---

## `CenterLoss`

Minimises the mean squared distance from each embedding to its nearest cluster
center.  Encourages compact, geometrically tight clusters.

```
L_center = λ · (1/2) · Σ_i ‖z_i − c_{ŷ_i}‖²
```

where `ŷ_i = argmin_j ‖z_i − c_j‖`.

### YAML

```yaml
loss:
  _target_: ...CenterLoss
  n_clusters: 8
  embedding_dim: 128
  lambda_center: 0.01
```

---

## `ClusteringAwareVAELoss` *(recommended entry point)*

Composite loss combining all three terms:

```
L = L_reconstruction + β·L_KL + γ·L_DEC + δ·L_center
```

Owns a single `cluster_centers` parameter shared by both sub-losses.

### YAML

```yaml
loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_clustering_losses.ClusteringAwareVAELoss
  n_clusters: 8
  embedding_dim: 128
  gamma: 0.1          # DEC weight
  delta: 0.01         # center weight
  vae_latent: mu      # "mu" or "z"
  latent_reduction: adaptive_avg_pool
  reconstruction_loss: mse
  reconstruction_weight: 1.0
  beta: 0.0001        # small KL — keeps PSNR intact
```

### Phase-2 training protocol (automatic)

Add `ClusterCentersWarmStartCallback` to your callback list.  It runs once
before the first epoch and handles the warm-start automatically.

```yaml
callbacks:
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.ClusterCentersWarmStartCallback
    max_samples: 4096
    image_key: image
    vae_latent: mu
```

The callback iterates `trainer.train_dataloader`, collects up to `max_samples`
embeddings, fits K-Means, and writes the centroids into `loss_function.cluster_centers`
before training starts.  It is a no-op if `loss_function` is not a
`ClusteringAwareVAELoss`.

### Phase-2 training protocol (manual)

```python
# 1. Collect embeddings from pre-trained encoder (no gradient)
embeddings = []
with torch.no_grad():
    for batch in dataloader:
        out = model(batch["image"])
        embeddings.append(out.mu)
embeddings = torch.cat(embeddings, dim=0)

# 2. Warm-start centers
loss_fn.initialize_centers_from_embeddings(embeddings)

# 3. Train Phase 2 with ClusteringAwareVAELoss
trainer.fit(model)
```

### Monitoring

Watch these metrics during Phase 2:

| Metric | Direction | Healthy sign |
|--------|-----------|--------------|
| `latent_calinski_harabasz` | ↑ | clusters separating |
| `latent_davies_bouldin` | ↓ | clusters tightening |
| `latent_silhouette` | ↑ | less ambiguous assignment |
| PSNR | → stable | reconstruction preserved |

If PSNR drops > 2 dB, reduce `gamma` or `delta`.

---

## Loss output keys

`ClusteringAwareVAELoss.forward` returns all keys from
`VariationalAutoencoderLoss` plus:

| Key | Description |
|-----|-------------|
| `dec_loss` | raw DEC KL divergence |
| `weighted_dec_loss` | `gamma × dec_loss` |
| `center_loss` | raw center loss |
| `weighted_center_loss` | `delta × center_loss` |
