---
sidebar_position: 10
title: Dataset Distillation (DDOQ)
---

# Technical Documentation: DDOQ (Dataset Distillation by Optimal Quantization)

The Dataset Distillation pipeline in `pytorch_smt` implements the **DDOQ (Dataset Distillation by Optimal Quantization)** method. This approach reframes the compression of massive datasets as an "optimal quantization" problem within latent spaces.

## 1. Method Overview

DDOQ is a cutting-edge dataset distillation method that avoids the massive memory and processing costs of optimizing image pixels directly. Instead, it maps data into a low-dimensional latent space and clusters it to find the most representative points (centroids).

The core innovation of DDOQ (supported by **Theorem 1** in the study) is the mathematical proof that the use of generative diffusion models preserves the proximity of data distributions from latent space to image space. By adding automatic statistical weights calculated during clustering, the method drastically reduces the **Wasserstein Distance**.

## 2. The DDOQ Algorithm (Step-by-Step)

1.  **Encoding**: A pre-trained VAE maps original high-dimensional images into a low-dimensional latent space.
2.  **Clustering & Weight Calculation**: Mini-Batch k-means (or CLVQ) finds $K$ centroids. Cluster mass is captured as weights, with a **square root heuristic** applied for variance reduction.
3.  **Image Synthesis / Selection (Decoding)**:
    *   *Original Paper*: Latent centroids are transformed into synthetic images via a Diffusion Model.
    *   *LULC Adaptation*: We perform a **Nearest Neighbor** search to select the real original image (Medoid) closest to the centroid.
4.  **Weighted Training**: The Student model is trained by multiplying the loss of each image by its corresponding weight ($\min_{\theta} \sum w \cdot \ell(x, y, \theta)$).

## 3. Semantic Segmentation Adaptation: Medoids vs. Soft-labels

While the original DDOQ was designed for image classification using synthetic data, our framework introduces critical adaptations for **Semantic Segmentation**:

### Medoid vs. Synthetic Images (Avoiding Artifacts)
In standard DDOQ, decoding a centroid into a synthetic image works well for global classification. However, for **Semantic Segmentation**, synthetic images often suffer from:
*   **Edge Hallucinations**: Blurred or unrealistic boundaries that confuse contour-sensitive models.
*   **Radiometric Shift**: Loss of multispectral fidelity (e.g., NIR bands) crucial for LULC.

**Our Choice**: We use the **Medoid (Nearest Neighbor)**. By selecting the real image closest to the centroid, we ensure 100% sharp boundaries and radiometric accuracy, creating a "Perfect Coreset" for pixel-level tasks.

### Spatial Soft-labels (Knowledge Transfer)
Standard DDOQ uses global soft-labels (class probabilities). In our adaptation:
*   **The Teacher Role**: If a pre-trained Teacher model is available, we pass the **Medoids** through it to generate **Spatial Soft-labels** (probability maps for every pixel).
*   **Dark Knowledge**: This transfers the Teacher's uncertainty and probabilistic edge knowledge to the Student, while the **Medoid** provides the structural fidelity that synthetic images lack.
*   **Weighted KL-Divergence**: The spatial loss is calculated pixel-by-pixel via KL Divergence and then scaled by the DDOQ cluster weight.

## 4. Workflow & Usage

### Step 1: Latent Extraction
Obtain fixed-size embedding vectors using `GenericAutoencoder`.

### Step 2: Optimal Quantization
Partition the latent space into $K$ clusters. Use `find_optimal_k_elbow_method` to find the optimal budget mathematically.

### Step 3: Medoid Search & Weighting
Find the real medoids and calculate Voronoi weights (`uniform`, `density`, or `sqrt`).

### Step 4: Training
Train a `StudentSegmentationModel` with a `DDOQDistilledDataset` (supporting both Hard-labels for Active Learning or Soft-labels for Distillation).

## 5. Configuration

```yaml
dataset_distillation:
  num_clusters: 500             # Coreset budget
  adaptive_k: true              # Use elbow method
  use_sqrt_heuristic: true      # Variance reduction (DDOQ-LULC)
  device: "cuda"
  output_ddoq_results_path: "ddoq_results.pt"
```

## 6. Python API

### Distillation (Offline Phase)
```python
# 1. Extract and find Medoids
latents = extract_all_latents(autoencoder, loader, device)
tool = KMeansClusteringTool(n_clusters=500, device=device)
tool.fit(latents)
weights = tool.get_cluster_weights(mode="sqrt")
indices = tool.get_medoids_from_dataloader(loader)

# 2. Save
save_ddoq_results(indices, weights, "ddoq_results.pt")
```

### Training (Student Phase)
```python
# Load results and train
indices, weights = load_ddoq_results("ddoq_results.pt")
dataset = DDOQDistilledDataset(original_ds, indices, weights, soft_labels=teacher_masks)
model = StudentSegmentationModel(cfg)
```

## 7. Operational Scenarios

*   **Scenario A: Active Learning**: Unsupervised clustering identifies the most representative real samples for human labeling. Training uses Weighted Cross-Entropy with Hard-labels.
*   **Scenario B: Knowledge Distillation**: A Teacher model generates Spatial Soft-labels for the medoids. Training uses Weighted KL-Divergence to transfer knowledge to a lightweight Student.
