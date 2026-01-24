# Self-Supervised Pipeline for Cloud Detection & Image Reconstruction

**Sentinel-2 (13 bands) with optional Sentinel-1 (SAR)**

This document describes a **fully self-supervised pipeline** that:

* Learns **cloud masks** (no labels)
* Performs **cloud removal / image reconstruction**
* Supports **Sentinel-1 on/off via config**
* Matches the provided dataset structure and constraints

---

## 1. Dataset Assumptions & Mapping

### Dataset Interface (given)

```python
Dataset(
    root_dir,
    seasons=None,
    s2_bands=None,
    patch_size=256,
    data_fraction=1.0,
    min_cloud_fraction=0.05,
    max_cloud_fraction=0.9,
    random_seed=42,
    cloud_mask_mode="simple",
    deep_model_kwargs=None
)
```

---

## 2. High-Level Pipeline Overview

```python
Data Sampling
   ↓
Preprocessing
   ↓
Self-Supervised Pretraining (Masked Reconstruction)
   ↓
Cloud Uncertainty Estimation
   ↓
Pseudo Cloud Mask Generation
   ↓
Self-Supervised Cloud Removal Training
   ↓
Inference (Cloud Mask + Reconstructed Image)
```

---

## 3. Configuration (Central Switches)

```yaml
model:
  use_s1: true | false
  encoder_type: unet | vit | hybrid
  latent_dim: 512

training:
  patch_size: 256
  batch_size: 8
  epochs_pretrain: 100
  epochs_reconstruction: 200

losses:
  spectral_l1: true
  sam_loss: true
  temporal_consistency: true
  uncertainty_weight: 0.3
```

---

## 4. Data Sampling Logic

### Scene-Level Sampling

* Shuffle scene keys using `random_seed`
* Select `data_fraction × scenes`
* Flatten samples:

```python
self.samples = [s for k in selected_scenes for s in scenes[k]]
```


---

## 5. Preprocessing Stage

### Sentinel-2

* Per-band normalization (mean/std or percentile)
* Optional band subset via `s2_bands`
* All bands resampled to `patch_size`

### Sentinel-1 (Optional)

* Normalize VV/VH separately
* Stack as additional channels only if `use_s1=True`

### Output Tensor

```text
X = [S2_BANDS, (S1_VV, S1_VH)?]
```

---

## 6. Stage 1 — Self-Supervised Pretraining (No Cloud Labels)

### Objective

Learn spectral + spatial consistency of land surfaces.

### Method: Masked Multispectral Autoencoding

**Input**

* Random spatial masks (block-wise)
* Random spectral masks (drop bands)

**Target**

* Original unmasked Sentinel-2 image

### Loss

```text
L_pretrain =
  L1_reconstruction +
  SAM_loss +
  (optional) temporal_consistency_loss
```

### Model

* Shared encoder
* Lightweight decoder
* No cloud head yet

**Result:**

The encoder learns land-surface priors independent of clouds.

---

## 7. Stage 2 — Cloud Uncertainty Estimation (Implicit Detection)

### Key Idea

Clouds are hard to reconstruct → high error / uncertainty.

### Procedure

For each training patch:

1. Run pretrained model
2. Compute per-pixel reconstruction error:

```text
E(x) = ||x - x̂||
```

(Per band, aggregated)

### Cloud Probability Map

```text
P_cloud = normalize(E(x))
```

**Optional refinement:**

* Spatial smoothing
* Percentile-based thresholding

This replaces `cloud_mask_mode="simple"` with model-driven masks.

---

## 8. Stage 3 — Pseudo Cloud Mask Generation

### Mask Definition

```text
cloud_mask = P_cloud > τ
```

Where:

* `τ` chosen to match `min_cloud_fraction / max_cloud_fraction`
* Can be adaptive per scene

These masks are noisy but improve with training iterations.

---

## 9. Stage 4 — Self-Supervised Cloud Removal Training

### Training Data

* **Input:** Cloudy image + pseudo cloud mask
* **Target:** Same image (identity), loss applied only on masked pixels

### Optional Synthetic Augmentation

* Apply fake cloud masks to clear images
* Forces stronger inpainting ability

### Loss

```text
L_recon =
  L1(masked pixels) +
  SAM(masked pixels) +
  spatial_smoothness
```

### Sentinel-1 (If Enabled)

* Concatenate S1 features at encoder input **OR**
* Inject at bottleneck as a structural prior

No reconstruction loss applied to S1 channels.

---

## 10. Joint Model Architecture

```text
           ┌─────────────┐
S2 (+S1) → │   Encoder   │
           └─────┬───────┘
                 │
     ┌───────────┴───────────┐
     │                       │
┌────▼─────┐           ┌─────▼─────┐
│ Cloud    │           │ Image     │
│ Head     │           │ Decoder   │
│ (Pcloud) │           │ (S2 clean)│
└──────────┘           └───────────┘
```

* Encoder is shared
* Cloud head predicts uncertainty
* Decoder reconstructs cloud-free surface

---

## 11. Inference Pipeline

For each patch:

1. Forward pass
2. Outputs:

   * Cloud probability map
   * Reconstructed cloud-free image

**Optional post-processing:**

* Morphological filtering
* Temporal blending across scenes

---

## 12. Outputs

### Per Patch

* `cloud_mask.npy`
* `reconstructed_s2.npy`

### Per Scene

* Mosaicked cloud-free image
* Temporal uncertainty statistics
