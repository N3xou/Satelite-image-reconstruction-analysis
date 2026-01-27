# Satellite Image Cloud Removal - ML/DL Comparison Project

A comprehensive machine learning and deep learning framework for removing clouds from satellite imagery using Sentinel-1 SAR and Sentinel-2 optical data.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Usage](#usage)

---

## 🎯 Project Overview

This project compares multiple machine learning and deep learning approaches for reconstructing cloud-covered satellite imagery. It uses the **SEN12MS-CR dataset**, which combines:
- **Sentinel-1 (SAR)**: 2 bands (VV, VH) - penetrates clouds
- **Sentinel-2 (Optical)**: Up to 13 spectral bands - affected by clouds
- **Cloud masks**: Soft probability masks derived from SAR-optical disagreement

### Problem Statement
Remove clouds from optical satellite imagery by leveraging all-weather SAR data and temporal patterns.

---

## 🏗️ Architecture

```
project/
├── main.py                      # Main execution pipeline
├── data_loader.py               # Dataset loading and preprocessing
├── Models.py                    # Model architectures
├── training_functions.py        # Training logic and utilities
├── Visualization.py             # Plotting and visualization
├── sen12mscr_dataset/          # Raw dataset (extracted from tar)
└── results/                    # Training outputs
    ├── model_comparison.csv
    ├── metrics_comparison.png
    ├── efficiency_analysis.png
    ├── all_models_comparison.png
    └── saved_models/
```

---

## 🔑 Key Components

### 1. Configuration (`main.py::Config`)

Central configuration class managing all hyperparameters:

```python
class Config:
    # Dataset
    DATASET_ROOT = "./sen12mscr_dataset"
    SEASONS = None  # None = all seasons
    S2_BANDS = list(range(1, 14))  # All 13 Sentinel-2 bands
    PATCH_SIZE = 256
    DATA_FRACTION = 0.15  # Use 15% of data
    
    # Training
    FOLDS = 3
    BATCH_SIZE = 6
    EPOCHS = 5
    LEARNING_RATE = 0.001
    PATIENCE = 2
    USE_AMP = True  # Automatic Mixed Precision
    
    # Models to train
    MODELS = ['SimpleCNN', 'UNet', 'GAN', 'LSTM', 'Diffusion', 'RandomForest']
```

**Key Parameters:**
- `DATA_FRACTION`: Controls used dataset size
- `min_cloud_fraction/max_cloud_fraction`: Filter patches by cloud coverage
- `USE_AMP`: Enable mixed precision training for speed

---

## 📊 Data Pipeline

### Core Dataset Class: `SEN12MSCRDataset`

**Location:** `data_loader.py`

**Purpose:** Loads and preprocesses Sentinel-1/2 data with cloud masks

#### Key Methods

```python
def __init__(self, root_dir, seasons=None, s2_bands=None, 
             patch_size=256, data_fraction=1.0,
             min_cloud_fraction=0.05, max_cloud_fraction=0.9)
```
**Purpose:** Initialize dataset with scene-level sampling and cloud filtering

**Important Features:**
- Scene-level data fraction control (prevents split leakage)
- Cloud coverage filtering (min/max thresholds)
- Random cropping to `patch_size`
- Automatic band selection

```python
def __getitem__(self, idx) -> tuple
```
**Returns:**
- `s1`: Sentinel-1 SAR (2, H, W) - normalized to [0,1]
- `s2_cloudy`: Cloudy optical (13, H, W) - normalized to [0,1]
- `s2_clean`: Clean optical (13, H, W) - normalized to [0,1]
- `cloud_mask`: Soft mask (1, H, W) - probability in [0,1]
```python
@staticmethod
def compute_cloud_mask(s1, s2_cloudy) -> np.ndarray
```
**Purpose:** Generates soft cloud likelihood mask

**Algorithm:**
1. Optical brightness: `opt = s2_cloudy.mean(axis=0)`
2. SAR energy: `sar = mean(|s1|)`
3. Normalize both locally
4. Cloud likelihood: `opt × (1 - sar)` (bright optical + weak SAR = cloud)

## 🧠 Models

### Model Architectures (`Models.py`)

#### 1. UNet
**Architecture:**
- Encoder: 4 conv blocks with max pooling (64→128→256→512)
- Bottleneck: 1024 channels
- Decoder: 4 upsampling blocks with skip connections
- Output: Sigmoid activation
#### 2. SimpleCNN
**Architecture:** 5-layer CNN (64→128→128→64→out)
#### 3. GAN (Generator + Discriminator)
**Generator Architecture:**
- ResNet-based with reflection padding
- Downsampling: 2 blocks (64→128→256)
- Residual blocks: 6 blocks at 256 channels
- Upsampling: 2 blocks (256→128→64)

```python
class Discriminator(nn.Module):
    def __init__(self, in_channels=8)  # Input + output concatenated
```
**Discriminator:** PatchGAN (local patches, not whole image)
#### 5. Diffusion Model
**Architecture:**
- Time embedding: Sinusoidal positional encoding
- U-Net structure with ResBlocks
- Downsampling + Middle + Upsampling paths
**Training:** Learns to denoise images step-by-step (DDPM)

#### 6. Random Forest
**Features:** Concatenates `[S1, S2_cloudy, cloud_mask]` per pixel

**Training:**
```python
def fit(self, train_loader, max_samples=100000):
    # Flattens spatial data to [N_pixels, N_features]
    # Trains sklearn RandomForestRegressor
```

**Prediction:**
```python
def predict(self, s1, s2_cloudy, cloud_mask) -> torch.Tensor:
    # Predicts clean S2 pixel-by-pixel
    # Returns [B, C, H, W]
```
## 🎓 Training Pipeline

### ModelTrainer (`training_functions.py`)

**Core Class:** Handles K-fold cross-validation with scene-level splits

#### Key Methods

```python
def train_kfold(self, dataset, k=5, epochs=20, batch_size=8, 
                lr=0.001, patience=7, use_amp=False)
```
**Purpose:** Train model with K-fold CV at scene level

**Process:**
1. Create train/val DataLoaders per fold based on scene-split
2. Train each fold with early stopping
3. Return all trained models + histories

**Features:**
- Mixed precision training (`torch.cuda.amp`)
- Automatic early stopping (monitors val_loss)
- Best model checkpoint restoration
- Epoch timing statistics

**Loss Function:** MSE between predicted and clean S2

#### Specialized Training Methods

```python
def _train_gan_fold(self, train_loader, val_loader, ...)
```
**GAN-specific training:**
- Alternating discriminator/generator updates
- Combined loss: `gan_loss + 100 × l1_loss`
- Separate optimizers and scalers

```python
def _train_diffusion_fold(self, train_loader, val_loader, ...)
```
**Diffusion-specific training:**
- Noise schedule: Linear from 1e-4 to 0.02
- Forward diffusion: Add noise to clean images
- Training objective: Predict added noise
- Validation: Simplified single-step reconstruction

```python
def _train_rf_fold(self, train_loader, val_loader)
```
**Random Forest training:**
- Pixel-wise flattening of spatial data
- Limited to 100k pixels for memory efficiency
- Wrapped in PyTorch-compatible interface

## 📈 Evaluation

### ModelEvaluator (`main.py`)

**Purpose:** Comprehensive model comparison across multiple metrics

#### Key Methods

```python
def evaluate_model(self, model, test_loader, model_name) -> dict
```
**Returns metrics:**
- **MSE/MAE/RMSE**: Pixel-wise reconstruction error
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **Inference Time**: Average time per batch (seconds)
- **Throughput**: Images processed per second

```python
def _calculate_metrics(self, targets, preds, inference_times):
    """
    Computes all evaluation metrics
    
    PSNR: 20 × log10(max_val / sqrt(MSE))
    SSIM: Structural similarity (simplified version)
    """
```

```python
def generate_comparison_report(self, training_times, output_dir):
    """
    Generates:
    - CSV comparison table
    - Metrics bar charts
    - Training/inference efficiency scatter plots
    - Best model analysis
    """
```

**Outputs:**
- `model_comparison.csv`: Complete results table
- `metrics_comparison.png`: Bar charts for all metrics
- `efficiency_analysis.png`: Time vs quality trade-offs

---

## 🎨 Visualization

### Key Functions (`Visualization.py`)

```python
def get_s1_viz(s1_band_tensor, speckle_filter="median", 
               filter_size=3, gamma=0.7) -> np.ndarray
```
**Purpose:** Make SAR imagery human-readable

**Process:**
1. Speckle reduction (median/gaussian filter)
2. Robust contrast stretch (2nd-98th percentile)
3. Gamma correction (lifts mid-tones)

```python
def visualize_predictions(models, dataset, device, 
                         output_dir, n_samples=5)
```
**Purpose:** Generate side-by-side comparison of all models

**Layout:** `[Cloudy | S1 | Mask | Diff | GT | Model1 | Model2 | ...]`

**Features:**
- RGB stretch for Sentinel-2 (bands [B04, B03, B02])
- SAR visualization with speckle filtering
- Difference heatmaps

## 🚀 Usage

### Quick Start

```bash
# 1. Extract dataset
python data_loader.py
# Select option 1: Extract tar files
# Provide paths to: ROIsXXXX_season_s1.tar, _s2.tar, _s2_cloudy.tar

# 2. Verify structure
python data_loader.py
# Select option 2: Verify dataset structure

# 3. Train models
python main.py
```

### Main Pipeline (`main.py::main()`)

**Execution Flow:**

1. **Load Dataset**
   ```python
   full_dataset = SEN12MSCRDataset(
       root_dir='./sen12mscr_dataset',
       s2_bands=list(range(1, 14)),  # All 13 bands
       patch_size=256,
       data_fraction=0.15,  # 15% of data
       min_cloud_fraction=0.1,
       max_cloud_fraction=0.7
   )
   ```

2. **Scene-Level Split**
   ```python
   # Group by scene, then split 80/20
   train_scenes, val_scenes = train_test_split(scene_keys, test_size=0.2)
   ```

3. **Train Models**
   ```python
   for model_name in Config.MODELS:
       trainer = ModelTrainer(model_name, device='cuda')
       models, histories = trainer.train_kfold(
           dataset=train_dataset,
           k=3,  # 3-fold CV
           epochs=5,
           batch_size=6
       )
   ```

4. **Evaluate**
   ```python
   evaluator = ModelEvaluator(device='cuda')
   for model_name, model in trained_models.items():
       evaluator.evaluate_model(model, val_loader, model_name)
   ```

5. **Generate Reports**
   ```python
   comparison_df = evaluator.generate_comparison_report(
       training_times, Config.OUTPUT_DIR
   )
   ```

6. **Visualize**
   ```python
   visualize_predictions(trained_models, val_dataset, 
                        device, Config.OUTPUT_DIR, n_samples=5)
   ```
## 📊 Expected Results

### Model Comparison

| Model | PSNR (dB) | Training Time | Inference Speed |
|-------|-----------|---------------|-----------------|
| SimpleCNN | ~26-28 | 5-10 min | Fast |
| UNet | ~28-30 | 15-25 min | Medium |
| GAN | ~30-32 | 30-50 min | Medium |
| LSTM | ~27-29 | 20-35 min | Slow |
| Diffusion | ~29-31 | 40-70 min | Very Slow |
| RandomForest | ~25-27 | 10-15 min | Fast |

## 🔧 Key Design Decisions

### 1. Scene-Level Splitting
**Why:** Prevents data leakage between train/val sets (patches from same scene are correlated)

## 🐛 Common Issues

### Out of Memory
- Reduce `BATCH_SIZE`
- Reduce `PATCH_SIZE` (e.g., 128 instead of 256)
- Use fewer bands: `S2_BANDS = [2,3,4,8]`

### Slow Training
- Increase `DATA_FRACTION` to < 0.15
- Reduce `FOLDS` (e.g., 2 instead of 3)
- Enable `USE_AMP = True`
- Increase `NUM_WORKERS`

### Poor Results
- Increase `DATA_FRACTION`
- Increase `EPOCHS`
- Adjust cloud filtering: `min_cloud_fraction`, `max_cloud_fraction`
- Check normalization (view sample statistics)

---

## 📚 Dependencies

```
torch >= 2.0
numpy
pandas
matplotlib
scikit-learn
rasterio (for .tif files)
tqdm
```

---

## 🎯 Future Improvements

1. **Improvement of loss function**
2. **PCA**
3. **SSL**
4. **ML for cloud mask detection/creation**

---

## 📖 References

- **SEN12MS-CR Dataset**: Sentinel-1/2 Cloud Removal Dataset

---

## 📝 License

This project is for educational and research purposes.

---

**Last Updated:** 2026-01-27
