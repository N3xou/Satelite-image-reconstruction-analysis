"""
Satellite Image Cloud Removal - ML/DL Comparison Project
========================================================
Compares multiple approaches for reconstructing cloud-covered satellite imagery
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from synthetic_data_generator import SatelliteDatasetPreparer
import warnings
from data_loader import SEN12MSCRDataset
from training_functions import ModelTrainer
from Visualization import visualize_dataset_samples,visualize_predictions
import time
warnings.filterwarnings('ignore')

# Try to import rasterio (optional, needed only for real Sentinel-2 data)
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Note: rasterio not installed. Install for real Sentinel-2 data: pip install rasterio")

class Config:
    # Dataset
    DATASET_ROOT = "./sen12mscr_dataset"
    SEASONS = None  # None = all seasons
    S2_BANDS = list(range(1, 14))
    PATCH_SIZE = 256
    DATA_FRACTION = 0.05 # Use % of data for quick training
    min_cloud_fraction = 0.4
    max_cloud_fraction = 0.6
    # Training
    FOLDS = 3
    BATCH_SIZE = 6
    EPOCHS = 5
    LEARNING_RATE = 0.001
    PATIENCE = 2
    USE_AMP = True  # Automatic Mixed Precision
    NUM_WORKERS = 4

    # Models to train
    MODELS = ['UNet']
    #MODELS = ['SimpleCNN', 'UNet', 'RandomForest']  # Fast models for demo
    #MODELS = ['SimpleCNN', 'UNet', 'GAN', 'RandomForest', 'Diffusion', 'DSen2CR']  # All models

    # Output
    OUTPUT_DIR = Path("./resultsTestCloudM50")
    SAVE_MODELS = True

# ==================== 2. DATA EXPLORATION & INSIGHTS ====================

class DataExplorer:
    """Generate insights about the satellite sen12mscr_dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def analyze_dataset(self):
        """Comprehensive sen12mscr_dataset analysis"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)

        # Basic statistics
        print(f"\nDataset size: {len(self.dataset)} image pairs")

        # Sample images
        sample_cloudy, sample_clean = self.dataset[0]
        print(f"Image shape: {sample_clean.shape}")
        print(f"Number of spectral bands: {sample_clean.shape[0]}")
        print(f"Spatial resolution: {sample_clean.shape[1]}x{sample_clean.shape[2]}")

        # Statistical analysis
        self._compute_statistics()
        self._plot_spectral_profiles()
        self._visualize_samples()

    def _compute_statistics(self):
        """Compute sen12mscr_dataset statistics"""
        all_clean = []
        all_cloudy = []

        for i in range(min(50, len(self.dataset))):
            cloudy, clean = self.dataset[i]
            all_clean.append(clean.numpy())
            all_cloudy.append(cloudy.numpy())

        all_clean = np.array(all_clean)
        all_cloudy = np.array(all_cloudy)

        print("\n--- Statistical Summary ---")
        print(f"Clean images - Mean: {all_clean.mean():.4f}, Std: {all_clean.std():.4f}")
        print(f"Cloudy images - Mean: {all_cloudy.mean():.4f}, Std: {all_cloudy.std():.4f}")

        # Cloud coverage estimation
        cloud_coverage = (all_cloudy > 0.7).mean(axis=(1,2,3)) * 100
        print(f"Average cloud coverage: {cloud_coverage.mean():.2f}%")

    def _plot_spectral_profiles(self):
        """Plot spectral profiles across bands"""
        cloudy, clean = self.dataset[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for band in range(clean.shape[0]):
            axes[0].plot(clean[band, 128, :].numpy(), label=f'Band {band+1}')
            axes[1].plot(cloudy[band, 128, :].numpy(), label=f'Band {band+1}')

        axes[0].set_title('Clean Image - Spectral Profile')
        axes[1].set_title('Cloudy Image - Spectral Profile')
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.savefig('spectral_profiles.png')
        plt.close()

    def _visualize_samples(self, n_samples=3):
        """Visualize sample image pairs"""
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))

        for i in range(n_samples):
            cloudy, clean = self.dataset[i]

            # Use first 3 bands as RGB
            cloudy_rgb = cloudy[:3].permute(1, 2, 0).numpy()
            clean_rgb = clean[:3].permute(1, 2, 0).numpy()
            diff = np.abs(cloudy_rgb - clean_rgb)

            axes[i, 0].imshow(cloudy_rgb)
            axes[i, 0].set_title('Cloudy Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(clean_rgb)
            axes[i, 1].set_title('Clean Target')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(diff)
            axes[i, 2].set_title('Difference')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig('sample_pairs.png')
        plt.close()
        print("\nVisualizations saved: spectral_profiles.png, sample_pairs.png")

# ==================== 5. EVALUATION & COMPARISON ====================

class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate single model with comprehensive metrics"""
        model.eval()

        all_preds = []
        all_targets = []
        inference_times = []

        print(f"\nEvaluating {model_name}...")

        with torch.no_grad():
            for batch_idx, (s1, s2_cloudy, s2_clean, cloud_mask) in enumerate(test_loader):
                if batch_idx >= 20:
                    break
                s1 = s1.to(self.device, non_blocking=True)
                s2_cloudy = s2_cloudy.to(self.device, non_blocking=True)
                s2_clean = s2_clean.to(self.device, non_blocking=True)
                cloud_mask = cloud_mask.to(self.device, non_blocking=True)

                model_input = torch.cat([s1, s2_cloudy], dim=1)

                # Measure inference time
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                if model_name == 'RandomForest':
                    # Use predict() instead of forward()
                    output = model.predict(s1, s2_cloudy, cloud_mask, device=self.device)
                else:
                    output = model(model_input)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                all_preds.append(output.cpu().numpy())
                all_targets.append(s2_clean.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics
        metrics = self._calculate_metrics(targets, preds, inference_times)
        self.results[model_name] = metrics

        # Print results
        print(f"\n{model_name} Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"  {metric:.<30} {value:.6f}")

        return metrics

    def _calculate_metrics(self, targets, preds, inference_times):
        """Calculate all evaluation metrics"""
        # Basic metrics
        mse = mean_squared_error(targets.flatten(), preds.flatten())
        mae = mean_absolute_error(targets.flatten(), preds.flatten())
        rmse = np.sqrt(mse)

        # PSNR
        psnr_scores = [self._psnr(targets[i], preds[i]) for i in range(len(preds))]
        psnr = np.mean(psnr_scores)

        # SSIM (simplified)
        ssim = self._ssim(targets, preds)

        # Timing
        avg_inference = np.mean(inference_times)
        throughput = len(preds) / np.sum(inference_times)

        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'PSNR_dB': psnr,
            'SSIM': ssim,
            'Inference_Time_s': avg_inference,
            'Throughput_img_s': throughput
        }

    @staticmethod
    def _psnr(target, pred, max_val=1.0):
        """Peak Signal-to-Noise Ratio"""
        mse = np.mean((target - pred) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def _ssim(targets, preds):
        """Simplified Structural Similarity Index"""
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x, mu_y = np.mean(targets), np.mean(preds)
        sigma_x, sigma_y = np.std(targets), np.std(preds)
        sigma_xy = np.mean((targets - mu_x) * (preds - mu_y))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        return ssim

    def generate_comparison_report(self, training_times, output_dir):
        """Generate comprehensive comparison visualizations"""
        if not self.results:
            print("\n⚠️ No results found to generate comparison report.")
            return None
        df = pd.DataFrame(self.results).T

        # Add training times
        df['Training_Time_min'] = [training_times.get(m, 0) / 60 for m in df.index]

        # Save CSV
        csv_path = output_dir / 'model_comparison.csv'
        df.to_csv(csv_path)
        print(f"\n✓ Saved comparison CSV: {csv_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(df.to_string())
        print("=" * 70)

        # Generate plots
        self._plot_metrics_comparison(df, output_dir)
        self._plot_training_efficiency(df, output_dir)

        # Best model analysis
        self._print_best_models(df)

        return df

    def _plot_metrics_comparison(self, df, output_dir):
        """Plot all metrics comparison"""
        metrics = [col for col in df.columns if col != 'Model']
        num_metrics = len(metrics)

        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))
        if num_metrics == 1:
            axes = [axes]

        # Define a robust color palette
        default_colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

        for i, (metric, ax) in enumerate(zip(metrics, axes)):
            # Ensure we have enough colors for the number of bars
            colors = default_colors[:len(df)]
            if not colors:
                colors = 'skyblue'  # Fallback to a single string color

            df[metric].plot(kind='bar', ax=ax, color=colors)
            ax.set_title(f'Model Comparison: {metric}')
            ax.set_ylabel('Value')
            if 'Model' in df.columns and not df.empty:
                ax.set_xticklabels(df['Model'], rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plot_path = output_dir / 'metrics_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved metrics plot: {plot_path}")

    def _plot_training_efficiency(self, df, output_dir):
        """Plot training time vs quality"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Training time vs PSNR
        ax1.scatter(df['Training_Time_min'], df['PSNR_dB'],
                    s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
        for i, model in enumerate(df.index):
            ax1.annotate(model,
                         (df.loc[model, 'Training_Time_min'], df.loc[model, 'PSNR_dB']),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax1.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Efficiency', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Inference time vs PSNR
        ax2.scatter(df['Inference_Time_s'] * 1000, df['PSNR_dB'],
                    s=200, alpha=0.6, c=range(len(df)), cmap='plasma')
        for i, model in enumerate(df.index):
            ax2.annotate(model,
                         (df.loc[model, 'Inference_Time_s'] * 1000, df.loc[model, 'PSNR_dB']),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax2.set_xlabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax2.set_title('Inference Efficiency', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'efficiency_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved efficiency plot: {plot_path}")

    def _print_best_models(self, df):
        """Print best model analysis"""
        print(f"\n{'=' * 70}")
        print("BEST MODEL ANALYSIS")
        print(f"{'=' * 70}")

        best_quality = df['PSNR_dB'].idxmax()
        best_speed = df['Throughput_img_s'].idxmax()
        fastest_train = df['Training_Time_min'].idxmin()

        print(f"Best Quality (PSNR):     {best_quality:.<20} {df.loc[best_quality, 'PSNR_dB']:.2f} dB")
        print(f"Fastest Inference:       {best_speed:.<20} {df.loc[best_speed, 'Throughput_img_s']:.2f} img/s")
        print(f"Fastest Training:        {fastest_train:.<20} {df.loc[fastest_train, 'Training_Time_min']:.2f} min")
        print(f"{'=' * 70}")


# ==================== MAIN EXECUTION ====================

def main():
    """Fully automated training pipeline"""

    print(f"\nConfiguration:")
    print(f"  Dataset:         {Config.DATASET_ROOT}")
    print(f"  Bands:           {len(Config.S2_BANDS)}")
    print(f"  Data Fraction:   {Config.DATA_FRACTION * 100:.0f}%")
    print(f"  Patch Size:      {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"  Batch Size:      {Config.BATCH_SIZE}")
    print(f"  Epochs:          {Config.EPOCHS}")
    print(f"  Models:          {', '.join(Config.MODELS)}")
    print(f"min_cloud_fraction: {Config.min_cloud_fraction:.2%}")
    print(f"max_cloud_fraction: {Config.max_cloud_fraction:.2%}")
    print("=" * 70)

    # Create output directory
    Config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # ==================== LOAD DATASET ====================
    print("\n### STEP 1: Loading Dataset ###")

    try:
        # Training dataset (5% of data)
        full_dataset = SEN12MSCRDataset(
            root_dir=Config.DATASET_ROOT,
            seasons=Config.SEASONS,
            s2_bands=Config.S2_BANDS,
            patch_size=Config.PATCH_SIZE,
            data_fraction=Config.DATA_FRACTION,
            min_cloud_fraction=Config.min_cloud_fraction,
            max_cloud_fraction=Config.max_cloud_fraction,
            cloud_mask_mode="gt_diff",
            random_seed=42
        )

        scene_to_indices = defaultdict(list)
        for idx, sample in enumerate(full_dataset.samples):
            # Extract season and scene_id from sample paths
            s2_clean_path = sample['s2_clean']
            # Path format: .../ROIsXXXX_season_s2/s2_X/patch.tif
            parts = s2_clean_path.parts
            season = parts[-3].split('_s2')[0]  # ROIsXXXX_season
            scene_id = parts[-2].split('_')[1]  # s2_X -> X
            scene_key = f"{season}_{scene_id}"
            scene_to_indices[scene_key].append(idx)

        # Split scenes 80/20
        scene_keys = list(scene_to_indices.keys())
        train_scenes, val_scenes = train_test_split(
            scene_keys,
            test_size=0.2,
            random_state=42
        )

        # Get indices for each split
        train_indices = [idx for scene in train_scenes for idx in scene_to_indices[scene]]
        val_indices = [idx for scene in val_scenes for idx in scene_to_indices[scene]]

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Total scenes:       {len(scene_keys)}")
        print(f"  Training scenes:    {len(train_scenes)} ({len(train_scenes) / len(scene_keys) * 100:.1f}%)")
        print(f"  Validation scenes:  {len(val_scenes)} ({len(val_scenes) / len(scene_keys) * 100:.1f}%)")
        print(f"  Training samples:   {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Bands:              {len(Config.S2_BANDS)}")

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nPlease ensure:")
        print("  1. Dataset extracted to ./sen12mscr_dataset/")
        print("  2. Directory structure is correct")
        print("  3. Run: python data_loader.py (option 1) to extract tar files")
        return

        # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS // 2,
        pin_memory=True
    )
    print("\nChecking sample statistics...")
    for i in range(min(3, len(full_dataset))):
        s1, s2_cloudy, s2_clean, cloud_mask = full_dataset[i]
        print(f"\nSample {i + 1}:")
        print(f"  S2 Cloudy - min: {s2_cloudy.min():.4f}, max: {s2_cloudy.max():.4f}, mean: {s2_cloudy.mean():.4f}")
        print(f"  S2 Clean  - min: {s2_clean.min():.4f}, max: {s2_clean.max():.4f}, mean: {s2_clean.mean():.4f}")
        print(f"  Cloud Mask - mean coverage: {cloud_mask.mean():.2%}")
    # Visualize dataset
    visualize_dataset_samples(full_dataset, Config.OUTPUT_DIR, n_samples=3)
    # ==================== TRAIN MODELS ====================
    print("\n### STEP 2: Training Models ###")

    trained_models = {}
    training_times = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for model_name in Config.MODELS:
        print(f"\n{'=' * 70}")
        print(f"Training {model_name}")
        print(f"{'=' * 70}")

        start_time = time.time()

        try:
            # Initialize trainer
            trainer = ModelTrainer(model_name, device=device)
            models, histories = trainer.train_kfold(
                dataset=train_dataset,
                k=Config.FOLDS,
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                lr=Config.LEARNING_RATE,
                patience=Config.PATIENCE,
                use_amp=Config.USE_AMP,
                num_workers=Config.NUM_WORKERS,
                persistent_workers=True,
                prefetch_factor=3
            )

            trained_models[model_name] = models[0]
            training_times[model_name] = time.time() - start_time

            print(f"\n✓ {model_name} training complete: {training_times[model_name] / 60:.2f} min")

        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not trained_models:
        print("\n✗ No models were trained successfully!")
        return

    # ==================== EVALUATE MODELS ====================
    print("\n### STEP 3: Evaluating Models ###")

    evaluator = ModelEvaluator(device=device)

    for model_name, model in trained_models.items():
        try:
            evaluator.evaluate_model(model, val_loader, model_name)
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            continue

    # ==================== GENERATE REPORTS ====================
    print("\n### STEP 4: Generating Comparison Reports ###")

    comparison_df = evaluator.generate_comparison_report(
        training_times,
        Config.OUTPUT_DIR
    )
    if comparison_df is None:
        print("\n✗ Skipping summary: No evaluation results available.")
        return

    # ==================== VISUALIZE PREDICTIONS ====================
    print("\n### STEP 5: Visualizing Predictions ###")

    visualize_predictions(
        trained_models,
        val_dataset,
        device,
        Config.OUTPUT_DIR,
        n_samples=5
    )

    # ==================== SAVE MODELS ====================
    if Config.SAVE_MODELS:
        print("\n### STEP 6: Saving Models ###")
        model_dir = Config.OUTPUT_DIR / 'saved_models'
        model_dir.mkdir(exist_ok=True)

        for model_name, model in trained_models.items():
            model_path = model_dir / f'{model_name}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved {model_name}: {model_path}")

    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nGenerated outputs in {Config.OUTPUT_DIR}:")
    print("  ✓ model_comparison.csv       - Detailed metrics comparison")
    print("  ✓ metrics_comparison.png     - All metrics visualized")
    print("  ✓ efficiency_analysis.png    - Training/inference efficiency")
    print("  ✓ predictions_comparison.png - Model predictions")
    print("  ✓ dataset_samples.png        - Dataset visualization")
    if Config.SAVE_MODELS:
        print("  ✓ saved_models/              - Trained model weights")

    print(f"\nTraining Summary:")
    for model_name, train_time in training_times.items():
            try:
                psnr = comparison_df.loc[model_name, 'PSNR_dB']
                throughput = comparison_df.loc[model_name, 'Throughput_img_s']
                print(f"  {model_name:.<15} {train_time / 60:>6.2f} min  │  {psnr:>6.2f} dB  │  {throughput:>6.2f} img/s")
            except KeyError:
                print(f"  {model_name:.<15} {train_time / 60:>6.2f} min  │  Metrics missing")

    print("=" * 70)
    print("\n✓ All done! Check the results directory for outputs.")


if __name__ == '__main__':
    main()

# NOTES
#
# funkcja straty ?
# 1 model do wykrywania i rekonstrukcji vs 2 modele (wykrywanie / rekonstrukcja) nauka na tylko czystych obrazach?
# Redukcja wymiarowosci pasm
#  Wykrywanie chmur ML
#  Testing self supervised learning (partly clear imaes, no ground truth)
# https://www.sciencedirect.com/science/article/pii/S2667393223000157