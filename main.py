"""
Satellite Image Cloud Removal - ML/DL Comparison Project
========================================================
Compares multiple approaches for reconstructing cloud-covered satellite imagery
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from synthetic_data_generator import SatelliteDatasetPreparer
import warnings

from training_functions import ModelTrainer
warnings.filterwarnings('ignore')

# Try to import rasterio (optional, needed only for real Sentinel-2 data)
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Note: rasterio not installed. Install for real Sentinel-2 data: pip install rasterio")

# ==================== 1. DATA ACQUISITION & PREPARATION ===================


class SatelliteDataset(Dataset):
    """PyTorch Dataset for satellite image pairs"""

    def __init__(self, clean_dir, cloudy_dir, transform=None, preload=True, cache_size=50):
        self.clean_dir = Path(clean_dir)
        self.cloudy_dir = Path(cloudy_dir)
        self.files = sorted(list(self.clean_dir.glob('*.npy')))
        self.transform = transform
        self.preload = preload
        self.cache = {}
        self.cache_size = cache_size

        # Preload all data to RAM if dataset is small (CPU bottleneck fix #1)
        if preload and len(self.files) <= cache_size:
            print(f"Preloading {len(self.files)} images to RAM...")
            for idx in range(len(self.files)):
                self.cache[idx] = self._load_from_disk(idx)
            print("Preloading complete!")

    def _load_from_disk(self, idx):
        """Load image pair from disk"""
        clean_path = self.files[idx]
        cloudy_path = self.cloudy_dir / clean_path.name

        clean = torch.from_numpy(np.load(clean_path))
        cloudy = torch.from_numpy(np.load(cloudy_path))

        if self.transform:
            clean = self.transform(clean)
            cloudy = self.transform(cloudy)

        return cloudy, clean

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Use cached data if available (CPU bottleneck fix #2)
        if idx in self.cache:
            return self.cache[idx]

        # Load from disk
        data = self._load_from_disk(idx)

        # Cache if within cache size
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data

        return data


# ==================== 2. DATA EXPLORATION & INSIGHTS ====================

class DataExplorer:
    """Generate insights about the satellite dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
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
        """Compute dataset statistics"""
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
    """Evaluate and compare model performance"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        self.timing_results = {}  # Store timing information

    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate single model"""
        import time

        model.eval()

        all_preds = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for cloudy, clean in test_loader:
                cloudy = cloudy.to(self.device, non_blocking=True)

                if 'LSTM' in model_name:
                    cloudy = cloudy.unsqueeze(1)

                # Measure inference time
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                output = model(cloudy)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                all_preds.append(output.cpu().numpy())
                all_targets.append(clean.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics
        mse = mean_squared_error(targets.flatten(), preds.flatten())
        mae = mean_absolute_error(targets.flatten(), preds.flatten())
        rmse = np.sqrt(mse)

        # PSNR
        psnr_scores = []
        for i in range(len(preds)):
            psnr = self._calculate_psnr(targets[i], preds[i])
            psnr_scores.append(psnr)

        psnr = np.mean(psnr_scores)

        # SSIM (simplified version)
        ssim = self._calculate_ssim(targets, preds)

        # Timing statistics
        avg_inference_time = np.mean(inference_times)
        total_inference_time = np.sum(inference_times)
        images_per_second = len(preds) / total_inference_time

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'PSNR': psnr,
            'SSIM': ssim,
            'Avg_Inference_Time': avg_inference_time,
            'Images_Per_Second': images_per_second
        }

        self.results[model_name] = metrics

        print(f"\n{model_name} Results:")
        print("-" * 40)
        for metric, value in metrics.items():
            if 'Time' in metric or 'Second' in metric:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value:.6f}")

        return metrics

    def _calculate_psnr(self, target, pred, max_val=1.0):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((target - pred) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))

    def _calculate_ssim(self, targets, preds):
        """Simplified SSIM calculation"""
        # Simplified version - full SSIM requires more complex windowing
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = np.mean(targets)
        mu_y = np.mean(preds)
        sigma_x = np.std(targets)
        sigma_y = np.std(preds)
        sigma_xy = np.mean((targets - mu_x) * (preds - mu_y))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))

        return ssim

    def compare_models(self, training_times=None):
        """Generate comparison visualizations including training time"""
        if not self.results:
            print("No results to compare")
            return

        df = pd.DataFrame(self.results).T

        # Add training times if provided
        if training_times:
            df['Training_Time_Minutes'] = [training_times.get(model, 0) / 60
                                           for model in df.index]

        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(df.to_string())

        # Plot comparison with training time
        n_metrics = len(df.columns)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        metrics = list(df.columns)
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

        for i, metric in enumerate(metrics):
            if i < len(axes):
                df[metric].plot(kind='bar', ax=axes[i], color=colors[:len(df)])
                axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create training time vs quality plot
        if training_times:
            fig, ax = plt.subplots(figsize=(10, 6))

            train_times = [training_times.get(model, 0) / 60 for model in df.index]
            psnr_values = df['PSNR'].values

            ax.scatter(train_times, psnr_values, s=200, alpha=0.6, c=range(len(df)), cmap='viridis')

            for i, model in enumerate(df.index):
                ax.annotate(model, (train_times[i], psnr_values[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')

            ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
            ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
            ax.set_title('Training Time vs Quality', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('time_vs_quality.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("\nTime vs Quality plot saved to: time_vs_quality.png")

        print("\nComparison saved to: model_comparison.png")

        # Best model analysis
        best_psnr_model = df['PSNR'].idxmax()
        best_speed_model = df['Images_Per_Second'].idxmax()

        print(f"\n{'='*60}")
        print("BEST MODEL ANALYSIS")
        print(f"{'='*60}")
        print(f"Best Quality (PSNR): {best_psnr_model} ({df.loc[best_psnr_model, 'PSNR']:.2f} dB)")
        print(f"Fastest Inference: {best_speed_model} ({df.loc[best_speed_model, 'Images_Per_Second']:.2f} img/s)")

        if training_times:
            fastest_training = df['Training_Time_Minutes'].idxmin()
            print(f"Fastest Training: {fastest_training} ({df.loc[fastest_training, 'Training_Time_Minutes']:.2f} min)")

        print(f"{'='*60}")

        return df


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution pipeline"""

    print("=" * 60)
    print("SATELLITE CLOUD REMOVAL - ML/DL COMPARISON PROJECT")
    print("=" * 60)

    # Choose dataset type
    print("\n### STEP 0: Dataset Selection ###")
    print("\nChoose your dataset:")
    print("1. Synthetic data (quick test, ~100 samples)")
    print("2. SEN12MS-CR dataset (real data, spring + winter)")

    choice = input("\nEnter choice (1-2, or press Enter for synthetic): ").strip()

    if choice == '2':
        # Use SEN12MS-CR dataset
        try:
            import sys
            # Try to import from sen12mscr_dataset.py
            try:
                from sen12mscr_dataset import SEN12MSCRDataset
            except ImportError:
                print("\n⚠ sen12mscr_dataset.py not found in current directory")
                print("Please ensure sen12mscr_dataset.py is in the same folder")
                choice = '1'

            if choice == '2':
                dataset_path = Path('./sen12mscr_dataset/organized')

                if not dataset_path.exists():
                    print("\n⚠ SEN12MS-CR organized dataset not found!")
                    print("\nSetup required:")
                    print("1. Place your tar files in current directory:")
                    print("   - ROIs1158_spring_s1.tar")
                    print("   - ROIs1158_spring_s2_cloudy.tar")
                    print("   - ROIs2017_winter_s1.tar")
                    print("   - ROIs2017_winter_s2_cloudy.tar")
                    print("\n2. Run: python sen12mscr_dataset.py")
                    print("3. Choose option 1: Extract and organize")
                    print("\nUsing synthetic data instead...")
                    choice = '1'
                else:
                    print(f"\n✓ Loading SEN12MS-CR dataset from {dataset_path}")

                    # Ask about bands
                    print("\nSelect bands to use:")
                    print("1. RGB + NIR (bands 2,3,4,8 - recommended, fastest)")
                    print("2. All 13 bands (complete Sentinel-2, slower)")
                    print("3. Custom bands")

                    band_choice = input("Enter choice (1-3): ").strip()

                    if band_choice == '2':
                        bands = list(range(1, 14))  # All bands
                        print("Using all 13 Sentinel-2 bands")
                    elif band_choice == '3':
                        bands_input = input("Enter band numbers (e.g., 2,3,4,8): ")
                        bands = [int(b.strip()) for b in bands_input.split(',')]
                        print(f"Using bands: {bands}")
                    else:
                        bands = [2, 3, 4, 8]  # Default: Blue, Green, Red, NIR
                        print("Using RGB + NIR (4 bands)")

                    # Load dataset
                    train_dataset = SEN12MSCRDataset(
                        dataset_path,
                        split='train',
                        bands=bands,
                        use_s1=False,  # Not using SAR for now
                        preload=False  # Don't preload large dataset
                    )

                    val_dataset = SEN12MSCRDataset(
                        dataset_path,
                        split='val',
                        bands=bands,
                        use_s1=False,
                        preload=False
                    )

                    # Create test dataset from validation
                    test_dataset = val_dataset

                    print(f"\nDataset loaded successfully!")
                    print(f"Training samples: {len(train_dataset)}")
                    print(f"Validation samples: {len(val_dataset)}")
                    print(f"Bands: {len(bands)}")

        except Exception as e:
            print(f"\n✗ Error loading SEN12MS-CR dataset: {e}")
            print("Using synthetic data instead...")
            choice = '1'

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Train Models with K-Fold
    print("\n### STEP 3: Model Training with K-Fold CV ###")

    # Adjust settings based on dataset
    if choice == '2':
        # SEN12MS-CR: Large dataset, adjust parameters
        print("\n⚙ Optimized settings for SEN12MS-CR dataset")
        k_folds = 3
        epochs = 30
        batch_size = 4
        patience = 7

        # Adjust input channels based on bands
        input_channels = len(bands) if choice == '2' else 4
        print(f"Model input channels: {input_channels}")
    else:
        # Synthetic: Small dataset
        k_folds = 3
        epochs = 30
        batch_size = 4
        patience = 5
        input_channels = 4

    # Ask which models to train
    print("\nWhich models would you like to train?")
    print("1. All models (SimpleCNN, UNet, GAN, LSTM, Diffusion)")
    print("2. Fast models only (SimpleCNN, UNet)")
    print("3. Best models only (UNet, GAN)")
    print("4. Single model (choose one)")

    model_choice = input("Enter choice (1-4): ").strip()

    if model_choice == '2':
        model_types = ['SimpleCNN', 'UNet']
    elif model_choice == '3':
        model_types = ['UNet', 'GAN']
    elif model_choice == '4':
        print("\nAvailable models:")
        print("1. SimpleCNN  2. UNet  3. GAN  4. LSTM  5. Diffusion")
        single = input("Enter number: ").strip()
        model_map = {'1': 'SimpleCNN', '2': 'UNet', '3': 'GAN',
                     '4': 'LSTM', '5': 'Diffusion'}
        model_types = [model_map.get(single, 'UNet')]
    else:
        model_types = ['SimpleCNN', 'UNet', 'GAN', 'LSTM', 'Diffusion']

    print(f"\nTraining models: {', '.join(model_types)}")

    trained_models = {}
    training_times = {}

    for model_type in model_types:
        trainer = ModelTrainer(model_type)
        models, histories = trainer.train_kfold(
            train_dataset,
            k=k_folds,
            epochs=epochs,
            batch_size=batch_size,
            lr=0.001,
            patience=patience,
            use_amp=True,
            num_workers=6,
            persistent_workers=True,
            prefetch_factor=4
        )
        trained_models[model_type] = models[0]
        training_times[model_type] = trainer.training_time

    # Evaluate and Compare
    print("\n### STEP 4: Model Evaluation and Comparison ###")

    evaluator = ModelEvaluator()

    for model_name, model in trained_models.items():
        evaluator.evaluate_model(model, test_loader, model_name)

    comparison_df = evaluator.compare_models(training_times=training_times)

    # Visualize Predictions
    print("\n### STEP 5: Visualizing Predictions ###")
    visualize_predictions(trained_models, test_dataset)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("- model_comparison.png")
    print("- time_vs_quality.png")
    print("- predictions_comparison.png")

    # Dataset-specific notes
    if choice == '2':
        print("\n✓ Trained on SEN12MS-CR dataset (real Sentinel-2 data)")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")
        print(f"  - Spectral bands: {len(bands)}")
        print(f"  - Seasons: Spring + Winter")
    else:
        print("\n📝 Note: You used synthetic data. For better results:")
        print("   1. Download SEN12MS-CR dataset from TUM")
        print("   2. Extract tar files to current directory")
        print("   3. Run: python sen12mscr_dataset.py (option 1)")
        print("   4. Run this script again and choose option 2")


def visualize_predictions(models, test_dataset, n_samples=3):
    """Visualize predictions from all models"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fig, axes = plt.subplots(n_samples, len(models) + 2, figsize=(20, 4*n_samples))

    for i in range(n_samples):
        cloudy, clean = test_dataset[i]
        cloudy_input = cloudy.unsqueeze(0).to(device)

        # Display cloudy input
        cloudy_rgb = cloudy[:3].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(cloudy_rgb, 0, 1))
        axes[i, 0].set_title('Cloudy Input')
        axes[i, 0].axis('off')

        # Display clean target
        clean_rgb = clean[:3].permute(1, 2, 0).numpy()
        axes[i, 1].imshow(np.clip(clean_rgb, 0, 1))
        axes[i, 1].set_title('Clean Target')
        axes[i, 1].axis('off')

        # Display predictions from each model
        for j, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                if 'LSTM' in model_name:
                    cloudy_input_lstm = cloudy_input.unsqueeze(1)
                    pred = model(cloudy_input_lstm)
                else:
                    pred = model(cloudy_input)

                pred_rgb = pred[0, :3].cpu().permute(1, 2, 0).numpy()
                axes[i, j + 2].imshow(np.clip(pred_rgb, 0, 1))
                axes[i, j + 2].set_title(f'{model_name} Output')
                axes[i, j + 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Predictions visualization saved to: predictions_comparison.png")


# ==================== ADDITIONAL UTILITIES ====================

class ExperimentLogger:
    """Log experiment results for reproducibility"""

    def __init__(self, log_file='experiment_log.txt'):
        self.log_file = log_file

    def log_experiment(self, config, results):
        """Log experiment configuration and results"""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"Experiment Date: {pd.Timestamp.now()}\n")
            f.write("\nConfiguration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nResults:\n")
            for model, metrics in results.items():
                f.write(f"\n{model}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.6f}\n")

        print(f"Experiment logged to: {self.log_file}")


def save_models(models, save_dir='./saved_models'):
    """Save trained models"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for model_name, model in models.items():
        path = save_dir / f'{model_name}.pth'
        torch.save(model.state_dict(), path)
        print(f"Saved {model_name} to {path}")





# ==================== ADVANCED FEATURES ====================

class CloudDetector:
    """Detect and mask clouds in images"""

    @staticmethod
    def detect_clouds(image, threshold=0.7):
        """Simple cloud detection based on brightness"""
        # Average across spectral bands
        brightness = image.mean(axis=0)
        cloud_mask = brightness > threshold
        return cloud_mask

    @staticmethod
    def create_cloud_free_composite(image_stack):
        """Create cloud-free composite from time series"""
        # Use median of non-cloudy pixels across time
        masks = [CloudDetector.detect_clouds(img) for img in image_stack]

        composite = np.zeros_like(image_stack[0])
        for band in range(image_stack[0].shape[0]):
            band_stack = np.array([img[band] for img in image_stack])
            # Mask cloudy pixels
            for i, mask in enumerate(masks):
                band_stack[i][mask] = np.nan
            # Take median
            composite[band] = np.nanmedian(band_stack, axis=0)

        return composite


class TemporalDataGenerator:
    """Generate temporal sequences for LSTM training"""

    def __init__(self, base_dataset, sequence_length=3):
        self.base_dataset = base_dataset
        self.sequence_length = sequence_length

    def create_sequences(self):
        """Create temporal sequences from dataset"""
        sequences = []

        for i in range(len(self.base_dataset) - self.sequence_length):
            seq_cloudy = []
            for j in range(self.sequence_length):
                cloudy, _ = self.base_dataset[i + j]
                seq_cloudy.append(cloudy)

            # Target is the clean version of last image
            _, clean = self.base_dataset[i + self.sequence_length - 1]

            sequences.append((torch.stack(seq_cloudy), clean))

        return sequences


# ==================== USAGE EXAMPLES ====================

def quick_start_example():
    """Quick start example with minimal dataset"""
    print("\n### QUICK START EXAMPLE ###\n")

    # Create small dataset
    preparer = SatelliteDatasetPreparer('./quick_test')
    clean_dir, cloudy_dir = preparer.create_synthetic_dataset(n_samples=20)

    # Load dataset
    dataset = SatelliteDataset(clean_dir, cloudy_dir)

    # Train single model
    trainer = ModelTrainer('SimpleCNN')
    models, _ = trainer.train_kfold(dataset, k=2, epochs=5, batch_size=4)

    # Quick evaluation
    test_loader = DataLoader(dataset, batch_size=4)
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(models[0], test_loader, 'SimpleCNN')

    print("\nQuick start completed!")


def advanced_example():
    """Advanced example with custom configurations"""
    print("\n### ADVANCED EXAMPLE ###\n")

    # Custom configuration
    config = {
        'n_samples': 200,
        'img_size': (512, 512),
        'n_bands': 13,  # Full Sentinel-2 bands
        'k_folds': 3,
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.0001
    }

    print("Advanced configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nTo run advanced example:")
    print("1. Increase dataset size and image resolution")
    print("2. Use all 13 Sentinel-2 bands")
    print("3. Train longer with more folds")
    print("4. Implement data augmentation")
    print("5. Use learning rate scheduling")
    print("6. Add early stopping")


if __name__ == '__main__':
    # Run main pipeline
    main()

    # Uncomment for quick start or advanced examples
    #quick_start_example()
    # advanced_example()

    """
    
    
    
    
    Validation methods . pixel similiarity error, brightness,contrast,
    
    autotune vs no autotune
    cpu/ram bottleneck

    2. Implementing Sentinel-2 data:
        - 
    
    3. Model improvements:
       - Add attention mechanisms
       - Use perceptual loss functions
       - Add temporal consistency constraints
    
    4. Evaluation enhancements:
       - Calculate per-band metrics
       - Add cloud coverage stratified analysis
       - Implement visual quality metrics (FID, LPIPS)
       - Compare with traditional methods (linear interpolation)

    """