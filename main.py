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
import warnings
import time

from config import Config
from data_loader import SEN12MSCRDataset
from training_functions import ModelTrainer
from Visualization import visualize_dataset_samples, visualize_predictions

warnings.filterwarnings('ignore')

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Note: rasterio not installed. Install for real Sentinel-2 data: pip install rasterio")


# ==================== EVALUATION & COMPARISON ====================

class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device  = device
        self.results = {}

    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate single model with comprehensive metrics"""
        model.eval()

        all_preds      = []
        all_targets    = []
        inference_times = []

        print(f"\nEvaluating {model_name}...")

        with torch.no_grad():
            for batch_idx, (s1, s2_cloudy, s2_clean, cloud_mask) in enumerate(test_loader):
                if batch_idx >= 20:
                    break
                s1         = s1.to(self.device,        non_blocking=True)
                s2_cloudy  = s2_cloudy.to(self.device,  non_blocking=True)
                s2_clean   = s2_clean.to(self.device,   non_blocking=True)
                cloud_mask = cloud_mask.to(self.device, non_blocking=True)

                model_input = torch.cat([s1, s2_cloudy], dim=1)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                if model_name == 'RandomForest':
                    output = model.predict(s1, s2_cloudy, cloud_mask, device=self.device)
                elif model_name == 'Diffusion':
                    T              = 1000
                    betas          = torch.linspace(1e-4, 0.02, T).to(self.device)
                    alphas         = 1.0 - betas
                    alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=1e-5)
                
                    x_cond = torch.cat([s1, s2_cloudy, cloud_mask], dim=1)  # [B, 2+13+1=16, H, W]
                    x      = torch.randn_like(s2_cloudy)                     # [B, 13, H, W]
                
                    step_size = T // 10
                    for i in reversed(range(0, T, step_size)):
                        t_tensor   = torch.full((s2_cloudy.shape[0],), i,
                                                dtype=torch.long, device=self.device)
                        x_input    = torch.cat([x, x_cond], dim=1)          # [B, 29, H, W]
                        pred_noise = model(x_input, t_tensor)
                        alpha      = alphas[i]
                        acp        = alphas_cumprod[i]
                        x = (1.0 / alpha.sqrt()) * (
                            x - (1.0 - alpha) / (1.0 - acp).sqrt() * pred_noise
                        )
                        if i > 0:
                            x = x + betas[i].sqrt() * torch.randn_like(x)
                    output = x.clamp(0, 1)
                else:
                    output = model(model_input)

                inference_times.append(time.time() - start_time)
                all_preds.append(output.cpu().numpy())
                all_targets.append(s2_clean.cpu().numpy())

        preds   = np.concatenate(all_preds,   axis=0)
        targets = np.concatenate(all_targets, axis=0)

        metrics = self._calculate_metrics(targets, preds, inference_times)
        self.results[model_name] = metrics

        print(f"\n{model_name} Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"  {metric:.<30} {value:.6f}")

        return metrics

    def _calculate_metrics(self, targets, preds, inference_times):
        mse        = mean_squared_error(targets.flatten(), preds.flatten())
        mae        = mean_absolute_error(targets.flatten(), preds.flatten())
        rmse       = np.sqrt(mse)
        psnr       = np.mean([self._psnr(targets[i], preds[i]) for i in range(len(preds))])
        ssim       = self._ssim(targets, preds)
        avg_inf    = np.mean(inference_times)
        throughput = len(preds) / np.sum(inference_times)
        return {
            'MSE': mse, 'MAE': mae, 'RMSE': rmse,
            'PSNR_dB': psnr, 'SSIM': ssim,
            'Inference_Time_s': avg_inf, 'Throughput_img_s': throughput,
        }

    @staticmethod
    def _psnr(target, pred, max_val=1.0):
        mse = np.mean((target - pred) ** 2)
        return 100.0 if mse == 0 else 20 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def _ssim(targets, preds):
        c1, c2   = 0.01 ** 2, 0.03 ** 2
        mu_x, mu_y       = np.mean(targets), np.mean(preds)
        sigma_x, sigma_y = np.std(targets),  np.std(preds)
        sigma_xy = np.mean((targets - mu_x) * (preds - mu_y))
        return ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / (
               (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))

    def generate_comparison_report(self, training_times, output_dir):
        if not self.results:
            print("\n⚠️ No results found to generate comparison report.")
            return None

        df = pd.DataFrame(self.results).T
        df['Training_Time_min'] = [training_times.get(m, 0) / 60 for m in df.index]

        csv_path = output_dir / 'model_comparison.csv'
        df.to_csv(csv_path)
        print(f"\n✓ Saved comparison CSV: {csv_path}")

        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(df.to_string())
        print("=" * 70)

        self._plot_metrics_comparison(df, output_dir)
        self._plot_training_efficiency(df, output_dir)
        self._print_best_models(df)

        return df

    def _plot_metrics_comparison(self, df, output_dir):
        metrics = [c for c in df.columns if c != 'Model']
        colors  = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for metric, ax in zip(metrics, axes):
            df[metric].plot(kind='bar', ax=ax, color=colors[:len(df)])
            ax.set_title(f'Model Comparison: {metric}')
            ax.set_ylabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_efficiency(self, df, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for ax, xcol, xlabel in [
            (ax1, 'Training_Time_min', 'Training Time (min)'),
            (ax2, 'Inference_Time_s',  'Inference Time (ms)'),
        ]:
            x = df[xcol] * (1000 if xcol == 'Inference_Time_s' else 1)
            ax.scatter(x, df['PSNR_dB'], s=200, alpha=0.6,
                       c=range(len(df)), cmap='viridis')
            for model in df.index:
                ax.annotate(model, (x[model], df.loc[model, 'PSNR_dB']),
                            xytext=(5, 5), textcoords='offset points', fontsize=10)
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _print_best_models(self, df):
        print(f"\n{'='*70}")
        print("BEST MODEL ANALYSIS")
        print(f"{'='*70}")
        bq = df['PSNR_dB'].idxmax()
        bs = df['Throughput_img_s'].idxmax()
        bt = df['Training_Time_min'].idxmin()
        print(f"Best Quality (PSNR):  {bq:.<20} {df.loc[bq, 'PSNR_dB']:.2f} dB")
        print(f"Fastest Inference:    {bs:.<20} {df.loc[bs, 'Throughput_img_s']:.2f} img/s")
        print(f"Fastest Training:     {bt:.<20} {df.loc[bt, 'Training_Time_min']:.2f} min")
        print(f"{'='*70}")


# ==================== MAIN EXECUTION ====================
def run_pipeline(models_override, output_dir_override, train_loader, val_loader,
                 train_dataset, val_dataset, full_dataset, device):
    """Run training, evaluation, visualization and saving for a given model list."""
    output_dir_override.mkdir(exist_ok=True, parents=True)

    # ── STEP 2: Train Models ──────────────────────────────────────────
    print(f"\n### TRAINING: {models_override} → {output_dir_override} ###")
    trained_models = {}
    training_times = {}

    for model_name in models_override:
        print(f"\n{'='*70}")
        print(f"Training {model_name}  [loss={Config.LOSS_TYPE}]")
        print(f"{'='*70}")
        start = time.time()
        try:
            trainer = ModelTrainer(
                model_name,
                device    = device,
                loss_type = Config.LOSS_TYPE,
            )
            model, history = trainer.train(
                train_loader = train_loader,
                val_loader   = val_loader,
                epochs       = Config.EPOCHS,
                lr           = Config.LEARNING_RATE,
                patience     = Config.PATIENCE,
                use_amp      = Config.USE_AMP,
            )
            trained_models[model_name] = model
            training_times[model_name] = time.time() - start
            print(f"\n✓ {model_name} done: {training_times[model_name]/60:.2f} min")
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            import traceback; traceback.print_exc()

    if not trained_models:
        print("\n✗ No models trained successfully.")
        return

    # ── STEP 3: Evaluate ─────────────────────────────────────────────
    print("\n### STEP 3: Evaluating Models ###")
    evaluator = ModelEvaluator(device=device)
    for model_name, model in trained_models.items():
        try:
            evaluator.evaluate_model(model, val_loader, model_name)
        except Exception as e:
            print(f"\n✗ Evaluation error for {model_name}: {e}")

    # ── STEP 4: Reports ───────────────────────────────────────────────
    print("\n### STEP 4: Generating Reports ###")
    comparison_df = evaluator.generate_comparison_report(training_times, output_dir_override)
    if comparison_df is None:
        print("\n✗ No results to report.")
        return

    # ── STEP 5: Visualise ─────────────────────────────────────────────
    print("\n### STEP 5: Visualising Predictions ###")
    visualize_predictions(trained_models, val_dataset, device, output_dir_override, n_samples=5)

    # ── STEP 6: Save Models ───────────────────────────────────────────
    if Config.SAVE_MODELS:
        print("\n### STEP 6: Saving Models ###")
        model_dir = output_dir_override / 'saved_models'
        model_dir.mkdir(exist_ok=True)
        for model_name, model in trained_models.items():
            path = model_dir / f'{model_name}.pth'
            torch.save(model.state_dict(), path)
            print(f"✓ Saved {model_name}: {path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — {output_dir_override}")
    print("=" * 70)
    print(f"Loss type used: {Config.LOSS_TYPE}")
    for model_name, t in training_times.items():
        try:
            psnr = comparison_df.loc[model_name, 'PSNR_dB']
            thr  = comparison_df.loc[model_name, 'Throughput_img_s']
            print(f"  {model_name:.<15} {t/60:>6.2f} min  │  {psnr:>6.2f} dB  │  {thr:>6.2f} img/s")
        except KeyError:
            print(f"  {model_name:.<15} {t/60:>6.2f} min  │  metrics missing")
    print("=" * 70)


def main():
    # Validate and print config
    Config.validate()
    Config.summary()

    Config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # ── STEP 1: Load Dataset (shared across both pipelines) ───────────
    print("\n### STEP 1: Loading Dataset ###")
    try:
        full_dataset = SEN12MSCRDataset(
            root_dir            = Config.DATASET_ROOT,
            seasons             = Config.SEASONS,
            s2_bands            = Config.S2_BANDS,
            patch_size          = Config.PATCH_SIZE,
            data_fraction       = Config.DATA_FRACTION,
            min_cloud_fraction  = Config.MIN_CLOUD_FRACTION,
            max_cloud_fraction  = Config.MAX_CLOUD_FRACTION,
            cloud_mask_mode     = Config.CLOUD_MASK_MODE,
            gt_diff_weight      = Config.GT_DIFF_WEIGHT,
            cloud_threshold     = Config.FD_CLOUD_THRESHOLD,
            use_moist_check     = Config.FD_USE_MOIST_CHECK,
            shadow_as_cloud     = Config.FD_SHADOW_AS_CLOUD,
            random_seed         = 42,
        )

        scene_to_indices = defaultdict(list)
        for idx, sample in enumerate(full_dataset.samples):
            parts    = sample['s2_clean'].parts
            season   = parts[-3].split('_s2')[0]
            scene_id = parts[-2].split('_')[1]
            scene_to_indices[f"{season}_{scene_id}"].append(idx)

        scene_keys = list(scene_to_indices.keys())
        train_scenes, val_scenes = train_test_split(
            scene_keys, test_size=0.2, random_state=42)

        train_indices = [i for s in train_scenes for i in scene_to_indices[s]]
        val_indices   = [i for s in val_scenes   for i in scene_to_indices[s]]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset   = torch.utils.data.Subset(full_dataset, val_indices)

        print(f"\n✓ Dataset loaded")
        print(f"  Total scenes:      {len(scene_keys)}")
        print(f"  Train scenes:      {len(train_scenes)}  ({len(train_dataset)} samples)")
        print(f"  Val scenes:        {len(val_scenes)}  ({len(val_dataset)} samples)")

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback; traceback.print_exc()
        return

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS // 2, pin_memory=True,
    )

    print("\nSample statistics (first 3):")
    for i in range(min(3, len(full_dataset))):
        s1, s2_cloudy, s2_clean, cloud_mask = full_dataset[i]
        print(f"  [{i+1}] cloudy={s2_cloudy.mean():.4f}  clean={s2_clean.mean():.4f} "
              f" mask_cov={(cloud_mask > 0.25).float().mean():.1%}")

    visualize_dataset_samples(full_dataset, Config.OUTPUT_DIR, n_samples=3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ── Pipeline 1: Config models ─────────────────────────────────────
    run_pipeline(
        models_override  = Config.MODELS,
        output_dir_override = Config.OUTPUT_DIR,
        train_loader     = train_loader,
        val_loader       = val_loader,
        train_dataset    = train_dataset,
        val_dataset      = val_dataset,
        full_dataset     = full_dataset,
        device           = device,
    )

    # ── Pipeline 2: DSen2CR only ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING DSen2CR STANDALONE PIPELINE")
    print("=" * 70)
    run_pipeline(
        models_override     = ["DSen2CR"],
        output_dir_override = Path(str(Config.OUTPUT_DIR) + '_dsen2cr'),
        train_loader        = train_loader,
        val_loader          = val_loader,
        train_dataset       = train_dataset,
        val_dataset         = val_dataset,
        full_dataset        = full_dataset,
        device              = device,
    )


if __name__ == '__main__':
    main()
