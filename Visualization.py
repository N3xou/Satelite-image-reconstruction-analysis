import numpy as np
import matplotlib.pyplot as plt
import random
import torch


def get_s1_viz(s1_band_tensor, speckle_filter="median", filter_size=3, gamma=0.7):
    """Make Sentinel-1 (SAR) easier to look at (visualization only)."""
    x = s1_band_tensor.detach().cpu().numpy().astype(np.float32)

    if speckle_filter:
        try:
            from scipy.ndimage import median_filter, gaussian_filter
            if speckle_filter == "median":
                x = median_filter(x, size=filter_size)
            elif speckle_filter == "gaussian":
                x = gaussian_filter(x, sigma=max(0.5, filter_size / 2.0))
        except Exception:
            pass

    p2, p98 = np.percentile(x, [2, 98])
    x = np.clip((x - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    x = x ** gamma
    return x


def get_stretched_rgb(img_tensor):
    n_bands = img_tensor.shape[0]
    if n_bands >= 13:
        rgb_indices = [3, 2, 1]
    elif n_bands >= 3:
        rgb_indices = [2, 1, 0]
    else:
        return img_tensor[0].cpu().numpy()
    rgb = img_tensor[rgb_indices].permute(1, 2, 0).cpu().numpy()
    p2, p98 = np.percentile(rgb, [2, 98])
    return np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)


def _mask_coverage(cloud_mask_tensor, threshold: float = 0.25) -> float:
    """
    Estimate cloud coverage as the fraction of pixels above `threshold`.

    Uses a binary threshold on the soft mask rather than taking the raw mean,
    which was producing ~42-45% coverage regardless of actual cloud amount
    because the per-image normalisation in the old gt_diff compressed all
    values toward the centre of [0,1].

    threshold=0.25 corresponds to a mean absolute reflectance diff of ~0.075
    (750 DN in raw S2 units), which reliably separates thin cloud/haze from
    clear sky.
    """
    if isinstance(cloud_mask_tensor, torch.Tensor):
        mask = cloud_mask_tensor.numpy()
    else:
        mask = cloud_mask_tensor
    return float((mask > threshold).mean())


def visualize_predictions(models, dataset, device, output_dir, n_samples=5):
    """Generate comprehensive comparison for all trained models across samples."""
    if not models:
        print("⚠️ No models available for visualization.")
        return

    print(f"\nGenerating all-model comparison ({n_samples} samples, {len(models) + 5} columns)...")

    n_cols = 5 + len(models)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    sample_indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for i, idx in enumerate(sample_indices):
        s1, s2_cloudy, s2_clean, cloud_mask = dataset[idx]
        s1_viz = get_s1_viz(s1[0], speckle_filter="median", filter_size=5, gamma=0.7)

        model_input = torch.cat([s1, s2_cloudy], dim=0).unsqueeze(0).to(device)

        if s2_cloudy.shape[0] >= 4:
            cloudy_rgb = s2_cloudy[[3, 2, 1]].permute(1, 2, 0).numpy()
            clean_rgb  = s2_clean[[3, 2, 1]].permute(1, 2, 0).numpy()
        else:
            cloudy_rgb = s2_cloudy[:3].permute(1, 2, 0).numpy()
            clean_rgb  = s2_clean[:3].permute(1, 2, 0).numpy()

        p2, p98 = np.percentile(cloudy_rgb, [5, 95])
        cloudy_stretched = np.clip((cloudy_rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
        p2, p98 = np.percentile(clean_rgb, [5, 95])
        clean_stretched  = np.clip((clean_rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Binary coverage estimate (threshold-based, not raw mean)
        coverage = _mask_coverage(cloud_mask[0].numpy())

        axes[i, 0].imshow(cloudy_stretched)
        axes[i, 0].set_title('1. Cloudy Input' if i == 0 else '')

        axes[i, 1].imshow(s1_viz, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('2. S1 (SAR)' if i == 0 else '')

        axes[i, 2].imshow(cloud_mask[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('3. Cloud Mask' if i == 0 else '')
        axes[i, 2].set_xlabel(f'coverage={coverage:.1%}', fontsize=8)

        axes[i, 3].imshow(np.abs(cloudy_stretched - clean_stretched))
        diff_mean = np.abs(cloudy_stretched - clean_stretched).mean()
        axes[i, 3].set_title('4. Difference' if i == 0 else '')
        axes[i, 3].set_xlabel(f'mean={diff_mean:.3f}', fontsize=8)

        axes[i, 4].imshow(np.clip(clean_stretched, 0, 1))
        axes[i, 4].set_title('5. Ground Truth' if i == 0 else '')

        for j, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                input_tensor = model_input.unsqueeze(1) if 'LSTM' in model_name else model_input
                if 'Diffusion' in model_name:
                    bs          = model_input.shape[0]
                    n_s1        = s1.shape[0]
                    s1_t        = model_input[:, :n_s1]
                    s2_cloudy_t = model_input[:, n_s1:]
                    cm_t        = cloud_mask.unsqueeze(0).to(device)
                
                    # Simple 10-step reverse diffusion for visualization
                    T              = 1000
                    betas          = torch.linspace(1e-4, 0.02, T).to(device)
                    alphas         = 1.0 - betas
                    alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=1e-5)
                
                    x = torch.randn_like(s2_cloudy_t)  # start from pure noise
                    x_cond = torch.cat([s1_t, s2_cloudy_t, cm_t], dim=1)
                
                    step_size = T // 10  # 10 denoising steps
                    for i in reversed(range(0, T, step_size)):
                        t_tensor = torch.full((bs,), i, dtype=torch.long, device=device)
                        x_input  = torch.cat([x, x_cond], dim=1)
                        with torch.no_grad():
                            pred_noise = model(x_input, t_tensor)
                        # DDPM reverse step
                        alpha     = alphas[i]
                        acp       = alphas_cumprod[i]
                        x = (1.0 / alpha.sqrt()) * (
                            x - (1.0 - alpha) / (1.0 - acp).sqrt() * pred_noise
                        )
                        if i > 0:
                            x = x + betas[i].sqrt() * torch.randn_like(x)
                    pred = x.clamp(0, 1)
                else:
                    pred = model(input_tensor)
                pred_img = pred[0].detach().cpu()

                if pred_img.shape[0] >= 13:
                    pred_rgb = pred_img[[3, 2, 1]].permute(1, 2, 0).numpy()
                elif pred_img.shape[0] >= 3:
                    pred_rgb = pred_img[[2, 1, 0]].permute(1, 2, 0).numpy()
                else:
                    pred_rgb = pred_img[0].numpy()

                p2, p98 = np.percentile(pred_rgb, [5, 95])
                pred_stretched = np.clip((pred_rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
                axes[i, 5 + j].imshow(pred_stretched)
                axes[i, 5 + j].set_title(f'Output: {model_name}' if i == 0 else '')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plot_path = output_dir / 'all_models_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved multi-model comparison: {plot_path}")


def visualize_dataset_samples(dataset, output_dir, n_samples=3):
    """Visualize dataset samples with accurate cloud coverage reporting."""
    print(f"\nVisualizing dataset samples...")

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    sample_indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for i, idx in enumerate(sample_indices):
        s1, s2_cloudy, s2_clean, cloud_mask = dataset[idx]

        if s2_cloudy.shape[0] >= 4:
            cloudy_rgb = s2_cloudy[[3, 2, 1]].permute(1, 2, 0).numpy()
            clean_rgb  = s2_clean[[3, 2, 1]].permute(1, 2, 0).numpy()
        else:
            cloudy_rgb = s2_cloudy[:3].permute(1, 2, 0).numpy()
            clean_rgb  = s2_clean[:3].permute(1, 2, 0).numpy()

        p2, p98 = np.percentile(cloudy_rgb, [2, 98])
        cloudy_stretched = np.clip((cloudy_rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
        p2, p98 = np.percentile(clean_rgb, [2, 98])
        clean_stretched  = np.clip((clean_rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Use threshold-based binary estimate for displayed coverage
        coverage = _mask_coverage(cloud_mask[0].numpy())

        axes[i, 0].imshow(cloudy_stretched)
        axes[i, 0].set_title('Cloudy (2-98% stretch)')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(clean_stretched)
        axes[i, 1].set_title('Clean (2-98% stretch)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cloud_mask[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Cloud Mask\n(coverage={coverage:.1%})')
        axes[i, 2].axis('off')

        diff = np.abs(cloudy_stretched - clean_stretched)
        axes[i, 3].imshow(diff)
        axes[i, 3].set_title(f'Difference\n(mean={diff.mean():.3f})')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plot_path = output_dir / 'dataset_samples.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dataset samples: {plot_path}")
