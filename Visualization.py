import numpy as np
import matplotlib.pyplot as plt
import random
import torch
def get_s1_viz(s1_band_tensor, speckle_filter="median", filter_size=3, gamma=0.7):
    """
    Make Sentinel-1 (SAR) easier to look at (visualization only).

    Args:
        s1_band_tensor: torch tensor [H, W] (already normalized to [0,1] in your pipeline)
        speckle_filter: "median", "gaussian", or None
        filter_size: kernel size (median) or sigma-ish control (gaussian via sigma=filter_size/2)
        gamma: <1 brightens mid-tones, >1 darkens

    Returns:
        2D numpy array [H, W] in [0,1] suitable for imshow(..., cmap="gray")
    """
    x = s1_band_tensor.detach().cpu().numpy().astype(np.float32)

    # 1) Speckle reduction
    if speckle_filter:
        try:
            from scipy.ndimage import median_filter, gaussian_filter
            if speckle_filter == "median":
                x = median_filter(x, size=filter_size)
            elif speckle_filter == "gaussian":
                sigma = max(0.5, filter_size / 2.0)
                x = gaussian_filter(x, sigma=sigma)
        except Exception:
            # If scipy isn't available for some reason, fall back to no filtering
            pass

    # 2) Robust contrast stretch (helps a lot for SAR)
    p2, p98 = np.percentile(x, [2, 98])
    x = (x - p2) / (p98 - p2 + 1e-8)
    x = np.clip(x, 0.0, 1.0)

    # 3) Gamma to lift mid-tones (structure becomes clearer)
    x = np.clip(x, 0.0, 1.0) ** gamma
    return x

def get_stretched_rgb(img_tensor):
    # RGB needs indices [3, 2, 1] for B04, B03, B02
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

def visualize_predictions(models, dataset, device, output_dir, n_samples=5):
    """Generate comprehensive comparison for all trained models across samples"""
    if not models:
        print("⚠️ No models available for visualization.")
        return

    print(f"\nGenerating all-model comparison ({n_samples} samples, {len(models) + 5} columns)...")

    # We need 5 columns for inputs/GT + 1 column for each model
    n_cols = 5 + len(models)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))

    # Ensure axes is 2D even if n_samples=1
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    sample_indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    #for i in range(min(n_samples, len(dataset))): < for static sample
    for i, idx in enumerate(sample_indices):
        s1, s2_cloudy, s2_clean, cloud_mask = dataset[idx]
        s1_viz = get_s1_viz(s1[0], speckle_filter="median", filter_size=5, gamma=0.7)
        model_input = torch.cat([s1, s2_cloudy], dim=0).unsqueeze(0).to(device)
        if s2_cloudy.shape[0] >= 4:
            # Band order: [B02=Blue, B03=Green, B04=Red, B08=NIR]
            # RGB needs: [Red, Green, Blue] = [B04, B03, B02] = [2, 1, 0]
            cloudy_rgb = s2_cloudy[[3, 2, 1]].permute(1, 2, 0).numpy()
            clean_rgb = s2_clean[[3, 2, 1]].permute(1, 2, 0).numpy()
        else:
            cloudy_rgb = s2_cloudy[:3].permute(1, 2, 0).numpy()
            clean_rgb = s2_clean[:3].permute(1, 2, 0).numpy()

        p2, p98 = np.percentile(cloudy_rgb, [2, 98])
        cloudy_stretched = np.clip((cloudy_rgb - p2) / (p98 - p2), 0, 1)
        clean_stretched = np.clip((clean_rgb - p2) / (p98 - p2), 0, 1)



        # --- Input Columns ---
        axes[i, 0].imshow(cloudy_stretched)
        axes[i, 0].set_title('1. Cloudy Input' if i == 0 else '')

        axes[i, 1].imshow(s1_viz, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('2. S1 (SAR)' if i == 0 else '')

        axes[i, 2].imshow(cloud_mask[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('3. Cloud Mask' if i == 0 else '')

        axes[i, 3].imshow(np.abs(cloudy_stretched - clean_stretched))
        axes[i, 3].set_title('4. Difference' if i == 0 else '')

        axes[i, 4].imshow(np.clip(clean_stretched, 0, 1))
        axes[i, 4].set_title('5. Ground Truth' if i == 0 else '')

        # --- Model Output Columns ---
        for j, (model_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                # Handle LSTM special shape if needed
                input_tensor = model_input.unsqueeze(1) if 'LSTM' in model_name else model_input

                pred = model(input_tensor)
                pred_img = pred[0].detach().cpu()  # [C, H, W]
                if pred_img.shape[0] >= 13:
                    rgb_indices = [3, 2, 1]  # B04, B03, B02
                    pred_rgb = pred_img[rgb_indices].permute(1, 2, 0).numpy()
                elif pred_img.shape[0] >= 3:
                    rgb_indices = [2, 1, 0]
                    pred_rgb = pred_img[rgb_indices].permute(1, 2, 0).numpy()
                else:
                    pred_rgb = pred_img[0].numpy()
                pred_stretched = np.clip((pred_rgb - p2) / (p98 - p2), 0, 1)
                col_idx = 5 + j
                axes[i, col_idx].imshow(pred_stretched)
                axes[i, col_idx].set_title(f'Output: {model_name}' if i == 0 else '')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plot_path = output_dir / 'all_models_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved multi-model comparison: {plot_path}")


def visualize_dataset_samples(dataset, output_dir, n_samples=3):
    """Visualize dataset samples"""
    print(f"\nVisualizing dataset samples...")

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    sample_indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    #for i in range(min(n_samples, len(dataset))): < for static samples
    for i, idx in enumerate(sample_indices):
        s1, s2_cloudy, s2_clean, cloud_mask = dataset[idx]
        if s2_cloudy.shape[0] >= 4:
            # Band order: [B02=Blue, B03=Green, B04=Red, B08=NIR]
            # RGB needs: [Red, Green, Blue] = [B04, B03, B02] = [2, 1, 0]
            cloudy_rgb = s2_cloudy[[3, 2, 1]].permute(1, 2, 0).numpy()
            clean_rgb = s2_clean[[3, 2, 1]].permute(1, 2, 0).numpy()
        else:
            cloudy_rgb = s2_cloudy[:3].permute(1, 2, 0).numpy()
            clean_rgb = s2_clean[:3].permute(1, 2, 0).numpy()

        p2, p98 = np.percentile(cloudy_rgb, [2, 98])
        cloudy_stretched = np.clip((cloudy_rgb - p2) / (p98 - p2), 0, 1)
        axes[i, 0].imshow(cloudy_stretched)
        axes[i, 0].set_title('Cloudy (2-98% stretch)')
        axes[i, 0].axis('off')

        # Clean with same stretching
        p2, p98 = np.percentile(clean_rgb, [2, 98])
        clean_stretched = np.clip((clean_rgb - p2) / (p98 - p2), 0, 1)
        axes[i, 1].imshow(clean_stretched)
        axes[i, 1].set_title('Clean (2-98% stretch)')
        axes[i, 1].axis('off')

        # Cloud mask
        axes[i, 2].imshow(cloud_mask[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Cloud Mask\n(coverage={cloud_mask.mean():.1%})')
        axes[i, 2].axis('off')

        # Difference (stretched)
        diff = np.abs(cloudy_stretched - clean_stretched)
        axes[i, 3].imshow(diff)
        axes[i, 3].set_title(f'Difference\n(mean={diff.mean():.3f})')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plot_path = output_dir / 'dataset_samples.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dataset samples: {plot_path}")