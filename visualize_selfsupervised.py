import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from SelfSupervisedTest import JointCloudModel, SelfSupervisedConfig
from data_loader import SEN12MSCRDataset


def to_rgb(s2_tensor, stretch_percentile=98):
    """
    Convert S2 tensor (13, H, W) to RGB (H, W, 3)
    Bands 4, 3, 2 are Red, Green, Blue (indices 3, 2, 1)
    """
    rgb = s2_tensor[[3, 2, 1]].permute(1, 2, 0).cpu().numpy()

    # Contrast stretching
    p_low, p_high = np.percentile(rgb, [100 - stretch_percentile, stretch_percentile])
    rgb = np.clip((rgb - p_low) / (p_high - p_low + 1e-8), 0, 1)
    return rgb


def visualize_reconstructions(model_path, dataset_root, n_samples=5, device='cuda'):
    # Setup configuration (must match training)
    config = SelfSupervisedConfig(use_s1=False, encoder_type='unet')

    # Load Model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = JointCloudModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Dataset (Validation samples)
    dataset = SEN12MSCRDataset(
        root_dir=dataset_root,
        patch_size=config.patch_size,
        data_fraction=0.1,  # Just a small portion to find samples
    )

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    cols = ['Cloudy Input (S2)', 'Pseudo Cloud Mask', 'Reconstructed (Clean)', 'Ground Truth (S2)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i, idx in enumerate(indices):
            s1, s2_cloudy, s2_clean, _ = dataset[idx]

            # Prepare input
            x = torch.cat([s2_cloudy], dim=0).unsqueeze(0).to(device)

            # Predict
            cloud_prob, reconstructed, _ = model(x)

            # Convert to RGB for visualization
            input_rgb = to_rgb(s2_cloudy)
            recon_rgb = to_rgb(reconstructed[0])
            target_rgb = to_rgb(s2_clean)
            mask = cloud_prob[0, 0].cpu().numpy()

            # Plotting
            axes[i, 0].imshow(input_rgb)
            axes[i, 1].imshow(mask, cmap='viridis', vmin=0, vmax=1)
            axes[i, 2].imshow(recon_rgb)
            axes[i, 3].imshow(target_rgb)

            for j in range(4):
                axes[i, j].axis('off')

    save_path = 'reconstruction_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visual comparison to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Point to your actual file paths
    MODEL_PATH = "results/selfsupervised/joint_model_best.pth"
    DATASET_ROOT = "./sen12mscr_dataset"

    if Path(MODEL_PATH).exists():
        visualize_reconstructions(MODEL_PATH, DATASET_ROOT)
    else:
        print(f"✗ Model not found at {MODEL_PATH}")