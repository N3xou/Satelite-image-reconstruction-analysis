"""
Single Image Cloud Removal Inference
====================================
Load a single scene (S1 SAR + S2 cloudy) and generate cloud-free RGB output.

At inference time we have no paired clean image, so cloud masks must be
computed without ground-truth.  "feature_detector" is the recommended mode
because it uses the full DSen2-CR reference algorithm (progressive min-score
elimination, snow guard, morphological smoothing) without needing s2_clean.

Usage:
    python inference.py --s1 path/to/s1.tif --s2-cloudy path/to/s2_cloudy.tif \
                        --model path/to/model.pth --model-type UNet
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import rasterio
from Models import UNet, SimpleCNN, Generator, DSen2CR
from data_loader import CloudMaskComputer


class SingleImageInference:
    """Cloud removal inference on a single image"""

    def __init__(self, model_path, model_type='UNet', device=None,
                 n_s2_bands=13, cloud_mask_mode='feature_detector',
                 cloud_threshold=0.35, use_moist_check=False,
                 shadow_as_cloud=False):
        """
        Parameters
        ----------
        model_path        : Path to trained model .pth file.
        model_type        : 'UNet' | 'SimpleCNN' | 'GAN' | 'DSen2CR'
        device            : 'cuda' | 'cpu' (auto-detect if None)
        n_s2_bands        : Number of S2 bands the model was trained with.
                            Must match the checkpoint (default 13).
        cloud_mask_mode   : Cloud mask algorithm to use.
                            'feature_detector' (default) — DSen2-CR reference
                            algorithm; no clean image needed.
                            'spectral' — simpler fallback.
                            'sar_optical' — uses SAR backscatter.
                            Note: 'gt_diff', 'gt_threshold', and 'combined'
                            require the clean image and cannot be used here.
        cloud_threshold   : Binarisation threshold for coverage estimation.
        use_moist_check   : Enable NDMI moisture check in feature_detector.
        shadow_as_cloud   : Count detected shadows as cloud in the mask.
        """
        self.device      = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type  = model_type
        self.n_s2_bands  = n_s2_bands

        # Validate that the chosen mode doesn't need ground truth
        _gt_modes = {'gt_diff', 'gt_threshold', 'combined'}
        if cloud_mask_mode in _gt_modes:
            raise ValueError(
                f"cloud_mask_mode='{cloud_mask_mode}' requires a paired clean "
                f"image and cannot be used at inference time. "
                f"Choose 'feature_detector', 'spectral', or 'sar_optical'."
            )

        self.mask_computer = CloudMaskComputer(
            mode            = cloud_mask_mode,
            cloud_threshold = cloud_threshold,
            use_moist_check = use_moist_check,
            shadow_as_cloud = shadow_as_cloud,
        )

        # s2_bands list: 1-based indices for the bands that were loaded
        # We assume contiguous 1..n_s2_bands unless the user passes otherwise.
        self.s2_bands = list(range(1, n_s2_bands + 1))

        print(f"Loading {model_type} model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        print(f"  Cloud mask mode: {cloud_mask_mode}")

    # ------------------------------------------------------------------

    def _load_model(self, model_path):
        """Load trained model from checkpoint."""
        in_channels  = 2 + self.n_s2_bands   # S1(2) + S2(n)
        out_channels = self.n_s2_bands

        if self.model_type == 'UNet':
            model = UNet(in_channels=in_channels, out_channels=out_channels)
        elif self.model_type == 'SimpleCNN':
            model = SimpleCNN(in_channels=in_channels, out_channels=out_channels)
        elif self.model_type == 'GAN':
            model = Generator(in_channels=in_channels, out_channels=out_channels)
        elif self.model_type == 'DSen2CR':
            model = DSen2CR(in_channels=in_channels, out_channels=out_channels)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    # ------------------------------------------------------------------

    @staticmethod
    def load_sentinel2(s2_path, bands=None):
        """
        Load Sentinel-2 image.

        Parameters
        ----------
        s2_path : Path to .tif file
        bands   : 1-based band indices to read; None = all bands

        Returns
        -------
        numpy array [C, H, W] normalised to [0, 1]
        """
        print(f"Loading S2 image: {s2_path}")
        with rasterio.open(s2_path) as src:
            img = (src.read(bands) if bands else src.read()).astype(np.float32)

            print(f"  Shape: {img.shape}")
            print(f"  Raw range: [{img.min():.2f}, {img.max():.2f}]")

            if img.max() > 100:
                img = img / 10000.0
                print("  Normalised by 10000")
            elif img.max() > 2:
                img = np.clip(img / img.max(), 0, 1)
                print("  Normalised by max value")

            img = np.clip(img, 0, 1)
            print(f"  Final range: [{img.min():.4f}, {img.max():.4f}]")
            return img

    @staticmethod
    def load_sentinel1(s1_path):
        """
        Load Sentinel-1 SAR image.

        Returns
        -------
        numpy array [2, H, W] (VV, VH) normalised to [0, 1]
        """
        print(f"Loading S1 SAR: {s1_path}")
        with rasterio.open(s1_path) as src:
            img = src.read().astype(np.float32)
            img = np.clip(img, -25, 0)
            img = (img + 25) / 25
            print(f"  Shape: {img.shape}")
            print(f"  Range: [{img.min():.4f}, {img.max():.4f}]")
            return img

    # ------------------------------------------------------------------

    def compute_cloud_mask(self, s1, s2_cloudy):
        """
        Compute cloud mask at inference time (no clean image available).

        Uses whichever inference-safe mode was configured at construction.
        The mask is returned as numpy [1, H, W] in [0, 1].
        """
        mask = self.mask_computer.compute(
            s2_cloudy = s2_cloudy,
            s2_clean  = None,      # not available at inference
            s1        = s1,
            s2_bands  = self.s2_bands,
        )
        coverage = float((mask > 0.25).mean())
        print(f"  Cloud coverage (>0.25): {coverage:.1%}")
        return mask

    # ------------------------------------------------------------------

    def predict(self, s1, s2_cloudy):
        """
        Run cloud removal on a single image.

        Parameters
        ----------
        s1        : numpy [2, H, W] normalised SAR
        s2_cloudy : numpy [C, H, W] normalised S2

        Returns
        -------
        s2_clean  : numpy [C, H, W]
        """
        print("\nRunning prediction...")

        # Build the same input as training: cat([s1, s2_cloudy])
        model_input = np.concatenate([s1, s2_cloudy], axis=0)     # [2+C, H, W]
        tensor_in   = torch.from_numpy(model_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            s2_clean = self.model(tensor_in)

        s2_clean = s2_clean[0].cpu().numpy()
        print(f"✓ Prediction complete  shape={s2_clean.shape}  "
              f"range=[{s2_clean.min():.4f}, {s2_clean.max():.4f}]")
        return s2_clean

    # ------------------------------------------------------------------

    @staticmethod
    def to_rgb(s2_image, bands_order='RGB', stretch_percentile=98):
        """
        Convert multi-band S2 to display RGB.

        Assumes input band order [B02=Blue, B03=Green, B04=Red, B08=NIR, …]
        matching the pipeline's default loading order.
        """
        if bands_order == 'RGB':
            rgb = s2_image[[2, 1, 0]]   # R=B04, G=B03, B=B02 → indices 2,1,0
        elif bands_order == 'NIR':
            rgb = s2_image[[3, 2, 1]]   # NIR-R-G false colour
        else:
            raise ValueError(f"Unknown bands_order: {bands_order}")

        rgb = rgb.transpose(1, 2, 0)

        if stretch_percentile:
            p_lo  = 100 - stretch_percentile
            p2, p98 = np.percentile(rgb, [p_lo, stretch_percentile])
            rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

        return rgb

    # ------------------------------------------------------------------

    def process_and_save(self, s2_cloudy_path, output_dir, s1_path=None,
                         save_formats=None, stretch_percentile=98):
        """
        Complete pipeline: load → compute mask → predict → visualise → save.

        Parameters
        ----------
        s2_cloudy_path    : Path to cloudy S2 .tif
        output_dir        : Directory to save outputs
        s1_path           : Optional path to S1 SAR .tif
        save_formats      : list from ['rgb', 'geotiff', 'npy'] (default: rgb + geotiff)
        stretch_percentile: Percentile for contrast stretching

        Returns
        -------
        dict mapping format name → output path
        """
        if save_formats is None:
            save_formats = ['rgb', 'geotiff']

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Load imagery
        s2_cloudy = self.load_sentinel2(s2_cloudy_path, bands=self.s2_bands)

        if s1_path:
            s1 = self.load_sentinel1(s1_path)
        else:
            # Dummy SAR if not available — model still runs but quality lower
            print("⚠ No S1 SAR provided. Using zero SAR (reduced quality).")
            H, W = s2_cloudy.shape[1:]
            s1   = np.zeros((2, H, W), dtype=np.float32)

        # Compute cloud mask (inference-safe — no clean image)
        cloud_mask = self.compute_cloud_mask(s1, s2_cloudy)

        # Predict
        s2_clean = self.predict(s1, s2_cloudy)

        outputs = {}

        # --- RGB visualisation ---
        if 'rgb' in save_formats or 'png' in save_formats:
            cloudy_rgb = self.to_rgb(s2_cloudy, stretch_percentile=stretch_percentile)
            clean_rgb  = self.to_rgb(s2_clean,  stretch_percentile=stretch_percentile)
            mask_disp  = cloud_mask[0]   # [H, W]

            fig, axes = plt.subplots(1, 4, figsize=(22, 6))

            axes[0].imshow(cloudy_rgb)
            axes[0].set_title('Input: Cloudy S2', fontsize=13, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(mask_disp, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f'Cloud Mask ({self.mask_computer.mode})', fontsize=13, fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(clean_rgb)
            axes[2].set_title(f'Output: Cloud-Free ({self.model_type})', fontsize=13, fontweight='bold')
            axes[2].axis('off')

            diff = np.abs(clean_rgb - cloudy_rgb)
            axes[3].imshow(diff)
            axes[3].set_title('Difference', fontsize=13, fontweight='bold')
            axes[3].axis('off')

            plt.tight_layout()
            rgb_path = output_dir / 'cloud_removal_result.png'
            plt.savefig(rgb_path, dpi=300, bbox_inches='tight')
            plt.close()
            outputs['rgb_comparison'] = rgb_path
            print(f"✓ Saved RGB comparison: {rgb_path}")

            clean_rgb_path = output_dir / 'clean_rgb.png'
            plt.imsave(clean_rgb_path, clean_rgb)
            outputs['clean_rgb'] = clean_rgb_path
            print(f"✓ Saved clean RGB: {clean_rgb_path}")

        # --- GeoTIFF ---
        if 'geotiff' in save_formats:
            geotiff_path = output_dir / 'clean_s2.tif'
            with rasterio.open(s2_cloudy_path) as src:
                meta = src.meta.copy()
                meta.update({'count': s2_clean.shape[0], 'dtype': 'float32'})
                with rasterio.open(geotiff_path, 'w', **meta) as dst:
                    dst.write(s2_clean.astype(np.float32))
            outputs['geotiff'] = geotiff_path
            print(f"✓ Saved GeoTIFF: {geotiff_path}")

        # --- NumPy ---
        if 'npy' in save_formats:
            npy_path = output_dir / 'clean_s2.npy'
            np.save(npy_path, s2_clean)
            outputs['npy'] = npy_path
            print(f"✓ Saved NumPy array: {npy_path}")

        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        for key, path in outputs.items():
            print(f"  {key}: {path.name}")

        return outputs


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description='Cloud removal on a single Sentinel-2 image'
    )
    parser.add_argument('--s2-cloudy', required=True,
                        help='Path to cloudy S2 .tif file')
    parser.add_argument('--s1', default=None,
                        help='Path to S1 SAR .tif file (optional)')
    parser.add_argument('--model', required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--model-type', default='UNet',
                        choices=['UNet', 'SimpleCNN', 'GAN', 'DSen2CR'],
                        help='Model architecture')
    parser.add_argument('--n-s2-bands', type=int, default=13,
                        help='Number of S2 bands the model was trained with')
    parser.add_argument('--cloud-mask-mode', default='feature_detector',
                        choices=['feature_detector', 'spectral', 'sar_optical'],
                        help='Cloud mask algorithm (gt_diff/combined require clean image)')
    parser.add_argument('--cloud-threshold', type=float, default=0.35,
                        help='Binarisation threshold for cloud coverage reporting')
    parser.add_argument('--use-moist-check', action='store_true',
                        help='Enable NDMI moisture check in feature_detector')
    parser.add_argument('--shadow-as-cloud', action='store_true',
                        help='Treat detected shadow pixels as cloud in the mask')
    parser.add_argument('--output-dir', default='./inference_output',
                        help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['rgb', 'geotiff'],
                        choices=['rgb', 'png', 'geotiff', 'npy'],
                        help='Output formats to save')
    parser.add_argument('--stretch', type=int, default=98,
                        help='Percentile for contrast stretching (0-100)')

    args = parser.parse_args()

    inferencer = SingleImageInference(
        model_path       = args.model,
        model_type       = args.model_type,
        n_s2_bands       = args.n_s2_bands,
        cloud_mask_mode  = args.cloud_mask_mode,
        cloud_threshold  = args.cloud_threshold,
        use_moist_check  = args.use_moist_check,
        shadow_as_cloud  = args.shadow_as_cloud,
    )

    inferencer.process_and_save(
        s2_cloudy_path    = args.s2_cloudy,
        output_dir        = args.output_dir,
        s1_path           = args.s1,
        save_formats      = args.formats,
        stretch_percentile = args.stretch,
    )

    print("\n✓ Done!")


def interactive_example():
    """Interactive mode — no command-line arguments needed."""
    print("=" * 70)
    print("INTERACTIVE SINGLE IMAGE CLOUD REMOVAL")
    print("=" * 70)

    s2_cloudy_path = input("\nPath to cloudy S2 image (.tif): ").strip()
    model_path     = input("Path to trained model (.pth): ").strip()
    model_type     = input("Model type (UNet/SimpleCNN/GAN/DSen2CR) [UNet]: ").strip() or 'UNet'
    n_s2_bands     = int(input("Number of S2 bands model was trained with [13]: ").strip() or '13')
    output_dir     = input("Output directory [./inference_output]: ").strip() or './inference_output'
    s1_path        = input("Path to S1 SAR (optional, press Enter to skip): ").strip() or None

    try:
        inferencer = SingleImageInference(
            model_path      = model_path,
            model_type      = model_type,
            n_s2_bands      = n_s2_bands,
            cloud_mask_mode = 'feature_detector',
        )

        inferencer.process_and_save(
            s2_cloudy_path    = s2_cloudy_path,
            output_dir        = output_dir,
            s1_path           = s1_path,
            save_formats      = ['rgb', 'geotiff'],
            stretch_percentile = 98,
        )

        print("\n✓ Success! Check output directory for results.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        interactive_example()