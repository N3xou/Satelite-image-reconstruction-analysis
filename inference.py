"""
Single Image Cloud Removal Inference
====================================
Load a single scene (S1 SAR + S2 cloudy) and generate cloud-free RGB output

Usage:
    python inference_single_image.py --s1 path/to/s1.tif --s2-cloudy path/to/s2_cloudy.tif --model path/to/model.pth
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import rasterio
from Models import UNet, SimpleCNN, Generator


class SingleImageInference:
    """Cloud removal inference on a single image"""

    def __init__(self, model_path, model_type='UNet', device=None):
        """
        Args:
            model_path: Path to trained model (.pth file)
            model_type: 'UNet', 'SimpleCNN', or 'GAN'
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        # Load model
        print(f"Loading {model_type} model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")

    def _load_model(self, model_path):
        """Load trained model"""
        # Infer number of channels from first layer
        # For now, assume 4 channels (RGB + NIR)
        n_channels = 4

        if self.model_type == 'UNet':
            model = UNet(in_channels=n_channels, out_channels=n_channels)
        elif self.model_type == 'SimpleCNN':
            model = SimpleCNN(in_channels=n_channels, out_channels=n_channels)
        elif self.model_type == 'GAN':
            model = Generator(in_channels=n_channels, out_channels=n_channels)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model

    @staticmethod
    def load_sentinel2(s2_path, bands=[2, 3, 4, 8]):
        """
        Load Sentinel-2 image

        Args:
            s2_path: Path to .tif file
            bands: Band indices to read (1-13)

        Returns:
            numpy array [bands, H, W], normalized to [0, 1]
        """
        print(f"Loading S2 image: {s2_path}")

        with rasterio.open(s2_path) as src:
            # Read specified bands
            img = src.read(bands).astype(np.float32)

            print(f"  Shape: {img.shape}")
            print(f"  Raw range: [{img.min():.2f}, {img.max():.2f}]")

            # Normalize
            # Check if already normalized (max < 2) or needs scaling (max > 100)
            if img.max() > 100:
                # Likely digital numbers (0-10000)
                img = img / 10000.0
                print(f"  Normalized by 10000")
            elif img.max() > 2:
                # Likely in wrong units, scale to reasonable range
                img = np.clip(img / img.max(), 0, 1)
                print(f"  Normalized by max value")

            img = np.clip(img, 0, 1)
            print(f"  Final range: [{img.min():.4f}, {img.max():.4f}]")

            return img

    @staticmethod
    def load_sentinel1(s1_path):
        """
        Load Sentinel-1 SAR image (optional, not used for inference)

        Returns:
            numpy array [2, H, W] (VV, VH), normalized to [0, 1]
        """
        print(f"Loading S1 SAR: {s1_path}")

        with rasterio.open(s1_path) as src:
            img = src.read().astype(np.float32)

            # S1 is in dB, typical range: -25 to 0
            img = np.clip(img, -25, 0)
            img = (img + 25) / 25  # Normalize to [0, 1]

            print(f"  Shape: {img.shape}")
            print(f"  Range: [{img.min():.4f}, {img.max():.4f}]")

            return img

    def predict(self, s2_cloudy):
        """
        Run cloud removal prediction

        Args:
            s2_cloudy: numpy array [bands, H, W] or torch tensor

        Returns:
            s2_clean: numpy array [bands, H, W]
        """
        print("\nRunning prediction...")

        # Convert to tensor if needed
        if isinstance(s2_cloudy, np.ndarray):
            s2_cloudy = torch.from_numpy(s2_cloudy)

        # Add batch dimension
        s2_cloudy = s2_cloudy.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            s2_clean = self.model(s2_cloudy)

        # Remove batch dimension and convert to numpy
        s2_clean = s2_clean[0].cpu().numpy()

        print(f"✓ Prediction complete")
        print(f"  Output shape: {s2_clean.shape}")
        print(f"  Output range: [{s2_clean.min():.4f}, {s2_clean.max():.4f}]")

        return s2_clean

    @staticmethod
    def to_rgb(s2_image, bands_order='RGB', stretch_percentile=98):
        """
        Convert multi-band S2 to RGB

        Args:
            s2_image: numpy array [bands, H, W]
                     Assumes bands = [B02=Blue, B03=Green, B04=Red, B08=NIR]
            bands_order: 'RGB' (natural color) or 'NIR' (false color NIR-R-G)
            stretch_percentile: Percentile for contrast stretching

        Returns:
            RGB image [H, W, 3] in range [0, 1]
        """
        if bands_order == 'RGB':
            # Natural color: Red, Green, Blue = B04, B03, B02 = indices 2, 1, 0
            rgb = s2_image[[2, 1, 0]]
        elif bands_order == 'NIR':
            # False color: NIR, Red, Green = B08, B04, B03 = indices 3, 2, 1
            rgb = s2_image[[3, 2, 1]]
        else:
            raise ValueError(f"Unknown bands_order: {bands_order}")

        # Transpose to [H, W, 3]
        rgb = rgb.transpose(1, 2, 0)

        # Contrast stretching
        if stretch_percentile:
            p_low = 100 - stretch_percentile
            p_high = stretch_percentile
            p2, p98 = np.percentile(rgb, [p_low, p_high])
            rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

        return rgb

    def process_and_save(self, s2_cloudy_path, output_dir, s1_path=None,
                         save_formats=['rgb', 'geotiff'], stretch_percentile=98):
        """
        Complete pipeline: load, predict, visualize, save

        Args:
            s2_cloudy_path: Path to cloudy S2 image
            output_dir: Directory to save outputs
            s1_path: Optional path to S1 SAR image
            save_formats: List of formats ['rgb', 'geotiff', 'npy']
            stretch_percentile: Percentile for RGB contrast stretching

        Returns:
            Dictionary with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Load images
        s2_cloudy = self.load_sentinel2(s2_cloudy_path)

        if s1_path:
            s1 = self.load_sentinel1(s1_path)

        # Predict
        s2_clean = self.predict(s2_cloudy)

        # Generate outputs
        outputs = {}

        # 1. RGB visualization
        if 'rgb' in save_formats or 'png' in save_formats:
            cloudy_rgb = self.to_rgb(s2_cloudy, stretch_percentile=stretch_percentile)
            clean_rgb = self.to_rgb(s2_clean, stretch_percentile=stretch_percentile)

            # Save comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(cloudy_rgb)
            axes[0].set_title('Input: Cloudy S2', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(clean_rgb)
            axes[1].set_title(f'Output: Cloud-Free ({self.model_type})', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            diff = np.abs(clean_rgb - cloudy_rgb)
            axes[2].imshow(diff)
            axes[2].set_title('Difference', fontsize=14, fontweight='bold')
            axes[2].axis('off')

            plt.tight_layout()
            rgb_path = output_dir / 'cloud_removal_result.png'
            plt.savefig(rgb_path, dpi=300, bbox_inches='tight')
            plt.close()
            outputs['rgb_comparison'] = rgb_path
            print(f"✓ Saved RGB comparison: {rgb_path}")

            # Save individual images
            clean_rgb_path = output_dir / 'clean_rgb.png'
            plt.imsave(clean_rgb_path, clean_rgb)
            outputs['clean_rgb'] = clean_rgb_path
            print(f"✓ Saved clean RGB: {clean_rgb_path}")

        # 2. GeoTIFF with georeferencing
        if 'geotiff' in save_formats:
            geotiff_path = output_dir / 'clean_s2.tif'

            # Copy metadata from input
            with rasterio.open(s2_cloudy_path) as src:
                meta = src.meta.copy()
                meta.update({
                    'count': s2_clean.shape[0],
                    'dtype': 'float32'
                })

                with rasterio.open(geotiff_path, 'w', **meta) as dst:
                    dst.write(s2_clean.astype(np.float32))

            outputs['geotiff'] = geotiff_path
            print(f"✓ Saved GeoTIFF: {geotiff_path}")

        # 3. NumPy array
        if 'npy' in save_formats:
            npy_path = output_dir / 'clean_s2.npy'
            np.save(npy_path, s2_clean)
            outputs['npy'] = npy_path
            print(f"✓ Saved NumPy array: {npy_path}")

        print(f"\n{'=' * 70}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Output directory: {output_dir}")
        for key, path in outputs.items():
            print(f"  {key}: {path.name}")

        return outputs


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Cloud removal on single Sentinel-2 image')

    parser.add_argument('--s2-cloudy', required=True, help='Path to cloudy S2 .tif file')
    parser.add_argument('--s1', default=None, help='Path to S1 SAR .tif file (optional)')
    parser.add_argument('--model', required=True, help='Path to trained model .pth file')
    parser.add_argument('--model-type', default='UNet', choices=['UNet', 'SimpleCNN', 'GAN'],
                        help='Model architecture')
    parser.add_argument('--output-dir', default='./inference_output', help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['rgb', 'geotiff'],
                        choices=['rgb', 'png', 'geotiff', 'npy'],
                        help='Output formats')
    parser.add_argument('--stretch', type=int, default=98,
                        help='Percentile for contrast stretching (0-100)')

    args = parser.parse_args()

    # Run inference
    inferencer = SingleImageInference(
        model_path=args.model,
        model_type=args.model_type
    )

    outputs = inferencer.process_and_save(
        s2_cloudy_path=args.s2_cloudy,
        output_dir=args.output_dir,
        s1_path=args.s1,
        save_formats=args.formats,
        stretch_percentile=args.stretch
    )

    print("\n✓ Done!")


def interactive_example():
    """Interactive example without command line"""
    print("=" * 70)
    print("INTERACTIVE SINGLE IMAGE CLOUD REMOVAL")
    print("=" * 70)

    # Get inputs
    s2_cloudy_path = input("\nPath to cloudy S2 image (.tif): ").strip()
    model_path = input("Path to trained model (.pth): ").strip()

    model_type = input("Model type (UNet/SimpleCNN/GAN) [UNet]: ").strip() or 'UNet'
    output_dir = input("Output directory [./inference_output]: ").strip() or './inference_output'

    # Optional S1
    s1_path = input("Path to S1 SAR (optional, press Enter to skip): ").strip() or None

    # Run
    try:
        inferencer = SingleImageInference(
            model_path=model_path,
            model_type=model_type
        )

        outputs = inferencer.process_and_save(
            s2_cloudy_path=s2_cloudy_path,
            output_dir=output_dir,
            s1_path=s1_path,
            save_formats=['rgb', 'geotiff'],
            stretch_percentile=98
        )

        print("\n✓ Success! Check output directory for results.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Command-line mode
        main()
    else:
        # Interactive mode
        interactive_example()