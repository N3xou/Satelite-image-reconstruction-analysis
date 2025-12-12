"""
WHUS2-CR Dataset Loader
=======================
Wuhan University Sentinel-2 Cloud Removal Dataset
- 17,358 image pairs (cloudy + clean)
- 4 bands: Red, Green, Blue, NIR
- Resolution: 256x256
- Format: GeoTIFF
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    print("WARNING: rasterio not installed")
    print("Install: pip install rasterio")
    RASTERIO_AVAILABLE = False


# ==================== WHUS2-CR DATASET ====================

class WHUS2CRDataset(Dataset):
    """
    PyTorch Dataset for WHUS2-CR (Wuhan University Sentinel-2 Cloud Removal)

    Dataset Information:
    - Size: 17,358 image pairs
    - Bands: 4 (Red, Green, Blue, NIR)
    - Resolution: 256x256 pixels
    - Format: GeoTIFF (.tif files)
    - Source: IEEE DataPort

    Download:
    1. Register at https://ieee-dataport.org/ (free)
    2. Visit: https://ieee-dataport.org/open-access/whus2-cr-wuhan-university-sentinel-2-cloud-removal-dataset
    3. Download and extract to your data directory


    """

    def __init__(self, root_dir, split='train', transform=None,
                 use_rgb_only=False, preload=False, cache_size=100):
        """
        Args:
            root_dir: Path to whus2cr_dataset directory
            split: 'train' or 'val'
            transform: Optional transforms
            use_rgb_only: If True, use only RGB bands (ignore NIR)
            preload: If True, load all images to RAM (faster but uses ~4GB)
            cache_size: Number of images to keep in cache
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required. Install: pip install rasterio")

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_rgb_only = use_rgb_only
        self.preload = preload
        self.cache = {}
        self.cache_size = cache_size

        # Directory structure
        self.cloudy_dir = self.root_dir / split / 'cloud_image'
        self.clean_dir = self.root_dir / split / 'no_cloud_image'

        # Check if directories exist
        if not self.cloudy_dir.exists():
            raise FileNotFoundError(
                f"Cloudy image directory not found: {self.cloudy_dir}\n"
                f"Please download WHUS2-CR dataset and extract to {self.root_dir}\n"
                f"See download instructions in the docstring."
            )

        if not self.clean_dir.exists():
            raise FileNotFoundError(
                f"Clean image directory not found: {self.clean_dir}"
            )

        # Get list of images
        self.image_files = sorted(list(self.cloudy_dir.glob('*.tif')))

        if len(self.image_files) == 0:
            raise ValueError(
                f"No .tif files found in {self.cloudy_dir}\n"
                f"Please check the dataset structure."
            )

        # Verify corresponding clean images exist
        valid_files = []
        for cloudy_file in self.image_files:
            clean_file = self.clean_dir / cloudy_file.name
            if clean_file.exists():
                valid_files.append(cloudy_file)

        self.image_files = valid_files

        print(f"\n{'=' * 60}")
        print(f"WHUS2-CR Dataset - {split.upper()} Split")
        print(f"{'=' * 60}")
        print(f"Directory: {self.root_dir}")
        print(f"Split: {split}")
        print(f"Cloudy images: {self.cloudy_dir}")
        print(f"Clean images: {self.clean_dir}")
        print(f"Total image pairs: {len(self.image_files)}")
        print(f"Bands: {'RGB (3)' if use_rgb_only else 'RGB + NIR (4)'}")
        print(f"Resolution: 256x256 pixels")

        # Preload if requested
        if preload and len(self.image_files) <= cache_size:
            print(f"\nPreloading {len(self.image_files)} images to RAM...")
            for idx in range(len(self.image_files)):
                if idx % 1000 == 0:
                    print(f"  Loaded {idx}/{len(self.image_files)} images...")
                self.cache[idx] = self._load_from_disk(idx)
            print(f"✓ Preloading complete! ({len(self.cache)} images in RAM)")

        print(f"{'=' * 60}\n")

    def _load_from_disk(self, idx):
        """Load image pair from disk"""
        cloudy_path = self.image_files[idx]
        clean_path = self.clean_dir / cloudy_path.name

        try:
            # Load with rasterio (for GeoTIFF)
            with rasterio.open(cloudy_path) as src:
                cloudy = src.read()  # Shape: (4, H, W)

            with rasterio.open(clean_path) as src:
                clean = src.read()  # Shape: (4, H, W)

            # Convert to float32 and normalize to [0, 1]
            # WHUS2-CR uses 0-10000 range for reflectance values
            cloudy = cloudy.astype(np.float32) / 10000.0
            clean = clean.astype(np.float32) / 10000.0

            # Clip to [0, 1] range
            cloudy = np.clip(cloudy, 0, 1)
            clean = np.clip(clean, 0, 1)

            # Use only RGB if requested (first 3 bands)
            if self.use_rgb_only:
                cloudy = cloudy[:3]
                clean = clean[:3]

            # Convert to PyTorch tensors
            cloudy = torch.from_numpy(cloudy)
            clean = torch.from_numpy(clean)

            if self.transform:
                cloudy = self.transform(cloudy)
                clean = self.transform(clean)

            return cloudy, clean

        except Exception as e:
            print(f"Error loading {cloudy_path.name}: {e}")
            raise

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            cloudy: Tensor of shape [4, 256, 256] or [3, 256, 256] if use_rgb_only
            clean: Tensor of shape [4, 256, 256] or [3, 256, 256] if use_rgb_only
        """
        # Use cached data if available
        if idx in self.cache:
            return self.cache[idx]

        # Load from disk
        data = self._load_from_disk(idx)

        # Cache if within cache size
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data

        return data

    def get_sample_info(self, idx):
        """Get information about a specific sample"""
        cloudy_path = self.image_files[idx]
        clean_path = self.clean_dir / cloudy_path.name

        with rasterio.open(cloudy_path) as src:
            info = {
                'filename': cloudy_path.name,
                'shape': src.shape,
                'bands': src.count,
                'dtype': src.dtypes[0],
                'crs': src.crs,
                'transform': src.transform,
                'cloudy_path': str(cloudy_path),
                'clean_path': str(clean_path)
            }

        return info


class WHUS2CRDownloader:
    """
    Helper class for WHUS2-CR dataset download instructions
    """

    def __init__(self, output_dir='./whus2cr_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def show_download_instructions(self):
        """Display download instructions"""
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  WHUS2-CR Dataset Download Instructions                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

Dataset: Wuhan University Sentinel-2 Cloud Removal (WHUS2-CR)
Size: ~8 GB (17,358 image pairs)
Format: GeoTIFF (.tif files)
Bands: 4 (Red, Green, Blue, NIR)
Resolution: 256×256 pixels

STEP 1: Create IEEE Account (Free)
────────────────────────────────────────────────────────────────────────────
1. Visit: https://ieee-dataport.org/
2. Click "Sign Up" in top right
3. Fill registration form
4. Verify email
5. Log in to your account

STEP 2: Download Dataset
────────────────────────────────────────────────────────────────────────────
1. Visit: https://ieee-dataport.org/open-access/whus2-cr-wuhan-university-sentinel-2-cloud-removal-dataset

2. Scroll down and click "Download Dataset" button
   (You must be logged in)

3. Download will start (approximately 8 GB)
   - File name: WHUS2-CR.zip or similar
   - Time: 10-30 minutes depending on connection

STEP 3: Extract Dataset
────────────────────────────────────────────────────────────────────────────
Extract the downloaded ZIP file to your project directory:

Windows (PowerShell):
  Expand-Archive -Path WHUS2-CR.zip -DestinationPath ./whus2cr_dataset

Windows (7-Zip):
  Right-click WHUS2-CR.zip -> 7-Zip -> Extract to "whus2cr_dataset"

Linux/Mac:
  unzip WHUS2-CR.zip -d ./whus2cr_dataset

Expected Directory Structure:
────────────────────────────────────────────────────────────────────────────
whus2cr_dataset/
├── train/
│   ├── cloud_image/
│   │   ├── 1.tif
│   │   ├── 2.tif
│   │   ├── 3.tif
│   │   └── ... (14,686 images)
│   └── no_cloud_image/
│       ├── 1.tif
│       ├── 2.tif
│       ├── 3.tif
│       └── ... (14,686 images)
└── val/
    ├── cloud_image/
    │   └── ... (2,672 images)
    └── no_cloud_image/
        └── ... (2,672 images)

STEP 4: Verify Installation
────────────────────────────────────────────────────────────────────────────
Run this Python code to verify:

    from whus2cr_dataset import WHUS2CRDataset

    dataset = WHUS2CRDataset('./whus2cr_dataset', split='train')
    print(f"✓ Loaded {len(dataset)} training samples")

    # Test loading one sample
    cloudy, clean = dataset[0]
    print(f"✓ Image shape: {cloudy.shape}")
    print(f"✓ Dataset ready to use!")

STEP 5: Train Models
────────────────────────────────────────────────────────────────────────────
    python satellite_cloud_removal.py
    # Choose option: WHUS2-CR dataset

Alternative Sources:
────────────────────────────────────────────────────────────────────────────
If IEEE DataPort is down, try:
- Paper authors: https://github.com/dr-lizhiwei/WHUS2-CR
- Contact: lizhiwei@whu.edu.cn

Citation:
────────────────────────────────────────────────────────────────────────────
If you use this dataset, please cite:
Li, Z., Shen, H., Cheng, Q., Liu, Y., You, S., & He, Z. (2019).
"Deep learning based cloud detection for medium and high resolution 
remote sensing images of different sensors."
ISPRS Journal of Photogrammetry and Remote Sensing, 150, 197-212.

╚══════════════════════════════════════════════════════════════════════════╝
        """)

    def verify_dataset(self):
        """Verify dataset structure"""
        print("\nVerifying dataset structure...")

        issues = []

        # Check directories
        train_cloudy = self.output_dir / 'train' / 'cloud_image'
        train_clean = self.output_dir / 'train' / 'no_cloud_image'
        val_cloudy = self.output_dir / 'val' / 'cloud_image'
        val_clean = self.output_dir / 'val' / 'no_cloud_image'

        dirs_to_check = [
            ('Training cloudy images', train_cloudy),
            ('Training clean images', train_clean),
            ('Validation cloudy images', val_cloudy),
            ('Validation clean images', val_clean)
        ]

        for name, dir_path in dirs_to_check:
            if dir_path.exists():
                count = len(list(dir_path.glob('*.tif')))
                print(f"✓ {name}: {count} images")
            else:
                print(f"✗ {name}: NOT FOUND")
                issues.append(f"Missing directory: {dir_path}")

        if issues:
            print("\n⚠ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nPlease check download instructions above.")
            return False
        else:
            print("\n✓ Dataset structure verified!")
            print("✓ Ready to use with WHUS2CRDataset")
            return True


# ==================== VISUALIZATION TOOLS ====================

def visualize_whus2cr_samples(dataset, n_samples=3, save_path='whus2cr_samples.png'):
    """
    Visualize sample image pairs from WHUS2-CR dataset

    Args:
        dataset: WHUS2CRDataset instance
        n_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    for i in range(n_samples):
        cloudy, clean = dataset[i]

        # Convert to numpy and take RGB bands only
        cloudy_rgb = cloudy[:3].permute(1, 2, 0).numpy()
        clean_rgb = clean[:3].permute(1, 2, 0).numpy()

        # Difference map
        diff = np.abs(cloudy_rgb - clean_rgb)

        # Plot
        axes[i, 0].imshow(np.clip(cloudy_rgb, 0, 1))
        axes[i, 0].set_title(f'Sample {i + 1}: Cloudy Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.clip(clean_rgb, 0, 1))
        axes[i, 1].set_title(f'Sample {i + 1}: Clean Target')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f'Sample {i + 1}: Difference')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved visualization to: {save_path}")


def analyze_whus2cr_statistics(dataset, n_samples=1000):
    """
    Analyze dataset statistics

    Args:
        dataset: WHUS2CRDataset instance
        n_samples: Number of samples to analyze
    """
    import matplotlib.pyplot as plt

    print(f"\nAnalyzing {min(n_samples, len(dataset))} samples...")

    all_cloudy = []
    all_clean = []
    cloud_coverage = []

    for i in range(min(n_samples, len(dataset))):
        if i % 100 == 0:
            print(f"  Processed {i}/{min(n_samples, len(dataset))} samples...")

        cloudy, clean = dataset[i]

        all_cloudy.append(cloudy.numpy())
        all_clean.append(clean.numpy())

        # Estimate cloud coverage (bright pixels in difference)
        diff = np.abs(cloudy.numpy() - clean.numpy())
        coverage = (diff.mean(axis=0) > 0.1).mean() * 100
        cloud_coverage.append(coverage)

    all_cloudy = np.array(all_cloudy)
    all_clean = np.array(all_clean)
    cloud_coverage = np.array(cloud_coverage)

    print("\n" + "=" * 60)
    print("WHUS2-CR Dataset Statistics")
    print("=" * 60)
    print(f"\nSamples analyzed: {len(all_cloudy)}")
    print(f"\nClean images:")
    print(f"  Mean: {all_clean.mean():.4f}")
    print(f"  Std:  {all_clean.std():.4f}")
    print(f"  Min:  {all_clean.min():.4f}")
    print(f"  Max:  {all_clean.max():.4f}")

    print(f"\nCloudy images:")
    print(f"  Mean: {all_cloudy.mean():.4f}")
    print(f"  Std:  {all_cloudy.std():.4f}")
    print(f"  Min:  {all_cloudy.min():.4f}")
    print(f"  Max:  {all_cloudy.max():.4f}")

    print(f"\nCloud Coverage:")
    print(f"  Mean: {cloud_coverage.mean():.2f}%")
    print(f"  Std:  {cloud_coverage.std():.2f}%")
    print(f"  Min:  {cloud_coverage.min():.2f}%")
    print(f"  Max:  {cloud_coverage.max():.2f}%")
    print("=" * 60)

    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(all_clean.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Clean Image Pixel Distribution')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(all_cloudy.flatten(), bins=50, alpha=0.7, color='gray')
    axes[0, 1].set_title('Cloudy Image Pixel Distribution')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')

    axes[1, 0].hist(cloud_coverage, bins=30, alpha=0.7, color='red')
    axes[1, 0].set_title('Cloud Coverage Distribution')
    axes[1, 0].set_xlabel('Cloud Coverage (%)')
    axes[1, 0].set_ylabel('Number of Images')

    axes[1, 1].scatter(all_clean.mean(axis=(1, 2, 3)),
                       all_cloudy.mean(axis=(1, 2, 3)), alpha=0.3)
    axes[1, 1].set_title('Clean vs Cloudy Mean Brightness')
    axes[1, 1].set_xlabel('Clean Image Mean')
    axes[1, 1].set_ylabel('Cloudy Image Mean')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('whus2cr_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ Saved statistics plot to: whus2cr_statistics.png")


# ==================== MAIN ====================

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  WHUS2-CR Dataset Loader                                                ║
║  Wuhan University Sentinel-2 Cloud Removal Dataset                      ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    print("\nWhat would you like to do?")
    print("1. Show download instructions")
    print("2. Verify dataset structure")
    print("3. Test dataset loading")
    print("4. Visualize samples")
    print("5. Analyze dataset statistics")
    print("6. Exit")

    choice = input("\nEnter choice (1-6): ").strip()

    downloader = WHUS2CRDownloader('./whus2cr_dataset')

    if choice == '1':
        downloader.show_download_instructions()

    elif choice == '2':
        downloader.verify_dataset()

    elif choice == '3':
        print("\nTesting dataset loading...")
        try:
            train_dataset = WHUS2CRDataset('./whus2cr_dataset', split='train')
            val_dataset = WHUS2CRDataset('./whus2cr_dataset', split='val')

            print(f"\n✓ Successfully loaded datasets:")
            print(f"  Training: {len(train_dataset)} samples")
            print(f"  Validation: {len(val_dataset)} samples")

            # Test loading one sample
            print("\nTesting sample loading...")
            cloudy, clean = train_dataset[0]
            print(f"✓ Cloudy image shape: {cloudy.shape}")
            print(f"✓ Clean image shape: {clean.shape}")
            print(f"✓ Value range: [{cloudy.min():.3f}, {cloudy.max():.3f}]")
            print("\n✓ Dataset is ready to use!")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("\nPlease check:")
            print("1. Dataset is downloaded")
            print("2. Dataset is extracted correctly")
            print("3. Directory structure matches expected format")

    elif choice == '4':
        print("\nGenerating sample visualizations...")
        try:
            dataset = WHUS2CRDataset('./whus2cr_dataset', split='train')
            visualize_whus2cr_samples(dataset, n_samples=3)
            print("\n✓ Visualization complete!")
        except Exception as e:
            print(f"\n✗ Error: {e}")

    elif choice == '5':
        print("\nAnalyzing dataset statistics...")
        try:
            dataset = WHUS2CRDataset('./whus2cr_dataset', split='train')
            analyze_whus2cr_statistics(dataset, n_samples=1000)
            print("\n✓ Analysis complete!")
        except Exception as e:
            print(f"\n✗ Error: {e}")

    else:
        print("Exiting...")

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Next Steps:                                                             ║
║  1. Download dataset from IEEE DataPort                                  ║
║  2. Extract to ./whus2cr_dataset/                                        ║
║  3. Run: python satellite_cloud_removal.py                               ║
║  4. Choose WHUS2-CR dataset option                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)