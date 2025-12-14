"""
SEN12MS-CR Dataset Loader
=========================
Sentinel-1/2 Cloud Removal Dataset from TUM
Supports: Spring and Winter ROIs with cloudy Sentinel-2 data
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tarfile
import shutil
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    print("WARNING: rasterio not installed")
    print("Install: pip install rasterio")
    RASTERIO_AVAILABLE = False


# ==================== SEN12MS-CR DATASET EXTRACTOR ====================

class SEN12MSCRExtractor:
    """
    Extract and organize SEN12MS-CR dataset from tar files
    """

    def __init__(self, output_dir='./sen12mscr_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def extract_all(self, tar_files):
        """
        Extract all tar files and organize structure

        Args:
            tar_files: List of paths to tar files

        Expected files:
        - ROIs1158_spring_s1.tar
        - ROIs1158_spring_s2_cloudy.tar
        - ROIs2017_winter_s1.tar
        - ROIs2017_winter_s2_cloudy.tar
        """
        print("\n" + "="*70)
        print("SEN12MS-CR DATASET EXTRACTION")
        print("="*70)

        extracted_dirs = []

        for tar_file in tar_files:
            tar_path = Path(tar_file)

            if not tar_path.exists():
                print(f"⚠ File not found: {tar_file}")
                continue

            print(f"\nExtracting {tar_path.name}...")

            # Extract to output directory
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(self.output_dir)
                print(f"  ✓ Extracted to {self.output_dir}")

                # Get extracted directory name
                members = tar.getmembers()
                if members:
                    root_dir = members[0].name.split('/')[0]
                    extracted_dirs.append(self.output_dir / root_dir)

        print("\n" + "="*70)
        print(f"Extraction complete! Extracted {len(extracted_dirs)} directories")
        print("="*70)

        return extracted_dirs

    def organize_structure(self):
        """
        Organize extracted files into train/val structure

        Expected after extraction:
        sen12mscr_dataset/
        ├── ROIs1158_spring_s1/
        ├── ROIs1158_spring_s2_cloudy/
        ├── ROIs2017_winter_s1/
        └── ROIs2017_winter_s2_cloudy/

        Will create:
        sen12mscr_dataset/
        ├── organized/
        │   ├── train/
        │   │   ├── cloudy/
        │   │   └── s1/
        │   └── val/
        │       ├── cloudy/
        │       └── s1/
        """
        print("\n" + "="*70)
        print("ORGANIZING DATASET STRUCTURE")
        print("="*70)

        # Find all S2 cloudy and S1 directories
        spring_s2_cloudy = self.output_dir / 'ROIs1158_spring_s2_cloudy' / 'ROIs1158_spring' / 's2_cloudy'
        spring_s1 = self.output_dir / 'ROIs1158_spring_s1' / 'ROIs1158_spring' / 's1'

        winter_s2_cloudy = self.output_dir / 'ROIs2017_winter_s2_cloudy' / 'ROIs2017_winter' / 's2_cloudy'
        winter_s1 = self.output_dir / 'ROIs2017_winter_s1' / 'ROIs2017_winter' / 's1'

        # Collect all image paths
        all_cloudy_images = []
        all_s1_images = []

        print("\nCollecting images...")

        # Spring data
        if spring_s2_cloudy.exists():
            spring_cloudy = sorted(list(spring_s2_cloudy.glob('**/*.tif')))
            all_cloudy_images.extend(spring_cloudy)
            print(f"  Spring S2 cloudy: {len(spring_cloudy)} images")

        if spring_s1.exists():
            spring_s1_imgs = sorted(list(spring_s1.glob('**/*.tif')))
            all_s1_images.extend(spring_s1_imgs)
            print(f"  Spring S1: {len(spring_s1_imgs)} images")

        # Winter data
        if winter_s2_cloudy.exists():
            winter_cloudy = sorted(list(winter_s2_cloudy.glob('**/*.tif')))
            all_cloudy_images.extend(winter_cloudy)
            print(f"  Winter S2 cloudy: {len(winter_cloudy)} images")

        if winter_s1.exists():
            winter_s1_imgs = sorted(list(winter_s1.glob('**/*.tif')))
            all_s1_images.extend(winter_s1_imgs)
            print(f"  Winter S1: {len(winter_s1_imgs)} images")

        print(f"\nTotal cloudy images: {len(all_cloudy_images)}")
        print(f"Total S1 images: {len(all_s1_images)}")

        # Create organized directory structure
        organized_dir = self.output_dir / 'organized'
        train_cloudy_dir = organized_dir / 'train' / 'cloudy'
        train_s1_dir = organized_dir / 'train' / 's1'
        val_cloudy_dir = organized_dir / 'val' / 'cloudy'
        val_s1_dir = organized_dir / 'val' / 's1'

        for dir_path in [train_cloudy_dir, train_s1_dir, val_cloudy_dir, val_s1_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        # Split into train/val (80/20)
        train_cloudy, val_cloudy = train_test_split(
            all_cloudy_images,
            test_size=0.2,
            random_state=42
        )

        train_s1, val_s1 = train_test_split(
            all_s1_images,
            test_size=0.2,
            random_state=42
        )

        print(f"\nSplit:")
        print(f"  Train: {len(train_cloudy)} cloudy, {len(train_s1)} S1")
        print(f"  Val:   {len(val_cloudy)} cloudy, {len(val_s1)} S1")

        # Create symbolic links or copy files
        print("\nCreating organized structure...")

        self._create_links(train_cloudy, train_cloudy_dir)
        self._create_links(train_s1, train_s1_dir)
        self._create_links(val_cloudy, val_cloudy_dir)
        self._create_links(val_s1, val_s1_dir)

        print("\n✓ Organization complete!")
        print(f"Organized dataset location: {organized_dir}")

        # Save file list for reference
        self._save_file_lists(
            organized_dir,
            train_cloudy, val_cloudy,
            train_s1, val_s1
        )

        return organized_dir

    def _create_links(self, file_list, target_dir):
        """Create symbolic links or copy files"""
        for i, file_path in enumerate(file_list):
            target_path = target_dir / f"{i:06d}.tif"

            # Try symbolic link first (faster)
            try:
                if not target_path.exists():
                    os.symlink(file_path, target_path)
            except OSError:
                # If symlink fails, copy instead
                if not target_path.exists():
                    shutil.copy2(file_path, target_path)

            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{len(file_list)} files...")

    def _save_file_lists(self, organized_dir, train_cloudy, val_cloudy, train_s1, val_s1):
        """Save lists of original file paths"""
        lists_dir = organized_dir / 'file_lists'
        lists_dir.mkdir(exist_ok=True)

        def save_list(file_list, filename):
            with open(lists_dir / filename, 'w') as f:
                for file_path in file_list:
                    f.write(f"{file_path}\n")

        save_list(train_cloudy, 'train_cloudy.txt')
        save_list(val_cloudy, 'val_cloudy.txt')
        save_list(train_s1, 'train_s1.txt')
        save_list(val_s1, 'val_s1.txt')

        print(f"  ✓ Saved file lists to {lists_dir}")


# ==================== SEN12MS-CR DATASET ====================

class SEN12MSCRDataset(Dataset):
    """
    PyTorch Dataset for SEN12MS-CR

    Dataset Information:
    - Sentinel-2 cloudy images (13 bands)
    - Sentinel-1 SAR images (2 bands: VV, VH)
    - Resolution: 256x256 pixels
    - Format: GeoTIFF

    Note: This dataset contains cloudy S2 and S1 (for cloud removal using SAR)
          Clean S2 images are in separate files (not downloaded)
    """

    def __init__(self, root_dir, split='train', bands=[2,3,4,8],
                 use_s1=False, transform=None, preload=False, cache_size=100):
        """
        Args:
            root_dir: Path to organized dataset directory
            split: 'train' or 'val'
            bands: Which S2 bands to use (1-13). Default: [2,3,4,8] = Blue,Green,Red,NIR
            use_s1: If True, also load Sentinel-1 SAR data
            transform: Optional transforms
            preload: If True, load all to RAM
            cache_size: Number of images to cache
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required. Install: pip install rasterio")

        self.root_dir = Path(root_dir)
        self.split = split
        self.bands = bands
        self.use_s1 = use_s1
        self.transform = transform
        self.preload = preload
        self.cache = {}
        self.cache_size = cache_size

        # Paths
        self.cloudy_dir = self.root_dir / split / 'cloudy'
        self.s1_dir = self.root_dir / split / 's1'

        if not self.cloudy_dir.exists():
            raise FileNotFoundError(
                f"Cloudy directory not found: {self.cloudy_dir}\n"
                f"Please run extraction and organization first."
            )

        # Get image files
        self.cloudy_files = sorted(list(self.cloudy_dir.glob('*.tif')))

        if use_s1:
            self.s1_files = sorted(list(self.s1_dir.glob('*.tif')))
            # Match S2 and S1 pairs
            self.cloudy_files = self.cloudy_files[:min(len(self.cloudy_files), len(self.s1_files))]
            self.s1_files = self.s1_files[:len(self.cloudy_files)]

        print(f"\n{'='*60}")
        print(f"SEN12MS-CR Dataset - {split.upper()} Split")
        print(f"{'='*60}")
        print(f"Directory: {self.root_dir}")
        print(f"Split: {split}")
        print(f"S2 Cloudy images: {len(self.cloudy_files)}")
        if use_s1:
            print(f"S1 SAR images: {len(self.s1_files)}")
        print(f"Selected S2 bands: {bands}")
        print(f"Band names: {self._get_band_names(bands)}")
        print(f"Resolution: 256x256 pixels")

        # Preload if requested
        if preload and len(self.cloudy_files) <= cache_size:
            print(f"\nPreloading {len(self.cloudy_files)} images to RAM...")
            for idx in range(len(self.cloudy_files)):
                if idx % 100 == 0:
                    print(f"  Loaded {idx}/{len(self.cloudy_files)} images...")
                self.cache[idx] = self._load_from_disk(idx)
            print(f"✓ Preloading complete!")

        print(f"{'='*60}\n")

    def _get_band_names(self, bands):
        """Get band names for selected bands"""
        band_names = {
            1: 'Coastal', 2: 'Blue', 3: 'Green', 4: 'Red',
            5: 'RedEdge1', 6: 'RedEdge2', 7: 'RedEdge3', 8: 'NIR',
            9: 'NIR_Narrow', 10: 'WaterVapor', 11: 'SWIR_Cirrus',
            12: 'SWIR1', 13: 'SWIR2'
        }
        return [band_names.get(b, f'B{b}') for b in bands]

    def _load_from_disk(self, idx):
        """Load image from disk"""
        cloudy_path = self.cloudy_files[idx]

        try:
            # Load S2 cloudy image
            with rasterio.open(cloudy_path) as src:
                # Read selected bands (bands are 1-indexed in rasterio)
                cloudy = src.read(self.bands)

            # Normalize to [0, 1]
            # SEN12MS-CR uses different encoding, typically 0-10000
            cloudy = cloudy.astype(np.float32)

            # Handle different value ranges
            if cloudy.max() > 10:
                cloudy = cloudy / 10000.0

            cloudy = np.clip(cloudy, 0, 1)

            # Load S1 if requested
            s1 = None
            if self.use_s1:
                s1_path = self.s1_files[idx]
                with rasterio.open(s1_path) as src:
                    s1 = src.read()  # 2 bands: VV, VH

                # Normalize S1 (SAR data, typically in dB)
                s1 = s1.astype(np.float32)
                # Convert from dB to linear scale
                s1 = np.clip((s1 + 25) / 35, 0, 1)  # Approximate normalization

            # Convert to PyTorch tensors
            cloudy = torch.from_numpy(cloudy)

            if s1 is not None:
                s1 = torch.from_numpy(s1)

            if self.transform:
                cloudy = self.transform(cloudy)
                if s1 is not None:
                    s1 = self.transform(s1)

            # For cloud removal, we use cloudy as both input and target
            # (since we don't have clean reference in this subset)
            # OR use S1 as auxiliary input
            if self.use_s1:
                return (cloudy, s1), cloudy  # Input: (S2_cloudy, S1), Target: S2_cloudy
            else:
                return cloudy, cloudy  # Input: S2_cloudy, Target: S2_cloudy (self-supervised)

        except Exception as e:
            print(f"Error loading {cloudy_path.name}: {e}")
            # Return dummy data on error
            dummy = torch.zeros(len(self.bands), 256, 256)
            if self.use_s1:
                dummy_s1 = torch.zeros(2, 256, 256)
                return (dummy, dummy_s1), dummy
            return dummy, dummy

    def __len__(self):
        return len(self.cloudy_files)

    def __getitem__(self, idx):
        """
        Returns:
            If use_s1=False:
                cloudy: Tensor [bands, 256, 256]
                cloudy: Tensor [bands, 256, 256] (same as input for self-supervised)

            If use_s1=True:
                (cloudy, s1): Tuple of tensors
                cloudy: Target tensor
        """
        if idx in self.cache:
            return self.cache[idx]

        data = self._load_from_disk(idx)

        if len(self.cache) < self.cache_size:
            self.cache[idx] = data

        return data

    def get_sample_info(self, idx):
        """Get information about a sample"""
        cloudy_path = self.cloudy_files[idx]

        with rasterio.open(cloudy_path) as src:
            info = {
                'filename': cloudy_path.name,
                'shape': src.shape,
                'bands': src.count,
                'dtype': src.dtypes[0],
                'crs': src.crs,
                'transform': src.transform,
                'path': str(cloudy_path)
            }

        return info


# ==================== VISUALIZATION ====================

def visualize_sen12mscr_samples(dataset, n_samples=3, save_path='sen12mscr_samples.png'):
    """Visualize sample images from SEN12MS-CR"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(n_samples, len(dataset))):
        data = dataset[i]

        # Handle both formats
        if isinstance(data[0], tuple):
            # (cloudy, s1), target
            cloudy, s1 = data[0]
            target = data[1]
        else:
            # cloudy, target
            cloudy = data[0]
            target = data[1]
            s1 = None

        # Take RGB bands (assuming bands 2,3,4 are included)
        # Bands are 0-indexed in tensor, so if bands=[2,3,4,8], RGB is [0,1,2]
        if cloudy.shape[0] >= 3:
            cloudy_rgb = cloudy[:3].permute(1, 2, 0).numpy()
        else:
            cloudy_rgb = cloudy[0].numpy()

        axes[i, 0].imshow(np.clip(cloudy_rgb, 0, 1))
        axes[i, 0].set_title(f'Sample {i+1}: S2 Cloudy')
        axes[i, 0].axis('off')

        if s1 is not None:
            # Show S1 (VV band)
            s1_img = s1[0].numpy()
            axes[i, 1].imshow(s1_img, cmap='gray')
            axes[i, 1].set_title(f'Sample {i+1}: S1 SAR')
        else:
            # Show same cloudy image
            axes[i, 1].imshow(np.clip(cloudy_rgb, 0, 1))
            axes[i, 1].set_title(f'Sample {i+1}: S2 Cloudy')

        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved visualization to: {save_path}")


def analyze_sen12mscr_statistics(dataset, n_samples=500):
    """Analyze dataset statistics"""
    import matplotlib.pyplot as plt

    print(f"\nAnalyzing {min(n_samples, len(dataset))} samples...")

    all_images = []

    for i in range(min(n_samples, len(dataset))):
        if i % 100 == 0:
            print(f"  Processed {i}/{min(n_samples, len(dataset))} samples...")

        data = dataset[i]
        if isinstance(data[0], tuple):
            img = data[0][0].numpy()  # S2 cloudy
        else:
            img = data[0].numpy()

        all_images.append(img)

    all_images = np.array(all_images)

    print("\n" + "="*60)
    print("SEN12MS-CR Dataset Statistics")
    print("="*60)
    print(f"\nSamples analyzed: {len(all_images)}")
    print(f"Image shape: {all_images.shape}")
    print(f"\nPixel values:")
    print(f"  Mean: {all_images.mean():.4f}")
    print(f"  Std:  {all_images.std():.4f}")
    print(f"  Min:  {all_images.min():.4f}")
    print(f"  Max:  {all_images.max():.4f}")

    # Per-band statistics
    print(f"\nPer-band statistics:")
    for i in range(all_images.shape[1]):
        print(f"  Band {i+1}: mean={all_images[:,i].mean():.4f}, std={all_images[:,i].std():.4f}")

    print("="*60)

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_images.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title('Pixel Value Distribution')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')

    # Per-band distribution
    for i in range(min(4, all_images.shape[1])):
        axes[1].hist(all_images[:,i].flatten(), bins=50, alpha=0.5,
                    label=f'Band {i+1}')
    axes[1].set_title('Per-Band Distribution')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('sen12mscr_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ Saved statistics to: sen12mscr_statistics.png")


# ==================== MAIN ====================

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  SEN12MS-CR Dataset Loader                                               ║
║  Sentinel-1/2 Multi-Spectral Cloud Removal Dataset                      ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    print("\nWhat would you like to do?")
    print("1. Extract and organize tar files")
    print("2. Verify dataset structure")
    print("3. Test dataset loading")
    print("4. Visualize samples")
    print("5. Analyze statistics")
    print("6. Exit")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '1':
        print("\n" + "="*70)
        print("TAR FILE EXTRACTION")
        print("="*70)

        # Prompt for tar file locations
        print("\nEnter paths to your downloaded tar files:")
        print("(Press Enter after each path, empty line when done)")

        tar_files = []
        while True:
            path = input(f"  Tar file {len(tar_files)+1}: ").strip()
            if not path:
                break
            if Path(path).exists():
                tar_files.append(path)
                print(f"    ✓ Found: {Path(path).name}")
            else:
                print(f"    ✗ Not found: {path}")

        if not tar_files:
            print("\nNo valid tar files provided. Using default:")
            tar_files = [
                './ROIs1158_spring_s1.tar',
                './ROIs1158_spring_s2_cloudy.tar',
                './ROIs2017_winter_s1.tar',
                './ROIs2017_winter_s2_cloudy.tar'
            ]
            print("\nDefault files:")
            for f in tar_files:
                print(f"  - {f}")

        # Extract
        extractor = SEN12MSCRExtractor('./sen12mscr_dataset')
        extractor.extract_all(tar_files)

        # Organize
        print("\nWould you like to organize into train/val split? (y/n)")
        if input().strip().lower() == 'y':
            organized_dir = extractor.organize_structure()
            print(f"\n✓ Dataset ready at: {organized_dir}")

    elif choice == '2':
        print("\nVerifying dataset structure...")

        dataset_dir = Path('./sen12mscr_dataset/organized')

        if not dataset_dir.exists():
            print(f"✗ Organized dataset not found at: {dataset_dir}")
            print("Run option 1 to extract and organize tar files first.")
        else:
            train_cloudy = dataset_dir / 'train' / 'cloudy'
            train_s1 = dataset_dir / 'train' / 's1'
            val_cloudy = dataset_dir / 'val' / 'cloudy'
            val_s1 = dataset_dir / 'val' / 's1'

            print(f"\n✓ Dataset structure verified:")
            print(f"  Train cloudy: {len(list(train_cloudy.glob('*.tif')))} images")
            print(f"  Train S1: {len(list(train_s1.glob('*.tif')))} images")
            print(f"  Val cloudy: {len(list(val_cloudy.glob('*.tif')))} images")
            print(f"  Val S1: {len(list(val_s1.glob('*.tif')))} images")

    elif choice == '3':
        print("\nTesting dataset loading...")
        try:
            train_dataset = SEN12MSCRDataset(
                './sen12mscr_dataset/organized',
                split='train',
                bands=[2, 3, 4, 8]  # Blue, Green, Red, NIR
            )

            val_dataset = SEN12MSCRDataset(
                './sen12mscr_dataset/organized',
                split='val',
                bands=[2, 3, 4, 8]
            )

            print(f"\n✓ Successfully loaded datasets:")
            print(f"  Training: {len(train_dataset)} samples")
            print(f"  Validation: {len(val_dataset)} samples")

            # Test loading one sample
            print("\nTesting sample loading...")
            cloudy, target = train_dataset[0]
            print(f"✓ Cloudy image shape: {cloudy.shape}")
            print(f"✓ Target image shape: {target.shape}")
            print(f"✓ Value range: [{cloudy.min():.3f}, {cloudy.max():.3f}]")
            print("\n✓ Dataset is ready to use!")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("\nPlease check:")
            print("1. Dataset is extracted and organized")
            print("2. Directory structure is correct")

    elif choice == '4':
        print("\nGenerating sample visualizations...")
        try:
            dataset = SEN12MSCRDataset(
                './sen12mscr_dataset/organized',
                split='train',
                bands=[2, 3, 4, 8]
            )
            visualize_sen12mscr_samples(dataset, n_samples=3)
            print("\n✓ Visualization complete!")
        except Exception as e:
            print(f"\n✗ Error: {e}")

    elif choice == '5':
        print("\nAnalyzing dataset statistics...")
        try:
            dataset = SEN12MSCRDataset(
                './sen12mscr_dataset/organized',
                split='train',
                bands=[2, 3, 4, 8]
            )
            analyze_sen12mscr_statistics(dataset, n_samples=500)
            print("\n✓ Analysis complete!")
        except Exception as e:
            print(f"\n✗ Error: {e}")

    else:
        print("Exiting...")

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Next Steps:                                                             ║
║  1. Extract tar files (option 1)                                         ║
║  2. Verify structure (option 2)                                          ║
║  3. Run: python satellite_cloud_removal.py                               ║
║  4. Choose SEN12MS-CR dataset option                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)