"""
SEN12MS-CR Dataset Loader
=========================
Complete implementation for Sentinel-1/2 Cloud Removal Dataset
Supports: All seasons (Spring, Summer, Fall, Winter)
Reference: Official SEN12MS-CR data loader
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import tarfile
from sklearn.model_selection import train_test_split
from enum import Enum
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    print("WARNING: rasterio not installed")
    print("Install: pip install rasterio")
    RASTERIO_AVAILABLE = False


# ==================== ENUMS FROM OFFICIAL LOADER ====================

class S1Bands(Enum):
    """Sentinel-1 SAR bands"""
    VV = 1
    VH = 2
    ALL = [VV, VH]


class S2Bands(Enum):
    """Sentinel-2 multispectral bands"""
    B01 = 1   # Coastal Aerosol
    B02 = 2   # Blue
    B03 = 3   # Green
    B04 = 4   # Red
    B05 = 5   # Red Edge 1
    B06 = 6   # Red Edge 2
    B07 = 7   # Red Edge 3
    B08 = 8   # NIR
    B08A = 9  # Narrow NIR
    B09 = 10  # Water Vapor
    B10 = 11  # SWIR Cirrus
    B11 = 12  # SWIR 1
    B12 = 13  # SWIR 2
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]


class Seasons(Enum):
    """Available seasons in SEN12MS-CR"""
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]


# ==================== TAR EXTRACTOR ====================

class SEN12MSCRExtractor:
    """Extract and organize SEN12MS-CR tar files"""

    def __init__(self, output_dir='./sen12mscr_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def extract_all(self, tar_files):
        """
        Extract all tar files

        Expected tar file patterns:
        - ROIsXXXX_season_s1.tar (Sentinel-1 SAR)
        - ROIsXXXX_season_s2.tar (Sentinel-2 clean)
        - ROIsXXXX_season_s2_cloudy.tar (Sentinel-2 cloudy)

        Where XXXX is the ROI number (1158, 1868, 1970, 2017)
        and season is spring, summer, fall, or winter
        """
        print("\n" + "="*70)
        print("SEN12MS-CR DATASET EXTRACTION")
        print("="*70)

        extracted_seasons = set()

        for tar_file in tar_files:
            tar_path = Path(tar_file)

            if not tar_path.exists():
                print(f"⚠ File not found: {tar_file}")
                continue

            print(f"\nExtracting {tar_path.name}...")

            # Extract season name from filename
            # Format: ROIsXXXX_season_sensor.tar
            parts = tar_path.stem.split('_')
            if len(parts) >= 2:
                season = parts[1]  # spring, summer, fall, winter
                extracted_seasons.add(season)

            # Extract to output directory
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(self.output_dir)
                print(f"  ✓ Extracted to {self.output_dir}")

        print("\n" + "="*70)
        print(f"Extraction complete!")
        print(f"Extracted seasons: {', '.join(sorted(extracted_seasons))}")
        print("="*70)

        return extracted_seasons

    def verify_structure(self):
        """Verify the extracted sen12mscr_dataset structure"""
        print("\n" + "="*70)
        print("VERIFYING DATASET STRUCTURE")
        print("="*70)

        found_seasons = []

        for season_enum in Seasons:
            if season_enum == Seasons.ALL:
                continue

            season_name = season_enum.value
            season_path = self.output_dir / season_name

            if season_path.exists():
                print(f"\n✓ Found season: {season_name}")
                found_seasons.append(season_name)

                # Count scenes (directories like s1_XXX, s2_XXX)
                s1_scenes = list(season_path.glob('s1_*'))
                s2_scenes = list(season_path.glob('s2_*'))

                print(f"  S1 scenes: {len(s1_scenes)}")
                print(f"  S2 scenes: {len(s2_scenes)}")

                # Count patches in first scene
                if s1_scenes:
                    patches = list(s1_scenes[0].glob('*.tif'))
                    print(f"  Patches per scene (sample): {len(patches)}")

        if not found_seasons:
            print("\n✗ No season directories found!")
            print("Expected structure after extraction:")
            print("  sen12mscr_dataset/")
            print("    ROIs1158_spring/")
            print("    ROIs1868_summer/")
            print("    ROIs1970_fall/")
            print("    ROIs2017_winter/")

        return found_seasons


# ==================== PYTORCH DATASET ====================

class SEN12MSCRDataset(Dataset):
    """
    PyTorch Dataset for SEN12MS-CR
    Input:  S2_cloudy (+ optional S1)
    Target: S2 clean
    """

    def __init__(self, base_dir, seasons=None, s1_bands=None, s2_bands=None,
                 split='train', val_split=0.2, random_state=42,
                 transform=None, use_s1=False):
        """
        Args:
            base_dir: Path to sen12mscr_dataset
            seasons: List of season names or 'all'. e.g., ['spring', 'winter']
            s1_bands: List of S1 band numbers [1,2] or None for all
            s2_bands: List of S2 band numbers [1-13] or None for all
            split: 'train', 'val', or 'small'
            val_split: Validation split ratio (0.2 = 20%)
            random_state: Random seed for reproducibility
            transform: Optional transforms
            use_s1: If True, return (s1, s2_cloudy) as input
            use_small: If True and split='small', use 10% of training data
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio required: pip install rasterio")

        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.use_s1 = use_s1

        # Default to all bands if not specified
        self.s1_bands = s1_bands if s1_bands else [1, 2]
        self.s2_bands = s2_bands if s2_bands else list(range(1, 14))

        # Parse seasons
        if seasons is None or seasons == "all":
            self.seasons = [
                "ROIs1158_spring",
                "ROIs1868_summer",
                "ROIs1970_fall",
                "ROIs2017_winter",
            ]
        else:
            season_map = {
                "spring": "ROIs1158_spring",
                "summer": "ROIs1868_summer",
                "fall": "ROIs1970_fall",
                "winter": "ROIs2017_winter",
            }
            self.seasons = [season_map.get(s.lower(), s) for s in seasons]

        print(f"\nLoading SEN12MS-CR [{split}]")

        self.samples = self._collect_samples()

        # Split into train/val/small
        if len(self.samples) == 0:
            raise ValueError("No samples found! Check dataset structure.")

        scenes = defaultdict(list)
        for s in self.samples:
            scenes[(s["season"], s["scene_id"])].append(s)

        scene_keys = list(scenes.keys())
        train_keys, val_keys = train_test_split(
            scene_keys, test_size=val_split, random_state=random_state
        )

        if split == "train":
            selected_keys = train_keys
        elif split == "val":
            selected_keys = val_keys
        elif split == "small":
            small_keys, _ = train_test_split(
                train_keys, test_size=0.9, random_state=random_state
            )
            selected_keys = small_keys
        else:
            raise ValueError("split must be train / val / small")

        self.samples = [
            sample
            for k in selected_keys
            for sample in scenes[k]
        ]

        print(f"Samples in {split}: {len(self.samples)}")

    def _collect_samples(self):
        """
        Collect all valid (s1, s2, s2_cloudy) triplets

        Expected structure:
        sen12mscr_dataset/
        ├── ROIsXXXX_season_s1/
        │   └── s1_X/*.tif
        ├── ROIsXXXX_season_s2/
        │   └── s2_X/*.tif
        └── ROIsXXXX_season_s2_cloudy/
            └── s2_cloudy_X/*.tif
        """
        samples = []

        for season in self.seasons:
            # Build directory paths based on actual structure
            # Pattern: ROIsXXXX_season_sensor
            s1_root = self.base_dir / f'{season}_s1'
            s2_root = self.base_dir / f'{season}_s2'
            s2cloudy_root = self.base_dir / f'{season}_s2_cloudy'

            print(f"\nSearching in season: {season}")
            print(f"  S1 dir: {s1_root} - {'✓ exists' if s1_root.exists() else '✗ missing'}")
            print(f"  S2 dir: {s2_root} - {'✓ exists' if s2_root.exists() else '✗ missing'}")
            print(f"  S2 cloudy dir: {s2cloudy_root} - {'✓ exists' if s2cloudy_root.exists() else '✗ missing'}")

            if not s2_root.exists() or not s2cloudy_root.exists():
                print(f"  ⚠ Skipping {season} - missing required directories")
                continue

            for s2_scene_dir in sorted(s2_root.glob('s2_*')):
                # Extract scene ID from directory name (s2_X -> X)
                scene_id = s2_scene_dir.name.split('_')[1]

                # Corresponding directories
                s1_scene_dir = s1_root / f's1_{scene_id}'
                s2cloudy_scene_dir = s2cloudy_root / f's2_cloudy_{scene_id}'

                # S1 is optional, but s2_cloudy is required
                if not s2cloudy_scene_dir.exists():
                    continue

                for s2_patch_path in sorted(s2_scene_dir.glob('*.tif')):
                    # Extract patch filename
                    patch_filename = s2_patch_path.name

                    # Build corresponding paths
                    # Filename pattern: ROIsXXXX_season_s2_X_pY.tif
                    # Need to convert to: ROIsXXXX_season_s1_X_pY.tif
                    s1_filename = patch_filename.replace('_s2_', '_s1_')
                    s2cloudy_filename = patch_filename.replace('_s2_', '_s2_cloudy_')

                    # Full paths
                    s1_path = s1_scene_dir / s1_filename if s1_scene_dir.exists() else None
                    s2cloudy_path = s2cloudy_scene_dir / s2cloudy_filename

                    # Verify required files exist
                    if not s2cloudy_path.exists():
                        continue

                    # S1 is optional
                    if s1_path and not s1_path.exists():
                        s1_path = None

                    samples.append({
                        's1': s1_path,
                        's2': s2_patch_path,
                        's2_cloudy': s2cloudy_path,
                        'season': season,
                        'scene_id': scene_id
                    })

            season_count = len([s for s in samples if s['season'] == season])
            print(f"  → Collected {season_count} valid triplets from {season}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            If use_s1=False:
                s2_cloudy: Input tensor [bands, H, W]
                s2: Target tensor [bands, H, W]
            If use_s1=True:
                (s1, s2_cloudy): Input tuple
                s2: Target tensor
        """
        sample = self.samples[idx]

        try:
            # Load S2 cloudy (input)
            with rasterio.open(sample['s2_cloudy']) as src:
                s2_cloudy = src.read(self.s2_bands)

            # Load S2 clean (target)
            with rasterio.open(sample['s2']) as src:
                s2 = src.read(self.s2_bands)

            # Normalize to [0, 1]
            s2_cloudy = s2_cloudy.astype(np.float32) / 10000.0
            s2 = s2.astype(np.float32) / 10000.0

            s2_cloudy = np.clip(s2_cloudy, 0, 1)
            s2 = np.clip(s2, 0, 1)

            # Convert to tensors
            s2_cloudy = torch.from_numpy(s2_cloudy)
            s2 = torch.from_numpy(s2)

            if self.use_s1:
                # Load S1 (auxiliary input)
                with rasterio.open(sample['s1']) as src:
                    s1 = src.read(self.s1_bands)

                # S1 is in dB, normalize
                s1 = s1.astype(np.float32)
                s1 = np.clip((s1 + 25) / 35, 0, 1)
                s1 = torch.from_numpy(s1)

                if self.transform:
                    s1 = self.transform(s1)
                    s2_cloudy = self.transform(s2_cloudy)
                    s2 = self.transform(s2)

                return (s1, s2_cloudy), s2
            else:
                if self.transform:
                    s2_cloudy = self.transform(s2_cloudy)
                    s2 = self.transform(s2)

                return s2_cloudy, s2

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data
            dummy_shape = (len(self.s2_bands), 256, 256)
            dummy_cloudy = torch.zeros(dummy_shape)
            dummy_clean = torch.zeros(dummy_shape)

            if self.use_s1:
                dummy_s1 = torch.zeros(len(self.s1_bands), 256, 256)
                return (dummy_s1, dummy_cloudy), dummy_clean
            return dummy_cloudy, dummy_clean

    def get_sample_info(self, idx):
        """Get information about a sample"""
        sample = self.samples[idx]
        return {
            'season': sample['season'],
            'scene_id': sample['scene_id'],
            's1_path': str(sample['s1']),
            's2_path': str(sample['s2']),
            's2_cloudy_path': str(sample['s2_cloudy'])
        }


# ==================== VISUALIZATION ====================

def visualize_sen12mscr_samples(dataset, n_samples=3, save_path='sen12mscr_samples.png'):
    """Visualize sample triplets"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(n_samples, len(dataset))):
        data = dataset[i]

        if isinstance(data[0], tuple):
            s1, s2_cloudy = data[0]
            s2 = data[1]
        else:
            s2_cloudy = data[0]
            s2 = data[1]
            s1 = None

        # Take RGB bands (B04=Red, B03=Green, B02=Blue = bands 4,3,2 = indices 3,2,1)
        # Assuming full 13 bands, RGB are at indices [3,2,1]
        if s2_cloudy.shape[0] >= 13:
            cloudy_rgb = s2_cloudy[[3, 2, 1]].permute(1, 2, 0).numpy()
            clean_rgb = s2[[3, 2, 1]].permute(1, 2, 0).numpy()
        elif s2_cloudy.shape[0] >= 3:
            cloudy_rgb = s2_cloudy[:3].permute(1, 2, 0).numpy()
            clean_rgb = s2[:3].permute(1, 2, 0).numpy()
        else:
            cloudy_rgb = s2_cloudy[0].numpy()
            clean_rgb = s2[0].numpy()

        axes[i, 0].imshow(np.clip(cloudy_rgb, 0, 1))
        axes[i, 0].set_title(f'Sample {i + 1}: S2 Cloudy')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.clip(clean_rgb, 0, 1))
        axes[i, 1].set_title(f'Sample {i + 1}: S2 Clean')
        axes[i, 1].axis('off')

        if s1 is not None:
            s1_img = s1[0].numpy()
            axes[i, 2].imshow(s1_img, cmap='gray')
            axes[i, 2].set_title(f'Sample {i + 1}: S1 SAR (VV)')
        else:
            diff = np.abs(cloudy_rgb - clean_rgb)
            axes[i, 2].imshow(diff)
            axes[i, 2].set_title(f'Sample {i + 1}: Difference')

        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved visualization to: {save_path}")


# ==================== MAIN ====================

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  SEN12MS-CR Dataset Loader                                               ║
║  Sentinel-1/2 Cloud Removal Dataset                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    print("\nWhat would you like to do?")
    print("1. Extract tar files")
    print("2. Verify dataset structure")
    print("3. Test dataset loading")
    print("4. Visualize samples")
    print("5. Show dataset statistics")
    print("6. Exit")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '1':
        print("\nEnter paths to your tar files:")
        print("(Press Enter after each path, empty line when done)")
        print("\nExpected files (example):")
        print("  ROIs2017_winter_s1.tar")
        print("  ROIs2017_winter_s2.tar")
        print("  ROIs2017_winter_s2_cloudy.tar")

        tar_files = []
        while True:
            path = input(f"  Tar file {len(tar_files) + 1}: ").strip()
            if not path:
                break
            if Path(path).exists():
                tar_files.append(path)
                print(f"    ✓ Found: {Path(path).name}")
            else:
                print(f"    ✗ Not found: {path}")

        if tar_files:
            extractor = SEN12MSCRExtractor('./sen12mscr_dataset')
            extractor.extract_all(tar_files)
            extractor.verify_structure()
        else:
            print("No valid tar files provided.")

    elif choice == '2':
        extractor = SEN12MSCRExtractor('./sen12mscr_dataset')
        found = extractor.verify_structure()

        if found:
            print(f"\n✓ Dataset is ready to use!")
            print(f"  Found {len(found)} season(s)")

    elif choice == '3':
        print("\nTesting dataset loading...")
        print("Which seasons do you have? (comma-separated, or 'all')")
        print("Available: winter, spring, summer, fall")

        seasons_input = input("Seasons: ").strip()
        if seasons_input.lower() == 'all':
            seasons = None
        else:
            seasons = [s.strip() for s in seasons_input.split(',')]

        try:
            # Test with RGB + NIR
            print("\nLoading with RGB + NIR bands [2,3,4,8]...")
            train_dataset = SEN12MSCRDataset(
                './sen12mscr_dataset',
                seasons=seasons,
                s2_bands=[2, 3, 4, 8],  # Blue, Green, Red, NIR
                split='train'
            )

            val_dataset = SEN12MSCRDataset(
                './sen12mscr_dataset',
                seasons=seasons,
                s2_bands=[2, 3, 4, 8],
                split='val'
            )

            print(f"\n✓ Successfully loaded!")
            print(f"  Training: {len(train_dataset)} samples")
            print(f"  Validation: {len(val_dataset)} samples")

            # Test loading one sample
            print("\nTesting sample loading...")
            s2_cloudy, s2_clean = train_dataset[0]
            print(f"✓ S2 cloudy shape: {s2_cloudy.shape}")
            print(f"✓ S2 clean shape: {s2_clean.shape}")
            print(f"✓ Value range: [{s2_cloudy.min():.3f}, {s2_cloudy.max():.3f}]")

            # Show sample info
            info = train_dataset.get_sample_info(0)
            print(f"\n✓ Sample info:")
            print(f"  Season: {info['season']}")
            print(f"  Scene ID: {info['scene_id']}")

            print("\n✓ Dataset is ready to use!")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback

            traceback.print_exc()

    elif choice == '4':
        print("\nVisualizing samples...")
        seasons_input = input("Seasons (comma-separated or 'all'): ").strip()
        if seasons_input.lower() == 'all':
            seasons = None
        else:
            seasons = [s.strip() for s in seasons_input.split(',')]

        try:
            dataset = SEN12MSCRDataset(
                './sen12mscr_dataset',
                seasons=seasons,
                s2_bands=list(range(1, 14)),  # All bands for best RGB
                split='train'
            )
            visualize_sen12mscr_samples(dataset, n_samples=3)
        except Exception as e:
            print(f"✗ Error: {e}")

    elif choice == '5':
        print("\nDataset Statistics")
        print("=" * 70)

        extractor = SEN12MSCRExtractor('./sen12mscr_dataset')
        found_seasons = extractor.verify_structure()

        if found_seasons:
            print(f"\nTotal seasons found: {len(found_seasons)}")
            print("\nTo get full statistics, load the dataset with option 3")

    else:
        print("Exiting...")

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Next Steps:                                                             ║
║  1. Extract tar files (option 1)                                         ║
║  2. Verify structure (option 2)                                          ║
║  3. Test loading (option 3)                                              ║
║  4. Train models: python satellite_cloud_removal.py                      ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)