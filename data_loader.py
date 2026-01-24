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
from tqdm import tqdm
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

import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class SEN12MSCRDataset(Dataset):
    """
    SEN12MS-CR PyTorch Dataset

    Input:
        S1 SAR (VV, VH)            -> (2, H, W)
        S2 cloudy (13 bands)       -> (13, H, W)
        Soft cloud mask            -> (1, H, W)

    Target:
        S2 clean (13 bands)        -> (13, H, W)
    """

    def __init__(
        self,
        root_dir,
        seasons=None,
        s2_bands=None,
        patch_size=256,
        data_fraction=1.0,
        min_cloud_fraction=0.05,
        max_cloud_fraction=0.9,
        random_seed=42,
        cloud_mask_mode: str = "simple",
        deep_model_kwargs: dict | None = None
    ):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.min_cloud_fraction = min_cloud_fraction
        self.max_cloud_fraction = max_cloud_fraction
        self.rng = random.Random(random_seed)
        self.cloud_mask_mode = cloud_mask_mode
        self.deep_model_kwargs = deep_model_kwargs or {}

        # Seasons
        if seasons is None:
            seasons = [
                "ROIs1158_spring",
                "ROIs1868_summer",
                "ROIs1970_fall",
                "ROIs2017_winter"
            ]

        # Sentinel-2 bands (1–13)
        if s2_bands is None:
            self.s2_bands = list(range(1, 14))
        else:
            self.s2_bands = s2_bands

        # --------------------------------------------------
        # Collect samples grouped by scene (ROI + season)
        # --------------------------------------------------
        scenes = defaultdict(list)

        for season in seasons:
            s1_dir = self.root_dir / f"{season}_s1"
            s2_dir = self.root_dir / f"{season}_s2"
            s2c_dir = self.root_dir / f"{season}_s2_cloudy"

            if not (s1_dir.exists() and s2_dir.exists() and s2c_dir.exists()):
                continue

            for s2_scene in sorted(s2_dir.glob("s2_*")):
                scene_id = s2_scene.name.split("_")[1]

                s1_scene = s1_dir / f"s1_{scene_id}"
                s2c_scene = s2c_dir / f"s2_cloudy_{scene_id}"

                if not (s1_scene.exists() and s2c_scene.exists()):
                    continue

                for clean_patch in s2_scene.glob("*.tif"):
                    cloudy_patch = s2c_scene / clean_patch.name.replace(
                        "_s2_", "_s2_cloudy_"
                    )
                    s1_patch = s1_scene / clean_patch.name.replace(
                        "_s2_", "_s1_"
                    )

                    if cloudy_patch.exists() and s1_patch.exists():
                        scenes[(season, scene_id)].append({
                            "s1": s1_patch,
                            "s2_clean": clean_patch,
                            "s2_cloudy": cloudy_patch
                        })

        # --------------------------------------------------
        # Scene-level subsampling (data_fraction)
        # --------------------------------------------------
        scene_keys = list(scenes.keys())
        self.rng.shuffle(scene_keys)

        num_scenes = max(1, int(len(scene_keys) * data_fraction))
        selected_scenes = scene_keys[:num_scenes]

        self.samples = [
            s for k in selected_scenes for s in scenes[k]
        ]

        print(
            f"[SEN12MS-CR] Loaded {len(self.samples)} samples "
            f"from {len(selected_scenes)} scenes "
            f"({data_fraction * 100:.1f}%)"
        )

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    @staticmethod
    def normalize_s2(x):
        x = x.astype(np.float32) / 10000.0
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def normalize_s1(x):
        # SAR is in dB, typical SEN12MS-CR range
        x = np.clip(x, -25.0, 0.0)
        return (x + 25.0) / 25.0

    # --------------------------------------------------
    # Random crop (shared across modalities)
    # --------------------------------------------------
    def random_crop(self, *arrays):
        _, H, W = arrays[0].shape
        ps = self.patch_size

        if H <= ps or W <= ps:
            return arrays

        x = self.rng.randint(0, W - ps)
        y = self.rng.randint(0, H - ps)

        return [a[:, y:y + ps, x:x + ps] for a in arrays]

    # --------------------------------------------------
    # Soft cloud mask (SAR–optical disagreement)
    # --------------------------------------------------
    @staticmethod
    def compute_cloud_mask(s1, s2_cloudy):
        """
        Soft cloud likelihood mask in [0,1]
        """
        # Optical brightness
        opt = s2_cloudy.mean(axis=0)

        # SAR energy (VV + VH)
        sar = np.mean(np.abs(s1), axis=0)

        # Normalize locally
        opt = (opt - opt.min()) / (opt.max() - opt.min() + 1e-6)
        sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-6)

        # Clouds: bright in optical, weak in SAR
        cloud_likelihood = opt * (1.0 - sar)

        return cloud_likelihood[None, :, :].astype(np.float32)

    # --------------------------------------------------
    # Get item
    # --------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load S1
        with rasterio.open(sample["s1"]) as src:
            s1 = src.read().astype(np.float32)

        # Load S2 cloudy
        with rasterio.open(sample["s2_cloudy"]) as src:
            s2_cloudy = src.read(self.s2_bands).astype(np.float32)

        # Load S2 clean
        with rasterio.open(sample["s2_clean"]) as src:
            s2_clean = src.read(self.s2_bands).astype(np.float32)

        # Normalize
        s1 = self.normalize_s1(s1)
        s2_cloudy = self.normalize_s2(s2_cloudy)
        s2_clean = self.normalize_s2(s2_clean)


        cloud_mask = self.compute_cloud_mask(s1, s2_cloudy)

        # Random crop
        s1, s2_cloudy, s2_clean, cloud_mask = self.random_crop(
            s1, s2_cloudy, s2_clean, cloud_mask
        )

        # Cloud fraction filtering
        cf = cloud_mask.mean()
        if cf < self.min_cloud_fraction or cf > self.max_cloud_fraction:
            return self.__getitem__(self.rng.randint(0, len(self) - 1))

        return (
            torch.from_numpy(s1),          # (2, H, W)
            torch.from_numpy(s2_cloudy),   # (13, H, W)
            torch.from_numpy(s2_clean),    # (13, H, W)
            torch.from_numpy(cloud_mask)   # (1, H, W)
        )


class SEN12MSCRSplitter:
    """Split and organize SEN12MS-CR dataset into train/val/small directories"""

    def __init__(self, base_dir='./sen12mscr_dataset', output_dir='./processed_data'):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        # Create output structure
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.small_dir = self.output_dir / 'small'

        for dir_path in [self.train_dir, self.val_dir, self.small_dir]:
            (dir_path / 'clean').mkdir(parents=True, exist_ok=True)
            (dir_path / 'cloudy').mkdir(parents=True, exist_ok=True)

    def collect_all_samples(self, seasons=None, s2_bands=None):
        """
        Collect all valid sample pairs from the dataset

        Args:
            seasons: List of seasons to include (None = all)
            s2_bands: List of S2 band indices to extract (None = all 13)
        """
        if seasons is None:
            seasons = [
                "ROIs1158_spring",
                "ROIs1868_summer",
                "ROIs1970_fall",
                "ROIs2017_winter"
            ]

        if s2_bands is None:
            s2_bands = list(range(1, 14))  # All 13 bands

        self.s2_bands = s2_bands

        print("\n" + "=" * 70)
        print("COLLECTING SAMPLES")
        print("=" * 70)
        print(f"Seasons: {', '.join(seasons)}")
        print(f"Bands: {len(s2_bands)} bands")

        samples = []

        for season in seasons:
            s2_root = self.base_dir / f'{season}_s2'
            s2cloudy_root = self.base_dir / f'{season}_s2_cloudy'

            if not s2_root.exists() or not s2cloudy_root.exists():
                print(f"⚠ Skipping {season} - missing directories")
                continue

            print(f"\nProcessing {season}...")
            season_samples = 0

            for s2_scene_dir in sorted(s2_root.glob('s2_*')):
                scene_id = s2_scene_dir.name.split('_')[1]
                s2cloudy_scene_dir = s2cloudy_root / f's2_cloudy_{scene_id}'

                if not s2cloudy_scene_dir.exists():
                    continue

                for s2_patch_path in sorted(s2_scene_dir.glob('*.tif')):
                    patch_filename = s2_patch_path.name
                    s2cloudy_filename = patch_filename.replace('_s2_', '_s2_cloudy_')
                    s2cloudy_path = s2cloudy_scene_dir / s2cloudy_filename

                    if s2cloudy_path.exists():
                        samples.append({
                            's2': s2_patch_path,
                            's2_cloudy': s2cloudy_path,
                            'season': season,
                            'scene_id': scene_id
                        })
                        season_samples += 1

            print(f"  ✓ Collected {season_samples} samples from {season}")

        print(f"\nTotal samples collected: {len(samples)}")
        return samples

    def split_and_save(self, samples, train_ratio=0.8, small_ratio=0.05,
                       random_state=42):
        """
        Split samples into train/val/small and save as numpy arrays

        Args:
            samples: List of sample dictionaries
            train_ratio: Ratio for training set (0.8 = 80%)
            small_ratio: Ratio for small dataset (0.05 = 5% of total)
            random_state: Random seed for reproducibility
        """
        print("\n" + "=" * 70)
        print("SPLITTING AND SAVING DATASET")
        print("=" * 70)

        # Group by scene to avoid data leakage
        scenes = defaultdict(list)
        for s in samples:
            scenes[(s["season"], s["scene_id"])].append(s)

        scene_keys = list(scenes.keys())

        # Split: 80% train, 20% val
        train_keys, val_keys = train_test_split(
            scene_keys,
            test_size=(1 - train_ratio),
            random_state=random_state
        )

        # Create small dataset (5% of all data)
        small_keys, _ = train_test_split(
            scene_keys,
            test_size=(1 - small_ratio),
            random_state=random_state
        )

        # Reconstruct sample lists
        train_samples = [s for k in train_keys for s in scenes[k]]
        val_samples = [s for k in val_keys for s in scenes[k]]
        small_samples = [s for k in small_keys for s in scenes[k]]

        print(f"\nSplit statistics:")
        print(f"  Training:   {len(train_samples)} samples ({len(train_samples) / len(samples) * 100:.1f}%)")
        print(f"  Validation: {len(val_samples)} samples ({len(val_samples) / len(samples) * 100:.1f}%)")
        print(f"  Small:      {len(small_samples)} samples ({len(small_samples) / len(samples) * 100:.1f}%)")

        # Save each split
        self._save_split(train_samples, self.train_dir, "Training")
        self._save_split(val_samples, self.val_dir, "Validation")
        self._save_split(small_samples, self.small_dir, "Small")

        print("\n" + "=" * 70)
        print("DATASET PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nProcessed data saved to: {self.output_dir}")
        print("\nDirectory structure:")
        print("  processed_data/")
        print("    ├── train/")
        print("    │   ├── clean/")
        print("    │   └── cloudy/")
        print("    ├── val/")
        print("    │   ├── clean/")
        print("    │   └── cloudy/")
        print("    └── small/")
        print("        ├── clean/")
        print("        └── cloudy/")

    def _save_split(self, samples, output_dir, split_name):
        """Save a single split (train/val/small) as numpy arrays"""
        print(f"\nSaving {split_name} set...")

        clean_dir = output_dir / 'clean'
        cloudy_dir = output_dir / 'cloudy'

        for idx, sample in enumerate(tqdm(samples, desc=f"Processing {split_name}")):
            try:
                # Load S2 cloudy
                with rasterio.open(sample['s2_cloudy']) as src:
                    s2_cloudy = src.read(self.s2_bands)

                # Load S2 clean
                with rasterio.open(sample['s2']) as src:
                    s2_clean = src.read(self.s2_bands)

                # Normalize to [0, 1]
                s2_cloudy = s2_cloudy.astype(np.float32) / 10000.0
                s2_clean = s2_clean.astype(np.float32) / 10000.0

                s2_cloudy = np.clip(s2_cloudy, 0, 1)
                s2_clean = np.clip(s2_clean, 0, 1)

                # Save as numpy arrays
                filename = f"{sample['season']}_{sample['scene_id']}_{idx:05d}.npy"
                np.save(clean_dir / filename, s2_clean)
                np.save(cloudy_dir / filename, s2_cloudy)

            except Exception as e:
                print(f"\n⚠ Error processing sample {idx}: {e}")
                continue

        print(f"✓ Saved {len(samples)} samples to {output_dir}")


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
def split_interface():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  SEN12MS-CR Dataset Splitter                                             ║
║  One-time processing: Creates train/val/small directories                ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize splitter
    splitter = SEN12MSCRSplitter(
        base_dir='./sen12mscr_dataset',
        output_dir='./processed_data'
    )

    # Ask user for band selection
    print("\nSelect bands to process:")
    print("1. RGB + NIR (bands 2,3,4,8 - recommended, 4 bands)")
    print("2. All 13 bands (complete Sentinel-2)")
    print("3. Custom bands")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '2':
        bands = list(range(1, 14))
        print("Using all 13 Sentinel-2 bands")
    elif choice == '3':
        bands_input = input("Enter band numbers (e.g., 2,3,4,8): ")
        bands = [int(b.strip()) for b in bands_input.split(',')]
        print(f"Using bands: {bands}")
    else:
        bands = [2, 3, 4, 8]  # Default: Blue, Green, Red, NIR
        print("Using RGB + NIR (4 bands)")

    # Ask for seasons
    print("\nSelect seasons to process:")
    print("1. All seasons (spring, summer, fall, winter)")
    print("2. Specific seasons")

    season_choice = input("\nEnter choice (1-2): ").strip()

    if season_choice == '2':
        print("Available: spring, summer, fall, winter")
        seasons_input = input("Enter seasons (comma-separated): ")
        season_map = {
            "spring": "ROIs1158_spring",
            "summer": "ROIs1868_summer",
            "fall": "ROIs1970_fall",
            "winter": "ROIs2017_winter"
        }
        seasons = [season_map[s.strip().lower()] for s in seasons_input.split(',')]
    else:
        seasons = None  # All seasons

    # Collect samples
    samples = splitter.collect_all_samples(seasons=seasons, s2_bands=bands)

    if len(samples) == 0:
        print("\n✗ No samples found! Check your dataset structure.")
        return

    # Confirm before processing
    print(f"\nAbout to process {len(samples)} samples.")
    print("This will create numpy arrays for faster loading.")

    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm == 'y':
        # Split and save
        splitter.split_and_save(
            samples,
            train_ratio=0.8,
            small_ratio=0.05,
            random_state=42
        )

        print("\n✓ Dataset processing complete!")
        print("\nNext steps:")
        print("1. Run: python main_simple.py")
        print("2. The script will automatically load from processed_data/")
        print("3. No need to run this splitter again!")
    else:
        print("Cancelled.")


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
    print("6. Split interface")
    print("7. Exit")

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
    elif choice =='6':
        split_interface()
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