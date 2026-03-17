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

            parts = tar_path.stem.split('_')
            if len(parts) >= 2:
                season = parts[1]
                extracted_seasons.add(season)

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

                s1_scenes = list(season_path.glob('s1_*'))
                s2_scenes = list(season_path.glob('s2_*'))

                print(f"  S1 scenes: {len(s1_scenes)}")
                print(f"  S2 scenes: {len(s2_scenes)}")

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


# ==================== CLOUD MASKING ====================

class CloudMaskComputer:
    """
    Multi-method cloud mask computation for SEN12MS-CR.

    Modes
    -----
    "gt_diff"      : Ground-truth difference mask.
                     Uses |cloudy - clean| per pixel, averaged across bands.
                     Most accurate — only possible because we have paired data.

    "gt_threshold" : Hard binary mask from ground-truth difference.
                     Pixels where mean absolute diff > `gt_threshold` are
                     flagged as cloudy.  Returns float in {0, 1}.

    "spectral"     : Physics-based spectral mask.
                     Combines brightness, Blue/NIR haze ratio and SWIR
                     cirrus band (B10 at index 10 when all 13 bands present).
                     Doesn't need the clean image; useful at inference time.

    "sar_optical"  : Original SAR vs. optical disagreement heuristic
                     (kept for backward compatibility / ablation studies).

    "combined"     : Weighted blend of gt_diff + spectral.
                     Best of both worlds for training: supervised signal
                     guides the mask while spectral features keep it
                     physically interpretable.

    Parameters
    ----------
    mode          : one of the strings above (default: "combined")
    gt_diff_weight: weight of gt_diff component in "combined" mode (0-1)
    gt_threshold  : threshold for "gt_threshold" mode (default: 0.10)
    """

    MODES = {"gt_diff", "gt_threshold", "spectral", "sar_optical", "combined"}

    def __init__(
        self,
        mode: str = "combined",
        gt_diff_weight: float = 0.6,
        gt_threshold: float = 0.10,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown cloud_mask_mode '{mode}'. Choose from {self.MODES}")
        self.mode = mode
        self.gt_diff_weight = gt_diff_weight
        self.gt_threshold = gt_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        s2_cloudy: np.ndarray,
        s2_clean: np.ndarray,
        s1: np.ndarray,
        s2_bands: list,
    ) -> np.ndarray:
        """
        Compute cloud mask.

        Parameters
        ----------
        s2_cloudy : float32 array [C, H, W], normalised to [0,1]
        s2_clean  : float32 array [C, H, W], normalised to [0,1]
        s1        : float32 array [2, H, W], normalised to [0,1]
        s2_bands  : list of 1-based Sentinel-2 band indices loaded

        Returns
        -------
        mask : float32 array [1, H, W] in [0, 1]
        """
        if self.mode == "gt_diff":
            return self._gt_diff(s2_cloudy, s2_clean)
        elif self.mode == "gt_threshold":
            return self._gt_threshold(s2_cloudy, s2_clean)
        elif self.mode == "spectral":
            return self._spectral(s2_cloudy, s2_bands)
        elif self.mode == "sar_optical":
            return self._sar_optical(s1, s2_cloudy)
        elif self.mode == "combined":
            return self._combined(s2_cloudy, s2_clean, s1, s2_bands)

    # ------------------------------------------------------------------
    # Individual methods
    # ------------------------------------------------------------------

    @staticmethod
    def _gt_diff(s2_cloudy: np.ndarray, s2_clean: np.ndarray) -> np.ndarray:
        """
        Ground-truth difference mask.

        Cloud pixels are bright in s2_cloudy but not in s2_clean.
        We take the mean absolute difference across spectral bands, then
        clip and normalise to [0, 1].

        This is the most informative signal available during training:
        it tells the model *exactly* where and how strongly clouds affect
        each pixel, including semi-transparent / thin clouds that spectral
        methods miss.
        """
        diff = np.abs(s2_cloudy.astype(np.float32) - s2_clean.astype(np.float32))
        # Mean over spectral axis -> [H, W]
        mask = diff.mean(axis=0)
        # Robust normalisation: clip at 99th percentile to suppress outliers
        p99 = np.percentile(mask, 99)
        if p99 > 1e-6:
            mask = np.clip(mask / p99, 0.0, 1.0)
        return mask[None, :, :].astype(np.float32)   # [1, H, W]

    @staticmethod
    def _gt_threshold(s2_cloudy: np.ndarray, s2_clean: np.ndarray,
                      threshold: float = 0.10) -> np.ndarray:
        """
        Hard binary mask derived from ground-truth difference.

        Pixels with mean |cloudy - clean| > threshold are set to 1.
        A small morphological dilation (3×3 max-pool) catches cloud
        halos / shadow edges.
        """
        diff = np.abs(s2_cloudy.astype(np.float32) - s2_clean.astype(np.float32))
        mask = (diff.mean(axis=0) > threshold).astype(np.float32)

        # Simple 3×3 max-pool dilation (pure numpy, no scipy dependency)
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(mask, 1, mode='edge')
        windows = sliding_window_view(pad, (3, 3))
        mask = windows.max(axis=(-2, -1))

        return mask[None, :, :].astype(np.float32)

    @staticmethod
    def _spectral(s2_cloudy: np.ndarray, s2_bands: list) -> np.ndarray:
        """
        Physics-based spectral cloud mask.

        Uses three complementary signals:
        1. Visible brightness  – clouds are bright across all bands.
        2. Blue / NIR ratio    – haze and thin clouds raise blue relative
                                  to NIR (where vegetation absorbs).
        3. SWIR cirrus band    – B10 (1375 nm) specifically targets
                                  cirrus clouds.  Used when available.

        Each component is individually normalised then blended with
        empirically chosen weights.
        """
        band_idx = {b: i for i, b in enumerate(s2_bands)}

        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)

        # 1. Mean visible brightness (B02, B03, B04 = bands 2,3,4)
        vis_indices = [band_idx[b] for b in [2, 3, 4] if b in band_idx]
        if vis_indices:
            brightness = s2_cloudy[vis_indices].mean(axis=0)
        else:
            brightness = s2_cloudy.mean(axis=0)

        brightness_norm = _norm(brightness)

        # 2. Blue / NIR haze ratio
        b02_available = 2 in band_idx
        b08_available = 8 in band_idx

        if b02_available and b08_available:
            blue = s2_cloudy[band_idx[2]]
            nir  = s2_cloudy[band_idx[8]]
            # High ratio → haze / thin clouds; cap at 2 to avoid div-by-zero
            haze_ratio = np.clip(blue / (nir + 1e-6), 0.0, 2.0) / 2.0
            haze_norm = _norm(haze_ratio)
        else:
            haze_norm = np.zeros_like(brightness_norm)

        # 3. Cirrus band (B10, index 10 in 1-based numbering)
        if 10 in band_idx:
            cirrus = s2_cloudy[band_idx[10]]
            cirrus_norm = _norm(cirrus)
            weights = (0.45, 0.30, 0.25)
        else:
            cirrus_norm = np.zeros_like(brightness_norm)
            weights = (0.55, 0.45, 0.0)

        mask = (weights[0] * brightness_norm
                + weights[1] * haze_norm
                + weights[2] * cirrus_norm)

        return mask[None, :, :].astype(np.float32)

    @staticmethod
    def _sar_optical(s1: np.ndarray, s2_cloudy: np.ndarray) -> np.ndarray:
        """
        Original SAR–optical disagreement heuristic (backward-compatible).

        Clouds: bright in optical (high reflectance) but attenuate SAR
        backscatter, causing a disagreement.
        """
        opt = s2_cloudy.mean(axis=0)
        sar = np.mean(np.abs(s1), axis=0)

        opt = (opt - opt.min()) / (opt.max() - opt.min() + 1e-6)
        sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-6)

        mask = opt * (1.0 - sar)
        return mask[None, :, :].astype(np.float32)

    def _combined(
        self,
        s2_cloudy: np.ndarray,
        s2_clean: np.ndarray,
        s1: np.ndarray,
        s2_bands: list,
    ) -> np.ndarray:
        """
        Weighted blend: gt_diff + spectral.

        gt_diff provides an accurate supervised signal for training.
        spectral adds physically meaningful structure (spatial texture,
        cirrus vs. thick cloud differentiation) that pure pixel-diff
        can miss around cloud edges.

        gt_diff_weight controls the blend (default 0.6 / 0.4).
        """
        gt   = self._gt_diff(s2_cloudy, s2_clean)
        spec = self._spectral(s2_cloudy, s2_bands)

        w_gt   = self.gt_diff_weight
        w_spec = 1.0 - w_gt

        combined = w_gt * gt + w_spec * spec
        # Re-normalise to [0, 1]
        p99 = np.percentile(combined, 99)
        if p99 > 1e-6:
            combined = np.clip(combined / p99, 0.0, 1.0)

        return combined.astype(np.float32)


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
        Cloud mask                 -> (1, H, W)

    Target:
        S2 clean (13 bands)        -> (13, H, W)

    Cloud mask modes
    ----------------
    "combined"     : (default) Supervised gt_diff + spectral blend.
                      Most informative for training.
    "gt_diff"      : Soft mask from |cloudy - clean|.  Exact but soft.
    "gt_threshold" : Hard binary mask from |cloudy - clean|.
    "spectral"     : Physics-based (brightness + haze + cirrus).
                      Usable at inference time without ground truth.
    "sar_optical"  : Legacy SAR vs optical disagreement.
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
        cloud_mask_mode: str = "combined",
        gt_diff_weight: float = 0.6,
        gt_threshold: float = 0.10,
    ):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.min_cloud_fraction = min_cloud_fraction
        self.max_cloud_fraction = max_cloud_fraction
        self.rng = random.Random(random_seed)

        self.mask_computer = CloudMaskComputer(
            mode=cloud_mask_mode,
            gt_diff_weight=gt_diff_weight,
            gt_threshold=gt_threshold,
        )

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
            f"({data_fraction * 100:.1f}%) | "
            f"cloud_mask_mode='{cloud_mask_mode}'"
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
    # Legacy static method (kept for backward compatibility)
    # --------------------------------------------------
    @staticmethod
    def compute_cloud_mask(s1, s2_cloudy):
        """
        Backward-compatible static wrapper.
        Uses original SAR–optical heuristic.
        For the improved supervised / spectral masks use CloudMaskComputer.
        """
        opt = s2_cloudy.mean(axis=0)
        sar = np.mean(np.abs(s1), axis=0)
        opt = (opt - opt.min()) / (opt.max() - opt.min() + 1e-6)
        sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-6)
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

        # Compute cloud mask (uses clean image when available)
        cloud_mask = self.mask_computer.compute(
            s2_cloudy=s2_cloudy,
            s2_clean=s2_clean,
            s1=s1,
            s2_bands=self.s2_bands,
        )

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
            torch.from_numpy(s2_cloudy),   # (C, H, W)
            torch.from_numpy(s2_clean),    # (C, H, W)
            torch.from_numpy(cloud_mask)   # (1, H, W)
        )


class SEN12MSCRSplitter:
    """Split and organize SEN12MS-CR dataset into train/val/small directories"""

    def __init__(self, base_dir='./sen12mscr_dataset', output_dir='./processed_data'):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.small_dir = self.output_dir / 'small'

        for dir_path in [self.train_dir, self.val_dir, self.small_dir]:
            (dir_path / 'clean').mkdir(parents=True, exist_ok=True)
            (dir_path / 'cloudy').mkdir(parents=True, exist_ok=True)

    def collect_all_samples(self, seasons=None, s2_bands=None):
        if seasons is None:
            seasons = [
                "ROIs1158_spring",
                "ROIs1868_summer",
                "ROIs1970_fall",
                "ROIs2017_winter"
            ]

        if s2_bands is None:
            s2_bands = list(range(1, 14))

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
        print("\n" + "=" * 70)
        print("SPLITTING AND SAVING DATASET")
        print("=" * 70)

        scenes = defaultdict(list)
        for s in samples:
            scenes[(s["season"], s["scene_id"])].append(s)

        scene_keys = list(scenes.keys())

        train_keys, val_keys = train_test_split(
            scene_keys,
            test_size=(1 - train_ratio),
            random_state=random_state
        )

        small_keys, _ = train_test_split(
            scene_keys,
            test_size=(1 - small_ratio),
            random_state=random_state
        )

        train_samples = [s for k in train_keys for s in scenes[k]]
        val_samples = [s for k in val_keys for s in scenes[k]]
        small_samples = [s for k in small_keys for s in scenes[k]]

        print(f"\nSplit statistics:")
        print(f"  Training:   {len(train_samples)} samples ({len(train_samples) / len(samples) * 100:.1f}%)")
        print(f"  Validation: {len(val_samples)} samples ({len(val_samples) / len(samples) * 100:.1f}%)")
        print(f"  Small:      {len(small_samples)} samples ({len(small_samples) / len(samples) * 100:.1f}%)")

        self._save_split(train_samples, self.train_dir, "Training")
        self._save_split(val_samples, self.val_dir, "Validation")
        self._save_split(small_samples, self.small_dir, "Small")

        print("\n" + "=" * 70)
        print("DATASET PROCESSING COMPLETE!")
        print("=" * 70)

    def _save_split(self, samples, output_dir, split_name):
        print(f"\nSaving {split_name} set...")

        clean_dir = output_dir / 'clean'
        cloudy_dir = output_dir / 'cloudy'

        for idx, sample in enumerate(tqdm(samples, desc=f"Processing {split_name}")):
            try:
                with rasterio.open(sample['s2_cloudy']) as src:
                    s2_cloudy = src.read(self.s2_bands)

                with rasterio.open(sample['s2']) as src:
                    s2_clean = src.read(self.s2_bands)

                s2_cloudy = s2_cloudy.astype(np.float32) / 10000.0
                s2_clean = s2_clean.astype(np.float32) / 10000.0

                s2_cloudy = np.clip(s2_cloudy, 0, 1)
                s2_clean = np.clip(s2_clean, 0, 1)

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

    splitter = SEN12MSCRSplitter(
        base_dir='./sen12mscr_dataset',
        output_dir='./processed_data'
    )

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
        bands = [2, 3, 4, 8]
        print("Using RGB + NIR (4 bands)")

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
        seasons = None

    samples = splitter.collect_all_samples(seasons=seasons, s2_bands=bands)

    if len(samples) == 0:
        print("\n✗ No samples found! Check your dataset structure.")
        return

    print(f"\nAbout to process {len(samples)} samples.")
    confirm = input("\nProceed? (y/n): ").strip().lower()

    if confirm == 'y':
        splitter.split_and_save(
            samples,
            train_ratio=0.8,
            small_ratio=0.05,
            random_state=42
        )

        print("\n✓ Dataset processing complete!")
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

    choice = input("\nEnter choice (1-7): ").strip()

    if choice == '1':
        print("\nEnter paths to your tar files:")
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

    elif choice == '3':
        print("\nTesting dataset loading...")
        seasons_input = input("Seasons (comma-separated or 'all'): ").strip()
        seasons = None if seasons_input.lower() == 'all' else [s.strip() for s in seasons_input.split(',')]

        try:
            print("\nLoading with combined cloud mask (gt_diff + spectral)...")
            train_dataset = SEN12MSCRDataset(
                './sen12mscr_dataset',
                seasons=seasons,
                s2_bands=[2, 3, 4, 8],
                cloud_mask_mode='combined',
            )
            print(f"\n✓ Successfully loaded {len(train_dataset)} samples")

            s1, s2_cloudy, s2_clean, cloud_mask = train_dataset[0]
            print(f"✓ S1 shape:        {s1.shape}")
            print(f"✓ S2 cloudy shape: {s2_cloudy.shape}")
            print(f"✓ S2 clean shape:  {s2_clean.shape}")
            print(f"✓ Cloud mask shape:{cloud_mask.shape}")
            print(f"✓ Mask range:      [{cloud_mask.min():.3f}, {cloud_mask.max():.3f}]")
            print(f"✓ Mean cloud frac: {cloud_mask.mean():.2%}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

    elif choice == '4':
        seasons_input = input("Seasons (comma-separated or 'all'): ").strip()
        seasons = None if seasons_input.lower() == 'all' else [s.strip() for s in seasons_input.split(',')]

        try:
            dataset = SEN12MSCRDataset(
                './sen12mscr_dataset',
                seasons=seasons,
                s2_bands=list(range(1, 14)),
            )
            visualize_sen12mscr_samples(dataset, n_samples=3)
        except Exception as e:
            print(f"✗ Error: {e}")

    elif choice == '5':
        extractor = SEN12MSCRExtractor('./sen12mscr_dataset')
        extractor.verify_structure()

    elif choice == '6':
        split_interface()
    else:
        print("Exiting...")