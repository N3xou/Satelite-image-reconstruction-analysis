import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter


class SatelliteDatasetPreparer:
    """Handles synthetic satellite data generation with simulated clouds."""

    def __init__(self, data_dir='./satellite_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def create_synthetic_dataset(self, n_samples=100, img_size=(256, 256),
                                 n_bands=4, cloud_density=0.3):
        """
        Create synthetic multispectral image pairs with simulated clouds.

        Parameters
        ----------
        n_samples      : number of image pairs to generate
        img_size       : (H, W) of each patch
        n_bands        : spectral bands (4 = RGB+NIR, 13 = full S2)
        cloud_density  : fraction of pixels that are cloud before smoothing

        Output layout (matches SyntheticDataset expectations)
        ------------------------------------------------------
        <data_dir>/
            clean/   img_0000.npy … shape (n_bands, H, W) in [0, 1]
            cloudy/  img_0000.npy … shape (n_bands, H, W) in [0, 1]
        """
        clean_dir  = self.data_dir / 'clean'
        cloudy_dir = self.data_dir / 'cloudy'
        clean_dir.mkdir(exist_ok=True)
        cloudy_dir.mkdir(exist_ok=True)

        print(f"Generating {n_samples} synthetic pairs "
              f"({n_bands} bands, {img_size[0]}×{img_size[1]}, "
              f"cloud_density={cloud_density:.0%}) …")

        for i in range(n_samples):
            clean_img  = self._generate_landscape(img_size, n_bands)
            cloudy_img = self._add_clouds(clean_img, cloud_density=cloud_density)
            np.save(clean_dir  / f'img_{i:04d}.npy', clean_img)
            np.save(cloudy_dir / f'img_{i:04d}.npy', cloudy_img)

            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  {i + 1}/{n_samples}")

        print(f"✓ Saved to {self.data_dir}")
        return clean_dir, cloudy_dir

    # ------------------------------------------------------------------

    def _generate_landscape(self, size, n_bands):
        """
        Procedural multi-band landscape using layered sine noise.

        Each band gets a slightly different spectral response so the
        cloud signal (bright, spectrally flat) is distinguishable from
        the surface signal (spectrally variable).
        """
        h, w = size
        x    = np.linspace(0, 4, w)
        y    = np.linspace(0, 4, h)
        X, Y = np.meshgrid(x, y)

        # Base terrain shared across bands (structural variation)
        terrain = (
            np.sin(X * 2)           + np.cos(Y * 2) +
            0.5  * np.sin(X * 4 + Y * 4) +
            0.25 * np.random.randn(h, w)
        )
        terrain = np.clip((terrain + 3) / 6, 0, 1)   # → [0, 1]

        img = np.zeros((n_bands, h, w), dtype=np.float32)
        for band in range(n_bands):
            # Scale + small per-band offset to mimic spectral variation
            scale  = 0.7 + band * 0.05
            offset = 0.02 * np.random.randn(h, w)
            img[band] = np.clip(terrain * scale + offset, 0.0, 1.0)

        return img

    def _add_clouds(self, img, cloud_density=0.3):
        """
        Simulate physically plausible cloud cover.

        Strategy
        --------
        1. Random binary seed at `cloud_density` fraction of pixels.
        2. Gaussian smoothing (sigma=10) to create soft cloud blobs,
           then threshold at 0.3 to get a binary cloud footprint.
        3. A separate smooth opacity map (0.7–1.0) makes cloud edges
           semi-transparent rather than hard-edged.
        4. Cloud pixels are bright (0.7–1.0) and spectrally flat,
           blended with the surface using per-pixel opacity.
        5. A faint additive haze is applied to all pixels to simulate
           aerosol scattering outside the main cloud body.

        The resulting |cloudy - clean| difference in cloud pixels is
        typically 0.2–0.6, which maps cleanly to the gt_diff mask
        (scaled by 0.30) used in SyntheticDataset.
        """
        n_bands, h, w = img.shape
        cloudy = img.copy()

        # ── 1. Cloud footprint ────────────────────────────────────────
        seed       = (np.random.rand(h, w) < cloud_density).astype(np.float32)
        blob       = gaussian_filter(seed, sigma=10)
        cloud_mask = blob > 0.3                         # bool (H, W)

        # ── 2. Per-pixel opacity inside cloud (0.7 – 1.0) ────────────
        opacity_raw = gaussian_filter(
            np.random.uniform(0.5, 1.0, (h, w)).astype(np.float32),
            sigma=5,
        )
        # Remap so opacity is 0.7–1.0 inside cloud, 0 outside
        opacity = np.where(cloud_mask,
                           0.7 + 0.3 * np.clip(opacity_raw, 0, 1),
                           0.0).astype(np.float32)

        # ── 3. Cloud reflectance (spectrally flat, bright) ────────────
        cloud_brightness = np.random.uniform(0.75, 1.0, (h, w)).astype(np.float32)
        cloud_brightness = gaussian_filter(cloud_brightness, sigma=3)

        # ── 4. Blend surface + cloud per band ─────────────────────────
        for band in range(n_bands):
            cloudy[band] = np.where(
                cloud_mask,
                (1.0 - opacity) * img[band] + opacity * cloud_brightness,
                img[band],
            )

        # ── 5. Additive haze (subtle, global) ────────────────────────
        haze = np.random.uniform(0.0, 0.05)
        cloudy = np.clip(cloudy + haze, 0.0, 1.0)

        return cloudy.astype(np.float32)
