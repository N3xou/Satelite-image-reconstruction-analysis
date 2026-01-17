class SatelliteDatasetPreparer:
    """Handles sen12mscr_dataset acquisition and cloud simulation"""

    def __init__(self, data_dir='./satellite_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)


    def create_synthetic_dataset(self, n_samples=100, img_size=(256, 256), n_bands=4):
        """Create synthetic multispectral images with simulated clouds"""
        clean_dir = self.data_dir / 'clean'
        cloudy_dir = self.data_dir / 'cloudy'
        clean_dir.mkdir(exist_ok=True)
        cloudy_dir.mkdir(exist_ok=True)

        for i in range(n_samples):
            # Generate synthetic multispectral image (R, G, B, NIR)
            clean_img = self._generate_landscape(img_size, n_bands)

            # Add clouds
            cloudy_img = self._add_clouds(clean_img)

            # Save
            np.save(clean_dir / f'img_{i:04d}.npy', clean_img)
            np.save(cloudy_dir / f'img_{i:04d}.npy', cloudy_img)

        print(f"Created {n_samples} synthetic image pairs")
        return clean_dir, cloudy_dir

    def _generate_landscape(self, size, n_bands):
        """Generate realistic-looking landscape with multiple spectral bands"""
        h, w = size
        img = np.zeros((n_bands, h, w))

        for band in range(n_bands):
            # Create terrain with fractal noise
            x = np.linspace(0, 4, w)
            y = np.linspace(0, 4, h)
            X, Y = np.meshgrid(x, y)

            terrain = (np.sin(X * 2) + np.cos(Y * 2) +
                      0.5 * np.sin(X * 4 + Y * 4) +
                      0.25 * np.random.randn(h, w))

            # Different spectral responses per band
            img[band] = (terrain + 3) / 6 * (0.8 + band * 0.1)

        return np.clip(img, 0, 1).astype(np.float32)

    def _add_clouds(self, img, cloud_density=0.3):
        """Add realistic cloud cover to image"""
        n_bands, h, w = img.shape
        cloudy = img.copy()

        # Generate cloud mask
        cloud_mask = np.random.rand(h, w) < cloud_density

        # Smooth cloud edges
        from scipy.ndimage import gaussian_filter
        cloud_mask = gaussian_filter(cloud_mask.astype(float), sigma=10) > 0.3

        # Apply clouds (bright in all bands)
        for band in range(n_bands):
            cloud_values = np.random.uniform(0.7, 1.0, (h, w))
            cloudy[band] = np.where(cloud_mask, cloud_values, cloudy[band])

        return cloudy.astype(np.float32)