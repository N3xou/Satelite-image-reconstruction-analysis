"""
Sentinel-2 Cloud Removal - Real Data Implementation
===================================================
Complete implementation for working with actual Sentinel-2 satellite imagery
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Optional imports (install as needed)
try:
    import rasterio
    from rasterio.windows import Window

    RASTERIO_AVAILABLE = True
except ImportError:
    print("Install rasterio: pip install rasterio")
    RASTERIO_AVAILABLE = False

try:
    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

    SENTINELSAT_AVAILABLE = True
except ImportError:
    print("Install sentinelsat: pip install sentinelsat")
    SENTINELSAT_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter

    SCIPY_AVAILABLE = True
except ImportError:
    print("Install scipy: pip install scipy")
    SCIPY_AVAILABLE = False


# ==================== SENTINEL-2 DATA DOWNLOADER ====================

class Sentinel2Downloader:
    """
    Downloads Sentinel-2 data from Copernicus Hub

    Setup:
    1. Register at: https://scihub.copernicus.eu/dhus/
    2. Get username and password
    3. Create GeoJSON file with your area of interest
    """

    def __init__(self, username, password):
        if not SENTINELSAT_AVAILABLE:
            raise ImportError("Please install: pip install sentinelsat")

        self.api = SentinelAPI(
            username,
            password,
            'https://scihub.copernicus.eu/dhus'
        )

    def search_images(self, aoi_geojson, date_start, date_end,
                      max_cloud_cover=30, product_type='S2MSI2A'):
        """
        Search for Sentinel-2 images

        Args:
            aoi_geojson: Path to GeoJSON file with area of interest
            date_start: Start date (YYYYMMDD or 'YYYYMMDD')
            date_end: End date
            max_cloud_cover: Maximum cloud coverage percentage (0-100)
            product_type: 'S2MSI1C' (Level-1C) or 'S2MSI2A' (Level-2A, recommended)

        Returns:
            Dictionary of found products
        """
        footprint = geojson_to_wkt(read_geojson(aoi_geojson))

        products = self.api.query(
            footprint,
            date=(date_start, date_end),
            platformname='Sentinel-2',
            cloudcoverpercentage=(0, max_cloud_cover),
            producttype=product_type
        )

        print(f"Found {len(products)} products")

        # Display product info
        for product_id, product_info in products.items():
            print(f"\nProduct: {product_info['title']}")
            print(f"  Date: {product_info['beginposition']}")
            print(f"  Cloud cover: {product_info['cloudcoverpercentage']:.2f}%")
            print(f"  Size: {product_info['size']}")

        return products

    def download_images(self, products, output_dir='./sentinel2_data'):
        """
        Download products

        Args:
            products: Products from search_images()
            output_dir: Where to save downloaded files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nDownloading to: {output_dir}")
        self.api.download_all(products, directory_path=output_dir)

        print("Download complete!")
        return output_dir


# ==================== SENTINEL-2 BAND INFORMATION ====================

SENTINEL2_BANDS = {
    'B01': {'name': 'Coastal aerosol', 'resolution': 60, 'wavelength': '443nm'},
    'B02': {'name': 'Blue', 'resolution': 10, 'wavelength': '490nm'},
    'B03': {'name': 'Green', 'resolution': 10, 'wavelength': '560nm'},
    'B04': {'name': 'Red', 'resolution': 10, 'wavelength': '665nm'},
    'B05': {'name': 'Vegetation Red Edge', 'resolution': 20, 'wavelength': '705nm'},
    'B06': {'name': 'Vegetation Red Edge', 'resolution': 20, 'wavelength': '740nm'},
    'B07': {'name': 'Vegetation Red Edge', 'resolution': 20, 'wavelength': '783nm'},
    'B08': {'name': 'NIR', 'resolution': 10, 'wavelength': '842nm'},
    'B8A': {'name': 'Narrow NIR', 'resolution': 20, 'wavelength': '865nm'},
    'B09': {'name': 'Water vapour', 'resolution': 60, 'wavelength': '945nm'},
    'B10': {'name': 'SWIR - Cirrus', 'resolution': 60, 'wavelength': '1375nm'},
    'B11': {'name': 'SWIR', 'resolution': 20, 'wavelength': '1610nm'},
    'B12': {'name': 'SWIR', 'resolution': 20, 'wavelength': '2190nm'},
}


# ==================== CLOUD DETECTION METHODS ====================

class CloudDetector:
    """
    Multiple cloud detection methods for Sentinel-2
    """

    @staticmethod
    def brightness_threshold(image, threshold=0.3):
        """
        Method 1: Simple brightness thresholding

        Args:
            image: Multi-band image [bands, height, width]
            threshold: Brightness threshold (0-1)

        Returns:
            Boolean cloud mask
        """
        brightness = image.mean(axis=0)
        return brightness > threshold

    @staticmethod
    def multispectral_method(bands_dict):
        """
        Method 2: Multi-spectral cloud detection

        Args:
            bands_dict: Dictionary with keys 'B02', 'B03', 'B04', 'B08', 'B11', 'B12'
                       Each value is a 2D array

        Returns:
            Boolean cloud mask
        """
        B02 = bands_dict['B02']  # Blue
        B03 = bands_dict['B03']  # Green
        B04 = bands_dict['B04']  # Red
        B08 = bands_dict['B08']  # NIR
        B11 = bands_dict['B11']  # SWIR1
        B12 = bands_dict['B12']  # SWIR2

        # Clouds are bright in visible bands
        blue_bright = B02 > 0.175

        # NDSI (Normalized Difference Snow Index) - separates clouds from snow
        ndsi = (B03 - B11) / (B03 + B11 + 1e-10)
        not_snow = ndsi < 0.4

        # Clouds have low SWIR reflectance
        swir_low = B11 < 0.3

        # Combine criteria
        cloud_mask = blue_bright & not_snow & swir_low

        # Smooth the mask
        if SCIPY_AVAILABLE:
            cloud_mask = gaussian_filter(cloud_mask.astype(float), sigma=2) > 0.5

        return cloud_mask

    @staticmethod
    def load_scl_mask(scl_band_path):
        """
        Method 3: Use official Sentinel-2 Scene Classification (SCL) band

        Args:
            scl_band_path: Path to SCL band file

        Returns:
            Boolean cloud mask
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Install rasterio: pip install rasterio")

        with rasterio.open(scl_band_path) as src:
            scl = src.read(1)

        # SCL Classification:
        # 0 = No Data, 1 = Saturated/Defective, 2 = Dark Area
        # 3 = Cloud Shadows, 4 = Vegetation, 5 = Bare Soils
        # 6 = Water, 7 = Unclassified, 8 = Cloud Medium Probability
        # 9 = Cloud High Probability, 10 = Thin Cirrus, 11 = Snow/Ice

        # Define cloud classes
        cloud_classes = [3, 8, 9, 10]  # Shadows, Medium/High prob clouds, Cirrus

        cloud_mask = np.isin(scl, cloud_classes)

        return cloud_mask


# ==================== SENTINEL-2 DATASET LOADER ====================

class Sentinel2Dataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 imagery

    Supports:
    - Loading from .SAFE folders
    - Multiple spectral bands
    - Automatic cloud detection
    - Patch extraction for training
    """

    def __init__(self, data_dir, bands=[2, 3, 4, 8], tile_size=256,
                 cloud_detection='brightness', use_scl=True):
        """
        Args:
            data_dir: Directory containing .SAFE folders
            bands: List of band numbers to load (e.g., [2,3,4,8] for RGB+NIR)
            tile_size: Size of extracted patches
            cloud_detection: 'brightness', 'multispectral', or 'scl'
            use_scl: If True, use official cloud mask from SCL band
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("Install rasterio: pip install rasterio")

        self.data_dir = Path(data_dir)
        self.bands = bands
        self.tile_size = tile_size
        self.cloud_detection = cloud_detection
        self.use_scl = use_scl

        # Find all Sentinel-2 products
        self.products = self._find_products()
        print(f"Found {len(self.products)} Sentinel-2 products")

        # Generate tile positions for each product
        self.tiles = self._generate_tile_positions()
        print(f"Generated {len(self.tiles)} training tiles")

    def _find_products(self):
        """Find all .SAFE folders"""
        products = []

        for safe_dir in self.data_dir.glob('*.SAFE'):
            # Find granule directory
            granule_dirs = list(safe_dir.glob('GRANULE/*/'))

            if granule_dirs:
                product_info = {
                    'safe_dir': safe_dir,
                    'granule_dir': granule_dirs[0],
                    'band_files': {}
                }

                # Find band files (10m resolution)
                img_data_dir = granule_dirs[0] / 'IMG_DATA' / 'R10m'

                if img_data_dir.exists():
                    for band in self.bands:
                        pattern = f'*_B{band:02d}_10m.jp2'
                        band_files = list(img_data_dir.glob(pattern))

                        if band_files:
                            product_info['band_files'][band] = band_files[0]

                # Find SCL band (20m resolution)
                scl_dir = granule_dirs[0] / 'IMG_DATA' / 'R20m'
                if scl_dir.exists():
                    scl_files = list(scl_dir.glob('*_SCL_20m.jp2'))
                    if scl_files:
                        product_info['scl_file'] = scl_files[0]

                # Only add if all required bands found
                if len(product_info['band_files']) == len(self.bands):
                    products.append(product_info)

        return products

    def _generate_tile_positions(self, tiles_per_image=50):
        """Generate random tile positions for each product"""
        tiles = []

        for product in self.products:
            # Get image dimensions
            with rasterio.open(list(product['band_files'].values())[0]) as src:
                height, width = src.shape

            # Generate random positions
            for _ in range(tiles_per_image):
                if height > self.tile_size and width > self.tile_size:
                    row = np.random.randint(0, height - self.tile_size)
                    col = np.random.randint(0, width - self.tile_size)

                    tiles.append({
                        'product': product,
                        'row': row,
                        'col': col
                    })

        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        """
        Returns:
            cloudy_image: Tensor [bands, height, width]
            clean_image: Tensor [bands, height, width] (ground truth)
        """
        tile_info = self.tiles[idx]
        product = tile_info['product']
        row, col = tile_info['row'], tile_info['col']

        # Define window
        window = Window(col, row, self.tile_size, self.tile_size)

        # Load bands
        image = []
        for band in self.bands:
            with rasterio.open(product['band_files'][band]) as src:
                band_data = src.read(1, window=window)
                # Normalize to [0, 1]
                band_data = np.clip(band_data / 10000.0, 0, 1)
                image.append(band_data)

        image = np.stack(image).astype(np.float32)

        # Get cloud mask
        if self.use_scl and 'scl_file' in product:
            cloud_mask = self._load_scl_mask(product['scl_file'], window)
        else:
            cloud_mask = self._detect_clouds(image)

        # Create cloudy version
        cloudy_image = self._apply_cloud_mask(image, cloud_mask)

        return torch.from_numpy(cloudy_image), torch.from_numpy(image)

    def _load_scl_mask(self, scl_file, window):
        """Load and resize SCL mask to match 10m bands"""
        with rasterio.open(scl_file) as src:
            # SCL is 20m resolution, need to adjust window
            scl_window = Window(
                window.col_off // 2,
                window.row_off // 2,
                window.width // 2,
                window.height // 2
            )
            scl = src.read(1, window=scl_window)

        # Upsample to 10m resolution
        from scipy.ndimage import zoom
        scl_upsampled = zoom(scl, 2, order=0)  # Nearest neighbor

        # Crop/pad to exact tile size
        h, w = scl_upsampled.shape
        if h > self.tile_size:
            scl_upsampled = scl_upsampled[:self.tile_size, :self.tile_size]
        elif h < self.tile_size:
            pad_h = self.tile_size - h
            pad_w = self.tile_size - w
            scl_upsampled = np.pad(scl_upsampled, ((0, pad_h), (0, pad_w)))

        # Cloud classes
        cloud_classes = [3, 8, 9, 10]
        cloud_mask = np.isin(scl_upsampled, cloud_classes)

        return cloud_mask

    def _detect_clouds(self, image):
        """Detect clouds using specified method"""
        if self.cloud_detection == 'brightness':
            return CloudDetector.brightness_threshold(image)
        elif self.cloud_detection == 'multispectral':
            # Need to load additional bands
            bands_dict = {f'B{i:02d}': image[j] for j, i in enumerate(self.bands)}
            return CloudDetector.multispectral_method(bands_dict)
        else:
            # Default to brightness
            return CloudDetector.brightness_threshold(image)

    def _apply_cloud_mask(self, image, cloud_mask):
        """Apply cloud mask to create cloudy image"""
        cloudy = image.copy()

        # Add clouds with realistic spectral signature
        for i in range(len(self.bands)):
            cloud_value = np.random.uniform(0.6, 0.9)
            cloudy[i][cloud_mask] = cloud_value

        return cloudy


# ==================== USAGE EXAMPLES ====================

def example_1_download_data():
    """
    Example 1: Download Sentinel-2 data
    """
    print("\n=== EXAMPLE 1: Download Sentinel-2 Data ===\n")

    # You need to register at: https://scihub.copernicus.eu/dhus/
    downloader = Sentinel2Downloader(
        username='YOUR_USERNAME',  # Replace with your username
        password='YOUR_PASSWORD'  # Replace with your password
    )

    # Create area of interest (GeoJSON)
    # You can create this at: https://geojson.io/
    aoi_geojson = 'area_of_interest.geojson'

    # Search for images
    products = downloader.search_images(
        aoi_geojson=aoi_geojson,
        date_start='20240101',
        date_end='20240131',
        max_cloud_cover=30
    )

    # Download
    downloader.download_images(products, output_dir='./sentinel2_data')


def example_2_load_and_train():
    """
    Example 2: Load Sentinel-2 data and train models
    """
    print("\n=== EXAMPLE 2: Load and Train ===\n")

    # Load sen12mscr_dataset
    dataset = Sentinel2Dataset(
        data_dir='./sentinel2_data',
        bands=[2, 3, 4, 8],  # Blue, Green, Red, NIR
        tile_size=256,
        cloud_detection='brightness',
        use_scl=True  # Use official cloud mask
    )

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Train model (import from main script)
    from main import ModelTrainer, ModelEvaluator

    trainer = ModelTrainer('UNet')
    models, histories = trainer.train_kfold(
        train_dataset,
        k=5,
        epochs=30,
        batch_size=8,
        lr=0.001
    )

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(models[0], test_loader, 'UNet')

    print(f"\nResults: PSNR = {metrics['PSNR']:.2f} dB")


def example_3_visualize_data():
    """
    Example 3: Visualize Sentinel-2 data
    """
    print("\n=== EXAMPLE 3: Visualize Data ===\n")

    import matplotlib.pyplot as plt

    dataset = Sentinel2Dataset(
        data_dir='./sentinel2_data',
        bands=[2, 3, 4, 8],
        tile_size=256,
        use_scl=True
    )

    # Get sample
    cloudy, clean = dataset[0]

    # Convert to numpy and RGB
    cloudy_rgb = cloudy[:3].permute(1, 2, 0).numpy()
    clean_rgb = clean[:3].permute(1, 2, 0).numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cloudy_rgb)
    axes[0].set_title('Cloudy Image')
    axes[0].axis('off')

    axes[1].imshow(clean_rgb)
    axes[1].set_title('Clean Image (Ground Truth)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('sentinel2_sample.png')
    print("Saved: sentinel2_sample.png")


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("SENTINEL-2 CLOUD REMOVAL - REAL DATA IMPLEMENTATION")
    print("=" * 60)

    print("\nAvailable examples:")
    print("1. Download Sentinel-2 data")
    print("2. Load and train models")
    print("3. Visualize data")

    print("\nTo run examples:")
    print("- Uncomment the example function at the bottom")
    print("- Make sure you have Sentinel-2 data downloaded")

    # Uncomment to run:
    # example_1_download_data()
    # example_2_load_and_train()
    # example_3_visualize_data()

    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS")
    print("=" * 60)
    print("""
    1. Register at Copernicus:
       https://scihub.copernicus.eu/dhus/

    2. Install required packages:
       pip install sentinelsat rasterio scipy

    3. Create area of interest GeoJSON:
       https://geojson.io/

    4. Run example_1_download_data() to download images

    5. Run example_2_load_and_train() to train models

    6. Use trained models for cloud removal!
    """)
    print("=" * 60)