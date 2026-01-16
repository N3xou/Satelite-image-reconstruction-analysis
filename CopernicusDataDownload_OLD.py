"""
Copernicus Sentinel-2 Data Downloader
=====================================
Download 300 images including 30 images of the same area from different dates
Requires 4 spectral bands (Blue, Green, Red, NIR)
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import json

try:
    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

    SENTINELSAT_AVAILABLE = True
except ImportError:
    print("ERROR: sentinelsat not installed!")
    print("Install with: pip install sentinelsat")
    SENTINELSAT_AVAILABLE = False

try:
    import geopandas as gpd

    GEOPANDAS_AVAILABLE = True
except ImportError:
    print("WARNING: geopandas not installed (optional)")
    print("Install with: pip install geopandas")
    GEOPANDAS_AVAILABLE = False


class CopernicusDownloader:
    """
    Downloads Sentinel-2 imagery from Copernicus Open Access Hub

    Requirements:
    1. Register at: https://scihub.copernicus.eu/dhus/#/self-registration
    2. Verify email
    3. Wait 24 hours for account activation
    """

    def __init__(self, username, password, output_dir='./sentinel2_data'):
        """
        Args:
            username: Copernicus username (email)
            password: Copernicus password
            output_dir: Directory to save downloaded data
        """
        if not SENTINELSAT_AVAILABLE:
            raise ImportError("sentinelsat is required. Install: pip install sentinelsat")

        self.api = SentinelAPI(
            username,
            password,
            'https://scihub.copernicus.eu/dhus'
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Connected to Copernicus Hub")
        print(f"Output directory: {self.output_dir}")

    def create_area_of_interest(self, lat, lon, buffer_km=10):
        """
        Create a GeoJSON for area of interest

        Args:
            lat: Latitude (e.g., 52.5200 for Berlin)
            lon: Longitude (e.g., 13.4050 for Berlin)
            buffer_km: Size of area in kilometers

        Returns:
            Path to GeoJSON file
        """
        # Create a simple bounding box
        # Approximate: 1 degree latitude = 111 km
        # 1 degree longitude = 111 km * cos(latitude)
        import math

        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * math.cos(math.radians(lat)))

        bbox = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon - lon_buffer, lat - lat_buffer],  # SW
                        [lon + lon_buffer, lat - lat_buffer],  # SE
                        [lon + lon_buffer, lat + lat_buffer],  # NE
                        [lon - lon_buffer, lat + lat_buffer],  # NW
                        [lon - lon_buffer, lat - lat_buffer]  # Close polygon
                    ]]
                }
            }]
        }

        geojson_path = self.output_dir / 'area_of_interest.geojson'
        with open(geojson_path, 'w') as f:
            json.dump(bbox, f, indent=2)

        print(f"Created area of interest: {geojson_path}")
        print(f"Center: ({lat}, {lon})")
        print(f"Size: ~{buffer_km * 2}km x {buffer_km * 2}km")

        return geojson_path

    def download_temporal_sequence(self, aoi_geojson, start_date, end_date,
                                   n_images=30, max_cloud=30):
        """
        Download temporal sequence: 30 images of the same area from different dates

        Args:
            aoi_geojson: Path to GeoJSON file with area of interest
            start_date: Start date (YYYYMMDD or 'YYYY-MM-DD')
            end_date: End date
            n_images: Number of images to download (default: 30)
            max_cloud: Maximum cloud coverage percentage (0-100)

        Returns:
            List of downloaded product IDs
        """
        print(f"\n{'=' * 70}")
        print(f"DOWNLOADING TEMPORAL SEQUENCE")
        print(f"{'=' * 70}")
        print(f"Target: {n_images} images of the same area")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Max cloud cover: {max_cloud}%")
        print(f"{'=' * 70}\n")

        # Read area of interest
        footprint = geojson_to_wkt(read_geojson(aoi_geojson))

        # Search for products
        products = self.api.query(
            footprint,
            date=(start_date, end_date),
            platformname='Sentinel-2',
            producttype='S2MSI2A',  # Level-2A (atmospherically corrected)
            cloudcoverpercentage=(0, max_cloud)
        )

        print(f"Found {len(products)} products matching criteria")

        if len(products) == 0:
            print("ERROR: No products found. Try:")
            print("  - Increasing date range")
            print("  - Increasing max_cloud percentage")
            print("  - Different area of interest")
            return []

        # Sort by date and select up to n_images
        products_df = self.api.to_dataframe(products)
        products_df = products_df.sort_values('beginposition')

        if len(products_df) > n_images:
            print(f"Selecting {n_images} out of {len(products_df)} available products")
            products_df = products_df.head(n_images)

        # Display selection
        print(f"\nSelected {len(products_df)} images:")
        print("-" * 70)
        for idx, row in products_df.iterrows():
            print(f"  {row['title']}")
            print(f"    Date: {row['beginposition']}")
            print(f"    Cloud: {row['cloudcoverpercentage']:.1f}%")
            print(f"    Size: {row['size']}")

        # Download
        print(f"\n{'=' * 70}")
        print(f"Starting download to: {self.output_dir}")
        print(f"{'=' * 70}\n")

        downloaded_ids = []
        for idx, (product_id, product_info) in enumerate(products_df.iterrows(), 1):
            print(f"\n[{idx}/{len(products_df)}] Downloading: {product_info['title']}")

            try:
                self.api.download(product_id, directory_path=self.output_dir)
                downloaded_ids.append(product_id)
                print(f"  ✓ Downloaded successfully")
            except Exception as e:
                print(f"  ✗ Download failed: {e}")

        print(f"\n{'=' * 70}")
        print(f"Downloaded {len(downloaded_ids)} / {len(products_df)} products")
        print(f"{'=' * 70}\n")

        # Save download log
        log_file = self.output_dir / 'temporal_sequence_log.txt'
        with open(log_file, 'w') as f:
            f.write(f"Temporal Sequence Download Log\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"Download date: {datetime.now()}\n")
            f.write(f"AOI: {aoi_geojson}\n")
            f.write(f"Date range: {start_date} to {end_date}\n")
            f.write(f"Target images: {n_images}\n")
            f.write(f"Downloaded: {len(downloaded_ids)}\n\n")

            f.write("Product List:\n")
            f.write("-" * 70 + "\n")
            for product_id in downloaded_ids:
                info = products_df.loc[product_id]
                f.write(f"{info['title']}\n")
                f.write(f"  Date: {info['beginposition']}\n")
                f.write(f"  Cloud: {info['cloudcoverpercentage']:.1f}%\n\n")

        print(f"Log saved to: {log_file}")

        return downloaded_ids

    def download_diverse_areas(self, locations, date_range, images_per_location=27,
                               max_cloud=30):
        """
        Download images from multiple different areas

        Args:
            locations: List of (name, lat, lon) tuples
            date_range: Tuple of (start_date, end_date)
            images_per_location: Images to download per location
            max_cloud: Maximum cloud coverage

        Returns:
            Dictionary mapping location names to downloaded product IDs
        """
        print(f"\n{'=' * 70}")
        print(f"DOWNLOADING FROM MULTIPLE AREAS")
        print(f"{'=' * 70}")
        print(f"Locations: {len(locations)}")
        print(f"Images per location: {images_per_location}")
        print(f"Total target: {len(locations) * images_per_location} images")
        print(f"{'=' * 70}\n")

        start_date, end_date = date_range
        all_downloads = {}

        for loc_idx, (name, lat, lon) in enumerate(locations, 1):
            print(f"\n{'=' * 70}")
            print(f"LOCATION {loc_idx}/{len(locations)}: {name}")
            print(f"{'=' * 70}")

            # Create AOI for this location
            aoi_file = self.create_area_of_interest(lat, lon, buffer_km=10)

            # Download images
            downloaded = self.download_temporal_sequence(
                aoi_file,
                start_date,
                end_date,
                n_images=images_per_location,
                max_cloud=max_cloud
            )

            all_downloads[name] = downloaded

        # Summary
        total_downloaded = sum(len(ids) for ids in all_downloads.values())
        print(f"\n{'=' * 70}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total downloaded: {total_downloaded} images")
        print(f"Breakdown:")
        for name, ids in all_downloads.items():
            print(f"  {name}: {len(ids)} images")
        print(f"{'=' * 70}\n")

        return all_downloads

    def get_available_bands(self, product_path):
        """
        List available spectral bands in a downloaded product

        Args:
            product_path: Path to .SAFE folder

        Returns:
            Dictionary of band numbers to file paths
        """
        product_path = Path(product_path)

        if not product_path.exists():
            print(f"ERROR: Product not found: {product_path}")
            return {}

        # Find granule directory
        granule_dir = product_path / 'GRANULE'
        if not granule_dir.exists():
            print(f"ERROR: No GRANULE directory in {product_path}")
            return {}

        granules = list(granule_dir.iterdir())
        if not granules:
            print(f"ERROR: No granules found in {granule_dir}")
            return {}

        granule = granules[0]

        # Check different resolutions
        bands = {}

        # 10m bands (B02, B03, B04, B08)
        img_10m = granule / 'IMG_DATA' / 'R10m'
        if img_10m.exists():
            for band_file in img_10m.glob('*_B*.jp2'):
                band_name = band_file.stem.split('_')[-2]  # Extract B02, B03, etc.
                bands[band_name] = band_file

        # 20m bands
        img_20m = granule / 'IMG_DATA' / 'R20m'
        if img_20m.exists():
            for band_file in img_20m.glob('*_B*.jp2'):
                band_name = band_file.stem.split('_')[-2]
                bands[band_name] = band_file

        # 60m bands
        img_60m = granule / 'IMG_DATA' / 'R60m'
        if img_60m.exists():
            for band_file in img_60m.glob('*_B*.jp2'):
                band_name = band_file.stem.split('_')[-2]
                bands[band_name] = band_file

        print(f"\nAvailable bands in {product_path.name}:")
        print("-" * 50)

        band_info = {
            'B01': '60m - Coastal aerosol',
            'B02': '10m - Blue',
            'B03': '10m - Green',
            'B04': '10m - Red',
            'B05': '20m - Vegetation Red Edge',
            'B06': '20m - Vegetation Red Edge',
            'B07': '20m - Vegetation Red Edge',
            'B08': '10m - NIR',
            'B8A': '20m - Narrow NIR',
            'B09': '60m - Water vapour',
            'B10': '60m - SWIR - Cirrus',
            'B11': '20m - SWIR',
            'B12': '20m - SWIR',
        }

        for band in sorted(bands.keys()):
            info = band_info.get(band, 'Unknown')
            print(f"  {band}: {info}")
            print(f"    Path: {bands[band]}")

        return bands


# ==================== USAGE EXAMPLE ====================

def download_300_images_example():
    """
    Complete example: Download 300 images including 30 temporal sequence
    """

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Copernicus Sentinel-2 Data Downloader                          ║
    ║  Download 300 images (30 temporal + 270 from diverse areas)     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # ===== CONFIGURATION =====

    # 1. Your Copernicus credentials
    USERNAME = 'your_email@example.com'  # REPLACE WITH YOUR EMAIL
    PASSWORD = 'your_password'  # REPLACE WITH YOUR PASSWORD

    # 2. Output directory
    OUTPUT_DIR = './sentinel2_300_images'

    # 3. Date range (choose a period with good satellite coverage)
    START_DATE = '20230101'  # January 1, 2023
    END_DATE = '20231231'  # December 31, 2023

    # 4. Maximum cloud coverage (%)
    MAX_CLOUD = 30

    # ===== STEP 1: Initialize Downloader =====

    print("\n[STEP 1] Initializing downloader...")
    downloader = CopernicusDownloader(USERNAME, PASSWORD, OUTPUT_DIR)

    # ===== STEP 2: Download 30 Temporal Images (Same Area) =====

    print("\n[STEP 2] Downloading 30 temporal images (same area)...")

    # Example: Berlin, Germany
    primary_location = {
        'name': 'Berlin',
        'lat': 52.5200,
        'lon': 13.4050
    }

    # Create AOI for primary location
    primary_aoi = downloader.create_area_of_interest(
        primary_location['lat'],
        primary_location['lon'],
        buffer_km=10  # 20km x 20km area
    )

    # Download 30 images from different dates
    temporal_downloads = downloader.download_temporal_sequence(
        primary_aoi,
        START_DATE,
        END_DATE,
        n_images=30,
        max_cloud=MAX_CLOUD
    )

    print(f"\n✓ Downloaded {len(temporal_downloads)} temporal images")

    # ===== STEP 3: Download 270 Images from Diverse Areas =====

    print("\n[STEP 3] Downloading 270 images from diverse locations...")

    # Define 9 different locations (30 images each = 270 total)
    diverse_locations = [
        ('Paris_France', 48.8566, 2.3522),
        ('London_UK', 51.5074, -0.1278),
        ('Rome_Italy', 41.9028, 12.4964),
        ('Madrid_Spain', 40.4168, -3.7038),
        ('Amsterdam_Netherlands', 52.3676, 4.9041),
        ('Vienna_Austria', 48.2082, 16.3738),
        ('Prague_Czech', 50.0755, 14.4378),
        ('Warsaw_Poland', 52.2297, 21.0122),
        ('Budapest_Hungary', 47.4979, 19.0402),
    ]

    # Download images from each location
    diverse_downloads = downloader.download_diverse_areas(
        diverse_locations,
        (START_DATE, END_DATE),
        images_per_location=30,  # 30 images per location
        max_cloud=MAX_CLOUD
    )

    # ===== STEP 4: Verify Downloads =====

    print("\n[STEP 4] Verifying downloads...")

    total_temporal = len(temporal_downloads)
    total_diverse = sum(len(ids) for ids in diverse_downloads.values())
    total_downloaded = total_temporal + total_diverse

    print(f"\n{'=' * 70}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"Temporal sequence (same area): {total_temporal} images")
    print(f"Diverse locations: {total_diverse} images")
    print(f"Total downloaded: {total_downloaded} / 300 target images")
    print(f"{'=' * 70}")

    # ===== STEP 5: Check Bands in First Product =====

    print("\n[STEP 5] Checking available bands...")

    # Find first downloaded product
    safe_folders = list(Path(OUTPUT_DIR).glob('*.SAFE'))
    if safe_folders:
        first_product = safe_folders[0]
        bands = downloader.get_available_bands(first_product)

        # Verify we have the required 4 bands
        required_bands = ['B02', 'B03', 'B04', 'B08']
        missing_bands = [b for b in required_bands if b not in bands]

        if missing_bands:
            print(f"\n⚠ WARNING: Missing required bands: {missing_bands}")
        else:
            print(f"\n✓ All required bands available (Blue, Green, Red, NIR)")

    print(f"\n{'=' * 70}")
    print(f"DOWNLOAD COMPLETE!")
    print(f"Data location: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


def quick_test_download():
    """
    Quick test: Download just a few images to verify setup
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Quick Test Download (10 images)                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    USERNAME = 'your_email@example.com'  # REPLACE
    PASSWORD = 'your_password'  # REPLACE

    downloader = CopernicusDownloader(USERNAME, PASSWORD, './test_download')

    # Create small test area
    aoi = downloader.create_area_of_interest(52.5200, 13.4050, buffer_km=5)

    # Download just 10 images for testing
    downloads = downloader.download_temporal_sequence(
        aoi,
        '20230601',
        '20230630',
        n_images=10,
        max_cloud=20
    )

    print(f"\n✓ Test complete! Downloaded {len(downloads)} images")
    print("If successful, modify download_300_images_example() with your credentials")


# ==================== MAIN ====================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Copernicus Sentinel-2 Downloader                               ║
    ║  Requirements:                                                   ║
    ║  1. pip install sentinelsat                                      ║
    ║  2. Register at: https://scihub.copernicus.eu/dhus/              ║
    ║  3. Wait 24 hours for account activation                         ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    print("\nChoose an option:")
    print("1. Quick test (10 images)")
    print("2. Full download (300 images)")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        print("\n[WARNING] Remember to update USERNAME and PASSWORD in the code!")
        input("Press Enter to continue...")
        quick_test_download()
    elif choice == '2':
        print("\n[WARNING] Remember to update USERNAME and PASSWORD in the code!")
        print("[INFO] This will download ~100GB of data and may take several hours")
        input("Press Enter to continue...")
        download_300_images_example()
    else:
        print("Exiting...")

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Next Steps:                                                     ║
    ║  1. Use Sentinel2Dataset from sentinel2_implementation.py       ║
    ║  2. Point it to your downloaded .SAFE folders                    ║
    ║  3. Train models with real satellite data!                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)