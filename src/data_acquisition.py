"""
Data acquisition from Google Earth Engine
Downloads Sentinel-2 imagery and ESA WorldCover data
"""
import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from tqdm import tqdm
import json

from config import (
    STUDY_AREA, TIME_PERIODS, SENTINEL2_BANDS,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)

class SatelliteDataDownloader:
    """Download and preprocess satellite imagery from Google Earth Engine"""

    def __init__(self):
        """Initialize Earth Engine"""
        try:
            ee.Initialize()
            print("Earth Engine initialized successfully")
        except Exception as e:
            print(f"Error initializing Earth Engine: {e}")
            print("Please run: earthengine authenticate")
            raise

    def create_aoi(self):
        """Create Area of Interest from config coordinates"""
        coords = STUDY_AREA['coordinates']
        return ee.Geometry.Polygon(coords)

    def get_sentinel2_composite(self, start_date, end_date, aoi):
        """
        Get cloud-free Sentinel-2 composite for given time period

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            aoi (ee.Geometry): Area of interest

        Returns:
            ee.Image: Cloud-masked composite image
        """
        def mask_s2_clouds(image):
            """Mask clouds using Sentinel-2 QA band"""
            qa = image.select('QA60')
            # Bits 10 and 11 are clouds and cirrus
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                qa.bitwiseAnd(cirrus_bit_mask).eq(0)
            )
            return image.updateMask(mask).divide(10000)

        # Get Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                     .map(mask_s2_clouds))

        # Create median composite
        composite = collection.median().clip(aoi)

        return composite.select(SENTINEL2_BANDS)

    def calculate_indices(self, image):
        """
        Calculate spectral indices

        Args:
            image (ee.Image): Sentinel-2 image

        Returns:
            ee.Image: Image with added index bands
        """
        # NDVI: (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        # NDWI: (Green - NIR) / (Green + NIR)
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

        # NDBI: (SWIR1 - NIR) / (SWIR1 + NIR)
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

        # Add indices to image
        return image.addBands([ndvi, ndwi, ndbi])

    def get_esa_worldcover(self, aoi, year=2021):
        """
        Get ESA WorldCover land cover classification

        Args:
            aoi (ee.Geometry): Area of interest
            year (int): Year of WorldCover data (2020 or 2021)

        Returns:
            ee.Image: Land cover classification
        """
        # ESA WorldCover 10m resolution
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
        return worldcover.select('Map').clip(aoi)

    def download_image(self, image, filename, aoi, scale=10):
        """
        Download Earth Engine image as GeoTIFF

        Args:
            image (ee.Image): Image to download
            filename (str): Output filename
            aoi (ee.Geometry): Area of interest
            scale (int): Resolution in meters
        """
        filepath = RAW_DATA_DIR / filename

        # For small areas, we can use geemap.download
        # The 'region' parameter can take the AOI geometry directly.
        try:
            geemap.ee_export_image(
                image,
                filename=str(filepath),
                scale=scale,
                region=aoi,  # <-- THIS IS THE FIX
                file_per_band=False
            )
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    def download_all_data(self):
        """Download all required data"""
        print("Starting data download...")

        # Create AOI
        aoi = self.create_aoi()

        # Save AOI for reference
        aoi_info = {
            'name': STUDY_AREA['name'],
            'coordinates': STUDY_AREA['coordinates'],
            'bounds': aoi.bounds().getInfo()
        }
        with open(RAW_DATA_DIR / 'aoi_info.json', 'w') as f:
            json.dump(aoi_info, f, indent=2)

        # Download 2019 (before pandemic)
        print("\n1. Downloading 2019 Sentinel-2 data...")
        s2_2019 = self.get_sentinel2_composite(
            TIME_PERIODS['before_pandemic']['start'],
            TIME_PERIODS['before_pandemic']['end'],
            aoi
        )
        s2_2019_indices = self.calculate_indices(s2_2019)
        self.download_image(s2_2019_indices, 'sentinel2_2019.tif', aoi)

        # Download 2024 (after pandemic)
        print("\n2. Downloading 2024 Sentinel-2 data...")
        s2_2024 = self.get_sentinel2_composite(
            TIME_PERIODS['after_pandemic']['start'],
            TIME_PERIODS['after_pandemic']['end'],
            aoi
        )
        s2_2024_indices = self.calculate_indices(s2_2024)
        self.download_image(s2_2024_indices, 'sentinel2_2024.tif', aoi)

        # Download ESA WorldCover for training labels
        print("\n3. Downloading ESA WorldCover data...")
        worldcover = self.get_esa_worldcover(aoi)
        self.download_image(worldcover, 'esa_worldcover.tif', aoi)

        print(f"Data saved to: {RAW_DATA_DIR}")

def main():
    """Main function to run data acquisition"""
    downloader = SatelliteDataDownloader()
    downloader.download_all_data()

if __name__ == "__main__":
    main()
