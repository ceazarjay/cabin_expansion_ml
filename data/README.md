# Data Directory

This directory contains satellite imagery and processed datasets.

## Structure
- `raw/` - Raw Sentinel-2 imagery from Google Earth Engine
  - `sentinel2_2019_summer.tif` - Pre-pandemic baseline
  - `sentinel2_2024_summer.tif` - Recent imagery
  - `esa_worldcover_2021.tif` - Training labels
- `processed/` - Preprocessed training/validation/test datasets
  - `X_train.npy`, `y_train.npy` - Training data
  - `X_val.npy`, `y_val.npy` - Validation data
  - `X_test.npy`, `y_test.npy` - Test data

## Note
Due to size constraints, raw data is not included in this repository.
To reproduce the analysis:
1. Set up Google Earth Engine authentication
2. Run `python src/data_acquisition.py`
3. Run `python src/data_preprocessing.py`

## Data Sources
- **Sentinel-2**: Copernicus Programme (ESA)
- **WorldCover**: ESA WorldCover 10m 2021
- **Resolution**: 10m spatial resolution
- **Bands**: B2, B3, B4, B8, B11, B12 + NDVI, NDWI, NDBI
