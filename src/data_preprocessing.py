"""
Data preprocessing and feature extraction
Prepare training/validation/test datasets
"""
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LAND_COVER_CLASSES,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE
)

class DataPreprocessor:
    """Preprocess satellite imagery for ML models"""

    def __init__(self):
        self.scaler = StandardScaler()

    def load_geotiff(self, filepath):
        """Load GeoTIFF and return data + metadata"""
        with rasterio.open(filepath) as src:
            data = src.read()  # Shape: (bands, height, width)
            profile = src.profile
            transform = src.transform
        return data, profile, transform

    def remap_worldcover_to_classes(self, worldcover):
        """
        Remap ESA WorldCover classes to our simplified classes
        """
        output = np.zeros_like(worldcover)
        output[worldcover == 80] = 0
        output[(worldcover == 10) | (worldcover == 95)] = 1
        output[(worldcover == 20) | (worldcover == 30) | (worldcover == 40) |
               (worldcover == 90) | (worldcover == 100)] = 2
        output[worldcover == 50] = 3
        output[(worldcover == 60) | (worldcover == 70)] = 4
        return output

    def extract_pixels(self, sentinel_data, labels, n_samples_per_class=2000):
        """
        Extract pixel samples from imagery and labels
        """
        n_bands, height, width = sentinel_data.shape

        X_all = sentinel_data.reshape(n_bands, -1).T
        y_all = labels.flatten()

        valid_mask = ~np.isnan(X_all).any(axis=1) & (y_all != -9999)
        X_all = X_all[valid_mask]
        y_all = y_all[valid_mask]

        X_samples = []
        y_samples = []

        for class_id in LAND_COVER_CLASSES.keys():
            class_mask = y_all == class_id
            n_available = np.sum(class_mask)

            if n_available == 0:
                print(f"Warning: No samples found for class {LAND_COVER_CLASSES[class_id]}")
                continue

            n_sample = min(n_samples_per_class, n_available)
            indices = np.where(class_mask)[0]
            sampled_indices = np.random.choice(indices, n_sample, replace=False)

            X_samples.append(X_all[sampled_indices])
            y_samples.append(y_all[sampled_indices])

            print(f"Class {class_id} ({LAND_COVER_CLASSES[class_id]}): sampled {n_sample}/{n_available} pixels")

        X = np.vstack(X_samples)
        y = np.concatenate(y_samples)

        return X, y

    def prepare_train_test_split(self):
        """Prepare train/val/test splits"""
        print("\n" + "="*50)
        print("PREPARING TRAINING DATA")
        print("="*50)

        print("\n1. Loading Sentinel-2 2019 data...")
        s2_2019, profile, transform = self.load_geotiff(RAW_DATA_DIR / 'sentinel2_2019.tif')

        print("2. Loading ESA WorldCover labels...")
        worldcover, _, _ = self.load_geotiff(RAW_DATA_DIR / 'esa_worldcover.tif')
        worldcover = worldcover[0]

        # --- START: FIX ---
        # Ensure rasters have the same dimensions by cropping to the minimum
        s2_h, s2_w = s2_2019.shape[1], s2_2019.shape[2]
        wc_h, wc_w = worldcover.shape
        min_h = min(s2_h, wc_h)
        min_w = min(s2_w, wc_w)

        s2_2019 = s2_2019[:, :min_h, :min_w]
        worldcover = worldcover[:min_h, :min_w]
        print(f"Cropped arrays to common shape: ({min_h}, {min_w})")
        # --- END: FIX ---

        print("3. Remapping land cover classes...")
        labels = self.remap_worldcover_to_classes(worldcover)

        print("\n4. Extracting pixel samples...")
        X, y = self.extract_pixels(s2_2019, labels, n_samples_per_class=2000)

        print(f"\nTotal samples extracted: {len(X)}")
        print(f"Feature dimensions: {X.shape}")
        print(f"Class distribution:")
        for class_id, class_name in LAND_COVER_CLASSES.items():
            count = np.sum(y == class_id)
            print(f"  {class_name}: {count} ({100*count/len(y):.1f}%)")

        print("\n5. Splitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE),
            random_state=RANDOM_STATE, stratify=y_temp
        )

        print(f"Train set: {len(X_train)} samples")
        print(f"Val set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")

        print("\n6. Fitting StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n7. Saving processed data...")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        np.save(PROCESSED_DATA_DIR / 'X_train.npy', X_train)
        np.save(PROCESSED_DATA_DIR / 'X_val.npy', X_val)
        np.save(PROCESSED_DATA_DIR / 'X_test.npy', X_test)
        np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
        np.save(PROCESSED_DATA_DIR / 'y_val.npy', y_val)
        np.save(PROCESSED_DATA_DIR / 'y_test.npy', y_test)

        np.save(PROCESSED_DATA_DIR / 'X_train_scaled.npy', X_train_scaled)
        np.save(PROCESSED_DATA_DIR / 'X_val_scaled.npy', X_val_scaled)
        np.save(PROCESSED_DATA_DIR / 'X_test_scaled.npy', X_test_scaled)

        joblib.dump(self.scaler, PROCESSED_DATA_DIR / 'scaler.pkl')

        metadata = {
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'n_classes': len(LAND_COVER_CLASSES),
            'class_distribution': {
                'train': {LAND_COVER_CLASSES[i]: int(np.sum(y_train == i)) for i in LAND_COVER_CLASSES.keys()},
                'val': {LAND_COVER_CLASSES[i]: int(np.sum(y_val == i)) for i in LAND_COVER_CLASSES.keys()},
                'test': {LAND_COVER_CLASSES[i]: int(np.sum(y_test == i)) for i in LAND_COVER_CLASSES.keys()}
            }
        }

        with open(PROCESSED_DATA_DIR / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Processed data saved to: {PROCESSED_DATA_DIR}")

def main():
    """Main preprocessing function"""
    preprocessor = DataPreprocessor()
    preprocessor.prepare_train_test_split()

if __name__ == "__main__":
    main()
