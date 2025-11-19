"""
Configuration file for cabin expansion ML project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
METRICS_DIR = RESULTS_DIR / 'metrics'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Study area configuration (example coordinates for a Norwegian mountain region)
# You'll need to update these to your actual area of interest
# Study area configuration - Geilo Region
STUDY_AREA = {
    'name': 'Geilo_Region',
    'coordinates': [
        [8.50, 60.50],  # Southwest corner
        [8.65, 60.50],  # Southeast corner
        [8.65, 60.60],  # Northeast corner
        [8.50, 60.60],  # Northwest corner
        [8.50, 60.50]   # Close the polygon
    ]
}

# Time periods for analysis
# Time periods for analysis - UPDATED for better seasonal match
TIME_PERIODS = {
    'before_pandemic': {
        'start': '2019-07-15',  # Mid-July only
        'end': '2019-08-15'     # One month window
    },
    'after_pandemic': {
        'start': '2024-07-15',  # Same mid-July window
        'end': '2024-08-15'
    }
}

# Sentinel-2 bands to use
SENTINEL2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2

# Land cover classes
LAND_COVER_CLASSES = {
    0: 'Water',
    1: 'Forest',
    2: 'Grassland',
    3: 'Built_up',
    4: 'Bare_ground'
}

N_CLASSES = len(LAND_COVER_CLASSES)

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# SVM parameters
SVM_PARAMS = {
    'C': 10,
    'kernel': 'rbf',
    'gamma': 'scale',
    'random_state': RANDOM_STATE
}

# Neural Network parameters
NN_PARAMS = {
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'patience': 10
}

# CNN parameters (for patch-based classification)
CNN_PARAMS = {
    'patch_size': 32,
    'n_filters': [32, 64, 128],
    'kernel_size': 3,
    'pool_size': 2,
    'dropout_rate': 0.5,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'patience': 10
}

print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
