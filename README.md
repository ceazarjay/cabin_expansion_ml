# Quantifying Cabin Expansion in Norwegian Mountains Using Deep Learning

**A data-centric machine learning approach for detecting land cover changes in satellite imagery**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project uses machine learning with attention mechanisms to quantify cabin expansion in Norwegian mountain regions by analyzing Sentinel-2 satellite imagery. The methodology combines spectral indices with spatial feature extraction to detect land cover changes between 2019 and 2024.

**Key Finding:** Detected **21.1% increase** in built-up areas (cabins) in Trysil region over 5 years.

## Methodology

### Data-Centric Approach

Rather than focusing on model architecture, this project emphasizes **feature extraction and data quality**:

1. **Multi-Scale Feature Extraction** - Captures features at different spatial scales (3×3, 5×5, 7×7 kernels)
2. **Attention Mechanisms** - Learns to focus on discriminative spatial patterns
3. **Spectral Index Engineering** - Incorporates NDVI, NDWI, NDBI for domain knowledge
4. **Seasonal Consistency** - Mid-July imagery to minimize atmospheric variance

### Pipeline
```
Sentinel-2 Imagery (2019 & 2024)
         ↓
Feature Extraction (spectral + spatial)
         ↓
Model Training (RF, SVM, NN, Attention CNN, Multi-Scale CNN)
         ↓
Change Detection & Quantification
         ↓
Results: 21.1% cabin expansion
```

## Results

### Model Performance

| Model | Test Accuracy | Test Kappa | Test F1 | Reference |
|-------|--------------|------------|---------|-----------|
| Random Forest (Baseline) | 78.20% | 0.7275 | 0.7856 | — |
| SVM | 75.35% | 0.6919 | 0.7509 | — |
| Neural Network | 73.95% | 0.6744 | 0.7382 | — |
| **Attention-Enhanced CNN** | **81.50%** | **0.7688** | **0.8201** | Adegun et al. (2023) |
| **Multi-Scale CNN** | **80.90%** | **0.7612** | **0.8145** | Yang et al. (2019) |

**Best Model:** Attention-Enhanced CNN achieves **+3.3 percentage points** improvement over baseline.

### Land Cover Changes (Trysil Region, 2019→2024)

| Class | 2019 (km²) | 2024 (km²) | Change (km²) | Change (%) |
|-------|-----------|-----------|-------------|-----------|
| Water | 2.409 | 3.249 | +0.839 | +31.8% |
| Forest | 54.945 | 56.302 | +1.358 | +2.5% |
| Grassland | 19.726 | 16.724 | -3.002 | -15.2% |
| **Built-up** | **7.341** | **8.889** | **+1.548** | **+21.1%** ✓ |
| Bare ground | 2.582 | 1.839 | -0.743 | -28.8% |

## Requirements
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Python 3.10+
- PyTorch 2.0+ (GPU support recommended)
- scikit-learn 1.3+
- earthengine-api 0.1.360+
- rioxarray 0.15+
- pandas, numpy, matplotlib, seaborn

## Usage

### 1. Setup Google Earth Engine
```bash
earthengine authenticate
earthengine set_project forest-classification-473219
```

### 2. Configure Study Area

Edit `src/config.py`:
```python
STUDY_AREA = {
    'name': 'Trysil',
    'coordinates': [[lon, lat], ...]  # Polygon coordinates
}
```

### 3. Run Analysis Pipeline
```bash
# Download satellite imagery
python src/data_acquisition.py

# Preprocess data
python src/data_preprocessing.py

# Train baseline models
python src/train_models.py
python src/train_neural_network.py

# Train enhanced models (data-centric)
python src/train_with_attention.py
python src/train_multiscale.py

# Detect changes
python src/change_detection.py

# Generate visualizations
python src/visualize_enhanced_results.py
python src/generate_final_summary.py
```

## Project Structure
```
cabin_expansion_ml/
├── src/                          # Source code
│   ├── config.py                # Configuration
│   ├── data_acquisition.py      # Download Sentinel-2 data
│   ├── data_preprocessing.py    # Feature extraction
│   ├── train_models.py          # Baseline models
│   ├── train_with_attention.py  # Attention-enhanced CNN
│   ├── train_multiscale.py      # Multi-scale CNN
│   ├── change_detection.py      # Temporal analysis
│   └── visualize_enhanced_results.py
├── data/                         # Data directory
│   ├── raw/                     # Sentinel-2 imagery
│   └── processed/               # Preprocessed datasets
├── models/                       # Trained models
├── results/                      # Outputs
│   ├── figures/                 # Visualizations
│   ├── metrics/                 # Performance metrics
│   └── FINAL_PROJECT_SUMMARY.txt
├── requirements.txt
├── LICENSE
└── README.md
```

## Key Features

### 1. Attention Mechanisms (Adegun et al., 2023)
Spatial attention learns to focus on discriminative features like roof structures and access roads, addressing the challenge of grass-roofed cabins that are spectrally identical to vegetation.

### 2. Multi-Scale Feature Fusion (Yang et al., 2019)
Parallel convolutions at different kernel sizes capture both fine-grained details and broader spatial context, handling cabins of varying sizes (50m² to 200m²).

### 3. Spectral Index Engineering
- **NDVI**: Vegetation density
- **NDWI**: Water content  
- **NDBI**: Built-up areas

### 4. Regional Validation
Tested on two regions (Trysil and Geilo) to validate methodology robustness.

## Challenges & Solutions

**Challenge:** Norwegian cabins have grass roofs that are spectrally identical to surrounding vegetation.

**Solution:** Spatial attention mechanisms that learn discriminative patterns (roads, clearings, spatial clustering) rather than relying solely on spectral signatures.

## Dataset

- **Source**: Copernicus Sentinel-2 MultiSpectral Instrument (MSI)
- **Resolution**: 10m spatial resolution
- **Temporal**: July 2019 (pre-pandemic) vs. July 2024 (post-pandemic)
- **Bands**: B2, B3, B4, B8, B11, B12
- **Training Labels**: ESA WorldCover 2021 (10m global land cover)
- **Study Area**: Trysil mountain region, Norway

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{mamburam2025cabin,
  author = {Mamburam, Ceazar Jay},
  title = {Quantifying Cabin Expansion in Norwegian Mountains Using Deep Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ceazarjay/cabin_expansion_ml}}
}
```

## References

- Adegun, A. A., Viriri, S., & Tapamo, J. R. (2023). Review of deep learning methods for remote sensing satellite images classification. *Journal of Big Data*, 10(1), 93.
- Yang, R., Zhang, Y., Zhao, P., Ji, Z., & Deng, W. (2019). MSPPF-nets: A deep learning architecture for remote sensing image classification. *IGARSS 2019*.
- ESA WorldCover (2021). 10m resolution global land cover map.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Course**: UC3MAL102 Machine Learning, Noroff University College
- **Supervisor**: Isah A. Lawal
- **Data**: Copernicus Sentinel-2 (ESA), ESA WorldCover
- **Inspiration**: Research by Adegun et al. (2023) and Yang et al. (2019)

## Contact

Ceazar Jay Mamburam  
Applied Data Science, Noroff University College  
[ceamam00891@stud.noroff.no]

---

**Keywords:** Remote sensing, Satellite imagery, Deep learning, CNN, Attention mechanisms, Land cover change, Environmental monitoring, Norway
