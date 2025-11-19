# Models Directory

This directory contains trained machine learning models.

## Baseline Models
- `random_forest.pkl` - Random Forest classifier (78.2% accuracy)
- `svm.pkl` - Support Vector Machine (75.4% accuracy)
- `neural_network_best.pth` - Neural Network (74.0% accuracy)

## Enhanced Models (Data-Centric Approach)
- `attention_cnn_best.pth` - Attention-Enhanced CNN (81.5% accuracy)
  - Reference: Adegun et al. (2023)
  - Spatial attention mechanisms for feature extraction
- `multiscale_cnn_best.pth` - Multi-Scale CNN (80.9% accuracy)
  - Reference: Yang et al. (2019)
  - Multi-scale feature fusion (3×3, 5×5, 7×7 kernels)

## Model Sizes
- Random Forest: ~50 MB
- SVM: ~30 MB
- Neural Networks: ~10-20 MB each

## Note
Pre-trained models are not included due to size constraints.
To train models:
1. Complete data preprocessing
2. Run `python src/train_models.py` (baseline)
3. Run `python src/train_with_attention.py` (enhanced)
4. Run `python src/train_multiscale.py` (enhanced)
