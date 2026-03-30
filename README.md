# Beyond Visible Spectrum: AI for Agriculture 2026

**1st place solution** for [ICPR 2026 Task 1 — Automated Multimodal Crop Disease Diagnosis](https://kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026).

**Final OOF F1 (macro): 0.6993 | OOF Accuracy: 0.7083**

---

## Task

3-class classification of wheat disease patches (Healthy / Rust / Other) using three aligned UAV modalities:

| Modality | Resolution | Bands |
|----------|-----------|-------|
| RGB | 64×64 | 3 (true color) |
| Multispectral | 64×64 | 5 (Blue/Green/Red/RedEdge/NIR) |
| Hyperspectral | 32×32 | 125 (450–950 nm, 4 nm/band) |

Training set: 600 samples (200 per class). Validation/test: 300 samples.

---

## Winning Approach: Feature Engineering + Gradient Boosting Ensemble

With only 600 training samples, handcrafted spectral features + gradient boosting outperformed deep learning approaches that reached 0.765 local OOF accuracy but overfit to the training distribution.

### Feature Extraction (`train_gbm.py`) — ~500+ features per sample

**RGB features:**
- Per-channel stats (mean, std, min, max, median, Q25/Q75, skewness) in RGB and HSV color spaces
- Color histograms (16 bins/channel) + entropy
- LBP texture (32-bin histogram on grayscale)
- Sobel gradient + Laplacian spatial features
- Green Ratio, Excess Green Index (ExG = 2G − R − B)

**Multispectral features:**
- Per-band statistics for all 5 bands
- 10+ vegetation indices: NDVI, NDRE, GNDVI, SAVI, EVI, MCARI, OSAVI, MSAVI, CI-green, CI-rededge, NDBI (7 stats each)
- All 10 pairwise band ratios (mean + std)
- Spatial gradients on NDVI channel
- Cross-band Pearson correlations

**Hyperspectral features:**
- Noise trimming: drop first 10 + last 14 bands
- Sampled band stats at 20 evenly-spaced bands
- 1st and 2nd order spectral derivatives (slope, position of max)
- Spectral shape: area, entropy, peak/trough band position + value
- Key wavelength ratios: red-edge mean/slope, NIR/red, green peak
- PCA (20 components): mean + std per component, explained variance ratios
- Spatial std + Sobel/Laplacian at blue, green, red, red-edge, NIR bands

### Ensemble

4 models trained with 5-fold stratified CV, predictions averaged (weighted by OOF F1):

| Model | OOF F1 |
|-------|--------|
| LightGBM | 0.6879 |
| XGBoost | 0.6899 |
| sklearn GradientBoosting | 0.6935 |
| ExtraTrees | 0.6824 |
| **Ensemble** | **0.6993** |

---

## What Didn't Win (but got to 0.765 locally)

`train_improved.py` and variants (`train_v3` through `train_v13`) explored:
- ConvNeXt-Tiny pretrained encoders for RGB + adapted multispectral
- Custom spectral attention encoder for hyperspectral
- Cross-modal attention fusion gates
- MixUp/CutMix augmentation, label smoothing, OneCycleLR
- 5-fold CV with 8× test-time augmentation

These overfit at 600 training samples. The ~0.07 gap between deep learning local OOF (0.765) and GBM local OOF (0.699) was the signal to trust the GBM for generalization.

---

## Repository Structure

```
train_gbm.py          # Winning solution: feature engineering + GBM ensemble
train_gbm_v2.py       # GBM v2 experiments
train_improved.py     # Advanced deep learning pipeline (ConvNeXt + attention fusion)
train_v3.py           # ... through train_v13.py: iterative DL experiments
train_finetune.py     # Fine-tuning experiments
icpr-2026-baseline-code-for-submission.ipynb  # Official baseline (ConvNeXt fusion)
```

---

## Setup

```bash
pip install numpy pandas scikit-learn lightgbm xgboost opencv-python tifffile
```

Data from the [competition page](https://kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026). Update `ROOT` path in `train_gbm.py` to your local data directory.

```bash
python train_gbm.py
```

---

## Key Takeaway

> For small agricultural remote sensing datasets (~600 samples), domain-informed spectral feature engineering with gradient boosting reliably outperforms end-to-end deep learning. Vegetation indices and spectral derivatives encode expert knowledge that models cannot learn from limited data.
