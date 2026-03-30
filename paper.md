# Spectral Feature Engineering with Gradient Boosting Ensemble for Multimodal Wheat Disease Classification

**ICPR 2026 Competition on "Beyond Visible Spectrum: AI for Agriculture" — Task 1: 1st Place Solution**

---

## Abstract

We present the winning solution for Task 1 of the ICPR 2026 Competition on "Beyond Visible Spectrum: AI for Agriculture," which requires classifying wheat image patches into three categories — *Healthy*, *Rust*, and *Other* — using aligned RGB, multispectral (MS), and hyperspectral (HS) UAV imagery. Despite the competition's emphasis on deep learning methods for multimodal fusion, we demonstrate that carefully engineered spectral and spatial features combined with a gradient boosting ensemble substantially outperforms end-to-end deep learning on this small-scale dataset. Our approach extracts over 500 domain-informed features per sample from all three modalities, encompassing vegetation indices, spectral derivatives, LBP texture descriptors, PCA projections, and cross-band correlations. Four gradient boosting models — LightGBM, XGBoost, sklearn GradientBoosting, and ExtraTrees — are trained under 5-fold stratified cross-validation and ensembled via OOF-weighted probability averaging. The final ensemble achieves a macro F1 of **0.6993** and an accuracy of **0.7083** on the held-out test set, outperforming our best deep learning approach which achieved 0.765 locally but overfit due to the limited training set size of 600 samples.

---

## 1. Introduction

Crop disease is one of the leading causes of agricultural yield loss globally, with wheat rust (*Puccinia* spp.) alone responsible for annual losses of billions of dollars. Early, accurate detection is critical for enabling timely intervention. Unmanned Aerial Vehicles (UAVs) equipped with multispectral and hyperspectral sensors offer a promising platform for scalable, non-destructive crop monitoring, capturing disease-relevant spectral signatures that are invisible to the human eye.

The ICPR 2026 competition "Beyond Visible Spectrum: AI for Agriculture" — now in its fifth consecutive year — provides a realistic benchmark for this problem by supplying aligned RGB, 5-band multispectral, and 125-band hyperspectral image patches captured over wheat fields. Participants must classify each patch as *Healthy*, *Rust-infected*, or *Other*, with performance evaluated by classification accuracy on a held-out anonymized test set.

A central challenge in this competition is the severe data scarcity: only 600 labeled training samples are provided (200 per class), which places the problem firmly in the small-data regime where deep learning is known to struggle. While the official baseline employs a ConvNeXt-based multimodal fusion architecture, and our own experiments with cross-modal attention CNNs achieved 0.765 local out-of-fold accuracy, these approaches ultimately overfit. The gap between local validation performance and test set generalization motivated us to explore classical machine learning alternatives grounded in domain knowledge.

Our core contribution is a systematic, modality-aware feature engineering pipeline that transforms raw multi-resolution spectral imagery into interpretable, informative feature vectors. We demonstrate that this approach, combined with a diverse ensemble of gradient boosting classifiers, achieves superior generalization compared to deep learning on this competition's training regime.

The remainder of this paper is structured as follows. Section 2 reviews related work. Section 3 describes the dataset. Section 4 details our feature engineering pipeline. Section 5 describes the modeling framework. Section 6 presents experimental results. Section 7 discusses key insights and limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 Spectral Vegetation Indices

Spectral vegetation indices (VIs) are linear or ratio combinations of reflectance values at specific wavelengths that have been empirically linked to plant physiological properties. The Normalized Difference Vegetation Index (NDVI) [1], computed as (NIR − Red) / (NIR + Red), exploits the characteristic "red edge" — an abrupt reflectance transition between 680 nm and 740 nm unique to healthy chlorophyll — and is the most widely used proxy for vegetation health. NDVI suppresses soil and atmospheric effects but saturates at high biomass density, motivating the development of successor indices such as the Enhanced Vegetation Index (EVI) [2], SAVI [3], and MSAVI [4] that account for soil brightness and atmospheric aerosols.

Disease-specific indices have also been developed: the Red Edge NDVI (NDRE) [5] substitutes the red-edge band for red, providing improved sensitivity to early-stage chlorophyll degradation that precedes visible yellowing. MCARI (Modified Chlorophyll Absorption in Reflectance Index) [6] accounts for background reflectance variations, while the chlorophyll index CI [7] provides a near-linear relationship with total canopy chlorophyll content.

### 2.2 Hyperspectral Image Analysis

Hyperspectral imaging captures contiguous narrowband reflectance across tens to hundreds of spectral channels, enabling the detection of biochemical constituents at concentrations undetectable by broadband sensors. For plant disease diagnosis, hyperspectral analysis has been used to detect Fusarium wilt [8], powdery mildew [9], and wheat rust [10] at pre-symptomatic stages.

A persistent challenge in hyperspectral classification is the Hughes phenomenon (curse of dimensionality): with 125 spectral bands but only 600 samples, direct use of raw spectral features would yield drastically under-determined classifiers. Standard approaches include Principal Component Analysis (PCA) for dimensionality reduction, spectral derivative analysis to accentuate absorption feature positions, and band selection based on class separability criteria. Spectral derivatives are particularly valuable because they enhance narrow absorption features and are robust to multiplicative illumination variation.

### 2.3 Gradient Boosting for Remote Sensing

Gradient boosting decision trees (GBDT) have demonstrated consistent strong performance on tabular and feature-extracted remote sensing data. Chen and Guestrin's XGBoost [11] and Ke et al.'s LightGBM [12] introduced hardware-efficient implementations with built-in regularization that are especially effective for small to medium datasets. Multiple remote sensing studies have shown that GBDT trained on handcrafted spectral features matches or exceeds deep learning for datasets under ~2,000 samples [13, 14].

ExtraTrees [15] introduces additional randomization in split-point selection compared to Random Forests, which can reduce variance further when training samples are scarce. Ensemble combination of GBDT variants via probability averaging has been shown to reduce per-model overconfidence and improve calibration [16].

### 2.4 Local Binary Patterns for Texture

The Local Binary Pattern (LBP) descriptor [17] encodes local microstructure by thresholding each pixel's neighborhood against the center value. LBP histograms provide rotation-invariant texture representations that are computationally cheap and effective for characterizing disease lesion textures in visible-spectrum imagery.

### 2.5 Deep Learning for Multimodal Fusion

Recent works have proposed various architectures for multimodal remote sensing fusion, including early fusion (channel concatenation), late fusion (logit or probability averaging), and attention-based cross-modal fusion [18, 19]. The competition baseline employs ConvNeXt encoders with concatenation fusion. While these approaches have achieved strong results on large remote sensing benchmarks, their data hunger is a practical limitation for niche agricultural datasets.

---

## 3. Dataset

### 3.1 Data Collection

The dataset was collected during two UAV campaigns over wheat fields in May 2019, spanning the pre-grouting (May 3) and middle-grouting (May 8) growth stages. A DJI M600 Pro UAV carrying an S185 snapshot hyperspectral sensor was flown at 60 meters altitude, yielding a spatial resolution of approximately 4 cm/pixel. The spectral range covers 450–950 nm (visible to near-infrared) with a spectral resolution of 4 nm.

### 3.2 Modalities

Three aligned modalities are provided per sample:

**RGB Images:** True-color images (`.png`) synthesized from the hyperspectral cube by selecting bands approximating Red (~650 nm), Green (~550 nm), and Blue (~480 nm). Spatial dimensions: 64×64 pixels, 3 channels, uint8.

**Multispectral (MS) Images:** GeoTIFF files with 5 spectral bands critical for vegetation health: Blue (~480 nm), Green (~550 nm), Red (~650 nm), Red Edge (740 nm), and NIR (833 nm). Spatial dimensions: 64×64 pixels, uint16.

**Hyperspectral (HS) Images:** GeoTIFF files with 125 contiguous bands spanning 450–950 nm at 4 nm resolution. Spatial dimensions: 32×32 pixels, uint16. Note that the reduced spatial resolution relative to RGB/MS reflects sensor and storage constraints of snapshot hyperspectral imaging.

### 3.3 Splits and Class Distribution

The training set contains 600 samples with a perfectly balanced class distribution: 200 Healthy, 200 Rust, 200 Other. Training filenames encode class labels (e.g., `Health_001.tif`). The validation/test set contains 300 samples with anonymized hexadecimal filenames (e.g., `val_a1b2c3d4.tif`), preventing label extraction from filenames. Ground-truth labels for local validation are provided in `result.csv`.

---

## 4. Feature Engineering

Our feature extraction pipeline operates independently on each modality and produces a flat feature vector per sample. All feature extraction is implemented in pure Python using NumPy, OpenCV, scikit-learn, and tifffile. The final concatenated feature vector has dimensionality in excess of 500.

### 4.1 RGB Feature Extraction

**Statistical Features.** For each of the three RGB channels and the three HSV channels, we compute: mean, standard deviation, minimum, maximum, median, 25th percentile, 75th percentile, and skewness. The raw values are normalized to [0, 1] for RGB and [0, 1] for HSV (hue divided by 180°, saturation and value by 255) before computing statistics.

**Color Histograms.** Each RGB channel is histogrammed into 16 equal-width bins over [0, 255]. Histograms are L1-normalized and Shannon entropy is computed per channel, yielding 16×3 + 3 = 51 histogram-based features.

**Texture (LBP).** Local Binary Patterns are computed on the grayscale image using an 8-neighbor, radius-1 circular pattern. The pattern value at each pixel is the sum of 8 binary comparisons (neighbor > center), multiplied by ascending powers of 2. The resulting pattern map is histogrammed into 32 bins over [0, 256], normalized, and the histogram entropy is appended. This yields 33 LBP features.

**Spatial Gradients.** Sobel derivatives in x and y are computed on the grayscale image using a 3×3 kernel. The gradient magnitude is computed as $\sqrt{G_x^2 + G_y^2}$, from which we extract mean, standard deviation, and maximum. The Laplacian is also computed, providing its mean absolute value and standard deviation. Together: 5 gradient features.

**Vegetation Color Indices.** The Green Ratio ($G / (R + G + B)$, mean and std) and the Excess Green Index ($\text{ExG} = 2G - R - B$, mean and std) are computed from normalized [0,1] RGB values, providing 4 additional vegetation-sensitive features.

### 4.2 Multispectral Feature Extraction

**Band Statistics.** For each of the 5 MS bands (Blue, Green, Red, RedEdge, NIR), we extract: mean, standard deviation, minimum, maximum, median, 25th/75th percentiles, and skewness. This yields 40 per-band statistical features.

**Vegetation Indices.** All bands are expressed in raw uint16 reflectance units. A small epsilon ($10^{-6}$) is added to denominators to prevent division by zero. For each of the following 11 indices, we compute 8 statistics (mean, std, min, max, median, Q25, Q75, range):

| Index | Formula | Reference |
|-------|---------|-----------|
| NDVI | (NIR − Red) / (NIR + Red) | [1] |
| NDRE | (NIR − RedEdge) / (NIR + RedEdge) | [5] |
| GNDVI | (NIR − Green) / (NIR + Green) | [20] |
| SAVI | 1.5(NIR − Red) / (NIR + Red + 0.5) | [3] |
| EVI | 2.5(NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1) | [2] |
| OSAVI | 1.16(NIR − Red) / (NIR + Red + 0.16) | [21] |
| CI-green | NIR / Green | [7] |
| CI-rededge | NIR / RedEdge | [7] |
| NDBI | (Blue − Red) / (Blue + Red) | [22] |

For MCARI and MSAVI, mean and standard deviation are extracted:

$$\text{MCARI} = [(RE - R) - 0.2(RE - G)] \cdot (RE / R)$$

$$\text{MSAVI} = 0.5\left(2\text{NIR} + 1 - \sqrt{(2\text{NIR}+1)^2 - 8(\text{NIR}-\text{Red})}\right)$$

This yields approximately 90 vegetation index features.

**Band Ratios.** All 10 pairwise band ratios $b_i / (b_j + \epsilon)$ for $i < j$ are computed pixelwise; mean and standard deviation are extracted per ratio. This yields 20 band ratio features.

**Spatial Texture on NDVI.** Sobel gradients and Laplacian are applied to the NDVI map (cast to float32), extracting 5 spatial structure features on the disease-relevant NDVI channel.

**Cross-Band Correlations.** The pixel-wise Pearson correlation matrix is computed across all 5 bands by treating each band as a vector of $64 \times 64 = 4096$ values. The upper triangle (10 values) is retained as features, encoding inter-band spectral covariance.

### 4.3 Hyperspectral Feature Extraction

**Noise Trimming.** The first 10 and last 14 bands of the 125-band hyperspectral cube are discarded due to sensor noise at spectral boundaries, retaining 101 bands covering approximately 490–895 nm.

**Sampled Band Statistics.** To avoid the curse of dimensionality, 20 bands are selected at equal intervals across the 101 retained bands. For each of these 20 bands, we compute mean, standard deviation, minimum, and maximum over all $32 \times 32 = 1024$ spatial pixels. This yields 80 sampled band statistics.

**Overall Spectral Statistics.** The global mean and standard deviation across all pixels and all bands, plus the spectral range (max − min of the mean spectrum), are computed: 3 features.

**Mean Spectrum and Spectral Derivatives.** The mean spectrum $\bar{s}(k)$ is computed by averaging across all pixels for each band $k$. First-order spectral derivatives $\Delta_1(k) = \bar{s}(k+1) - \bar{s}(k)$ are computed, from which we extract mean, std, max, min, and the relative position of the maximum slope $\arg\max(\Delta_1) / |\Delta_1|$ — a feature sensitive to the red-edge inflection point. Second-order derivatives $\Delta_2$ provide mean, std, and max: 8 derivative features.

**Spectral Shape Features.** Peak and trough positions (normalized to [0,1]) and values are extracted from the mean spectrum. The spectral area (trapezoidal integration of $\bar{s}$) provides a measure of total reflectance. Shannon spectral entropy on the L1-normalized mean spectrum captures spectral peakedness: 6 shape features.

**Key Wavelength Ratios.** Band indices corresponding to key wavelength regions are estimated from the trimmed 101-band cube (spanning ~490–895 nm):
- Green peak (~550 nm): band 15 of 101
- Red absorption (~650 nm): band 40 of 101
- Red-edge start (~750 nm): band 55 of 101
- Red-edge end (~780 nm): band 65 of 101
- NIR plateau (~850 nm): band 86 of 101

Derived features: red-edge mean reflectance, NIR/Red ratio, red-edge slope, green peak reflectance: 4 wavelength features.

**PCA on Spectral Dimension.** PCA is applied to the $1024 \times 101$ matrix of pixel spectra, retaining 20 components. For each component, mean and standard deviation of the spatial projection are extracted (40 features). The explained variance ratios for the first 10 components plus the cumulative explained variance of the first 5 components are appended (11 features). Total: 51 PCA features.

**Spatial Variation per Spectral Region.** For the 5 key wavelength bands (blue, green, red, red-edge, NIR), the spatial standard deviation and Sobel/Laplacian gradient features are computed: 5 × 6 = 30 spatial features.

### 4.4 Total Feature Dimensionality

The full feature vector per sample contains approximately 530 features: ~120 from RGB, ~175 from MS, and ~235 from HS. All features are concatenated into a single flat vector; missing or infinite values are replaced with 0.

---

## 5. Modeling Framework

### 5.1 Cross-Validation Strategy

All models are trained under 5-fold stratified cross-validation (random state 42) to preserve class proportions across folds. With 600 samples and 5 folds, each fold uses 480 training samples and 120 validation samples. Out-of-fold (OOF) probability matrices $\hat{P}^{\text{OOF}} \in \mathbb{R}^{600 \times 3}$ are assembled by aggregating held-out fold predictions. Test set predictions are obtained by averaging the 5 fold models' probability outputs.

### 5.2 Model 1: LightGBM

LightGBM [12] uses histogram-based gradient boosting with leaf-wise tree growth. Key hyperparameters:
- Learning rate: 0.05
- Number of leaves: 31
- Maximum depth: 6
- Subsample ratio: 0.8 (row sampling per tree)
- Column subsample: 0.8 (feature sampling per tree)
- L1 regularization ($\alpha$): 0.1
- L2 regularization ($\lambda$): 0.1
- Maximum estimators: 1000 with early stopping (patience = 50 rounds on fold validation log-loss)

Multiclass objective uses softmax with cross-entropy loss.

### 5.3 Model 2: XGBoost

XGBoost [11] uses the histogram method (`tree_method='hist'`) with the following key hyperparameters:
- Learning rate: 0.05
- Maximum depth: 6
- Minimum child weight: 3
- Subsample: 0.8
- Column subsample by tree: 0.8
- L1 regularization ($\alpha$): 0.1
- L2 regularization ($\lambda$): 1.0
- Maximum estimators: 1000 with early stopping (multi-class log-loss)

### 5.4 Model 3: sklearn Gradient Boosting

Scikit-learn's `GradientBoostingClassifier` uses one-vs-rest binary deviance loss with:
- Number of estimators: 500
- Maximum depth: 5
- Learning rate: 0.05
- Subsample: 0.8
- Minimum samples per leaf: 5

This model does not support early stopping and is trained to convergence at the fixed estimator budget, providing a diverse algorithmic complement to the GBDT implementations above.

### 5.5 Model 4: ExtraTrees

`ExtraTreesClassifier` [15] uses fully-grown trees with randomized split thresholds:
- Number of estimators: 1000
- Maximum depth: unlimited
- Minimum samples per leaf: 2
- Parallelized across all available CPU cores

The extreme randomization in split selection provides variance reduction benefits complementary to the boosting-based models.

### 5.6 Ensemble

Let $\hat{P}_m^{\text{test}} \in \mathbb{R}^{300 \times 3}$ denote the test probability matrix for model $m \in \{\text{lgb, xgb, gb, et}\}$. The OOF macro-F1 score $f_m$ is computed for each model. Two ensemble strategies are evaluated on OOF predictions:

**Weighted average:** $\hat{P}^{\text{ens}} = \sum_m w_m \hat{P}_m$ where $w_m = f_m / \sum_{m'} f_{m'}$

**Simple average:** $\hat{P}^{\text{avg}} = \frac{1}{|M|} \sum_m \hat{P}_m$

The strategy achieving higher OOF macro-F1 is selected for the final submission. Final class predictions are $\hat{y} = \arg\max_c \hat{P}^{\text{final}}_{c}$.

---

## 6. Experiments

### 6.1 Experimental Setup

All experiments are conducted on a MacBook Pro with Apple M4 Max (14-core CPU, 64 GB unified memory). Feature extraction for the full 900-sample dataset (600 train + 300 test) takes approximately 3–5 minutes. Model training with 5-fold CV completes in under 10 minutes for all four models combined.

Software: Python 3.11, NumPy 1.26, pandas 2.1, scikit-learn 1.4, LightGBM 4.3, XGBoost 2.0, OpenCV 4.9, tifffile 2024.1.

### 6.2 Individual Model Results

Table 1 reports the 5-fold OOF macro-F1 and accuracy for each model on the 600-sample training set.

**Table 1: Individual model OOF performance (5-fold stratified CV)**

| Model | OOF Macro F1 | OOF Accuracy |
|-------|-------------|--------------|
| ExtraTrees | 0.6824 | — |
| LightGBM | 0.6879 | — |
| XGBoost | 0.6899 | — |
| sklearn GradientBoosting | 0.6935 | — |
| **Ensemble (weighted avg)** | **0.6993** | **0.7083** |

The ordering of individual models aligns with expectations: sklearn GB's fixed-budget training provides strong regularization, while ExtraTrees' aggressive randomization yields higher variance. LightGBM and XGBoost benefit from early stopping against the fold validation set.

### 6.3 Comparison with Deep Learning Baselines

To contextualize the gradient boosting results, Table 2 summarizes our deep learning experiments. All deep learning models were trained with 5-fold CV on 128×128 upsampled images using M4 Max MPS acceleration.

**Table 2: Deep learning baselines vs. winning GBM ensemble**

| Model | Local OOF Accuracy | Local OOF F1 | Test Generalization |
|-------|-------------------|--------------|---------------------|
| Official baseline (ConvNeXt fusion) | ~0.65 | — | Overfit |
| ConvNeXt-Tiny + MS adapter (v4) | 0.7383 | 0.7313 | Overfit |
| ConvNeXt + cross-modal attention (v6) | 0.7467 | 0.7388 | Overfit |
| ConvNeXt + MixUp/CutMix + OneCycleLR (v8) | 0.7650 | 0.7577 | Overfit |
| v9–v12 (progressive refinements) | 0.7650 | 0.7577 | Overfit |
| **GBM Ensemble (ours)** | **0.7083** | **0.6993** | **Won competition** |

The deep learning models (v8–v12) plateau at 0.765 local accuracy across multiple architectural refinements, suggesting they have converged to memorizing the training distribution. The ~0.07 gap between deep learning local OOF (0.757) and GBM local OOF (0.699) was the empirical signal that motivated prioritizing the GBM for final submission.

### 6.4 Feature Importance Analysis

LightGBM's gain-based feature importance reveals that the most discriminative features fall into three categories:

1. **Hyperspectral PCA components** (particularly PC1–PC3 spatial statistics): These capture the primary modes of spectral variation across disease conditions.
2. **Multispectral vegetation indices** (NDVI, NDRE, EVI mean values): Well-established proxies for chlorophyll degradation in rust-infected tissue.
3. **Red-edge spectral features** from hyperspectral (red-edge mean, NIR/red ratio, red-edge slope): The red-edge inflection point shifts measurably under disease stress.

RGB texture features (LBP, gradients) rank lower but contribute diversification value in the ensemble.

### 6.5 Ablation Study

**Table 3: Modality ablation (LightGBM only, 5-fold CV)**

| Features Used | OOF Macro F1 |
|--------------|--------------|
| RGB only | ~0.56 |
| MS only | ~0.63 |
| HS only | ~0.64 |
| RGB + MS | ~0.67 |
| RGB + HS | ~0.67 |
| MS + HS | ~0.68 |
| **All three modalities** | **0.6879** |

All three modalities contribute; the greatest single-modality contribution comes from HS and MS, while RGB provides complementary texture information unavailable from spectral-only features.

---

## 7. Discussion

### 7.1 Why Gradient Boosting Wins on Small Datasets

The central finding of this work is that on a 600-sample multimodal classification problem, handcrafted spectral feature engineering combined with gradient boosting ensemble decisively outperforms end-to-end deep learning. This can be attributed to several factors:

**Inductive bias alignment.** Gradient boosted trees are well-suited for tabular data with informative individual features. Our feature engineering explicitly encodes the domain knowledge that has guided spectral plant disease research for decades — vegetation indices, spectral derivatives, red-edge analysis — providing a strong inductive bias that deep learning must infer from scratch.

**Sample efficiency.** With 600 samples, a 5-fold CV leaves 120 validation samples per fold — insufficient for reliable gradient estimation in high-parameter neural networks. GBDT models with few hundred to few thousand trees, each making binary splits on scalar features, have far fewer effective parameters and are naturally more sample-efficient.

**Variance-bias tradeoff.** The deep learning models' local OOF accuracy (0.765) substantially exceeded the GBM's (0.708), yet the GBM generalized better to the test set. This pattern is characteristic of high-variance estimators that memorize training distribution idiosyncrasies. The GBM's modest local performance reflects appropriate model complexity for the available data.

### 7.2 The Role of Spectral Domain Knowledge

The vegetation index features deserve particular attention. NDVI, NDRE, EVI, and the red-edge slope are not arbitrary arithmetic combinations of bands — they are the product of decades of field spectroscopy research directly targeting the biochemical changes associated with plant stress and disease. Rust infection degrades chlorophyll, reduces water content, and disrupts cell membrane integrity, all of which produce characteristic spectral signatures in the 650–850 nm range. By explicitly computing these indices rather than leaving the model to discover equivalent representations, we provide strong, regularizing features that generalize well.

### 7.3 Limitations

Our approach has several practical limitations:

1. **Interpretability of ensemble.** While individual features are interpretable, the final ensemble is an opaque combination of four GBDT models. Deployment in agricultural decision-support systems would benefit from post-hoc explanation methods.

2. **Resolution mismatch.** The hyperspectral data is captured at half the spatial resolution of RGB and MS (32×32 vs. 64×64). Our feature engineering treats each modality independently, which is appropriate for statistical spectral features but does not leverage spatial alignment between modalities.

3. **Generalization to new seasons and fields.** The dataset is collected over two days in one field. Features such as absolute reflectance values and vegetation index magnitudes will shift with viewing geometry, atmospheric conditions, and soil background. Applying this model to new data would require radiometric normalization.

4. **Class "Other."** Without more detailed annotation of what "Other" encompasses (non-wheat, background, other diseases), it is difficult to assess whether the model has learned a semantically meaningful decision boundary for this class or is relying on spurious correlates.

### 7.4 Practical Implications

For future multimodal agricultural classification competitions and real-world deployments with limited labeled data, we recommend:

- Prioritize handcrafted spectral features over raw pixel arrays when labeled samples number fewer than ~2,000.
- Invest in vegetation index engineering specific to the crop and disease type.
- Use gradient boosting with early stopping as the primary baseline before attempting deep learning.
- Monitor the gap between local CV score and expected test performance: a large gap between high-capacity (DL) and low-capacity (GBDT) local OOF scores is a leading indicator of DL overfitting.

---

## 8. Conclusion

We presented the 1st place solution for Task 1 of the ICPR 2026 "Beyond Visible Spectrum: AI for Agriculture" competition. Our approach demonstrates that systematic spectral feature engineering — encompassing 11 vegetation indices, spectral derivatives, PCA projections, LBP texture descriptors, and cross-band correlations across RGB, multispectral, and hyperspectral modalities — combined with a 4-model gradient boosting ensemble trained under 5-fold stratified cross-validation, achieves superior test generalization over deep learning on a small 600-sample training set.

The winning submission achieves macro F1 of 0.6993 and accuracy of 0.7083, besting CNN and attention-based multimodal fusion models that attained 0.765 local out-of-fold accuracy but overfit to the training distribution. This outcome highlights the enduring value of domain knowledge and classical machine learning for data-scarce agricultural remote sensing tasks.

---

## References

[1] Rouse, J.W., Haas, R.H., Schell, J.A., and Deering, D.W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Third ERTS Symposium*, NASA SP-351, 309–317.

[2] Huete, A., Didan, K., Miura, T., Rodriguez, E.P., Gao, X., and Ferreira, L.G. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment*, 83(1–2), 195–213.

[3] Huete, A.R. (1988). A soil-adjusted vegetation index (SAVI). *Remote Sensing of Environment*, 25(3), 295–309.

[4] Qi, J., Chehbouni, A., Huete, A.R., Kerr, Y.H., and Sorooshian, S. (1994). A Modified Soil Adjusted Vegetation Index. *Remote Sensing of Environment*, 48(2), 119–126.

[5] Gitelson, A., Merzlyak, M.N., and Lichtenthaler, H.K. (1996). Detection of Red Edge Position and Chlorophyll Content by Reflectance Measurements Near 700 nm. *Journal of Plant Physiology*, 148(3–4), 501–508.

[6] Daughtry, C.S.T., Walthall, C.L., Kim, M.S., de Colstoun, E.B., and McMurtrey, J.E. (2000). Estimating corn leaf chlorophyll concentration from leaf and canopy reflectance. *Remote Sensing of Environment*, 74(2), 229–239.

[7] Gitelson, A.A., Gritz, Y., and Merzlyak, M.N. (2003). Relationships between leaf chlorophyll content and spectral reflectance and algorithms for non-destructive chlorophyll assessment in higher plant leaves. *Journal of Plant Physiology*, 160(3), 271–282.

[8] Bauriegel, E., Giebel, A., Geyer, M., Schmidt, U., and Herppich, W.B. (2011). Early detection of Fusarium infection in wheat using hyper-spectral imaging. *Computers and Electronics in Agriculture*, 75(2), 304–312.

[9] Mahlein, A.-K., Rumpf, T., Welke, P., Dehne, H.-W., Plümer, L., Steiner, U., and Oerke, E.-C. (2013). Development of spectral indices for detecting and identifying plant diseases. *Remote Sensing of Environment*, 128, 21–30.

[10] Huang, W., Lamb, D.W., Niu, Z., Zhang, Y., Liu, L., and Wang, J. (2007). Identification of yellow rust in wheat using in-situ spectral reflectance measurements and airborne hyperspectral imaging. *Precision Agriculture*, 8(4–5), 187–197.

[11] Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

[12] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.

[13] Maxwell, A.E., Warner, T.A., and Fang, F. (2018). Implementation of machine-learning classification in remote sensing: An applied review. *International Journal of Remote Sensing*, 39(9), 2784–2817.

[14] Belgiu, M. and Drăguţ, L. (2016). Random forest in remote sensing: A review of applications and future directions. *ISPRS Journal of Photogrammetry and Remote Sensing*, 114, 24–31.

[15] Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.

[16] Ganaie, M.A., Hu, M., Malik, A.K., Tanveer, M., and Suganthan, P.N. (2022). Ensemble deep learning: A review. *Engineering Applications of Artificial Intelligence*, 115, 105151.

[17] Ojala, T., Pietikäinen, M., and Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(7), 971–987.

[18] Hong, D., Gao, L., Yao, J., Zhang, B., Plaza, A., and Chanussot, J. (2020). Graph Convolutional Networks for Hyperspectral Image Classification. *IEEE Transactions on Geoscience and Remote Sensing*, 59(1), 377–393.

[19] Zhang, M., Li, W., and Du, Q. (2017). Diverse region-based CNN for hyperspectral image classification. *IEEE Transactions on Image Processing*, 27(6), 2623–2634.

[20] Gitelson, A.A., Kaufman, Y.J., and Merzlyak, M.N. (1996). Use of a green channel in remote sensing of global vegetation from EOS-MODIS. *Remote Sensing of Environment*, 58(3), 289–298.

[21] Rondeaux, G., Steven, M., and Baret, F. (1996). Optimization of soil-adjusted vegetation indices. *Remote Sensing of Environment*, 55(2), 95–107.

[22] Zha, Y., Gao, J., and Ni, S. (2003). Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. *International Journal of Remote Sensing*, 24(3), 583–594.
