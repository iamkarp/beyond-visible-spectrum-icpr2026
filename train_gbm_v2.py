"""
Beyond Visible Spectrum - GBM v2
Improved feature engineering with GLCM textures, better spectral features,
SVM model, and more aggressive hyperparameter tuning.
"""

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier,
)
from skimage.feature import graycomatrix, graycoprops

import cv2
import tifffile as tiff
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = "/Users/macbook/Library/CloudStorage/GoogleDrive-jason.karpeles@pmg.com/My Drive/Projects/Beyond Visible Spectrum"
OUT_DIR = os.path.join(ROOT, "output")
N_FOLDS = 5
SEED = 42

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ============================================================
# Data loading (reuse from v1)
# ============================================================
def list_files(folder, exts):
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)])

def base_id(path):
    return os.path.splitext(os.path.basename(path))[0]

def parse_label(bid):
    m = re.match(r"^(Health|Rust|Other)_", bid)
    return m.group(1) if m else None

def build_index(root, split):
    split_dir = os.path.join(root, split)
    idx = {}
    for mod, exts in [("rgb", (".png", ".jpg")), ("ms", (".tif", ".tiff")), ("hs", (".tif", ".tiff"))]:
        for p in list_files(os.path.join(split_dir, mod.upper()), exts):
            idx.setdefault(base_id(p), {})[mod] = p
    return idx

def make_df(idx, has_labels=True):
    rows = []
    for bid, paths in idx.items():
        row = {"base_id": bid, **paths}
        if has_labels:
            lab = parse_label(bid)
            if lab is None:
                continue
            row["label"] = lab
        rows.append(row)
    return pd.DataFrame(rows)

def read_tiff(path):
    arr = tiff.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D, got {arr.shape}")
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


# ============================================================
# GLCM Texture Features
# ============================================================
def glcm_features(gray_uint8, prefix, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract GLCM texture features from a grayscale uint8 image."""
    features = {}
    # Quantize to 32 levels to reduce computation
    gray_q = (gray_uint8 / 8).astype(np.uint8)

    try:
        glcm = graycomatrix(gray_q, distances=distances, angles=angles,
                            levels=32, symmetric=True, normed=True)
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
            vals = graycoprops(glcm, prop)
            features[f"{prefix}_glcm_{prop}_mean"] = np.mean(vals)
            features[f"{prefix}_glcm_{prop}_std"] = np.std(vals)
            features[f"{prefix}_glcm_{prop}_max"] = np.max(vals)
    except:
        pass
    return features


# ============================================================
# Feature extraction functions
# ============================================================
def band_statistics(arr, prefix):
    features = {}
    H, W, C = arr.shape
    for c in range(C):
        band = arr[:, :, c].ravel()
        features[f"{prefix}_b{c}_mean"] = np.mean(band)
        features[f"{prefix}_b{c}_std"] = np.std(band)
        features[f"{prefix}_b{c}_min"] = np.min(band)
        features[f"{prefix}_b{c}_max"] = np.max(band)
        features[f"{prefix}_b{c}_median"] = np.median(band)
        features[f"{prefix}_b{c}_q25"] = np.percentile(band, 25)
        features[f"{prefix}_b{c}_q75"] = np.percentile(band, 75)
        features[f"{prefix}_b{c}_skew"] = float(np.mean(((band - np.mean(band)) / (np.std(band) + 1e-8)) ** 3))
        features[f"{prefix}_b{c}_kurt"] = float(np.mean(((band - np.mean(band)) / (np.std(band) + 1e-8)) ** 4) - 3)
    return features


def spatial_features(arr_2d, prefix):
    features = {}
    arr = arr_2d.astype(np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    features[f"{prefix}_grad_mean"] = np.mean(mag)
    features[f"{prefix}_grad_std"] = np.std(mag)
    features[f"{prefix}_grad_max"] = np.max(mag)
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    features[f"{prefix}_lap_mean"] = np.mean(np.abs(lap))
    features[f"{prefix}_lap_std"] = np.std(lap)
    return features


def lbp_features(gray, prefix, radius=1, n_points=8):
    features = {}
    H, W = gray.shape
    padded = np.pad(gray, radius, mode="reflect")
    pattern = np.zeros_like(gray, dtype=np.float32)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dy = int(round(radius * np.sin(angle)))
        dx = int(round(radius * np.cos(angle)))
        neighbor = padded[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
        pattern += (neighbor > gray).astype(np.float32) * (2 ** i)
    hist = np.histogram(pattern, bins=32, range=(0, 256))[0]
    hist = hist / (hist.sum() + 1e-8)
    for b in range(32):
        features[f"{prefix}_lbp_{b}"] = hist[b]
    features[f"{prefix}_lbp_entropy"] = -np.sum(hist * np.log(hist + 1e-8))
    return features


def extract_rgb_features(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    features = {}

    # Band statistics
    features.update(band_statistics(img_rgb / 255.0, "rgb"))

    # Color histogram
    for c, name in enumerate(["r", "g", "b"]):
        hist = np.histogram(img_rgb[:, :, c], bins=16, range=(0, 256))[0]
        hist = hist / (hist.sum() + 1e-8)
        for b in range(16):
            features[f"rgb_hist_{name}_{b}"] = hist[b]
        features[f"rgb_entropy_{name}"] = -np.sum(hist * np.log(hist + 1e-8))

    # HSV features
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    features.update(band_statistics(hsv / np.array([180, 255, 255], dtype=np.float32), "hsv"))

    # LAB features (good for disease detection - separates color from intensity)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    features.update(band_statistics(lab / np.array([100, 255, 255], dtype=np.float32), "lab"))

    # Grayscale spatial + texture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features.update(spatial_features(gray.astype(np.float32), "rgb_gray"))
    features.update(lbp_features(gray.astype(np.float32), "rgb"))

    # GLCM on grayscale
    features.update(glcm_features(gray, "rgb"))

    # GLCM on each RGB channel
    for c, name in enumerate(["r", "g", "b"]):
        features.update(glcm_features(img_rgb[:, :, c].astype(np.uint8), f"rgb_{name}"))

    # Color ratios
    total = img_rgb.sum(axis=2) + 1e-8
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    features["rgb_green_ratio_mean"] = np.mean(g / total)
    features["rgb_green_ratio_std"] = np.std(g / total)
    features["rgb_red_ratio_mean"] = np.mean(r / total)
    features["rgb_rg_ratio_mean"] = np.mean(r / (g + 1e-8))
    features["rgb_rg_ratio_std"] = np.std(r / (g + 1e-8))

    # ExG and ExR
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    exg = 2 * gn - rn - bn
    exr = 1.4 * rn - gn
    features["rgb_exg_mean"] = np.mean(exg)
    features["rgb_exg_std"] = np.std(exg)
    features["rgb_exr_mean"] = np.mean(exr)
    features["rgb_exr_std"] = np.std(exr)
    features["rgb_exgr_mean"] = np.mean(exg - exr)

    # Spatial uniformity (std of 4 quadrants)
    H, W = gray.shape
    h2, w2 = H // 2, W // 2
    quads = [gray[:h2, :w2], gray[:h2, w2:], gray[h2:, :w2], gray[h2:, w2:]]
    quad_means = [np.mean(q) for q in quads]
    features["rgb_quad_std"] = np.std(quad_means)
    features["rgb_quad_range"] = max(quad_means) - min(quad_means)

    return features


def extract_ms_features(path):
    arr = read_tiff(path)  # (H,W,5): Blue, Green, Red, RedEdge, NIR
    features = {}
    eps = 1e-6

    # Band statistics
    features.update(band_statistics(arr, "ms"))

    blue, green, red, rededge, nir = [arr[:, :, i] for i in range(5)]

    def safe_idx(name, num, den):
        idx = num / (den + eps)
        features[f"ms_{name}_mean"] = np.mean(idx)
        features[f"ms_{name}_std"] = np.std(idx)
        features[f"ms_{name}_min"] = np.min(idx)
        features[f"ms_{name}_max"] = np.max(idx)
        features[f"ms_{name}_median"] = np.median(idx)
        features[f"ms_{name}_q25"] = np.percentile(idx, 25)
        features[f"ms_{name}_q75"] = np.percentile(idx, 75)
        features[f"ms_{name}_range"] = np.max(idx) - np.min(idx)
        features[f"ms_{name}_iqr"] = features[f"ms_{name}_q75"] - features[f"ms_{name}_q25"]
        return idx

    # Core vegetation indices
    ndvi = safe_idx("ndvi", nir - red, nir + red)
    ndre = safe_idx("ndre", nir - rededge, nir + rededge)
    gndvi = safe_idx("gndvi", nir - green, nir + green)
    savi = safe_idx("savi", 1.5 * (nir - red), nir + red + 0.5)
    evi = safe_idx("evi", 2.5 * (nir - red), nir + 6 * red - 7.5 * blue + 1)
    osavi = safe_idx("osavi", 1.16 * (nir - red), nir + red + 0.16)

    # Additional specialized indices
    safe_idx("ci_green", nir / (green + eps) - 1, np.ones_like(green))
    safe_idx("ci_rededge", nir / (rededge + eps) - 1, np.ones_like(rededge))
    safe_idx("ndbi", blue - red, blue + red)
    safe_idx("ngrdi", green - red, green + red)

    # MCARI / TCARI (targeted for disease detection)
    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    features["ms_mcari_mean"] = np.mean(mcari)
    features["ms_mcari_std"] = np.std(mcari)
    features["ms_mcari_median"] = np.median(mcari)

    # TCARI/OSAVI ratio (disease indicator)
    tcari = 3 * ((rededge - red) - 0.2 * (rededge - green) * (rededge / (red + eps)))
    osavi_arr = 1.16 * (nir - red) / (nir + red + 0.16 + eps)
    tcari_osavi = tcari / (osavi_arr + eps)
    features["ms_tcari_osavi_mean"] = np.mean(tcari_osavi)
    features["ms_tcari_osavi_std"] = np.std(tcari_osavi)

    # MSAVI
    msavi = 0.5 * (2 * nir + 1 - np.sqrt(np.clip((2 * nir + 1)**2 - 8 * (nir - red), 0, None) + eps))
    features["ms_msavi_mean"] = np.mean(msavi)
    features["ms_msavi_std"] = np.std(msavi)

    # RDVI (Renormalized Difference Vegetation Index)
    rdvi = (nir - red) / np.sqrt(nir + red + eps)
    features["ms_rdvi_mean"] = np.mean(rdvi)
    features["ms_rdvi_std"] = np.std(rdvi)

    # Band ratios (all pairs)
    band_names = ["blue", "green", "red", "rededge", "nir"]
    for i in range(5):
        for j in range(i + 1, 5):
            ratio = arr[:, :, i] / (arr[:, :, j] + eps)
            features[f"ms_ratio_{band_names[i]}_{band_names[j]}_mean"] = np.mean(ratio)
            features[f"ms_ratio_{band_names[i]}_{band_names[j]}_std"] = np.std(ratio)

    # Spatial features on key indices
    features.update(spatial_features(ndvi.astype(np.float32), "ms_ndvi_sp"))
    features.update(spatial_features(ndre.astype(np.float32), "ms_ndre_sp"))

    # GLCM on NDVI
    ndvi_uint8 = np.clip((ndvi + 1) * 127.5, 0, 255).astype(np.uint8)
    features.update(glcm_features(ndvi_uint8, "ms_ndvi"))

    # Cross-band correlations
    flat = arr.reshape(-1, 5)
    try:
        corr = np.corrcoef(flat.T)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"ms_corr_{band_names[i]}_{band_names[j]}"] = corr[i, j]
    except:
        pass

    # Spatial homogeneity per band
    for i, name in enumerate(band_names):
        band = arr[:, :, i]
        H, W = band.shape
        h2, w2 = H // 2, W // 2
        quads = [band[:h2, :w2], band[:h2, w2:], band[h2:, :w2], band[h2:, w2:]]
        features[f"ms_{name}_quad_std"] = np.std([np.mean(q) for q in quads])

    return features


def extract_hs_features(path, drop_first=10, drop_last=14, n_pca=25):
    arr = read_tiff(path)
    B = arr.shape[2]
    if B > (drop_first + drop_last + 1):
        arr = arr[:, :, drop_first:B - drop_last]

    features = {}
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)

    mean_spectrum = np.mean(flat, axis=0)
    std_spectrum = np.std(flat, axis=0)
    median_spectrum = np.median(flat, axis=0)

    # Sampled band statistics (20 evenly spaced bands)
    n_sample = min(20, C)
    sample_indices = np.linspace(0, C - 1, n_sample, dtype=int)
    for i, idx in enumerate(sample_indices):
        band = flat[:, idx]
        features[f"hs_b{i}_mean"] = np.mean(band)
        features[f"hs_b{i}_std"] = np.std(band)
        features[f"hs_b{i}_min"] = np.min(band)
        features[f"hs_b{i}_max"] = np.max(band)
        features[f"hs_b{i}_median"] = np.median(band)

    # Overall statistics
    features["hs_total_mean"] = np.mean(flat)
    features["hs_total_std"] = np.std(flat)
    features["hs_total_range"] = float(mean_spectrum.max() - mean_spectrum.min())

    # Spectral derivatives (1st and 2nd order)
    deriv1 = np.diff(mean_spectrum)
    features["hs_d1_mean"] = np.mean(deriv1)
    features["hs_d1_std"] = np.std(deriv1)
    features["hs_d1_max"] = np.max(deriv1)
    features["hs_d1_min"] = np.min(deriv1)
    features["hs_d1_max_pos"] = np.argmax(deriv1) / len(deriv1)
    features["hs_d1_min_pos"] = np.argmin(deriv1) / len(deriv1)
    features["hs_d1_zero_crossings"] = np.sum(np.diff(np.sign(deriv1)) != 0) / len(deriv1)

    deriv2 = np.diff(deriv1)
    features["hs_d2_mean"] = np.mean(deriv2)
    features["hs_d2_std"] = np.std(deriv2)
    features["hs_d2_max"] = np.max(deriv2)
    features["hs_d2_min"] = np.min(deriv2)

    # Spectral shape
    features["hs_peak_band"] = np.argmax(mean_spectrum) / C
    features["hs_trough_band"] = np.argmin(mean_spectrum) / C
    features["hs_peak_value"] = np.max(mean_spectrum)
    features["hs_trough_value"] = np.min(mean_spectrum)

    # Spectral area and entropy
    features["hs_spectral_area"] = np.trapezoid(mean_spectrum)
    spec_norm = mean_spectrum / (mean_spectrum.sum() + 1e-8)
    features["hs_spectral_entropy"] = -np.sum(spec_norm * np.log(spec_norm + 1e-8))

    # Spectral moments
    x = np.arange(C, dtype=np.float32)
    spec_sum = mean_spectrum.sum() + 1e-8
    centroid = np.sum(x * mean_spectrum) / spec_sum
    features["hs_spectral_centroid"] = centroid / C
    spread = np.sqrt(np.sum((x - centroid)**2 * mean_spectrum) / spec_sum)
    features["hs_spectral_spread"] = spread / C
    features["hs_spectral_skewness"] = np.sum((x - centroid)**3 * mean_spectrum) / (spec_sum * (spread + 1e-8)**3)
    features["hs_spectral_kurtosis"] = np.sum((x - centroid)**4 * mean_spectrum) / (spec_sum * (spread + 1e-8)**4)

    # Key wavelength regions (after trimming: ~490-890nm)
    # Approximate band indices
    n_remaining = C
    # Green peak (~550nm) → band ~15
    green_idx = int(n_remaining * 0.15)
    # Red (~650nm) → band ~40
    red_idx = int(n_remaining * 0.40)
    # Red edge start (~700nm) → band ~52
    re_start = int(n_remaining * 0.52)
    # Red edge peak (~740nm) → band ~62
    re_peak = int(n_remaining * 0.62)
    # NIR plateau (~800nm) → band ~77
    nir_idx = int(n_remaining * 0.77)
    # NIR (~850nm) → band ~90
    nir2_idx = int(n_remaining * 0.90)

    features["hs_green_peak"] = mean_spectrum[green_idx]
    features["hs_red_absorption"] = mean_spectrum[red_idx]
    features["hs_red_edge_start"] = mean_spectrum[re_start]
    features["hs_red_edge_peak"] = mean_spectrum[re_peak]
    features["hs_nir_plateau"] = mean_spectrum[nir_idx]
    features["hs_nir_value"] = mean_spectrum[nir2_idx]

    # Red edge slope (key disease indicator)
    if re_peak > re_start:
        re_slope = (mean_spectrum[re_peak] - mean_spectrum[re_start]) / (re_peak - re_start)
        features["hs_red_edge_slope"] = re_slope
    # Red edge inflection point (using max of 1st derivative in that region)
    re_deriv = deriv1[max(0, re_start-1):min(len(deriv1), re_peak)]
    if len(re_deriv) > 0:
        features["hs_red_edge_inflection"] = (np.argmax(re_deriv) + re_start) / C
        features["hs_red_edge_max_slope"] = np.max(re_deriv)

    # Spectral ratios at key bands
    features["hs_nir_red_ratio"] = mean_spectrum[nir_idx] / (mean_spectrum[red_idx] + 1e-8)
    features["hs_nir_green_ratio"] = mean_spectrum[nir_idx] / (mean_spectrum[green_idx] + 1e-8)
    features["hs_re_red_ratio"] = mean_spectrum[re_peak] / (mean_spectrum[red_idx] + 1e-8)

    # Chlorophyll absorption depth (around red region)
    if green_idx < red_idx < nir_idx:
        # Continuum removal at red absorption
        baseline = mean_spectrum[green_idx] + (mean_spectrum[nir_idx] - mean_spectrum[green_idx]) * (red_idx - green_idx) / (nir_idx - green_idx + 1e-8)
        features["hs_red_absorption_depth"] = baseline - mean_spectrum[red_idx]
        features["hs_red_absorption_relative"] = (baseline - mean_spectrum[red_idx]) / (baseline + 1e-8)

    # PCA features
    try:
        n_comp = min(n_pca, C, H * W)
        pca = PCA(n_components=n_comp)
        pca_result = pca.fit_transform(flat)
        for i in range(min(n_comp, 25)):
            comp = pca_result[:, i]
            features[f"hs_pca{i}_mean"] = np.mean(comp)
            features[f"hs_pca{i}_std"] = np.std(comp)
            features[f"hs_pca{i}_skew"] = float(np.mean(((comp - np.mean(comp)) / (np.std(comp) + 1e-8)) ** 3))
        for i, ev in enumerate(pca.explained_variance_ratio_[:15]):
            features[f"hs_pca_ev{i}"] = ev
        features["hs_pca_cumev_3"] = np.sum(pca.explained_variance_ratio_[:3])
        features["hs_pca_cumev_5"] = np.sum(pca.explained_variance_ratio_[:5])
        features["hs_pca_cumev_10"] = np.sum(pca.explained_variance_ratio_[:10])
    except:
        pass

    # Spatial features at key bands
    for name, idx in [("green", green_idx), ("red", red_idx), ("re", re_start), ("nir", nir_idx)]:
        if idx < C:
            features.update(spatial_features(arr[:, :, idx], f"hs_{name}"))

    # GLCM on PCA component 1 (captures most spatial variation)
    try:
        pc1_img = pca_result[:, 0].reshape(H, W)
        pc1_uint8 = np.clip((pc1_img - pc1_img.min()) / (pc1_img.max() - pc1_img.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
        features.update(glcm_features(pc1_uint8, "hs_pca1"))
    except:
        pass

    # Spectral angle features
    # Compare each pixel spectrum to class-typical spectra
    # (we use the mean spectrum as reference)
    pixel_norms = np.linalg.norm(flat, axis=1) + 1e-8
    ref_norm = np.linalg.norm(mean_spectrum) + 1e-8
    cos_sim = np.dot(flat, mean_spectrum) / (pixel_norms * ref_norm)
    features["hs_spectral_angle_mean"] = np.mean(np.arccos(np.clip(cos_sim, -1, 1)))
    features["hs_spectral_angle_std"] = np.std(np.arccos(np.clip(cos_sim, -1, 1)))

    return features


def extract_cross_modal_features(rgb_path, ms_path, hs_path):
    """Cross-modal features comparing information across modalities."""
    features = {}

    try:
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            return features
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        rgb_green = rgb[:, :, 1].astype(np.float32) / 255.0  # Green channel (BGR)

        ms = read_tiff(ms_path)
        # Normalize MS
        ms_norm = ms.copy()
        for c in range(ms.shape[2]):
            mn, mx = ms_norm[:, :, c].min(), ms_norm[:, :, c].max()
            ms_norm[:, :, c] = (ms_norm[:, :, c] - mn) / (mx - mn + 1e-8)

        # Compare RGB green to MS green band
        ms_green = ms_norm[:, :, 1]  # Green band
        features["cross_rgb_ms_green_corr"] = np.corrcoef(rgb_green.ravel(), ms_green.ravel())[0, 1]
        features["cross_rgb_ms_green_diff"] = np.mean(np.abs(rgb_green - ms_green))

        # Compare RGB gray to MS NIR
        ms_nir = ms_norm[:, :, 4]
        features["cross_rgb_ms_nir_corr"] = np.corrcoef(rgb_gray.ravel(), ms_nir.ravel())[0, 1]
        features["cross_rgb_ms_nir_diff"] = np.mean(np.abs(rgb_gray - ms_nir))

        # Spatial gradient correlation between modalities
        rgb_grad = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)
        ms_grad = cv2.Sobel(ms_green, cv2.CV_32F, 1, 0, ksize=3)
        features["cross_grad_rgb_ms_corr"] = np.corrcoef(rgb_grad.ravel(), ms_grad.ravel())[0, 1]

    except:
        pass

    return features


def extract_all_features(row):
    features = {"base_id": row["base_id"]}

    if pd.notna(row.get("rgb")):
        features.update(extract_rgb_features(row["rgb"]))
    if pd.notna(row.get("ms")):
        features.update(extract_ms_features(row["ms"]))
    if pd.notna(row.get("hs")):
        features.update(extract_hs_features(row["hs"]))

    # Cross-modal features
    if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
        features.update(extract_cross_modal_features(
            row.get("rgb"), row.get("ms"), row.get("hs")
        ))

    return features


# ============================================================
# Main
# ============================================================
def main():
    print("Building data index...")
    train_idx = build_index(ROOT, "train")
    val_idx = build_index(ROOT, "val")
    train_df = make_df(train_idx, has_labels=True)
    val_df = make_df(val_idx, has_labels=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Extract features
    print("\nExtracting training features...")
    train_features = []
    for i, (_, row) in enumerate(train_df.iterrows()):
        feats = extract_all_features(row)
        train_features.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(train_df)}")
    train_feat_df = pd.DataFrame(train_features)
    print(f"  Train features: {train_feat_df.shape}")

    print("\nExtracting validation features...")
    val_features = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        feats = extract_all_features(row)
        val_features.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val_df)}")
    val_feat_df = pd.DataFrame(val_features)
    print(f"  Val features: {val_feat_df.shape}")

    # Prepare matrices
    feature_cols = [c for c in train_feat_df.columns if c != "base_id"]
    X_train = train_feat_df[feature_cols].fillna(0).values.astype(np.float32)
    X_test = val_feat_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train = np.array([LBL2ID[l] for l in train_df["label"]])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nTotal features: {X_train.shape[1]}")

    results = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ====== LightGBM with tuned params ======
    print("\n" + "=" * 60)
    print("Training LightGBM (tuned)...")
    print("=" * 60)

    for config_name, lgb_params in [
        ("lgb_v1", {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.03, "num_leaves": 31, "max_depth": 5,
            "min_child_samples": 8, "subsample": 0.75, "colsample_bytree": 0.7,
            "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000,
            "verbose": -1, "random_state": SEED,
        }),
        ("lgb_v2", {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.02, "num_leaves": 20, "max_depth": 4,
            "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.6,
            "reg_alpha": 1.0, "reg_lambda": 1.0, "n_estimators": 3000,
            "verbose": -1, "random_state": SEED + 1,
        }),
        ("lgb_v3", {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.05, "num_leaves": 40, "max_depth": 6,
            "min_child_samples": 5, "subsample": 0.85, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 0.3, "n_estimators": 1500,
            "verbose": -1, "random_state": SEED + 2,
        }),
    ]:
        oof_probs = np.zeros((len(X_train), 3))
        test_probs = np.zeros((len(X_test), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[va_idx], y_train[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            oof_probs[va_idx] = model.predict_proba(X_train[va_idx])
            test_probs += model.predict_proba(X_test) / N_FOLDS

        f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
        acc = accuracy_score(y_train, oof_probs.argmax(1))
        print(f"  {config_name}: OOF F1={f1:.4f} Acc={acc:.4f}")
        results[config_name] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1}

    # ====== XGBoost variants ======
    print("\n" + "=" * 60)
    print("Training XGBoost variants...")
    print("=" * 60)

    for config_name, xgb_params in [
        ("xgb_v1", {
            "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
            "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
            "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 0.5,
            "reg_lambda": 1.0, "n_estimators": 2000, "tree_method": "hist",
            "random_state": SEED, "verbosity": 0,
        }),
        ("xgb_v2", {
            "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
            "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
            "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 1.0,
            "reg_lambda": 2.0, "n_estimators": 3000, "tree_method": "hist",
            "random_state": SEED + 1, "verbosity": 0,
        }),
    ]:
        oof_probs = np.zeros((len(X_train), 3))
        test_probs = np.zeros((len(X_test), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train[tr_idx], y_train[tr_idx],
                      eval_set=[(X_train[va_idx], y_train[va_idx])], verbose=False)
            oof_probs[va_idx] = model.predict_proba(X_train[va_idx])
            test_probs += model.predict_proba(X_test) / N_FOLDS

        f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
        acc = accuracy_score(y_train, oof_probs.argmax(1))
        print(f"  {config_name}: OOF F1={f1:.4f} Acc={acc:.4f}")
        results[config_name] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1}

    # ====== sklearn models ======
    print("\n" + "=" * 60)
    print("Training sklearn models...")
    print("=" * 60)

    # Gradient Boosting
    oof_probs = np.zeros((len(X_train), 3))
    test_probs = np.zeros((len(X_test), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        model = GradientBoostingClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            subsample=0.75, min_samples_leaf=8, random_state=SEED,
        )
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_train[va_idx])
        test_probs += model.predict_proba(X_test) / N_FOLDS
    f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
    print(f"  sklearn_gb: OOF F1={f1:.4f}")
    results["sklearn_gb"] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1}

    # ExtraTrees
    oof_probs = np.zeros((len(X_train), 3))
    test_probs = np.zeros((len(X_test), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        model = ExtraTreesClassifier(
            n_estimators=1500, max_depth=None, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        )
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_train[va_idx])
        test_probs += model.predict_proba(X_test) / N_FOLDS
    f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
    print(f"  extra_trees: OOF F1={f1:.4f}")
    results["extra_trees"] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1}

    # Random Forest
    oof_probs = np.zeros((len(X_train), 3))
    test_probs = np.zeros((len(X_test), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        model = RandomForestClassifier(
            n_estimators=1500, max_depth=None, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        )
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_train[va_idx])
        test_probs += model.predict_proba(X_test) / N_FOLDS
    f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
    print(f"  random_forest: OOF F1={f1:.4f}")
    results["random_forest"] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1}

    # ====== SVM (with scaled features) ======
    print("\n" + "=" * 60)
    print("Training SVM...")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for config_name, svm_params in [
        ("svm_rbf", {"kernel": "rbf", "C": 10.0, "gamma": "scale", "probability": True, "random_state": SEED}),
        ("svm_rbf2", {"kernel": "rbf", "C": 50.0, "gamma": "auto", "probability": True, "random_state": SEED}),
    ]:
        oof_probs = np.zeros((len(X_train), 3))
        test_probs = np.zeros((len(X_test), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            model = SVC(**svm_params)
            model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
            oof_probs[va_idx] = model.predict_proba(X_train_scaled[va_idx])
            test_probs += model.predict_proba(X_test_scaled) / N_FOLDS

        f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
        acc = accuracy_score(y_train, oof_probs.argmax(1))
        print(f"  {config_name}: OOF F1={f1:.4f} Acc={acc:.4f}")
        results[config_name] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1}

    # ====== ENSEMBLE ======
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)

    # Sort models by F1
    sorted_models = sorted(results.keys(), key=lambda m: results[m]["f1"], reverse=True)
    print("\nAll models ranked:")
    for m in sorted_models:
        print(f"  {m:20s} F1={results[m]['f1']:.4f}")

    # Try different ensemble strategies
    best_ens_f1 = -1
    best_ens_name = None
    best_oof = None
    best_test = None

    # Strategy 1: Top-K weighted average (K=3,5,7,all)
    for k in [3, 5, 7, len(sorted_models)]:
        top_k = sorted_models[:k]
        weights = np.array([results[m]["f1"] for m in top_k])
        weights = weights ** 2  # Square weights to emphasize better models
        weights = weights / weights.sum()

        oof_ens = sum(w * results[m]["oof_probs"] for m, w in zip(top_k, weights))
        test_ens = sum(w * results[m]["test_probs"] for m, w in zip(top_k, weights))
        f1 = f1_score(y_train, oof_ens.argmax(1), average="macro")
        acc = accuracy_score(y_train, oof_ens.argmax(1))
        ens_name = f"top{k}_weighted"
        print(f"\n  {ens_name}: OOF F1={f1:.4f} Acc={acc:.4f}")
        if f1 > best_ens_f1:
            best_ens_f1 = f1
            best_ens_name = ens_name
            best_oof = oof_ens
            best_test = test_ens

    # Strategy 2: Simple average of all
    oof_avg = sum(results[m]["oof_probs"] for m in sorted_models) / len(sorted_models)
    test_avg = sum(results[m]["test_probs"] for m in sorted_models) / len(sorted_models)
    f1 = f1_score(y_train, oof_avg.argmax(1), average="macro")
    acc = accuracy_score(y_train, oof_avg.argmax(1))
    print(f"\n  simple_avg: OOF F1={f1:.4f} Acc={acc:.4f}")
    if f1 > best_ens_f1:
        best_ens_f1 = f1
        best_ens_name = "simple_avg"
        best_oof = oof_avg
        best_test = test_avg

    # Strategy 3: Rank-based averaging
    oof_rank = np.zeros_like(oof_avg)
    test_rank = np.zeros_like(test_avg)
    for m in sorted_models:
        probs = results[m]["oof_probs"]
        # Convert to ranks within each class
        for c in range(3):
            oof_rank[:, c] += np.argsort(np.argsort(probs[:, c])).astype(np.float32) / len(probs)
        probs_t = results[m]["test_probs"]
        for c in range(3):
            test_rank[:, c] += np.argsort(np.argsort(probs_t[:, c])).astype(np.float32) / len(probs_t)
    oof_rank /= len(sorted_models)
    test_rank /= len(sorted_models)
    f1 = f1_score(y_train, oof_rank.argmax(1), average="macro")
    acc = accuracy_score(y_train, oof_rank.argmax(1))
    print(f"\n  rank_avg: OOF F1={f1:.4f} Acc={acc:.4f}")
    if f1 > best_ens_f1:
        best_ens_f1 = f1
        best_ens_name = "rank_avg"
        best_oof = oof_rank
        best_test = test_rank

    final_oof_f1 = best_ens_f1
    final_oof_acc = accuracy_score(y_train, best_oof.argmax(1))
    print(f"\n*** Best ensemble: {best_ens_name} ***")
    print(f"*** OOF F1: {final_oof_f1:.4f} | Acc: {final_oof_acc:.4f} ***")
    print(classification_report(y_train, best_oof.argmax(1), target_names=LABELS, digits=4))

    # Build submission
    test_preds = best_test.argmax(1)
    sub_ids = []
    for _, r in val_df.iterrows():
        if pd.notna(r.get("hs")):
            sub_ids.append(os.path.basename(r["hs"]))
        elif pd.notna(r.get("ms")):
            sub_ids.append(os.path.basename(r["ms"]))
        else:
            sub_ids.append(os.path.basename(r["rgb"]))

    pred_labels = [ID2LBL[p] for p in test_preds]
    sub = pd.DataFrame({"Id": sub_ids, "Category": pred_labels})

    f1_str = f"{final_oof_f1:.4f}".replace(".", "p")
    acc_str = f"{final_oof_acc:.4f}".replace(".", "p")
    sub_name = f"submission_gbm_v2_oof_f1_{f1_str}_acc_{acc_str}.csv"
    sub_path = os.path.join(OUT_DIR, sub_name)
    sub.to_csv(sub_path, index=False)
    print(f"\nBest ensemble submission: {sub_path}")

    # Also save best single model submission
    best_single = sorted_models[0]
    test_preds_single = results[best_single]["test_probs"].argmax(1)
    pred_labels_single = [ID2LBL[p] for p in test_preds_single]
    sub_single = pd.DataFrame({"Id": sub_ids, "Category": pred_labels_single})
    f1_s = f"{results[best_single]['f1']:.4f}".replace(".", "p")
    sub_single.to_csv(os.path.join(OUT_DIR, f"submission_best_single_{best_single}_f1_{f1_s}.csv"), index=False)

    # Save probabilities for stacking
    np.save(os.path.join(OUT_DIR, "gbm_v2_oof_probs.npy"), best_oof)
    np.save(os.path.join(OUT_DIR, "gbm_v2_test_probs.npy"), best_test)

    print("\nDone!")


if __name__ == "__main__":
    main()
