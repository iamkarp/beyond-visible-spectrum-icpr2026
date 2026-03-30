"""
Beyond Visible Spectrum v4 - Fixed leakage, proper cross-validated features

Fix from v3: Spectral library features (Mahalanobis, spectral angle)
are now computed PER-FOLD to prevent data leakage. The class prototypes
are built from the training fold only and applied to the validation fold.

Also adds stronger regularization and more diverse models.
"""

import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
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
HS_DROP_FIRST = 10
HS_DROP_LAST = 14
HS_TARGET_CH = 101  # standardize to 101 channels

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ============================================================
# Data loading
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
# Continuum Removal
# ============================================================
def continuum_removal(spectrum):
    n = len(spectrum)
    if n < 3:
        return spectrum
    hull_points = [0, n - 1]
    changed = True
    while changed:
        changed = False
        new_hull = [hull_points[0]]
        for i in range(1, len(hull_points)):
            start = hull_points[i - 1]
            end = hull_points[i]
            max_dist = 0
            max_idx = -1
            for j in range(start + 1, end):
                t = (j - start) / (end - start)
                interp = spectrum[start] + t * (spectrum[end] - spectrum[start])
                if spectrum[j] > interp + max_dist:
                    max_dist = spectrum[j] - interp
                    max_idx = j
            if max_idx >= 0:
                new_hull.append(max_idx)
                changed = True
            new_hull.append(end)
        hull_points = sorted(set(new_hull))
    hull_interp = np.interp(range(n), hull_points, spectrum[hull_points])
    return spectrum / (hull_interp + 1e-8)


def spectral_angle(spec1, spec2):
    dot = np.dot(spec1, spec2)
    norm1 = np.linalg.norm(spec1)
    norm2 = np.linalg.norm(spec2)
    cos_angle = dot / (norm1 * norm2 + 1e-8)
    return np.arccos(np.clip(cos_angle, -1, 1))


# ============================================================
# Feature extraction (non-leaky features only)
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

def glcm_features(gray_uint8, prefix):
    features = {}
    gray_q = (gray_uint8 / 8).astype(np.uint8)
    try:
        glcm = graycomatrix(gray_q, distances=[1, 2],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=32, symmetric=True, normed=True)
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
            vals = graycoprops(glcm, prop)
            features[f"{prefix}_glcm_{prop}_mean"] = np.mean(vals)
            features[f"{prefix}_glcm_{prop}_std"] = np.std(vals)
    except:
        pass
    return features

def lbp_features(gray, prefix):
    features = {}
    H, W = gray.shape
    padded = np.pad(gray, 1, mode="reflect")
    pattern = np.zeros_like(gray, dtype=np.float32)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dy = int(round(np.sin(angle)))
        dx = int(round(np.cos(angle)))
        neighbor = padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
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

    features.update(band_statistics(img_rgb / 255.0, "rgb"))

    for c, name in enumerate(["r", "g", "b"]):
        hist = np.histogram(img_rgb[:, :, c], bins=16, range=(0, 256))[0]
        hist = hist / (hist.sum() + 1e-8)
        for b in range(16):
            features[f"rgb_hist_{name}_{b}"] = hist[b]
        features[f"rgb_entropy_{name}"] = -np.sum(hist * np.log(hist + 1e-8))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    features.update(band_statistics(hsv / np.array([180, 255, 255], dtype=np.float32), "hsv"))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    features.update(band_statistics(lab / np.array([100, 255, 255], dtype=np.float32), "lab"))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features.update(spatial_features(gray.astype(np.float32), "rgb_gray"))
    features.update(lbp_features(gray.astype(np.float32), "rgb"))
    features.update(glcm_features(gray, "rgb"))
    for c, name in enumerate(["r", "g", "b"]):
        features.update(glcm_features(img_rgb[:, :, c].astype(np.uint8), f"rgb_{name}"))

    total = img_rgb.sum(axis=2) + 1e-8
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    features["rgb_green_ratio_mean"] = np.mean(g / total)
    features["rgb_green_ratio_std"] = np.std(g / total)
    features["rgb_red_ratio_mean"] = np.mean(r / total)
    features["rgb_rg_ratio_mean"] = np.mean(r / (g + 1e-8))
    features["rgb_rg_ratio_std"] = np.std(r / (g + 1e-8))

    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    exg = 2 * gn - rn - bn
    features["rgb_exg_mean"] = np.mean(exg)
    features["rgb_exg_std"] = np.std(exg)
    features["rgb_exr_mean"] = np.mean(1.4 * rn - gn)
    features["rgb_exgr_mean"] = np.mean(exg - (1.4 * rn - gn))

    H, W = gray.shape
    h2, w2 = H // 2, W // 2
    quads = [gray[:h2, :w2], gray[:h2, w2:], gray[h2:, :w2], gray[h2:, w2:]]
    features["rgb_quad_std"] = np.std([np.mean(q) for q in quads])

    return features


def extract_ms_features(path):
    arr = read_tiff(path)
    features = {}
    eps = 1e-6

    features.update(band_statistics(arr, "ms"))
    blue, green, red, rededge, nir = [arr[:, :, i] for i in range(5)]

    def safe_idx(name, num, den):
        v = num / (den + eps)
        features[f"ms_{name}_mean"] = np.mean(v)
        features[f"ms_{name}_std"] = np.std(v)
        features[f"ms_{name}_min"] = np.min(v)
        features[f"ms_{name}_max"] = np.max(v)
        features[f"ms_{name}_median"] = np.median(v)
        features[f"ms_{name}_q25"] = np.percentile(v, 25)
        features[f"ms_{name}_q75"] = np.percentile(v, 75)
        features[f"ms_{name}_range"] = np.max(v) - np.min(v)
        features[f"ms_{name}_iqr"] = features[f"ms_{name}_q75"] - features[f"ms_{name}_q25"]
        return v

    ndvi = safe_idx("ndvi", nir - red, nir + red)
    ndre = safe_idx("ndre", nir - rededge, nir + rededge)
    gndvi = safe_idx("gndvi", nir - green, nir + green)
    safe_idx("savi", 1.5 * (nir - red), nir + red + 0.5)
    safe_idx("evi", 2.5 * (nir - red), nir + 6 * red - 7.5 * blue + 1)
    safe_idx("osavi", 1.16 * (nir - red), nir + red + 0.16)
    safe_idx("ci_green", nir / (green + eps) - 1, np.ones_like(green))
    safe_idx("ci_rededge", nir / (rededge + eps) - 1, np.ones_like(rededge))
    safe_idx("ndbi", blue - red, blue + red)
    safe_idx("ngrdi", green - red, green + red)

    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    features["ms_mcari_mean"] = np.mean(mcari)
    features["ms_mcari_std"] = np.std(mcari)

    tcari = 3 * ((rededge - red) - 0.2 * (rededge - green) * (rededge / (red + eps)))
    osavi_arr = 1.16 * (nir - red) / (nir + red + 0.16 + eps)
    features["ms_tcari_osavi_mean"] = np.mean(tcari / (osavi_arr + eps))
    features["ms_tcari_osavi_std"] = np.std(tcari / (osavi_arr + eps))

    msavi = 0.5 * (2 * nir + 1 - np.sqrt(np.clip((2 * nir + 1)**2 - 8 * (nir - red), 0, None) + eps))
    features["ms_msavi_mean"] = np.mean(msavi)
    features["ms_msavi_std"] = np.std(msavi)

    features["ms_ndvi_positive_frac"] = np.mean(ndvi.ravel() > 0)
    features["ms_ndvi_high_frac"] = np.mean(ndvi.ravel() > 0.5)

    band_names = ["blue", "green", "red", "rededge", "nir"]
    for i in range(5):
        for j in range(i + 1, 5):
            ratio = arr[:, :, i] / (arr[:, :, j] + eps)
            features[f"ms_ratio_{band_names[i]}_{band_names[j]}_mean"] = np.mean(ratio)
            features[f"ms_ratio_{band_names[i]}_{band_names[j]}_std"] = np.std(ratio)

    features.update(spatial_features(ndvi.astype(np.float32), "ms_ndvi_sp"))
    features.update(spatial_features(ndre.astype(np.float32), "ms_ndre_sp"))

    ndvi_uint8 = np.clip((ndvi + 1) * 127.5, 0, 255).astype(np.uint8)
    features.update(glcm_features(ndvi_uint8, "ms_ndvi"))

    flat = arr.reshape(-1, 5)
    try:
        corr = np.corrcoef(flat.T)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"ms_corr_{band_names[i]}_{band_names[j]}"] = corr[i, j]
    except:
        pass

    for i, name in enumerate(band_names):
        band = arr[:, :, i]
        H, W = band.shape
        h2, w2 = H // 2, W // 2
        quads = [band[:h2, :w2], band[:h2, w2:], band[h2:, :w2], band[h2:, w2:]]
        features[f"ms_{name}_quad_std"] = np.std([np.mean(q) for q in quads])

    return features


def get_hs_spectrum(path):
    """Read HS and return standardized mean spectrum (101 channels)."""
    arr = read_tiff(path)
    B = arr.shape[2]
    if B > (HS_DROP_FIRST + HS_DROP_LAST + 1):
        arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST]
    # Standardize to 101 channels
    C = arr.shape[2]
    if C > HS_TARGET_CH:
        arr = arr[:, :, :HS_TARGET_CH]
    elif C < HS_TARGET_CH:
        pad = np.zeros((arr.shape[0], arr.shape[1], HS_TARGET_CH - C), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=2)
    return arr, np.mean(arr.reshape(-1, HS_TARGET_CH), axis=0)


def extract_hs_features(arr):
    """Extract non-leaky HS features from the image array."""
    features = {}
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)
    mean_spectrum = np.mean(flat, axis=0)
    std_spectrum = np.std(flat, axis=0)
    median_spectrum = np.median(flat, axis=0)

    # Continuum removal
    cr_spectrum = continuum_removal(mean_spectrum)
    features["hs_cr_min"] = np.min(cr_spectrum)
    features["hs_cr_min_pos"] = np.argmin(cr_spectrum) / C
    features["hs_cr_mean"] = np.mean(cr_spectrum)
    features["hs_cr_std"] = np.std(cr_spectrum)
    features["hs_cr_depth_red"] = 1.0 - cr_spectrum[int(C * 0.40)]
    features["hs_cr_depth_rededge"] = 1.0 - cr_spectrum[int(C * 0.55)]
    features["hs_cr_depth_green"] = 1.0 - cr_spectrum[int(C * 0.15)]
    features["hs_cr_absorption_width"] = np.sum(cr_spectrum < 0.95) / C

    cr_median = continuum_removal(median_spectrum)
    features["hs_cr_med_min"] = np.min(cr_median)
    features["hs_cr_med_depth_red"] = 1.0 - cr_median[int(C * 0.40)]

    # Sampled band statistics
    n_sample = 20
    sample_indices = np.linspace(0, C - 1, n_sample, dtype=int)
    for i, idx in enumerate(sample_indices):
        band = flat[:, idx]
        features[f"hs_b{i}_mean"] = np.mean(band)
        features[f"hs_b{i}_std"] = np.std(band)
        features[f"hs_b{i}_min"] = np.min(band)
        features[f"hs_b{i}_max"] = np.max(band)
        features[f"hs_b{i}_median"] = np.median(band)

    features["hs_total_mean"] = np.mean(flat)
    features["hs_total_std"] = np.std(flat)
    features["hs_total_range"] = float(mean_spectrum.max() - mean_spectrum.min())

    # Spectral derivatives
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

    std_deriv1 = np.diff(std_spectrum)
    features["hs_std_d1_mean"] = np.mean(std_deriv1)
    features["hs_std_d1_std"] = np.std(std_deriv1)

    features["hs_peak_band"] = np.argmax(mean_spectrum) / C
    features["hs_trough_band"] = np.argmin(mean_spectrum) / C
    features["hs_peak_value"] = np.max(mean_spectrum)
    features["hs_trough_value"] = np.min(mean_spectrum)

    features["hs_spectral_area"] = np.trapezoid(mean_spectrum)
    spec_norm = mean_spectrum / (mean_spectrum.sum() + 1e-8)
    features["hs_spectral_entropy"] = -np.sum(spec_norm * np.log(spec_norm + 1e-8))

    x = np.arange(C, dtype=np.float32)
    spec_sum = mean_spectrum.sum() + 1e-8
    centroid = np.sum(x * mean_spectrum) / spec_sum
    features["hs_spectral_centroid"] = centroid / C
    spread = np.sqrt(np.sum((x - centroid)**2 * mean_spectrum) / spec_sum)
    features["hs_spectral_spread"] = spread / C

    # Key wavelength regions
    green_idx = int(C * 0.15)
    red_idx = int(C * 0.40)
    re_start = int(C * 0.52)
    re_peak = int(C * 0.62)
    nir_idx = int(C * 0.77)
    nir2_idx = min(int(C * 0.90), C - 1)

    features["hs_green_peak"] = mean_spectrum[green_idx]
    features["hs_red_absorption"] = mean_spectrum[red_idx]
    features["hs_red_edge_start"] = mean_spectrum[re_start]
    features["hs_red_edge_peak"] = mean_spectrum[re_peak]
    features["hs_nir_plateau"] = mean_spectrum[nir_idx]
    features["hs_nir_value"] = mean_spectrum[nir2_idx]

    if re_peak > re_start:
        features["hs_red_edge_slope"] = (mean_spectrum[re_peak] - mean_spectrum[re_start]) / (re_peak - re_start)
    re_deriv = deriv1[max(0, re_start-1):min(len(deriv1), re_peak)]
    if len(re_deriv) > 0:
        features["hs_red_edge_inflection"] = (np.argmax(re_deriv) + re_start) / C
        features["hs_red_edge_max_slope"] = np.max(re_deriv)

    features["hs_nir_red_ratio"] = mean_spectrum[nir_idx] / (mean_spectrum[red_idx] + 1e-8)
    features["hs_nir_green_ratio"] = mean_spectrum[nir_idx] / (mean_spectrum[green_idx] + 1e-8)
    features["hs_re_red_ratio"] = mean_spectrum[re_peak] / (mean_spectrum[red_idx] + 1e-8)

    if green_idx < red_idx < nir_idx:
        baseline = mean_spectrum[green_idx] + (mean_spectrum[nir_idx] - mean_spectrum[green_idx]) * (red_idx - green_idx) / (nir_idx - green_idx + 1e-8)
        features["hs_red_absorption_depth"] = baseline - mean_spectrum[red_idx]
        features["hs_red_absorption_relative"] = (baseline - mean_spectrum[red_idx]) / (baseline + 1e-8)

    # Band region statistics
    regions = {
        "blue_vis": (0, int(C * 0.10)),
        "green_vis": (int(C * 0.10), int(C * 0.25)),
        "red_vis": (int(C * 0.25), int(C * 0.45)),
        "red_edge": (int(C * 0.45), int(C * 0.65)),
        "nir": (int(C * 0.65), C),
    }
    for rname, (start, end) in regions.items():
        region_data = mean_spectrum[start:end]
        features[f"hs_region_{rname}_mean"] = np.mean(region_data)
        features[f"hs_region_{rname}_std"] = np.std(region_data)
        features[f"hs_region_{rname}_slope"] = (region_data[-1] - region_data[0]) / (len(region_data) + 1e-8)

    for r1 in regions:
        for r2 in regions:
            if r1 < r2:
                s1, e1 = regions[r1]
                s2, e2 = regions[r2]
                m1 = np.mean(mean_spectrum[s1:e1])
                m2 = np.mean(mean_spectrum[s2:e2])
                features[f"hs_cross_{r1}_{r2}_ratio"] = m1 / (m2 + 1e-8)

    # PCA features
    try:
        n_comp = min(25, C, H * W)
        pca = PCA(n_components=n_comp)
        pca_result = pca.fit_transform(flat)
        for i in range(min(n_comp, 25)):
            comp = pca_result[:, i]
            features[f"hs_pca{i}_mean"] = np.mean(comp)
            features[f"hs_pca{i}_std"] = np.std(comp)
        for i, ev in enumerate(pca.explained_variance_ratio_[:15]):
            features[f"hs_pca_ev{i}"] = ev
        features["hs_pca_cumev_3"] = np.sum(pca.explained_variance_ratio_[:3])
        features["hs_pca_cumev_5"] = np.sum(pca.explained_variance_ratio_[:5])
        features["hs_pca_cumev_10"] = np.sum(pca.explained_variance_ratio_[:10])
    except:
        pass

    # Spatial features
    for name, idx in [("green", green_idx), ("red", red_idx), ("re", re_start), ("nir", nir_idx)]:
        if idx < C:
            features.update(spatial_features(arr[:, :, idx], f"hs_{name}"))

    # GLCM on PCA1
    try:
        pc1_img = pca_result[:, 0].reshape(H, W)
        pc1_uint8 = np.clip((pc1_img - pc1_img.min()) / (pc1_img.max() - pc1_img.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
        features.update(glcm_features(pc1_uint8, "hs_pca1"))
    except:
        pass

    # Pixel heterogeneity
    pixel_stds = np.std(flat, axis=0)
    features["hs_pixel_hetero_mean"] = np.mean(pixel_stds)
    features["hs_pixel_hetero_std"] = np.std(pixel_stds)
    cv = pixel_stds / (np.mean(flat, axis=0) + 1e-8)
    features["hs_cv_mean"] = np.mean(cv)
    features["hs_cv_std"] = np.std(cv)

    return features


def extract_cross_modal_features(rgb_path, ms_path):
    features = {}
    try:
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            return features
        rgb_green = rgb[:, :, 1].astype(np.float32) / 255.0
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        ms = read_tiff(ms_path)
        for c in range(ms.shape[2]):
            mn, mx = ms[:, :, c].min(), ms[:, :, c].max()
            ms[:, :, c] = (ms[:, :, c] - mn) / (mx - mn + 1e-8)

        features["cross_rgb_ms_green_corr"] = np.corrcoef(rgb_green.ravel(), ms[:, :, 1].ravel())[0, 1]
        features["cross_rgb_ms_green_diff"] = np.mean(np.abs(rgb_green - ms[:, :, 1]))
        features["cross_rgb_ms_nir_corr"] = np.corrcoef(rgb_gray.ravel(), ms[:, :, 4].ravel())[0, 1]
    except:
        pass
    return features


# ============================================================
# Spectral library features - PROPERLY CROSS-VALIDATED
# ============================================================
def build_spectral_library(spectra, labels):
    """Build class prototypes from a set of spectra and labels."""
    libraries = {}
    for label in LABELS:
        class_idx = [i for i, l in enumerate(labels) if l == LBL2ID[label]]
        if not class_idx:
            continue
        class_spectra = np.array([spectra[i] for i in class_idx])
        mean_spec = np.mean(class_spectra, axis=0)
        std_spec = np.std(class_spectra, axis=0) + 1e-6
        libraries[label] = {"mean": mean_spec, "std": std_spec}
        # Only compute covariance if we have enough samples
        if len(class_idx) > 20:
            try:
                # Use regularized covariance
                cov = np.cov(class_spectra.T)
                reg = np.eye(cov.shape[0]) * 0.01 * np.trace(cov) / cov.shape[0]
                libraries[label]["cov_inv"] = np.linalg.inv(cov + reg)
            except:
                libraries[label]["cov_inv"] = None
        else:
            libraries[label]["cov_inv"] = None
    return libraries


def compute_library_features(spectrum, libraries):
    """Compute distance features from a single spectrum to class prototypes."""
    features = {}
    if spectrum is None:
        return features

    for label, lib in libraries.items():
        diff = spectrum - lib["mean"]
        features[f"slib_{label}_euclid"] = np.sqrt(np.sum(diff**2))
        features[f"slib_{label}_norm_euclid"] = np.sqrt(np.sum((diff / lib["std"])**2))
        features[f"slib_{label}_angle"] = spectral_angle(spectrum, lib["mean"])
        features[f"slib_{label}_corr"] = np.corrcoef(spectrum, lib["mean"])[0, 1]
        if lib["cov_inv"] is not None:
            try:
                features[f"slib_{label}_mahal"] = np.sqrt(np.clip(diff @ lib["cov_inv"] @ diff, 0, None))
            except:
                pass

    # Relative distances
    for l1 in LABELS:
        for l2 in LABELS:
            if l1 < l2:
                k1 = f"slib_{l1}_euclid"
                k2 = f"slib_{l2}_euclid"
                if k1 in features and k2 in features:
                    features[f"slib_{l1}_{l2}_ratio"] = features[k1] / (features[k2] + 1e-8)
                k1a = f"slib_{l1}_angle"
                k2a = f"slib_{l2}_angle"
                if k1a in features and k2a in features:
                    features[f"slib_{l1}_{l2}_angle_ratio"] = features[k1a] / (features[k2a] + 1e-8)

    return features


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Beyond Visible Spectrum v4 - Fixed leakage")
    print("=" * 60)

    train_idx = build_index(ROOT, "train")
    val_idx = build_index(ROOT, "val")
    train_df = make_df(train_idx, has_labels=True)
    val_df = make_df(val_idx, has_labels=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Phase 1: Extract base features + spectra
    print("\nPhase 1: Extracting base features...")
    train_base_features = []
    train_spectra = []  # List of mean spectra, aligned with train_df
    train_hs_arrays = []  # Full HS arrays for spectral library

    for i, (_, row) in enumerate(train_df.iterrows()):
        feats = {"base_id": row["base_id"]}
        if pd.notna(row.get("rgb")):
            feats.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")):
            feats.update(extract_ms_features(row["ms"]))

        spectrum = None
        if pd.notna(row.get("hs")):
            hs_arr, spectrum = get_hs_spectrum(row["hs"])
            feats.update(extract_hs_features(hs_arr))
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            feats.update(extract_cross_modal_features(row["rgb"], row["ms"]))

        train_base_features.append(feats)
        train_spectra.append(spectrum)
        if (i + 1) % 100 == 0:
            print(f"  Train: {i + 1}/{len(train_df)}")

    val_base_features = []
    val_spectra = []

    for i, (_, row) in enumerate(val_df.iterrows()):
        feats = {"base_id": row["base_id"]}
        if pd.notna(row.get("rgb")):
            feats.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")):
            feats.update(extract_ms_features(row["ms"]))

        spectrum = None
        if pd.notna(row.get("hs")):
            hs_arr, spectrum = get_hs_spectrum(row["hs"])
            feats.update(extract_hs_features(hs_arr))
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            feats.update(extract_cross_modal_features(row["rgb"], row["ms"]))

        val_base_features.append(feats)
        val_spectra.append(spectrum)
        if (i + 1) % 100 == 0:
            print(f"  Val: {i + 1}/{len(val_df)}")

    # Convert to DataFrames
    train_base_df = pd.DataFrame(train_base_features)
    val_base_df = pd.DataFrame(val_base_features)

    base_feature_cols = [c for c in train_base_df.columns if c != "base_id"]
    X_train_base = train_base_df[base_feature_cols].fillna(0).values.astype(np.float32)
    X_test_base = val_base_df[base_feature_cols].fillna(0).values.astype(np.float32)
    X_train_base = np.nan_to_num(X_train_base, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_base = np.nan_to_num(X_test_base, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = np.array([LBL2ID[l] for l in train_df["label"]])
    train_spectra_arr = np.array(train_spectra)  # (N, 101)
    val_spectra_arr = np.array(val_spectra)

    print(f"  Base features: {X_train_base.shape[1]}")

    # Phase 2: Cross-validated spectral library features
    print("\nPhase 2: Cross-validated spectral library features...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # For OOF: compute library from train fold, apply to val fold
    oof_slib_features = [None] * len(train_df)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y_train)):
        # Build library from training fold only
        fold_spectra = [train_spectra[i] for i in tr_idx]
        fold_labels = [y_train[i] for i in tr_idx]
        library = build_spectral_library(fold_spectra, fold_labels)

        # Apply to validation fold
        for i in va_idx:
            feats = compute_library_features(train_spectra[i], library)
            oof_slib_features[i] = feats
        print(f"  Fold {fold + 1}: library built from {len(tr_idx)} samples, applied to {len(va_idx)}")

    # For test: compute library from ALL training data
    all_library = build_spectral_library(train_spectra, y_train.tolist())
    test_slib_features = []
    for spec in val_spectra:
        test_slib_features.append(compute_library_features(spec, all_library))

    # Also compute all-data library features for training (for final model fitting)
    train_slib_all = []
    for spec in train_spectra:
        train_slib_all.append(compute_library_features(spec, all_library))

    # Convert to arrays
    slib_df_oof = pd.DataFrame(oof_slib_features)
    slib_df_test = pd.DataFrame(test_slib_features)
    slib_df_train_all = pd.DataFrame(train_slib_all)

    slib_cols = [c for c in slib_df_oof.columns]
    X_slib_oof = slib_df_oof[slib_cols].fillna(0).values.astype(np.float32)
    X_slib_test = slib_df_test[slib_cols].fillna(0).values.astype(np.float32)
    X_slib_train_all = slib_df_train_all[slib_cols].fillna(0).values.astype(np.float32)

    X_slib_oof = np.nan_to_num(X_slib_oof, nan=0.0, posinf=0.0, neginf=0.0)
    X_slib_test = np.nan_to_num(X_slib_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_slib_train_all = np.nan_to_num(X_slib_train_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Spectral library features: {X_slib_oof.shape[1]}")

    # Combine: base features + OOF spectral library features
    X_train_combined_oof = np.hstack([X_train_base, X_slib_oof])
    X_test_combined = np.hstack([X_test_base, X_slib_test])
    # For final model training, use all-data library features
    X_train_combined_all = np.hstack([X_train_base, X_slib_train_all])

    all_feature_cols = base_feature_cols + slib_cols
    print(f"  Total combined features: {X_train_combined_oof.shape[1]}")

    # Phase 3: Train models WITH PROPER OOF EVALUATION
    print("\n" + "=" * 60)
    print("Phase 3: Training models (proper OOF)")
    print("=" * 60)

    results = {}

    # We need to be careful: for models using slib features,
    # the OOF evaluation must use X_train_combined_oof (fold-specific slib).
    # But for final test predictions, we train on X_train_combined_all and predict X_test_combined.

    model_configs = [
        ("lgb_v1", {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
            "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
            "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000,
            "verbose": -1, "random_state": SEED,
        }),
        ("lgb_v2", {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4,
            "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
            "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000,
            "verbose": -1, "random_state": SEED + 10,
        }),
        ("lgb_v3", {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
            "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500,
            "verbose": -1, "random_state": SEED + 20,
        }),
    ]

    for name, params in model_configs:
        oof_probs = np.zeros((len(y_train), 3))
        test_probs = np.zeros((len(X_test_combined), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y_train)):
            # For OOF: use fold-specific slib features
            X_tr_oof = X_train_combined_oof[tr_idx]
            X_va_oof = X_train_combined_oof[va_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr_oof, y_train[tr_idx],
                      eval_set=[(X_va_oof, y_train[va_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_probs[va_idx] = model.predict_proba(X_va_oof)

            # For test: train on all-data slib features for this fold
            X_tr_all = X_train_combined_all[tr_idx]
            model2 = lgb.LGBMClassifier(**params)
            model2.fit(X_tr_all, y_train[tr_idx],
                       eval_set=[(X_train_combined_all[va_idx], y_train[va_idx])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
            test_probs += model2.predict_proba(X_test_combined) / N_FOLDS

        f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
        acc = accuracy_score(y_train, oof_probs.argmax(1))
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1, "acc": acc}

    # XGBoost
    xgb_configs = [
        ("xgb_v1", {
            "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
            "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
            "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
            "reg_lambda": 1.0, "n_estimators": 2000, "tree_method": "hist",
            "random_state": SEED, "verbosity": 0,
        }),
        ("xgb_v2", {
            "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
            "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
            "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
            "reg_lambda": 2.0, "n_estimators": 3000, "tree_method": "hist",
            "random_state": SEED + 5, "verbosity": 0,
        }),
    ]

    for name, params in xgb_configs:
        oof_probs = np.zeros((len(y_train), 3))
        test_probs = np.zeros((len(X_test_combined), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y_train)):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_combined_oof[tr_idx], y_train[tr_idx],
                      eval_set=[(X_train_combined_oof[va_idx], y_train[va_idx])], verbose=False)
            oof_probs[va_idx] = model.predict_proba(X_train_combined_oof[va_idx])

            model2 = xgb.XGBClassifier(**params)
            model2.fit(X_train_combined_all[tr_idx], y_train[tr_idx],
                       eval_set=[(X_train_combined_all[va_idx], y_train[va_idx])], verbose=False)
            test_probs += model2.predict_proba(X_test_combined) / N_FOLDS

        f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
        acc = accuracy_score(y_train, oof_probs.argmax(1))
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1, "acc": acc}

    # sklearn GB
    oof_probs = np.zeros((len(y_train), 3))
    test_probs = np.zeros((len(X_test_combined), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y_train)):
        model = GradientBoostingClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            subsample=0.75, min_samples_leaf=8, random_state=SEED)
        model.fit(X_train_combined_oof[tr_idx], y_train[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_train_combined_oof[va_idx])

        model2 = GradientBoostingClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            subsample=0.75, min_samples_leaf=8, random_state=SEED)
        model2.fit(X_train_combined_all[tr_idx], y_train[tr_idx])
        test_probs += model2.predict_proba(X_test_combined) / N_FOLDS
    f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
    acc = accuracy_score(y_train, oof_probs.argmax(1))
    print(f"  sklearn_gb: Acc={acc:.4f} F1={f1:.4f}")
    results["sklearn_gb"] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1, "acc": acc}

    # ExtraTrees
    oof_probs = np.zeros((len(y_train), 3))
    test_probs = np.zeros((len(X_test_combined), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y_train)):
        model = ExtraTreesClassifier(n_estimators=2000, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        model.fit(X_train_combined_oof[tr_idx], y_train[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_train_combined_oof[va_idx])

        model2 = ExtraTreesClassifier(n_estimators=2000, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        model2.fit(X_train_combined_all[tr_idx], y_train[tr_idx])
        test_probs += model2.predict_proba(X_test_combined) / N_FOLDS
    f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
    acc = accuracy_score(y_train, oof_probs.argmax(1))
    print(f"  extra_trees: Acc={acc:.4f} F1={f1:.4f}")
    results["extra_trees"] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1, "acc": acc}

    # RandomForest
    oof_probs = np.zeros((len(y_train), 3))
    test_probs = np.zeros((len(X_test_combined), 3))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y_train)):
        model = RandomForestClassifier(n_estimators=2000, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        model.fit(X_train_combined_oof[tr_idx], y_train[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_train_combined_oof[va_idx])

        model2 = RandomForestClassifier(n_estimators=2000, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        model2.fit(X_train_combined_all[tr_idx], y_train[tr_idx])
        test_probs += model2.predict_proba(X_test_combined) / N_FOLDS
    f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
    acc = accuracy_score(y_train, oof_probs.argmax(1))
    print(f"  random_forest: Acc={acc:.4f} F1={f1:.4f}")
    results["random_forest"] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1, "acc": acc}

    # Also train base-only models (no slib features) for diversity
    print("\n  Training base-only models (no slib)...")
    for name, params in [
        ("lgb_base", {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
            "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.7,
            "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000,
            "verbose": -1, "random_state": SEED,
        }),
    ]:
        oof_probs = np.zeros((len(y_train), 3))
        test_probs = np.zeros((len(X_test_base), 3))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_base, y_train)):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_base[tr_idx], y_train[tr_idx],
                      eval_set=[(X_train_base[va_idx], y_train[va_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_probs[va_idx] = model.predict_proba(X_train_base[va_idx])
            test_probs += model.predict_proba(X_test_base) / N_FOLDS
        f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
        acc = accuracy_score(y_train, oof_probs.argmax(1))
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof_probs": oof_probs, "test_probs": test_probs, "f1": f1, "acc": acc}

    # Phase 4: Ensemble
    print("\n" + "=" * 60)
    print("Phase 4: Ensemble")
    print("=" * 60)

    sorted_models = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    print("\nAll models ranked:")
    for m in sorted_models:
        print(f"  {m:20s} Acc={results[m]['acc']:.4f} F1={results[m]['f1']:.4f}")

    best_ens_acc = -1
    best_ens_name = None
    best_oof = None
    best_test = None

    for k in [3, 5, 7, len(sorted_models)]:
        top_k = sorted_models[:min(k, len(sorted_models))]
        weights = np.array([results[m]["acc"] for m in top_k])
        weights = weights ** 3
        weights = weights / weights.sum()

        oof_ens = sum(w * results[m]["oof_probs"] for m, w in zip(top_k, weights))
        test_ens = sum(w * results[m]["test_probs"] for m, w in zip(top_k, weights))
        acc = accuracy_score(y_train, oof_ens.argmax(1))
        f1 = f1_score(y_train, oof_ens.argmax(1), average="macro")
        print(f"\n  top{k}_weighted: Acc={acc:.4f} F1={f1:.4f}")
        if acc > best_ens_acc:
            best_ens_acc = acc
            best_ens_name = f"top{k}_weighted"
            best_oof = oof_ens
            best_test = test_ens

    oof_avg = sum(results[m]["oof_probs"] for m in sorted_models) / len(sorted_models)
    test_avg = sum(results[m]["test_probs"] for m in sorted_models) / len(sorted_models)
    acc = accuracy_score(y_train, oof_avg.argmax(1))
    f1 = f1_score(y_train, oof_avg.argmax(1), average="macro")
    print(f"\n  simple_avg: Acc={acc:.4f} F1={f1:.4f}")
    if acc > best_ens_acc:
        best_ens_acc = acc
        best_ens_name = "simple_avg"
        best_oof = oof_avg
        best_test = test_avg

    final_acc = best_ens_acc
    final_f1 = f1_score(y_train, best_oof.argmax(1), average="macro")
    print(f"\n*** Best: {best_ens_name} | Acc={final_acc:.4f} F1={final_f1:.4f} ***")
    print(classification_report(y_train, best_oof.argmax(1), target_names=LABELS, digits=4))

    # Save submission
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

    acc_str = f"{final_acc:.4f}".replace(".", "p")
    f1_str = f"{final_f1:.4f}".replace(".", "p")
    sub_name = f"submission_v4_oof_acc_{acc_str}_f1_{f1_str}.csv"
    sub_path = os.path.join(OUT_DIR, sub_name)
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    # Also save individual best model submission
    best_single = sorted_models[0]
    tp = results[best_single]["test_probs"].argmax(1)
    sub_s = pd.DataFrame({"Id": sub_ids, "Category": [ID2LBL[p] for p in tp]})
    acc_s = f"{results[best_single]['acc']:.4f}".replace(".", "p")
    sub_s.to_csv(os.path.join(OUT_DIR, f"submission_v4_{best_single}_acc_{acc_s}.csv"), index=False)

    np.save(os.path.join(OUT_DIR, "v4_oof_probs.npy"), best_oof)
    np.save(os.path.join(OUT_DIR, "v4_test_probs.npy"), best_test)

    print("\nDone!")


if __name__ == "__main__":
    main()
