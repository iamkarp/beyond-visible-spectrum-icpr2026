"""
Beyond Visible Spectrum v7 - Health class focus + model diversity

Key improvements over v6:
1. New health-specific spectral features (Red Edge Position, PRI, PSRI, ARI, derivatives)
2. SVM (RBF) and kNN models for ensemble diversity
3. Feature selection via mutual information (reduce noise from 900+ features)
4. Use only EfficientNet CNN PCA (100 - proven better than 350)
5. Threshold optimization for final predictions
6. Calibrated probability blending
"""

import os, re, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter1d

import cv2
import tifffile as tiff
import lightgbm as lgb
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = "/Users/macbook/Library/CloudStorage/GoogleDrive-jason.karpeles@pmg.com/My Drive/Projects/Beyond Visible Spectrum"
OUT_DIR = os.path.join(ROOT, "output")
N_FOLDS = 5
SEED = 42
HS_DROP_FIRST = 10
HS_DROP_LAST = 14
HS_TARGET_CH = 101

# Wavelength mapping: 125 bands over 450-950nm, ~4.03nm spacing
# After dropping first 10: start wavelength ≈ 490nm
# After dropping last 14: end wavelength ≈ 893nm
# 101 bands spanning 490-893nm, ~3.99nm/band
HS_WL_START = 490.0
HS_WL_END = 893.0
HS_WL = np.linspace(HS_WL_START, HS_WL_END, HS_TARGET_CH)

def wl_to_band(nm):
    return int(np.clip(np.round((nm - HS_WL_START) / (HS_WL_END - HS_WL_START) * (HS_TARGET_CH - 1)), 0, HS_TARGET_CH - 1))

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
# Helpers
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
            start, end = hull_points[i - 1], hull_points[i]
            max_dist, max_idx = 0, -1
            for j in range(start + 1, end):
                t = (j - start) / (end - start)
                interp = spectrum[start] + t * (spectrum[end] - spectrum[start])
                if spectrum[j] > interp + max_dist:
                    max_dist, max_idx = spectrum[j] - interp, j
            if max_idx >= 0:
                new_hull.append(max_idx)
                changed = True
            new_hull.append(end)
        hull_points = sorted(set(new_hull))
    hull_interp = np.interp(range(n), hull_points, spectrum[hull_points])
    return spectrum / (hull_interp + 1e-8)

def spectral_angle(s1, s2):
    cos = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))

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
        dy, dx = int(round(np.sin(angle))), int(round(np.cos(angle)))
        neighbor = padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        pattern += (neighbor > gray).astype(np.float32) * (2 ** i)
    hist = np.histogram(pattern, bins=32, range=(0, 256))[0]
    hist = hist / (hist.sum() + 1e-8)
    for b in range(32):
        features[f"{prefix}_lbp_{b}"] = hist[b]
    features[f"{prefix}_lbp_entropy"] = -np.sum(hist * np.log(hist + 1e-8))
    return features


# ============================================================
# Feature extraction (v6 base + v7 additions)
# ============================================================
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
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    total = r + g + b + 1e-8
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

    def vidx(name, num, den):
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

    ndvi = vidx("ndvi", nir - red, nir + red)
    ndre = vidx("ndre", nir - rededge, nir + rededge)
    gndvi = vidx("gndvi", nir - green, nir + green)
    vidx("savi", 1.5 * (nir - red), nir + red + 0.5)
    vidx("evi", 2.5 * (nir - red), nir + 6 * red - 7.5 * blue + 1)
    vidx("osavi", 1.16 * (nir - red), nir + red + 0.16)
    vidx("ci_green", nir / (green + eps) - 1, np.ones_like(green))
    vidx("ci_rededge", nir / (rededge + eps) - 1, np.ones_like(rededge))
    vidx("ndbi", blue - red, blue + red)
    vidx("ngrdi", green - red, green + red)

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

    bnames = ["blue", "green", "red", "rededge", "nir"]
    for i in range(5):
        for j in range(i + 1, 5):
            r = arr[:, :, i] / (arr[:, :, j] + eps)
            features[f"ms_ratio_{bnames[i]}_{bnames[j]}_mean"] = np.mean(r)
            features[f"ms_ratio_{bnames[i]}_{bnames[j]}_std"] = np.std(r)

    features.update(spatial_features(ndvi.astype(np.float32), "ms_ndvi_sp"))
    features.update(spatial_features(ndre.astype(np.float32), "ms_ndre_sp"))
    ndvi_u8 = np.clip((ndvi + 1) * 127.5, 0, 255).astype(np.uint8)
    features.update(glcm_features(ndvi_u8, "ms_ndvi"))

    flat = arr.reshape(-1, 5)
    try:
        corr = np.corrcoef(flat.T)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"ms_corr_{bnames[i]}_{bnames[j]}"] = corr[i, j]
    except:
        pass

    for i, name in enumerate(bnames):
        band = arr[:, :, i]
        H, W = band.shape
        h2, w2 = H // 2, W // 2
        quads = [band[:h2, :w2], band[:h2, w2:], band[h2:, :w2], band[h2:, w2:]]
        features[f"ms_{name}_quad_std"] = np.std([np.mean(q) for q in quads])

    return features


def get_hs_data(path):
    arr = read_tiff(path)
    B = arr.shape[2]
    if B > (HS_DROP_FIRST + HS_DROP_LAST + 1):
        arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST]
    C = arr.shape[2]
    if C > HS_TARGET_CH:
        arr = arr[:, :, :HS_TARGET_CH]
    elif C < HS_TARGET_CH:
        pad = np.zeros((arr.shape[0], arr.shape[1], HS_TARGET_CH - C), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=2)
    return arr, np.mean(arr.reshape(-1, HS_TARGET_CH), axis=0)


def extract_hs_features(arr):
    """Comprehensive HS features (v6 base + v7 health-specific additions)."""
    features = {}
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)
    mean_spec = np.mean(flat, axis=0)
    std_spec = np.std(flat, axis=0)
    median_spec = np.median(flat, axis=0)

    # === Raw mean spectrum ===
    for i in range(C):
        features[f"hs_raw_{i}"] = mean_spec[i]

    # === Continuum removal ===
    cr = continuum_removal(mean_spec)
    features["hs_cr_min"] = np.min(cr)
    features["hs_cr_min_pos"] = np.argmin(cr) / C
    features["hs_cr_mean"] = np.mean(cr)
    features["hs_cr_std"] = np.std(cr)
    features["hs_cr_depth_red"] = 1.0 - cr[int(C * 0.40)]
    features["hs_cr_depth_rededge"] = 1.0 - cr[int(C * 0.55)]
    features["hs_cr_depth_green"] = 1.0 - cr[int(C * 0.15)]
    features["hs_cr_absorption_width"] = np.sum(cr < 0.95) / C
    cr_med = continuum_removal(median_spec)
    features["hs_cr_med_min"] = np.min(cr_med)
    features["hs_cr_med_depth_red"] = 1.0 - cr_med[int(C * 0.40)]

    # === Sampled band statistics ===
    n_sample = 20
    sample_idx = np.linspace(0, C - 1, n_sample, dtype=int)
    for i, idx in enumerate(sample_idx):
        band = flat[:, idx]
        features[f"hs_b{i}_mean"] = np.mean(band)
        features[f"hs_b{i}_std"] = np.std(band)
        features[f"hs_b{i}_min"] = np.min(band)
        features[f"hs_b{i}_max"] = np.max(band)
        features[f"hs_b{i}_median"] = np.median(band)

    # Overall stats
    features["hs_total_mean"] = np.mean(flat)
    features["hs_total_std"] = np.std(flat)
    features["hs_total_range"] = float(mean_spec.max() - mean_spec.min())

    # === Derivatives (comprehensive from v6 + v7 smoothed derivatives) ===
    d1 = np.diff(mean_spec)
    features["hs_d1_mean"] = np.mean(d1)
    features["hs_d1_std"] = np.std(d1)
    features["hs_d1_max"] = np.max(d1)
    features["hs_d1_min"] = np.min(d1)
    features["hs_d1_max_pos"] = np.argmax(d1) / len(d1)
    features["hs_d1_min_pos"] = np.argmin(d1) / len(d1)
    features["hs_d1_zero_crossings"] = np.sum(np.diff(np.sign(d1)) != 0) / len(d1)

    d2 = np.diff(d1)
    features["hs_d2_mean"] = np.mean(d2)
    features["hs_d2_std"] = np.std(d2)
    features["hs_d2_max"] = np.max(d2)

    std_d1 = np.diff(std_spec)
    features["hs_std_d1_mean"] = np.mean(std_d1)
    features["hs_std_d1_std"] = np.std(std_d1)

    # === NEW v7: Smoothed derivative features at key wavelengths ===
    smooth_spec = gaussian_filter1d(mean_spec, sigma=2)
    smooth_d1 = np.diff(smooth_spec)
    smooth_d2 = np.diff(smooth_d1)

    # Sample smoothed 1st derivative at 20 positions
    d1_sample_idx = np.linspace(0, len(smooth_d1) - 1, 20, dtype=int)
    for i, idx in enumerate(d1_sample_idx):
        features[f"hs_sd1_{i}"] = smooth_d1[idx]

    # Sample smoothed 2nd derivative at 15 positions
    d2_sample_idx = np.linspace(0, len(smooth_d2) - 1, 15, dtype=int)
    for i, idx in enumerate(d2_sample_idx):
        features[f"hs_sd2_{i}"] = smooth_d2[idx]

    # === NEW v7: Health-specific vegetation indices from HS ===
    # Key band indices
    b500 = wl_to_band(500)
    b531 = wl_to_band(531)
    b550 = wl_to_band(550)
    b570 = wl_to_band(570)
    b670 = wl_to_band(670)
    b680 = wl_to_band(680)
    b700 = wl_to_band(700)
    b710 = wl_to_band(710)
    b720 = wl_to_band(720)
    b740 = wl_to_band(740)
    b750 = wl_to_band(750)
    b800 = wl_to_band(800)
    b850 = wl_to_band(850)

    r500 = mean_spec[b500]
    r531 = mean_spec[b531]
    r550 = mean_spec[b550]
    r570 = mean_spec[b570]
    r670 = mean_spec[b670]
    r680 = mean_spec[b680]
    r700 = mean_spec[b700]
    r710 = mean_spec[b710]
    r720 = mean_spec[b720]
    r740 = mean_spec[b740]
    r750 = mean_spec[b750]
    r800 = mean_spec[b800]
    r850 = mean_spec[b850]
    eps = 1e-8

    # Photochemical Reflectance Index (PRI) - stress indicator
    features["hs_pri"] = (r531 - r570) / (r531 + r570 + eps)
    # PRI per-pixel statistics
    pri_img = (flat[:, b531] - flat[:, b570]) / (flat[:, b531] + flat[:, b570] + eps)
    features["hs_pri_std"] = np.std(pri_img)
    features["hs_pri_q25"] = np.percentile(pri_img, 25)
    features["hs_pri_q75"] = np.percentile(pri_img, 75)

    # Plant Senescence Reflectance Index (PSRI)
    features["hs_psri"] = (r680 - r500) / (r750 + eps)
    psri_img = (flat[:, b680] - flat[:, b500]) / (flat[:, b750] + eps)
    features["hs_psri_std"] = np.std(psri_img)

    # Anthocyanin Reflectance Index (ARI)
    features["hs_ari"] = 1.0 / (r550 + eps) - 1.0 / (r700 + eps)
    ari_img = 1.0 / (flat[:, b550] + eps) - 1.0 / (flat[:, b700] + eps)
    features["hs_ari_std"] = np.std(ari_img)

    # Carotenoid Reflectance Index (CRI)
    b510 = wl_to_band(510)
    r510 = mean_spec[b510]
    features["hs_cri1"] = 1.0 / (r510 + eps) - 1.0 / (r550 + eps)
    features["hs_cri2"] = 1.0 / (r500 + eps) - 1.0 / (r700 + eps)

    # Normalized Difference Water Index (NDWI) - approximation with available bands
    features["hs_ndwi"] = (r850 - r550) / (r850 + r550 + eps)

    # Modified Chlorophyll Absorption Ratio Index (MCARI from HS)
    features["hs_mcari"] = ((r700 - r670) - 0.2 * (r700 - r550)) * (r700 / (r670 + eps))

    # Transformed Chlorophyll Absorption Ratio Index (TCARI from HS)
    features["hs_tcari"] = 3 * ((r700 - r670) - 0.2 * (r700 - r550) * (r700 / (r670 + eps)))

    # TCARI/OSAVI ratio (strong chlorophyll indicator)
    osavi = 1.16 * (r800 - r670) / (r800 + r670 + 0.16 + eps)
    features["hs_tcari_osavi"] = features["hs_tcari"] / (osavi + eps)

    # Simple Ratio indices
    features["hs_sr_800_680"] = r800 / (r680 + eps)
    features["hs_sr_750_700"] = r750 / (r700 + eps)
    features["hs_sr_700_670"] = r700 / (r670 + eps)
    features["hs_sr_850_710"] = r850 / (r710 + eps)

    # === NEW v7: Red Edge Position (REP) - key health indicator ===
    # Find wavelength of max 1st derivative in red edge region (680-750nm)
    re_start, re_end = b680, b750
    if re_end > re_start + 2:
        re_d1 = smooth_d1[re_start:re_end]
        max_pos = np.argmax(re_d1)
        # Interpolate for sub-band precision
        if 0 < max_pos < len(re_d1) - 1:
            y0, y1, y2 = re_d1[max_pos-1], re_d1[max_pos], re_d1[max_pos+1]
            offset = 0.5 * (y2 - y0) / (2 * y1 - y0 - y2 + eps)
            rep_band = re_start + max_pos + offset
        else:
            rep_band = re_start + max_pos
        features["hs_rep"] = HS_WL_START + rep_band * (HS_WL_END - HS_WL_START) / (HS_TARGET_CH - 1)
        features["hs_rep_band"] = rep_band / C
        features["hs_rep_slope"] = np.max(re_d1)
        features["hs_rep_area"] = np.sum(re_d1)

    # Red edge inflection point characteristics
    features["hs_re_width"] = np.sum(smooth_d1[re_start:re_end] > 0.5 * np.max(smooth_d1[re_start:re_end])) / C
    features["hs_re_asymmetry"] = np.mean(smooth_d1[re_start:re_start + (re_end - re_start) // 2]) / (
        np.mean(smooth_d1[re_start + (re_end - re_start) // 2:re_end]) + eps)

    # === NEW v7: Per-pixel spectral derivative statistics ===
    # Compute per-pixel 1st derivative and get statistics
    pixel_d1 = np.diff(flat, axis=1)  # shape: (npix, C-1)
    re_pixel_d1 = pixel_d1[:, re_start:re_end]
    features["hs_pixel_re_d1_mean"] = np.mean(re_pixel_d1)
    features["hs_pixel_re_d1_std"] = np.std(re_pixel_d1)
    features["hs_pixel_re_d1_max_mean"] = np.mean(np.max(re_pixel_d1, axis=1))
    features["hs_pixel_re_d1_max_std"] = np.std(np.max(re_pixel_d1, axis=1))

    # === Shape features (from v6) ===
    features["hs_peak_band"] = np.argmax(mean_spec) / C
    features["hs_trough_band"] = np.argmin(mean_spec) / C
    features["hs_peak_value"] = np.max(mean_spec)
    features["hs_trough_value"] = np.min(mean_spec)
    features["hs_spectral_area"] = np.trapezoid(mean_spec)
    spec_norm = mean_spec / (mean_spec.sum() + 1e-8)
    features["hs_spectral_entropy"] = -np.sum(spec_norm * np.log(spec_norm + 1e-8))

    x = np.arange(C, dtype=np.float32)
    s = mean_spec.sum() + 1e-8
    centroid = np.sum(x * mean_spec) / s
    features["hs_spectral_centroid"] = centroid / C
    spread = np.sqrt(np.sum((x - centroid)**2 * mean_spec) / s)
    features["hs_spectral_spread"] = spread / C

    # Key wavelength features
    gi, ri = int(C*.15), int(C*.40)
    rsi, rpi = int(C*.52), int(C*.62)
    ni, ni2 = int(C*.77), min(int(C*.90), C-1)

    features["hs_green_peak"] = mean_spec[gi]
    features["hs_red_absorption"] = mean_spec[ri]
    features["hs_red_edge_start"] = mean_spec[rsi]
    features["hs_red_edge_peak"] = mean_spec[rpi]
    features["hs_nir_plateau"] = mean_spec[ni]
    features["hs_nir_value"] = mean_spec[ni2]
    features["hs_nir_red_ratio"] = mean_spec[ni] / (mean_spec[ri] + 1e-8)
    features["hs_nir_green_ratio"] = mean_spec[ni] / (mean_spec[gi] + 1e-8)
    features["hs_re_red_ratio"] = mean_spec[rpi] / (mean_spec[ri] + 1e-8)

    if rpi > rsi:
        features["hs_red_edge_slope"] = (mean_spec[rpi] - mean_spec[rsi]) / (rpi - rsi)
    re_d = d1[max(0, rsi-1):min(len(d1), rpi)]
    if len(re_d) > 0:
        features["hs_red_edge_inflection"] = (np.argmax(re_d) + rsi) / C
        features["hs_red_edge_max_slope"] = np.max(re_d)

    if gi < ri < ni:
        baseline = mean_spec[gi] + (mean_spec[ni] - mean_spec[gi]) * (ri - gi) / (ni - gi + 1e-8)
        features["hs_red_absorption_depth"] = baseline - mean_spec[ri]
        features["hs_red_absorption_relative"] = (baseline - mean_spec[ri]) / (baseline + 1e-8)

    # Region stats
    regions = {"blue_vis": (0, int(C*.10)), "green_vis": (int(C*.10), int(C*.25)),
               "red_vis": (int(C*.25), int(C*.45)), "red_edge": (int(C*.45), int(C*.65)), "nir": (int(C*.65), C)}
    for rn, (s, e) in regions.items():
        rd = mean_spec[s:e]
        features[f"hs_region_{rn}_mean"] = np.mean(rd)
        features[f"hs_region_{rn}_std"] = np.std(rd)
        features[f"hs_region_{rn}_slope"] = (rd[-1] - rd[0]) / (len(rd) + 1e-8)
    for r1 in regions:
        for r2 in regions:
            if r1 < r2:
                s1, e1 = regions[r1]
                s2, e2 = regions[r2]
                features[f"hs_cross_{r1}_{r2}_ratio"] = np.mean(mean_spec[s1:e1]) / (np.mean(mean_spec[s2:e2]) + 1e-8)

    # PCA (25 components)
    try:
        n_comp = min(25, C, H * W)
        pca = PCA(n_components=n_comp)
        pca_res = pca.fit_transform(flat)
        for i in range(n_comp):
            features[f"hs_pca{i}_mean"] = np.mean(pca_res[:, i])
            features[f"hs_pca{i}_std"] = np.std(pca_res[:, i])
        for i in range(min(15, n_comp)):
            features[f"hs_pca_ev{i}"] = pca.explained_variance_ratio_[i]
        features["hs_pca_cumev_3"] = np.sum(pca.explained_variance_ratio_[:3])
        features["hs_pca_cumev_5"] = np.sum(pca.explained_variance_ratio_[:5])
        features["hs_pca_cumev_10"] = np.sum(pca.explained_variance_ratio_[:10])
        pc1_img = pca_res[:, 0].reshape(H, W)
        pc1_u8 = np.clip((pc1_img - pc1_img.min()) / (pc1_img.max() - pc1_img.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
        features.update(glcm_features(pc1_u8, "hs_pca1"))
    except:
        pass

    # Spatial features
    for name, idx in [("green", gi), ("red", ri), ("re", rsi), ("nir", ni)]:
        if idx < C:
            features.update(spatial_features(arr[:, :, idx], f"hs_{name}"))

    # Pixel heterogeneity
    pstd = np.std(flat, axis=0)
    features["hs_pixel_hetero_mean"] = np.mean(pstd)
    features["hs_pixel_hetero_std"] = np.std(pstd)
    cv = pstd / (np.mean(flat, axis=0) + 1e-8)
    features["hs_cv_mean"] = np.mean(cv)
    features["hs_cv_std"] = np.std(cv)

    # K-Means clustering
    try:
        flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        for n_clusters in [2, 3]:
            km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42, max_iter=100)
            labels_km = km.fit_predict(flat_norm)
            sizes = np.sort([np.sum(labels_km == c) / len(labels_km) for c in range(n_clusters)])[::-1]
            for ci, sz in enumerate(sizes):
                features[f"hs_km{n_clusters}_size_{ci}"] = sz
            features[f"hs_km{n_clusters}_inertia"] = km.inertia_ / len(flat)
            centers = km.cluster_centers_
            for ci in range(n_clusters):
                for cj in range(ci + 1, n_clusters):
                    features[f"hs_km{n_clusters}_dist_{ci}_{cj}"] = np.linalg.norm(centers[ci] - centers[cj])
    except:
        pass

    # === NEW v7: Quartile spectra features ===
    q25_spec = np.percentile(flat, 25, axis=0)
    q75_spec = np.percentile(flat, 75, axis=0)
    iqr_spec = q75_spec - q25_spec
    features["hs_iqr_mean"] = np.mean(iqr_spec)
    features["hs_iqr_std"] = np.std(iqr_spec)
    features["hs_iqr_max"] = np.max(iqr_spec)
    features["hs_iqr_max_band"] = np.argmax(iqr_spec) / C

    # Ratio of IQR in different regions
    for rn, (s, e) in regions.items():
        features[f"hs_iqr_{rn}"] = np.mean(iqr_spec[s:e])

    return features, mean_spec


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
# Spectral library (cross-validated)
# ============================================================
def build_library(spectra, labels):
    libraries = {}
    for label in LABELS:
        cidx = [i for i, l in enumerate(labels) if l == LBL2ID[label]]
        if not cidx:
            continue
        cs = np.array([spectra[i] for i in cidx])
        mean_s = np.mean(cs, axis=0)
        std_s = np.std(cs, axis=0) + 1e-6
        lib = {"mean": mean_s, "std": std_s, "cov_inv": None}
        if len(cidx) > 20:
            try:
                cov = np.cov(cs.T)
                reg = np.eye(cov.shape[0]) * 0.01 * np.trace(cov) / cov.shape[0]
                lib["cov_inv"] = np.linalg.inv(cov + reg)
            except:
                pass
        libraries[label] = lib
    return libraries

def library_features(spectrum, libraries):
    features = {}
    if spectrum is None:
        return features
    for label, lib in libraries.items():
        diff = spectrum - lib["mean"]
        features[f"sl_{label}_euclid"] = np.sqrt(np.sum(diff**2))
        features[f"sl_{label}_norm_euclid"] = np.sqrt(np.sum((diff / lib["std"])**2))
        features[f"sl_{label}_angle"] = spectral_angle(spectrum, lib["mean"])
        features[f"sl_{label}_corr"] = np.corrcoef(spectrum, lib["mean"])[0, 1]
        if lib["cov_inv"] is not None:
            try:
                features[f"sl_{label}_mahal"] = np.sqrt(np.clip(diff @ lib["cov_inv"] @ diff, 0, None))
            except:
                pass
    for l1 in LABELS:
        for l2 in LABELS:
            if l1 < l2:
                for metric in ["euclid", "angle"]:
                    k1, k2 = f"sl_{l1}_{metric}", f"sl_{l2}_{metric}"
                    if k1 in features and k2 in features:
                        features[f"sl_{l1}_{l2}_{metric}_ratio"] = features[k1] / (features[k2] + 1e-8)
    return features


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Beyond Visible Spectrum v7 - Health focus + Model diversity")
    print("=" * 60)

    train_idx = build_index(ROOT, "train")
    val_idx = build_index(ROOT, "val")
    train_df = make_df(train_idx, has_labels=True)
    val_df = make_df(val_idx, has_labels=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # === Phase 1: CNN features (EfficientNet only - proven better) ===
    print("\nPhase 1: Loading CNN features (EfficientNet only)...")
    eff_data = np.load(os.path.join(OUT_DIR, "cnn_features.npz"), allow_pickle=True)
    eff_names_t = {n: i for i, n in enumerate(eff_data["train_names"])}
    eff_names_v = {n: i for i, n in enumerate(eff_data["val_names"])}

    n_train, n_val = len(train_df), len(val_df)
    train_cnn = np.zeros((n_train, 100), dtype=np.float32)
    val_cnn = np.zeros((n_val, 100), dtype=np.float32)
    for i, bid in enumerate(train_df["base_id"]):
        if bid in eff_names_t:
            train_cnn[i] = eff_data["train_pca"][eff_names_t[bid]]
    for i, bid in enumerate(val_df["base_id"]):
        if bid in eff_names_v:
            val_cnn[i] = eff_data["val_pca"][eff_names_v[bid]]
    print(f"  EfficientNet PCA: {train_cnn.shape[1]} features")

    # === Phase 2: Handcrafted features ===
    print("\nPhase 2: Extracting features (v6 base + v7 health-specific)...")
    train_feats, train_spectra = [], []
    for i, (_, row) in enumerate(train_df.iterrows()):
        f = {}
        spec = None
        if pd.notna(row.get("rgb")):
            f.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")):
            f.update(extract_ms_features(row["ms"]))
        if pd.notna(row.get("hs")):
            hs_arr, spec = get_hs_data(row["hs"])
            hs_f, spec = extract_hs_features(hs_arr)
            f.update(hs_f)
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            f.update(extract_cross_modal_features(row["rgb"], row["ms"]))
        train_feats.append(f)
        train_spectra.append(spec)
        if (i + 1) % 100 == 0:
            print(f"  Train: {i + 1}/{len(train_df)}")
    print(f"  Train: {len(train_df)}/{len(train_df)}")

    val_feats, val_spectra = [], []
    for i, (_, row) in enumerate(val_df.iterrows()):
        f = {}
        spec = None
        if pd.notna(row.get("rgb")):
            f.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")):
            f.update(extract_ms_features(row["ms"]))
        if pd.notna(row.get("hs")):
            hs_arr, spec = get_hs_data(row["hs"])
            hs_f, spec = extract_hs_features(hs_arr)
            f.update(hs_f)
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            f.update(extract_cross_modal_features(row["rgb"], row["ms"]))
        val_feats.append(f)
        val_spectra.append(spec)
        if (i + 1) % 100 == 0:
            print(f"  Val: {i + 1}/{len(val_df)}")
    print(f"  Val: {len(val_df)}/{len(val_df)}")

    train_feat_df = pd.DataFrame(train_feats)
    val_feat_df = pd.DataFrame(val_feats)
    base_cols = sorted(train_feat_df.columns.tolist())
    X_hand_train = train_feat_df[base_cols].fillna(0).values.astype(np.float32)
    X_hand_test = val_feat_df[base_cols].fillna(0).values.astype(np.float32)
    X_hand_train = np.nan_to_num(X_hand_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_hand_test = np.nan_to_num(X_hand_test, nan=0.0, posinf=0.0, neginf=0.0)

    X_base_train = np.hstack([X_hand_train, train_cnn])
    X_base_test = np.hstack([X_hand_test, val_cnn])
    y_train = np.array([LBL2ID[l] for l in train_df["label"]])
    print(f"  Handcrafted: {X_hand_train.shape[1]} | CNN: {train_cnn.shape[1]} | Total: {X_base_train.shape[1]}")

    # === Phase 3: CV spectral library ===
    print("\nPhase 3: CV spectral library...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_slib = [None] * len(train_df)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_base_train, y_train)):
        lib = build_library([train_spectra[i] for i in tr_idx], [y_train[i] for i in tr_idx])
        for i in va_idx:
            oof_slib[i] = library_features(train_spectra[i], lib)
        print(f"  Fold {fold + 1}: lib from {len(tr_idx)}, applied to {len(va_idx)}")

    all_lib = build_library(train_spectra, y_train.tolist())
    test_slib = [library_features(s, all_lib) for s in val_spectra]
    train_slib_all = [library_features(s, all_lib) for s in train_spectra]

    slib_df_oof = pd.DataFrame(oof_slib)
    slib_df_test = pd.DataFrame(test_slib)
    slib_df_all = pd.DataFrame(train_slib_all)
    X_slib_oof = slib_df_oof.fillna(0).values.astype(np.float32)
    X_slib_test = slib_df_test.fillna(0).values.astype(np.float32)
    X_slib_all = slib_df_all.fillna(0).values.astype(np.float32)
    for arr in [X_slib_oof, X_slib_test, X_slib_all]:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_full_oof = np.hstack([X_base_train, X_slib_oof])
    X_full_test = np.hstack([X_base_test, X_slib_test])
    X_full_all = np.hstack([X_base_train, X_slib_all])
    print(f"  Slib features: {X_slib_oof.shape[1]} | Total: {X_full_oof.shape[1]}")

    # === Phase 4: Feature selection ===
    print("\nPhase 4: Feature selection via mutual information...")
    mi = mutual_info_classif(X_full_all, y_train, random_state=SEED, n_neighbors=5)
    mi_threshold = np.sort(mi)[::-1][min(500, len(mi) - 1)]
    selected_mask = mi >= mi_threshold
    n_selected = selected_mask.sum()
    print(f"  MI threshold: {mi_threshold:.4f} | Selected: {n_selected}/{len(mi)} features")

    X_sel_oof = X_full_oof[:, selected_mask]
    X_sel_test = X_full_test[:, selected_mask]
    X_sel_all = X_full_all[:, selected_mask]

    # Also prepare handcrafted-only + slib for some models
    X_hand_oof = np.hstack([X_hand_train, X_slib_oof])
    X_hand_all_slib = np.hstack([X_hand_train, X_slib_all])
    X_hand_test_slib = np.hstack([X_hand_test, X_slib_test])

    # === Phase 5: Train Level-1 models ===
    print("\n" + "=" * 60)
    print("Phase 5: Level-1 models (trees + SVM + kNN)")
    print("=" * 60)

    results = {}
    health_w = {0: 1.3, 1: 1.0, 2: 1.0}

    def train_lgb(name, params, X_tr, X_te, use_weight=False):
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_te), 3))
        sw = np.ones(len(y_train))
        if use_weight:
            for i, y in enumerate(y_train):
                sw[i] = health_w[y]
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr[tri], y_train[tri], sample_weight=sw[tri],
                  eval_set=[(X_tr[vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_tr[vai])
            X_all_fold = X_full_all if X_tr.shape[1] == X_full_all.shape[1] else (
                X_sel_all if X_tr.shape[1] == X_sel_all.shape[1] else X_hand_all_slib)
            m2 = lgb.LGBMClassifier(**params)
            m2.fit(X_all_fold[tri], y_train[tri], sample_weight=sw[tri],
                   eval_set=[(X_all_fold[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            X_te_fold = X_te if X_te.shape[1] == X_all_fold.shape[1] else (
                X_sel_test if X_te.shape[1] == X_sel_test.shape[1] else X_hand_test_slib)
            test_p += m2.predict_proba(X_te_fold) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    def train_xgb(name, params, X_tr, X_te):
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_te), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr[tri], y_train[tri], eval_set=[(X_tr[vai], y_train[vai])], verbose=False)
            oof_p[vai] = m.predict_proba(X_tr[vai])
            X_all_fold = X_full_all if X_tr.shape[1] == X_full_all.shape[1] else (
                X_sel_all if X_tr.shape[1] == X_sel_all.shape[1] else X_hand_all_slib)
            m2 = xgb.XGBClassifier(**params)
            m2.fit(X_all_fold[tri], y_train[tri], eval_set=[(X_all_fold[vai], y_train[vai])], verbose=False)
            X_te_fold = X_te if X_te.shape[1] == X_all_fold.shape[1] else (
                X_sel_test if X_te.shape[1] == X_sel_test.shape[1] else X_hand_test_slib)
            test_p += m2.predict_proba(X_te_fold) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    lgb_base = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                "verbose": -1, "random_state": SEED}

    # Full features
    train_lgb("lgb_a", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
              "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
              "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}, X_full_oof, X_full_test)

    train_lgb("lgb_b", {**lgb_base, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4,
              "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
              "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000}, X_full_oof, X_full_test)

    train_lgb("lgb_c", {**lgb_base, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
              "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500}, X_full_oof, X_full_test)

    train_lgb("lgb_d", {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
              "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
              "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}, X_full_oof, X_full_test)

    # Cost-sensitive
    train_lgb("lgb_w", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
              "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
              "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}, X_full_oof, X_full_test, use_weight=True)

    # Feature-selected
    train_lgb("lgb_sel", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
              "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.7,
              "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}, X_sel_oof, X_sel_test)

    # Handcrafted only
    train_lgb("lgb_hand", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
              "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
              "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}, X_hand_oof, X_hand_test_slib)

    # XGBoost
    xgb_base = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                "tree_method": "hist", "random_state": SEED, "verbosity": 0}

    train_xgb("xgb_a", {**xgb_base, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
              "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
              "reg_lambda": 1.0, "n_estimators": 2000}, X_full_oof, X_full_test)

    train_xgb("xgb_b", {**xgb_base, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
              "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
              "reg_lambda": 2.0, "n_estimators": 3000}, X_full_oof, X_full_test)

    train_xgb("xgb_sel", {**xgb_base, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
              "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 0.5,
              "reg_lambda": 1.0, "n_estimators": 2000}, X_sel_oof, X_sel_test)

    # sklearn ensemble
    for name, model_cls, kw in [
        ("gb", GradientBoostingClassifier, {"n_estimators": 1000, "max_depth": 4,
                                            "learning_rate": 0.03, "subsample": 0.75,
                                            "min_samples_leaf": 8, "random_state": SEED}),
        ("et", ExtraTreesClassifier, {"n_estimators": 2000, "min_samples_leaf": 2,
                                      "random_state": SEED, "n_jobs": -1}),
        ("rf", RandomForestClassifier, {"n_estimators": 2000, "min_samples_leaf": 2,
                                        "random_state": SEED, "n_jobs": -1}),
    ]:
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_full_test), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = model_cls(**kw)
            m.fit(X_full_oof[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(X_full_oof[vai])
            m2 = model_cls(**kw)
            m2.fit(X_full_all[tri], y_train[tri])
            test_p += m2.predict_proba(X_full_test) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === NEW v7: SVM models ===
    print("\n  SVM models...")
    scaler_svm = StandardScaler()
    X_sel_oof_sc = scaler_svm.fit_transform(X_sel_oof)
    X_sel_test_sc = scaler_svm.transform(X_sel_test)

    for C_val, gamma, name in [(1.0, "scale", "svm_1"), (10.0, "scale", "svm_10"),
                                (5.0, "auto", "svm_5a"), (1.0, 0.001, "svm_1g")]:
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_sel_test_sc), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = SVC(C=C_val, kernel="rbf", gamma=gamma, probability=True, random_state=SEED)
            m.fit(X_sel_oof_sc[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(X_sel_oof_sc[vai])
            # For test: train on all (using oof features which include CV slib)
            scaler_fold = StandardScaler()
            X_sel_all_sc = scaler_fold.fit_transform(X_sel_all)
            X_sel_test_sc_fold = scaler_fold.transform(X_sel_test)
            m2 = SVC(C=C_val, kernel="rbf", gamma=gamma, probability=True, random_state=SEED)
            m2.fit(X_sel_all_sc[tri], y_train[tri])
            test_p += m2.predict_proba(X_sel_test_sc_fold) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === NEW v7: kNN models ===
    print("\n  kNN models...")
    for k, w in [(5, "distance"), (10, "distance"), (15, "uniform"), (20, "distance")]:
        name = f"knn_{k}_{w[0]}"
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_sel_test_sc), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = KNeighborsClassifier(n_neighbors=k, weights=w, metric="minkowski", p=2, n_jobs=-1)
            m.fit(X_sel_oof_sc[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(X_sel_oof_sc[vai])
            scaler_fold = StandardScaler()
            X_sel_all_sc = scaler_fold.fit_transform(X_sel_all)
            X_sel_test_sc_fold = scaler_fold.transform(X_sel_test)
            m2 = KNeighborsClassifier(n_neighbors=k, weights=w, metric="minkowski", p=2, n_jobs=-1)
            m2.fit(X_sel_all_sc[tri], y_train[tri])
            test_p += m2.predict_proba(X_sel_test_sc_fold) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Seed diversity
    print("\n  Seed diversity models...")
    for seed_off in [100, 200, 300]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_off)
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_full_test), 3))
        for fold, (tri, vai) in enumerate(skf_s.split(X_base_train, y_train)):
            lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
            slib_vai = np.array([list(library_features(train_spectra[i], lib).values()) for i in vai]).astype(np.float32)
            slib_tri = np.array([list(library_features(train_spectra[i], lib).values()) for i in tri]).astype(np.float32)
            X_tri = np.hstack([X_base_train[tri], slib_tri])
            X_vai = np.hstack([X_base_train[vai], slib_vai])
            seed_params = {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                           "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                           "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000,
                           "random_state": SEED + seed_off}
            m = lgb.LGBMClassifier(**seed_params)
            m.fit(X_tri, y_train[tri], eval_set=[(X_vai, y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_vai)
            m2 = lgb.LGBMClassifier(**seed_params)
            m2.fit(X_full_all[tri], y_train[tri], eval_set=[(X_full_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_full_test) / N_FOLDS
        name = f"lgb_s{seed_off}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 6: Stacking ===
    print("\n" + "=" * 60)
    print("Phase 6: Stacking")
    print("=" * 60)

    model_names = sorted(results.keys())
    stack_oof = np.hstack([results[m]["oof"] for m in model_names])
    stack_test = np.hstack([results[m]["test"] for m in model_names])

    pca_stack = PCA(n_components=50, random_state=SEED)
    X_oof_pca = pca_stack.fit_transform(X_full_oof)
    X_test_pca = pca_stack.transform(X_full_test)
    stack_oof_aug = np.hstack([stack_oof, X_oof_pca])
    stack_test_aug = np.hstack([stack_test, X_test_pca])

    scaler_s = StandardScaler()
    stack_oof_sc = scaler_s.fit_transform(stack_oof_aug)
    stack_test_sc = scaler_s.transform(stack_test_aug)

    for C_val, name in [(0.5, "stack_lr_05"), (1.0, "stack_lr_10"), (5.0, "stack_lr_50")]:
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_full_test), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = LogisticRegression(C=C_val, max_iter=2000, multi_class="multinomial", random_state=SEED)
            m.fit(stack_oof_sc[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(stack_oof_sc[vai])
            m2 = LogisticRegression(C=C_val, max_iter=2000, multi_class="multinomial", random_state=SEED)
            m2.fit(stack_oof_sc, y_train)
            test_p += m2.predict_proba(stack_test_sc) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    for lr, nl, name in [(0.05, 15, "stack_lgb_a"), (0.03, 20, "stack_lgb_b")]:
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_full_test), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(objective="multiclass", num_class=3, learning_rate=lr,
                                    num_leaves=nl, max_depth=4, min_child_samples=15,
                                    subsample=0.8, colsample_bytree=0.5, reg_alpha=1.0,
                                    reg_lambda=1.0, n_estimators=1000, verbose=-1, random_state=SEED)
            m.fit(stack_oof_aug[tri], y_train[tri], eval_set=[(stack_oof_aug[vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_p[vai] = m.predict_proba(stack_oof_aug[vai])
            m2 = lgb.LGBMClassifier(objective="multiclass", num_class=3, learning_rate=lr,
                                     num_leaves=nl, max_depth=4, min_child_samples=15,
                                     subsample=0.8, colsample_bytree=0.5, reg_alpha=1.0,
                                     reg_lambda=1.0, n_estimators=1000, verbose=-1, random_state=SEED)
            m2.fit(stack_oof_aug, y_train, eval_set=[(stack_oof_aug[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
            test_p += m2.predict_proba(stack_test_aug) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 7: Optuna ensemble optimization ===
    print("\n" + "=" * 60)
    print("Phase 7: Optuna ensemble optimization")
    print("=" * 60)

    all_models = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    print("\nAll models ranked:")
    for m in all_models:
        print(f"  {m:20s} Acc={results[m]['acc']:.4f} F1={results[m]['f1']:.4f}")

    top_n = min(15, len(all_models))
    top_models = all_models[:top_n]
    top_oofs = [results[m]["oof"] for m in top_models]
    top_tests = [results[m]["test"] for m in top_models]

    def objective(trial):
        weights = []
        for i, m in enumerate(top_models):
            w = trial.suggest_float(f"w_{m}", 0.0, 1.0)
            weights.append(w)
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8:
            return 0.0
        weights /= ws
        oof_e = sum(w * o for w, o in zip(weights, top_oofs))
        return accuracy_score(y_train, oof_e.argmax(1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=3000, show_progress_bar=False)

    best_weights = np.array([study.best_params[f"w_{m}"] for m in top_models])
    best_weights /= best_weights.sum()
    optuna_oof = sum(w * o for w, o in zip(best_weights, top_oofs))
    optuna_test = sum(w * o for w, o in zip(best_weights, top_tests))
    acc = accuracy_score(y_train, optuna_oof.argmax(1))
    f1 = f1_score(y_train, optuna_oof.argmax(1), average="macro")
    print(f"\n  Optuna ensemble: Acc={acc:.4f} F1={f1:.4f}")
    print(f"  Weights: {dict(zip(top_models, [f'{w:.3f}' for w in best_weights]))}")
    results["optuna_ens"] = {"oof": optuna_oof, "test": optuna_test, "acc": acc, "f1": f1}

    # Simple weighted averages
    for k in [3, 5, 7]:
        top = all_models[:k]
        w = np.array([results[m]["acc"] for m in top]) ** 3
        w /= w.sum()
        oof_e = sum(wi * results[m]["oof"] for m, wi in zip(top, w))
        test_e = sum(wi * results[m]["test"] for m, wi in zip(top, w))
        acc = accuracy_score(y_train, oof_e.argmax(1))
        f1 = f1_score(y_train, oof_e.argmax(1), average="macro")
        print(f"  top{k}_weighted: Acc={acc:.4f} F1={f1:.4f}")
        results[f"top{k}_w"] = {"oof": oof_e, "test": test_e, "acc": acc, "f1": f1}

    # === Phase 8: Threshold optimization ===
    print("\n" + "=" * 60)
    print("Phase 8: Threshold optimization")
    print("=" * 60)

    # Find best overall model first
    all_final = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    best = all_final[0]
    best_oof = results[best]["oof"]
    best_test = results[best]["test"]

    # Try class-specific threshold adjustments
    print(f"\n  Starting from best model: {best} (Acc={results[best]['acc']:.4f})")
    best_thresh_acc = results[best]["acc"]
    best_bias = np.zeros(3)

    for h_bias in np.arange(-0.15, 0.20, 0.01):
        for r_bias in np.arange(-0.10, 0.10, 0.01):
            o_bias = -(h_bias + r_bias)
            bias = np.array([h_bias, r_bias, o_bias])
            adjusted = best_oof + bias
            preds = adjusted.argmax(1)
            acc = accuracy_score(y_train, preds)
            if acc > best_thresh_acc:
                best_thresh_acc = acc
                best_bias = bias.copy()

    if np.any(best_bias != 0):
        adjusted_oof = best_oof + best_bias
        adjusted_test = best_test + best_bias
        adj_f1 = f1_score(y_train, adjusted_oof.argmax(1), average="macro")
        print(f"  Threshold optimized: Acc={best_thresh_acc:.4f} F1={adj_f1:.4f}")
        print(f"  Bias: Health={best_bias[0]:.3f} Rust={best_bias[1]:.3f} Other={best_bias[2]:.3f}")
        print(classification_report(y_train, adjusted_oof.argmax(1), target_names=LABELS, digits=4))
        results["thresh_opt"] = {"oof": adjusted_oof, "test": adjusted_test,
                                 "acc": best_thresh_acc, "f1": adj_f1}
    else:
        print("  No threshold improvement found.")

    # === Final results ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    all_final = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    best = all_final[0]
    best_acc = results[best]["acc"]
    best_f1 = results[best]["f1"]
    best_oof_final = results[best]["oof"]
    best_test_final = results[best]["test"]

    print(f"\n*** Best: {best} | Acc={best_acc:.4f} F1={best_f1:.4f} ***")
    print(classification_report(y_train, best_oof_final.argmax(1), target_names=LABELS, digits=4))

    # Save
    sub_ids = []
    for _, r in val_df.iterrows():
        if pd.notna(r.get("hs")):
            sub_ids.append(os.path.basename(r["hs"]))
        elif pd.notna(r.get("ms")):
            sub_ids.append(os.path.basename(r["ms"]))
        else:
            sub_ids.append(os.path.basename(r["rgb"]))

    preds = [ID2LBL[p] for p in best_test_final.argmax(1)]
    sub = pd.DataFrame({"Id": sub_ids, "Category": preds})
    a_s = f"{best_acc:.4f}".replace(".", "p")
    f_s = f"{best_f1:.4f}".replace(".", "p")
    sub_path = os.path.join(OUT_DIR, f"submission_v7_acc_{a_s}_f1_{f_s}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    # Save probabilities
    np.save(os.path.join(OUT_DIR, "v7_best_oof_probs.npy"), best_oof_final)
    np.save(os.path.join(OUT_DIR, "v7_best_test_probs.npy"), best_test_final)

    # Also save optuna ensemble if different
    if best != "optuna_ens":
        opt_preds = [ID2LBL[p] for p in results["optuna_ens"]["test"].argmax(1)]
        opt_sub = pd.DataFrame({"Id": sub_ids, "Category": opt_preds})
        oa = f"{results['optuna_ens']['acc']:.4f}".replace(".", "p")
        of = f"{results['optuna_ens']['f1']:.4f}".replace(".", "p")
        opt_sub.to_csv(os.path.join(OUT_DIR, f"submission_v7_optuna_acc_{oa}_f1_{of}.csv"), index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
