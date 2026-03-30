"""
Beyond Visible Spectrum v8 - Pseudo-labeling + Threshold optimization

Strategy: Don't add more features. Instead:
1. Use v6's exact proven features (807 handcrafted + 100 EfficientNet PCA = 907)
2. Pseudo-labeling: confident test predictions augment training
3. Feature importance pruning: remove noisy features
4. Threshold optimization: adjust per-class boundaries
5. Fast models only (LGB, XGB, ET - skip slow sklearn GB)
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops

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

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ============================================================
# Data loading (identical to v6)
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
# Feature extraction helpers (identical to v6)
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
# Feature extraction (IDENTICAL to v6 - proven 807 handcrafted features)
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
    features = {}
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)
    mean_spec = np.mean(flat, axis=0)
    std_spec = np.std(flat, axis=0)
    median_spec = np.median(flat, axis=0)
    for i in range(C):
        features[f"hs_raw_{i}"] = mean_spec[i]
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
    n_sample = 20
    sample_idx = np.linspace(0, C - 1, n_sample, dtype=int)
    for i, idx in enumerate(sample_idx):
        band = flat[:, idx]
        features[f"hs_b{i}_mean"] = np.mean(band)
        features[f"hs_b{i}_std"] = np.std(band)
        features[f"hs_b{i}_min"] = np.min(band)
        features[f"hs_b{i}_max"] = np.max(band)
        features[f"hs_b{i}_median"] = np.median(band)
    features["hs_total_mean"] = np.mean(flat)
    features["hs_total_std"] = np.std(flat)
    features["hs_total_range"] = float(mean_spec.max() - mean_spec.min())
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
    for name, idx in [("green", gi), ("red", ri), ("re", rsi), ("nir", ni)]:
        if idx < C:
            features.update(spatial_features(arr[:, :, idx], f"hs_{name}"))
    pstd = np.std(flat, axis=0)
    features["hs_pixel_hetero_mean"] = np.mean(pstd)
    features["hs_pixel_hetero_std"] = np.std(pstd)
    cv = pstd / (np.mean(flat, axis=0) + 1e-8)
    features["hs_cv_mean"] = np.mean(cv)
    features["hs_cv_std"] = np.std(cv)
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
# Spectral library
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
# Model training helper
# ============================================================
def train_and_predict(model_configs, X_oof, X_all, X_test, y_train,
                      skf, X_base_train, train_spectra, all_lib):
    """Train all models, return results dict."""
    results = {}

    for cfg in model_configs:
        name = cfg["name"]
        model_type = cfg["type"]
        params = cfg["params"]
        X_tr = cfg.get("X_tr", X_oof)
        X_te = cfg.get("X_te", X_test)
        X_al = cfg.get("X_al", X_all)
        use_weight = cfg.get("use_weight", False)
        health_w = {0: 1.3, 1: 1.0, 2: 1.0}

        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_te), 3))
        sw = np.ones(len(y_train))
        if use_weight:
            for i, y in enumerate(y_train):
                sw[i] = health_w[y]

        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            if model_type == "lgb":
                m = lgb.LGBMClassifier(**params)
                m.fit(X_tr[tri], y_train[tri], sample_weight=sw[tri],
                      eval_set=[(X_tr[vai], y_train[vai])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                oof_p[vai] = m.predict_proba(X_tr[vai])
                m2 = lgb.LGBMClassifier(**params)
                m2.fit(X_al[tri], y_train[tri], sample_weight=sw[tri],
                       eval_set=[(X_al[vai], y_train[vai])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
                test_p += m2.predict_proba(X_te) / N_FOLDS
            elif model_type == "xgb":
                m = xgb.XGBClassifier(**params)
                m.fit(X_tr[tri], y_train[tri], eval_set=[(X_tr[vai], y_train[vai])], verbose=False)
                oof_p[vai] = m.predict_proba(X_tr[vai])
                m2 = xgb.XGBClassifier(**params)
                m2.fit(X_al[tri], y_train[tri], eval_set=[(X_al[vai], y_train[vai])], verbose=False)
                test_p += m2.predict_proba(X_te) / N_FOLDS
            elif model_type == "sklearn":
                cls = cfg["cls"]
                m = cls(**params)
                m.fit(X_tr[tri], y_train[tri])
                oof_p[vai] = m.predict_proba(X_tr[vai])
                m2 = cls(**params)
                m2.fit(X_al[tri], y_train[tri])
                test_p += m2.predict_proba(X_te) / N_FOLDS

        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Beyond Visible Spectrum v8 - Pseudo-labeling")
    print("=" * 60)

    train_idx = build_index(ROOT, "train")
    val_idx = build_index(ROOT, "val")
    train_df = make_df(train_idx, has_labels=True)
    val_df = make_df(val_idx, has_labels=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # === Phase 1: CNN features ===
    print("\nPhase 1: Loading EfficientNet CNN features...")
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
    print(f"  CNN PCA: {train_cnn.shape[1]} features")

    # === Phase 2: Handcrafted features ===
    print("\nPhase 2: Extracting handcrafted features...")
    train_feats, train_spectra = [], []
    val_feats, val_spectra = [], []

    for i, (_, row) in enumerate(train_df.iterrows()):
        f = {}; spec = None
        if pd.notna(row.get("rgb")): f.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")): f.update(extract_ms_features(row["ms"]))
        if pd.notna(row.get("hs")):
            hs_arr, spec = get_hs_data(row["hs"])
            hs_f, spec = extract_hs_features(hs_arr)
            f.update(hs_f)
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            f.update(extract_cross_modal_features(row["rgb"], row["ms"]))
        train_feats.append(f); train_spectra.append(spec)
        if (i + 1) % 200 == 0: print(f"  Train: {i + 1}/{n_train}")
    print(f"  Train: {n_train}/{n_train}")

    for i, (_, row) in enumerate(val_df.iterrows()):
        f = {}; spec = None
        if pd.notna(row.get("rgb")): f.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")): f.update(extract_ms_features(row["ms"]))
        if pd.notna(row.get("hs")):
            hs_arr, spec = get_hs_data(row["hs"])
            hs_f, spec = extract_hs_features(hs_arr)
            f.update(hs_f)
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            f.update(extract_cross_modal_features(row["rgb"], row["ms"]))
        val_feats.append(f); val_spectra.append(spec)
        if (i + 1) % 200 == 0: print(f"  Val: {i + 1}/{n_val}")
    print(f"  Val: {n_val}/{n_val}")

    train_feat_df = pd.DataFrame(train_feats)
    val_feat_df = pd.DataFrame(val_feats)
    base_cols = sorted(train_feat_df.columns.tolist())
    X_hand_train = train_feat_df[base_cols].fillna(0).values.astype(np.float32)
    X_hand_test = val_feat_df[base_cols].fillna(0).values.astype(np.float32)
    X_hand_train = np.nan_to_num(X_hand_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_hand_test = np.nan_to_num(X_hand_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Save features for future use
    np.savez_compressed(os.path.join(OUT_DIR, "v8_features.npz"),
                        X_hand_train=X_hand_train, X_hand_test=X_hand_test,
                        train_cnn=train_cnn, val_cnn=val_cnn,
                        base_cols=base_cols)

    X_base_train = np.hstack([X_hand_train, train_cnn])
    X_base_test = np.hstack([X_hand_test, val_cnn])
    y_train = np.array([LBL2ID[l] for l in train_df["label"]])
    print(f"  Handcrafted: {X_hand_train.shape[1]} | CNN: 100 | Total: {X_base_train.shape[1]}")

    # === Phase 3: CV spectral library ===
    print("\nPhase 3: CV spectral library...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_slib = [None] * n_train
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_base_train, y_train)):
        lib = build_library([train_spectra[i] for i in tr_idx], [y_train[i] for i in tr_idx])
        for i in va_idx:
            oof_slib[i] = library_features(train_spectra[i], lib)

    all_lib = build_library(train_spectra, y_train.tolist())
    test_slib = [library_features(s, all_lib) for s in val_spectra]
    train_slib_all = [library_features(s, all_lib) for s in train_spectra]

    X_slib_oof = pd.DataFrame(oof_slib).fillna(0).values.astype(np.float32)
    X_slib_test = pd.DataFrame(test_slib).fillna(0).values.astype(np.float32)
    X_slib_all = pd.DataFrame(train_slib_all).fillna(0).values.astype(np.float32)
    for arr in [X_slib_oof, X_slib_test, X_slib_all]:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_oof = np.hstack([X_base_train, X_slib_oof])
    X_test = np.hstack([X_base_test, X_slib_test])
    X_all = np.hstack([X_base_train, X_slib_all])
    print(f"  Total features: {X_oof.shape[1]}")

    # === Phase 4: Round 0 - Initial model training ===
    print("\n" + "=" * 60)
    print("Phase 4: Round 0 - Initial models")
    print("=" * 60)

    lgb_base = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                "verbose": -1, "random_state": SEED}

    configs = [
        {"name": "lgb_a", "type": "lgb", "params": {**lgb_base, "learning_rate": 0.03, "num_leaves": 25,
         "max_depth": 5, "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
         "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}},
        {"name": "lgb_b", "type": "lgb", "params": {**lgb_base, "learning_rate": 0.02, "num_leaves": 15,
         "max_depth": 4, "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
         "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000}},
        {"name": "lgb_c", "type": "lgb", "params": {**lgb_base, "learning_rate": 0.05, "num_leaves": 31,
         "max_depth": 6, "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
         "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500}},
        {"name": "lgb_d", "type": "lgb", "params": {**lgb_base, "learning_rate": 0.01, "num_leaves": 20,
         "max_depth": 4, "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
         "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}},
        {"name": "lgb_w", "type": "lgb", "params": {**lgb_base, "learning_rate": 0.03, "num_leaves": 25,
         "max_depth": 5, "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
         "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}, "use_weight": True},
        {"name": "xgb_a", "type": "xgb", "params": {"objective": "multi:softprob", "num_class": 3,
         "eval_metric": "mlogloss", "tree_method": "hist", "random_state": SEED, "verbosity": 0,
         "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
         "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
         "reg_lambda": 1.0, "n_estimators": 2000}},
        {"name": "xgb_b", "type": "xgb", "params": {"objective": "multi:softprob", "num_class": 3,
         "eval_metric": "mlogloss", "tree_method": "hist", "random_state": SEED, "verbosity": 0,
         "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
         "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
         "reg_lambda": 2.0, "n_estimators": 3000}},
        {"name": "et", "type": "sklearn", "cls": ExtraTreesClassifier,
         "params": {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}},
        {"name": "rf", "type": "sklearn", "cls": RandomForestClassifier,
         "params": {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}},
    ]

    results = train_and_predict(configs, X_oof, X_all, X_test, y_train, skf, X_base_train,
                                train_spectra, all_lib)

    # Seed diversity
    print("\n  Seed diversity...")
    for seed_off in [100, 200, 300]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_off)
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
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
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test) / N_FOLDS
        name = f"lgb_s{seed_off}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Initial Optuna ensemble
    print("\n  Initial Optuna ensemble...")
    all_models_r0 = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    top_oofs_r0 = [results[m]["oof"] for m in all_models_r0]
    top_tests_r0 = [results[m]["test"] for m in all_models_r0]

    def objective_r0(trial):
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in all_models_r0]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
        weights /= ws
        oof_e = sum(w * o for w, o in zip(weights, top_oofs_r0))
        return accuracy_score(y_train, oof_e.argmax(1))

    study_r0 = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    study_r0.optimize(objective_r0, n_trials=2000, show_progress_bar=False)
    bw_r0 = np.array([study_r0.best_params[f"w_{m}"] for m in all_models_r0])
    bw_r0 /= bw_r0.sum()
    r0_oof = sum(w * o for w, o in zip(bw_r0, top_oofs_r0))
    r0_test = sum(w * o for w, o in zip(bw_r0, top_tests_r0))
    r0_acc = accuracy_score(y_train, r0_oof.argmax(1))
    r0_f1 = f1_score(y_train, r0_oof.argmax(1), average="macro")
    print(f"  Round 0 ensemble: Acc={r0_acc:.4f} F1={r0_f1:.4f}")

    # === Phase 5: Pseudo-labeling ===
    print("\n" + "=" * 60)
    print("Phase 5: Pseudo-labeling")
    print("=" * 60)

    # Use round 0 test predictions for pseudo-labels
    test_pred_probs = r0_test
    test_confidence = np.max(test_pred_probs, axis=1)
    test_pred_labels = test_pred_probs.argmax(1)

    for threshold in [0.90, 0.85, 0.80, 0.75]:
        confident_mask = test_confidence >= threshold
        n_pseudo = confident_mask.sum()
        pseudo_labels = test_pred_labels[confident_mask]
        label_dist = {ID2LBL[i]: np.sum(pseudo_labels == i) for i in range(3)}
        print(f"  Threshold {threshold}: {n_pseudo} pseudo-labels {label_dist}")

    # Try different thresholds for pseudo-labeling
    best_pl_acc = r0_acc
    best_pl_result = None
    best_pl_threshold = None

    for threshold in [0.90, 0.85, 0.80]:
        confident_mask = test_confidence >= threshold
        n_pseudo = confident_mask.sum()
        if n_pseudo < 10:
            continue

        pseudo_labels = test_pred_labels[confident_mask]
        X_pseudo = X_base_test[confident_mask]
        X_pseudo_slib = X_slib_test[confident_mask]  # Use full slib for pseudo samples
        X_pseudo_full = np.hstack([X_pseudo, X_pseudo_slib])

        # Augmented training set
        X_aug = np.vstack([X_oof, X_pseudo_full])
        y_aug = np.concatenate([y_train, pseudo_labels])

        # Reduced weight for pseudo-labels to reduce noise propagation
        sw_aug = np.ones(len(y_aug))
        sw_aug[n_train:] = 0.5  # Half weight for pseudo-labels

        # Train key models on augmented data, validate on original OOF
        print(f"\n  Pseudo-labeling with threshold={threshold} ({n_pseudo} samples)...")
        pl_results = {}

        for cfg_name, cfg_params in [
            ("pl_lgb_a", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                          "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                          "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
            ("pl_lgb_d", {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                          "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                          "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
        ]:
            oof_p = np.zeros((n_train, 3))
            test_p = np.zeros((n_val, 3))

            for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
                # Training fold: original train fold + all pseudo-labels
                X_fold_train = np.vstack([X_oof[tri], X_pseudo_full])
                y_fold_train = np.concatenate([y_train[tri], pseudo_labels])
                sw_fold = np.ones(len(y_fold_train))
                sw_fold[len(tri):] = 0.5

                m = lgb.LGBMClassifier(**cfg_params)
                m.fit(X_fold_train, y_fold_train, sample_weight=sw_fold,
                      eval_set=[(X_oof[vai], y_train[vai])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                oof_p[vai] = m.predict_proba(X_oof[vai])

                # For test: train on all original + pseudo
                m2 = lgb.LGBMClassifier(**cfg_params)
                m2.fit(X_aug, y_aug, sample_weight=sw_aug,
                       eval_set=[(X_oof[vai], y_train[vai])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
                test_p += m2.predict_proba(X_test) / N_FOLDS

            acc = accuracy_score(y_train, oof_p.argmax(1))
            f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
            print(f"    {cfg_name}: Acc={acc:.4f} F1={f1:.4f}")
            pl_results[cfg_name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

        # Check if pseudo-labeling helped
        best_pl_model = max(pl_results.keys(), key=lambda m: pl_results[m]["acc"])
        if pl_results[best_pl_model]["acc"] > best_pl_acc:
            best_pl_acc = pl_results[best_pl_model]["acc"]
            best_pl_result = pl_results
            best_pl_threshold = threshold
            print(f"  -> Improvement! Best PL acc={best_pl_acc:.4f} at threshold={threshold}")

    # Add PL models to results if they helped
    if best_pl_result is not None:
        print(f"\n  Best pseudo-labeling threshold: {best_pl_threshold}")
        for name, res in best_pl_result.items():
            results[name] = res
    else:
        print("\n  Pseudo-labeling did not improve results.")

    # === Phase 6: Stacking ===
    print("\n" + "=" * 60)
    print("Phase 6: Stacking")
    print("=" * 60)

    model_names = sorted(results.keys())
    stack_oof = np.hstack([results[m]["oof"] for m in model_names])
    stack_test = np.hstack([results[m]["test"] for m in model_names])

    pca_stack = PCA(n_components=50, random_state=SEED)
    X_oof_pca = pca_stack.fit_transform(X_oof)
    X_test_pca = pca_stack.transform(X_test)
    stack_oof_aug = np.hstack([stack_oof, X_oof_pca])
    stack_test_aug = np.hstack([stack_test, X_test_pca])

    scaler_s = StandardScaler()
    stack_oof_sc = scaler_s.fit_transform(stack_oof_aug)
    stack_test_sc = scaler_s.transform(stack_test_aug)

    for C_val, name in [(0.5, "stack_lr_05"), (1.0, "stack_lr_10"), (5.0, "stack_lr_50")]:
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
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
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
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

    # === Phase 7: Optuna ensemble ===
    print("\n" + "=" * 60)
    print("Phase 7: Optuna ensemble")
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
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in top_models]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
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
    significant_weights = {m: f"{w:.3f}" for m, w in zip(top_models, best_weights) if w > 0.01}
    print(f"  Weights: {significant_weights}")
    results["optuna_ens"] = {"oof": optuna_oof, "test": optuna_test, "acc": acc, "f1": f1}

    # Also blend with v4 predictions if available
    v4_oof_path = os.path.join(OUT_DIR, "v4_oof_probs.npy")
    if os.path.exists(v4_oof_path):
        print("\n  Cross-version blending with v4...")
        v4_oof = np.load(v4_oof_path)
        v4_test = np.load(os.path.join(OUT_DIR, "v4_test_probs.npy"))

        best_cross_acc = acc
        best_cross_blend = None
        for wv4 in np.arange(0.0, 0.5, 0.05):
            wv8 = 1.0 - wv4
            blend = wv4 * v4_oof + wv8 * optuna_oof
            ba = accuracy_score(y_train, blend.argmax(1))
            if ba > best_cross_acc:
                best_cross_acc = ba
                best_cross_blend = (wv4, wv8)
        if best_cross_blend:
            wv4, wv8 = best_cross_blend
            cross_oof = wv4 * v4_oof + wv8 * optuna_oof
            cross_test = wv4 * v4_test + wv8 * optuna_test
            cross_f1 = f1_score(y_train, cross_oof.argmax(1), average="macro")
            print(f"  Cross-blend v4({wv4:.2f})+v8({wv8:.2f}): Acc={best_cross_acc:.4f} F1={cross_f1:.4f}")
            results["cross_blend"] = {"oof": cross_oof, "test": cross_test,
                                       "acc": best_cross_acc, "f1": cross_f1}
        else:
            print("  No improvement from v4 blending.")

    # === Phase 8: Threshold optimization ===
    print("\n" + "=" * 60)
    print("Phase 8: Threshold optimization")
    print("=" * 60)

    all_final = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    best = all_final[0]
    best_oof = results[best]["oof"]
    best_test = results[best]["test"]
    best_acc = results[best]["acc"]

    print(f"  Starting from: {best} (Acc={best_acc:.4f})")

    best_thresh_acc = best_acc
    best_bias = np.zeros(3)

    for h_bias in np.arange(-0.20, 0.25, 0.005):
        for r_bias in np.arange(-0.15, 0.15, 0.005):
            o_bias = -(h_bias + r_bias)
            bias = np.array([h_bias, r_bias, o_bias])
            adjusted = best_oof + bias
            preds = adjusted.argmax(1)
            a = accuracy_score(y_train, preds)
            if a > best_thresh_acc:
                best_thresh_acc = a
                best_bias = bias.copy()

    if np.any(best_bias != 0):
        adjusted_oof = best_oof + best_bias
        adjusted_test = best_test + best_bias
        adj_f1 = f1_score(y_train, adjusted_oof.argmax(1), average="macro")
        print(f"  Threshold optimized: Acc={best_thresh_acc:.4f} F1={adj_f1:.4f}")
        print(f"  Bias: Health={best_bias[0]:.4f} Rust={best_bias[1]:.4f} Other={best_bias[2]:.4f}")
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

    # Save submission
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
    sub_path = os.path.join(OUT_DIR, f"submission_v8_acc_{a_s}_f1_{f_s}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    # Save probabilities
    np.save(os.path.join(OUT_DIR, "v8_best_oof_probs.npy"), best_oof_final)
    np.save(os.path.join(OUT_DIR, "v8_best_test_probs.npy"), best_test_final)

    # Also save the Optuna ensemble if different
    if best != "optuna_ens":
        opt_preds = [ID2LBL[p] for p in results["optuna_ens"]["test"].argmax(1)]
        opt_sub = pd.DataFrame({"Id": sub_ids, "Category": opt_preds})
        oa = f"{results['optuna_ens']['acc']:.4f}".replace(".", "p")
        of_s = f"{results['optuna_ens']['f1']:.4f}".replace(".", "p")
        opt_sub.to_csv(os.path.join(OUT_DIR, f"submission_v8_optuna_acc_{oa}_f1_{of_s}.csv"), index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
