"""
Beyond Visible Spectrum v5 - Push for 0.78

New additions over v4:
1. Raw mean spectrum as features (101-dim) + PCA-reduced version
2. Per-pixel K-Means clustering features (captures spatial heterogeneity)
3. Hierarchical classification (Rust vs non-Rust, then Health vs Other)
4. Multiple random seeds for ensemble diversity
5. Spectral unmixing features
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
HS_TARGET_CH = 101

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ============================================================
# Data loading (compact)
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

def spatial_features(arr_2d, prefix):
    features = {}
    arr = arr_2d.astype(np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    features[f"{prefix}_grad_mean"] = np.mean(mag)
    features[f"{prefix}_grad_std"] = np.std(mag)
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    features[f"{prefix}_lap_mean"] = np.mean(np.abs(lap))
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
# Feature extraction
# ============================================================
def extract_rgb_features(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    features = {}
    # Band stats (RGB, HSV, LAB)
    for c in range(3):
        band = (img_rgb[:, :, c] / 255.0).ravel()
        for stat, fn in [("mean", np.mean), ("std", np.std), ("min", np.min),
                         ("max", np.max), ("median", np.median)]:
            features[f"rgb_b{c}_{stat}"] = fn(band)
        features[f"rgb_b{c}_q25"] = np.percentile(band, 25)
        features[f"rgb_b{c}_q75"] = np.percentile(band, 75)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    for c in range(3):
        band = (hsv[:, :, c] / [180, 255, 255][c]).ravel()
        for stat, fn in [("mean", np.mean), ("std", np.std)]:
            features[f"hsv_b{c}_{stat}"] = fn(band)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        band = (lab[:, :, c] / [100, 255, 255][c]).ravel()
        for stat, fn in [("mean", np.mean), ("std", np.std)]:
            features[f"lab_b{c}_{stat}"] = fn(band)

    # Histograms
    for c, name in enumerate(["r", "g", "b"]):
        hist = np.histogram(img_rgb[:, :, c], bins=16, range=(0, 256))[0]
        hist = hist / (hist.sum() + 1e-8)
        for b in range(16):
            features[f"rgb_hist_{name}_{b}"] = hist[b]

    # Texture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features.update(spatial_features(gray.astype(np.float32), "rgb_gray"))
    features.update(lbp_features(gray.astype(np.float32), "rgb"))
    features.update(glcm_features(gray, "rgb"))

    # Color ratios
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    total = r + g + b + 1e-8
    features["rgb_green_ratio"] = np.mean(g / total)
    features["rgb_rg_ratio"] = np.mean(r / (g + 1e-8))
    features["rgb_exg"] = np.mean(2 * g / 255.0 - r / 255.0 - b / 255.0)

    return features


def extract_ms_features(path):
    arr = read_tiff(path)
    features = {}
    eps = 1e-6
    blue, green, red, rededge, nir = [arr[:, :, i] for i in range(5)]

    # Band stats
    for i in range(5):
        band = arr[:, :, i].ravel()
        for stat, fn in [("mean", np.mean), ("std", np.std), ("min", np.min),
                         ("max", np.max), ("median", np.median)]:
            features[f"ms_b{i}_{stat}"] = fn(band)
        features[f"ms_b{i}_q25"] = np.percentile(band, 25)
        features[f"ms_b{i}_q75"] = np.percentile(band, 75)

    # Vegetation indices
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

    # MCARI/TCARI
    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    features["ms_mcari_mean"] = np.mean(mcari)
    features["ms_mcari_std"] = np.std(mcari)

    # Band ratios
    bnames = ["blue", "green", "red", "rededge", "nir"]
    for i in range(5):
        for j in range(i + 1, 5):
            r = arr[:, :, i] / (arr[:, :, j] + eps)
            features[f"ms_ratio_{bnames[i]}_{bnames[j]}_mean"] = np.mean(r)
            features[f"ms_ratio_{bnames[i]}_{bnames[j]}_std"] = np.std(r)

    # NDVI spatial
    features.update(spatial_features(ndvi.astype(np.float32), "ms_ndvi_sp"))
    ndvi_u8 = np.clip((ndvi + 1) * 127.5, 0, 255).astype(np.uint8)
    features.update(glcm_features(ndvi_u8, "ms_ndvi"))

    # Correlations
    flat = arr.reshape(-1, 5)
    try:
        corr = np.corrcoef(flat.T)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"ms_corr_{bnames[i]}_{bnames[j]}"] = corr[i, j]
    except:
        pass

    return features


def get_hs_data(path):
    """Read and standardize HS data."""
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
    return arr


def extract_hs_features(arr):
    """Extract HS features."""
    features = {}
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)
    mean_spec = np.mean(flat, axis=0)
    std_spec = np.std(flat, axis=0)
    median_spec = np.median(flat, axis=0)

    # === Raw mean spectrum as features ===
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

    # Sampled band stats
    n_sample = 20
    sample_idx = np.linspace(0, C - 1, n_sample, dtype=int)
    for i, idx in enumerate(sample_idx):
        features[f"hs_b{i}_std"] = np.std(flat[:, idx])

    # Overall stats
    features["hs_total_mean"] = np.mean(flat)
    features["hs_total_std"] = np.std(flat)

    # Derivatives
    d1 = np.diff(mean_spec)
    features["hs_d1_mean"] = np.mean(d1)
    features["hs_d1_std"] = np.std(d1)
    features["hs_d1_max"] = np.max(d1)
    features["hs_d1_max_pos"] = np.argmax(d1) / len(d1)
    features["hs_d1_min"] = np.min(d1)
    features["hs_d1_min_pos"] = np.argmin(d1) / len(d1)

    d2 = np.diff(d1)
    features["hs_d2_mean"] = np.mean(d2)
    features["hs_d2_std"] = np.std(d2)

    # Shape
    features["hs_peak_band"] = np.argmax(mean_spec) / C
    features["hs_trough_band"] = np.argmin(mean_spec) / C
    features["hs_spectral_area"] = np.trapezoid(mean_spec)

    spec_norm = mean_spec / (mean_spec.sum() + 1e-8)
    features["hs_entropy"] = -np.sum(spec_norm * np.log(spec_norm + 1e-8))

    x = np.arange(C, dtype=np.float32)
    s = mean_spec.sum() + 1e-8
    centroid = np.sum(x * mean_spec) / s
    features["hs_centroid"] = centroid / C
    spread = np.sqrt(np.sum((x - centroid)**2 * mean_spec) / s)
    features["hs_spread"] = spread / C

    # Key bands
    gi, ri, rsi, rpi, ni = int(C*.15), int(C*.40), int(C*.52), int(C*.62), int(C*.77)
    features["hs_nir_red_ratio"] = mean_spec[ni] / (mean_spec[ri] + 1e-8)
    features["hs_nir_green_ratio"] = mean_spec[ni] / (mean_spec[gi] + 1e-8)
    features["hs_re_red_ratio"] = mean_spec[rpi] / (mean_spec[ri] + 1e-8)
    if rpi > rsi:
        features["hs_re_slope"] = (mean_spec[rpi] - mean_spec[rsi]) / (rpi - rsi)
    re_d = d1[max(0, rsi-1):min(len(d1), rpi)]
    if len(re_d) > 0:
        features["hs_re_inflection"] = (np.argmax(re_d) + rsi) / C
        features["hs_re_max_slope"] = np.max(re_d)

    # Absorption depth
    if gi < ri < ni:
        baseline = mean_spec[gi] + (mean_spec[ni] - mean_spec[gi]) * (ri - gi) / (ni - gi + 1e-8)
        features["hs_red_abs_depth"] = baseline - mean_spec[ri]

    # Region stats and ratios
    regions = {"bv": (0, int(C*.10)), "gv": (int(C*.10), int(C*.25)),
               "rv": (int(C*.25), int(C*.45)), "re": (int(C*.45), int(C*.65)), "nir": (int(C*.65), C)}
    for rn, (s, e) in regions.items():
        rd = mean_spec[s:e]
        features[f"hs_r_{rn}_mean"] = np.mean(rd)
        features[f"hs_r_{rn}_std"] = np.std(rd)
    for r1 in regions:
        for r2 in regions:
            if r1 < r2:
                s1, e1 = regions[r1]
                s2, e2 = regions[r2]
                features[f"hs_rr_{r1}_{r2}"] = np.mean(mean_spec[s1:e1]) / (np.mean(mean_spec[s2:e2]) + 1e-8)

    # PCA
    try:
        n_comp = min(20, C, H * W)
        pca = PCA(n_components=n_comp)
        pca_res = pca.fit_transform(flat)
        for i in range(n_comp):
            features[f"hs_pca{i}_mean"] = np.mean(pca_res[:, i])
            features[f"hs_pca{i}_std"] = np.std(pca_res[:, i])
        for i in range(min(10, n_comp)):
            features[f"hs_pca_ev{i}"] = pca.explained_variance_ratio_[i]
    except:
        pass

    # Spatial features
    for name, idx in [("green", gi), ("red", ri), ("nir", ni)]:
        features.update(spatial_features(arr[:, :, idx], f"hs_{name}"))

    # Pixel heterogeneity
    pstd = np.std(flat, axis=0)
    features["hs_hetero_mean"] = np.mean(pstd)
    features["hs_hetero_std"] = np.std(pstd)

    # === NEW: Per-pixel K-Means clustering ===
    try:
        # Normalize pixel spectra for clustering
        flat_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        for n_clusters in [2, 3]:
            km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42, max_iter=100)
            labels = km.fit_predict(flat_norm)
            # Cluster sizes (sorted)
            sizes = np.sort([np.sum(labels == c) / len(labels) for c in range(n_clusters)])[::-1]
            for ci, sz in enumerate(sizes):
                features[f"hs_km{n_clusters}_size_{ci}"] = sz
            # Inertia (how tight the clusters are)
            features[f"hs_km{n_clusters}_inertia"] = km.inertia_ / len(flat)
            # Distance between cluster centers
            centers = km.cluster_centers_
            for ci in range(n_clusters):
                for cj in range(ci + 1, n_clusters):
                    features[f"hs_km{n_clusters}_dist_{ci}_{cj}"] = np.linalg.norm(centers[ci] - centers[cj])
            # Dominant cluster spectral features
            dom_idx = np.argmax([np.sum(labels == c) for c in range(n_clusters)])
            dom_center = km.cluster_centers_[dom_idx]
            features[f"hs_km{n_clusters}_dom_centroid_peak"] = np.argmax(dom_center) / len(dom_center)
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
        libraries[label] = {
            "mean": np.mean(cs, axis=0),
            "std": np.std(cs, axis=0) + 1e-6,
        }
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
    for l1 in LABELS:
        for l2 in LABELS:
            if l1 < l2:
                k1, k2 = f"sl_{l1}_euclid", f"sl_{l2}_euclid"
                if k1 in features and k2 in features:
                    features[f"sl_{l1}_{l2}_ratio"] = features[k1] / (features[k2] + 1e-8)
    return features


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Beyond Visible Spectrum v5")
    print("=" * 60)

    train_idx = build_index(ROOT, "train")
    val_idx = build_index(ROOT, "val")
    train_df = make_df(train_idx, has_labels=True)
    val_df = make_df(val_idx, has_labels=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Extract features
    print("\nExtracting features...")
    train_feats, train_spectra = [], []
    for i, (_, row) in enumerate(train_df.iterrows()):
        f = {"base_id": row["base_id"]}
        spec = None
        if pd.notna(row.get("rgb")):
            f.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")):
            f.update(extract_ms_features(row["ms"]))
        if pd.notna(row.get("hs")):
            hs_arr = get_hs_data(row["hs"])
            hs_f, spec = extract_hs_features(hs_arr)
            f.update(hs_f)
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            f.update(extract_cross_modal_features(row["rgb"], row["ms"]))
        train_feats.append(f)
        train_spectra.append(spec)
        if (i + 1) % 200 == 0:
            print(f"  Train: {i + 1}/{len(train_df)}")
    print(f"  Train: {len(train_df)}/{len(train_df)}")

    val_feats, val_spectra = [], []
    for i, (_, row) in enumerate(val_df.iterrows()):
        f = {"base_id": row["base_id"]}
        spec = None
        if pd.notna(row.get("rgb")):
            f.update(extract_rgb_features(row["rgb"]))
        if pd.notna(row.get("ms")):
            f.update(extract_ms_features(row["ms"]))
        if pd.notna(row.get("hs")):
            hs_arr = get_hs_data(row["hs"])
            hs_f, spec = extract_hs_features(hs_arr)
            f.update(hs_f)
        if pd.notna(row.get("rgb")) and pd.notna(row.get("ms")):
            f.update(extract_cross_modal_features(row["rgb"], row["ms"]))
        val_feats.append(f)
        val_spectra.append(spec)
        if (i + 1) % 200 == 0:
            print(f"  Val: {i + 1}/{len(val_df)}")
    print(f"  Val: {len(val_df)}/{len(val_df)}")

    train_feat_df = pd.DataFrame(train_feats)
    val_feat_df = pd.DataFrame(val_feats)
    base_cols = [c for c in train_feat_df.columns if c != "base_id"]
    X_base_train = train_feat_df[base_cols].fillna(0).values.astype(np.float32)
    X_base_test = val_feat_df[base_cols].fillna(0).values.astype(np.float32)
    X_base_train = np.nan_to_num(X_base_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_base_test = np.nan_to_num(X_base_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.array([LBL2ID[l] for l in train_df["label"]])
    print(f"  Base features: {X_base_train.shape[1]}")

    # CV spectral library
    print("\nCV spectral library features...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_slib = [None] * len(train_df)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_base_train, y_train)):
        lib = build_library([train_spectra[i] for i in tr_idx], [y_train[i] for i in tr_idx])
        for i in va_idx:
            oof_slib[i] = library_features(train_spectra[i], lib)

    all_lib = build_library(train_spectra, y_train.tolist())
    test_slib = [library_features(s, all_lib) for s in val_spectra]
    train_slib_all = [library_features(s, all_lib) for s in train_spectra]

    slib_df_oof = pd.DataFrame(oof_slib)
    slib_df_test = pd.DataFrame(test_slib)
    slib_df_all = pd.DataFrame(train_slib_all)
    slib_cols = list(slib_df_oof.columns)

    X_slib_oof = slib_df_oof.fillna(0).values.astype(np.float32)
    X_slib_test = slib_df_test.fillna(0).values.astype(np.float32)
    X_slib_all = slib_df_all.fillna(0).values.astype(np.float32)
    for arr in [X_slib_oof, X_slib_test, X_slib_all]:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_oof = np.hstack([X_base_train, X_slib_oof])
    X_test = np.hstack([X_base_test, X_slib_test])
    X_all = np.hstack([X_base_train, X_slib_all])
    print(f"  Total features: {X_oof.shape[1]}")

    # Train models
    print("\n" + "=" * 60)
    print("Training models...")
    print("=" * 60)

    results = {}

    lgb_configs = [
        ("lgb_a", {"learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                   "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                   "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
        ("lgb_b", {"learning_rate": 0.02, "num_leaves": 15, "max_depth": 4,
                   "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
                   "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000}),
        ("lgb_c", {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                   "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
                   "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500}),
        ("lgb_d", {"learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                   "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                   "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
    ]

    for name, cfg in lgb_configs:
        params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                  "verbose": -1, "random_state": SEED, **cfg}
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_test), 3))

        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_oof[tri], y_train[tri], eval_set=[(X_oof[vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_oof[vai])
            m2 = lgb.LGBMClassifier(**params)
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test) / N_FOLDS

        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # XGBoost
    for name, cfg in [
        ("xgb_a", {"learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
                   "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
                   "reg_lambda": 1.0, "n_estimators": 2000}),
        ("xgb_b", {"learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
                   "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
                   "reg_lambda": 2.0, "n_estimators": 3000}),
    ]:
        params = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                  "tree_method": "hist", "random_state": SEED, "verbosity": 0, **cfg}
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_test), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = xgb.XGBClassifier(**params)
            m.fit(X_oof[tri], y_train[tri], eval_set=[(X_oof[vai], y_train[vai])], verbose=False)
            oof_p[vai] = m.predict_proba(X_oof[vai])
            m2 = xgb.XGBClassifier(**params)
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])], verbose=False)
            test_p += m2.predict_proba(X_test) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # sklearn models
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
        test_p = np.zeros((len(X_test), 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = model_cls(**kw)
            m.fit(X_oof[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(X_oof[vai])
            m2 = model_cls(**kw)
            m2.fit(X_all[tri], y_train[tri])
            test_p += m2.predict_proba(X_test) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Base-only LGB (no slib)
    oof_p = np.zeros((len(y_train), 3))
    test_p = np.zeros((len(X_base_test), 3))
    for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
        m = lgb.LGBMClassifier(objective="multiclass", num_class=3, learning_rate=0.03,
                                num_leaves=25, max_depth=5, min_child_samples=10,
                                subsample=0.75, colsample_bytree=0.7, reg_alpha=0.5,
                                reg_lambda=0.5, n_estimators=2000, verbose=-1, random_state=SEED)
        m.fit(X_base_train[tri], y_train[tri], eval_set=[(X_base_train[vai], y_train[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_p[vai] = m.predict_proba(X_base_train[vai])
        test_p += m.predict_proba(X_base_test) / N_FOLDS
    acc = accuracy_score(y_train, oof_p.argmax(1))
    f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
    print(f"  lgb_base: Acc={acc:.4f} F1={f1:.4f}")
    results["lgb_base"] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Hierarchical: Rust vs Non-Rust, then Health vs Other ===
    print("\n  Hierarchical approach...")
    y_binary = (y_train == LBL2ID["Rust"]).astype(int)  # 1=Rust, 0=non-Rust
    oof_rust = np.zeros(len(y_train))
    test_rust = np.zeros(len(X_test))
    for fold, (tri, vai) in enumerate(skf.split(X_oof, y_train)):
        m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=20,
                                max_depth=5, min_child_samples=10, subsample=0.8,
                                colsample_bytree=0.6, n_estimators=2000, verbose=-1, random_state=SEED)
        m.fit(X_oof[tri], y_binary[tri], eval_set=[(X_oof[vai], y_binary[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_rust[vai] = m.predict_proba(X_oof[vai])[:, 1]
        m2 = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=20,
                                  max_depth=5, min_child_samples=10, subsample=0.8,
                                  colsample_bytree=0.6, n_estimators=2000, verbose=-1, random_state=SEED)
        m2.fit(X_all[tri], y_binary[tri], eval_set=[(X_all[vai], y_binary[vai])],
               callbacks=[lgb.early_stopping(100, verbose=False)])
        test_rust += m2.predict_proba(X_test)[:, 1] / N_FOLDS

    # Health vs Other (on non-Rust samples only, but predict on all)
    non_rust = y_train != LBL2ID["Rust"]
    y_ho = (y_train[non_rust] == LBL2ID["Health"]).astype(int)
    oof_health = np.zeros(len(y_train))
    test_health = np.zeros(len(X_test))

    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 100)
    X_nr_oof = X_oof[non_rust]
    X_nr_all = X_all[non_rust]

    for fold, (tri, vai) in enumerate(skf2.split(X_nr_oof, y_ho)):
        m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=20,
                                max_depth=5, min_child_samples=8, subsample=0.8,
                                colsample_bytree=0.6, n_estimators=2000, verbose=-1, random_state=SEED)
        m.fit(X_nr_oof[tri], y_ho[tri], eval_set=[(X_nr_oof[vai], y_ho[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        # Predict on ALL train samples (not just non-Rust)
        nr_indices = np.where(non_rust)[0]
        oof_health[nr_indices[vai]] = m.predict_proba(X_nr_oof[vai])[:, 1]

        m2 = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=20,
                                  max_depth=5, min_child_samples=8, subsample=0.8,
                                  colsample_bytree=0.6, n_estimators=2000, verbose=-1, random_state=SEED)
        m2.fit(X_nr_all[tri], y_ho[tri], eval_set=[(X_nr_all[vai], y_ho[vai])],
               callbacks=[lgb.early_stopping(100, verbose=False)])
        test_health += m2.predict_proba(X_test)[:, 1] / N_FOLDS
    # For Rust samples, set health probability to 0
    oof_health[y_train == LBL2ID["Rust"]] = 0.0

    # Combine hierarchical predictions into 3-class probs
    hier_oof = np.zeros((len(y_train), 3))
    hier_oof[:, LBL2ID["Rust"]] = oof_rust
    hier_oof[:, LBL2ID["Health"]] = (1 - oof_rust) * oof_health
    hier_oof[:, LBL2ID["Other"]] = (1 - oof_rust) * (1 - oof_health)

    hier_test = np.zeros((len(X_test), 3))
    hier_test[:, LBL2ID["Rust"]] = test_rust
    hier_test[:, LBL2ID["Health"]] = (1 - test_rust) * test_health
    hier_test[:, LBL2ID["Other"]] = (1 - test_rust) * (1 - test_health)

    acc = accuracy_score(y_train, hier_oof.argmax(1))
    f1 = f1_score(y_train, hier_oof.argmax(1), average="macro")
    print(f"  hierarchical: Acc={acc:.4f} F1={f1:.4f}")
    results["hier"] = {"oof": hier_oof, "test": hier_test, "acc": acc, "f1": f1}

    # === Different seed models ===
    print("\n  Different seed models...")
    for seed_offset in [100, 200]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_offset)
        oof_p = np.zeros((len(y_train), 3))
        test_p = np.zeros((len(X_test), 3))
        for fold, (tri, vai) in enumerate(skf_s.split(X_base_train, y_train)):
            # Need fold-specific slib
            lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
            slib_vai = np.array([list(library_features(train_spectra[i], lib).values()) for i in vai]).astype(np.float32)
            slib_tri = np.array([list(library_features(train_spectra[i], lib).values()) for i in tri]).astype(np.float32)
            X_tri = np.hstack([X_base_train[tri], slib_tri])
            X_vai = np.hstack([X_base_train[vai], slib_vai])

            m = lgb.LGBMClassifier(objective="multiclass", num_class=3, learning_rate=0.03,
                                    num_leaves=25, max_depth=5, min_child_samples=10,
                                    subsample=0.75, colsample_bytree=0.6, reg_alpha=0.5,
                                    reg_lambda=0.5, n_estimators=2000, verbose=-1,
                                    random_state=SEED + seed_offset)
            m.fit(X_tri, y_train[tri], eval_set=[(X_vai, y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_vai)

            # Test: use all-data library
            m2 = lgb.LGBMClassifier(objective="multiclass", num_class=3, learning_rate=0.03,
                                     num_leaves=25, max_depth=5, min_child_samples=10,
                                     subsample=0.75, colsample_bytree=0.6, reg_alpha=0.5,
                                     reg_lambda=0.5, n_estimators=2000, verbose=-1,
                                     random_state=SEED + seed_offset)
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test) / N_FOLDS

        name = f"lgb_seed{seed_offset}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Ensemble
    print("\n" + "=" * 60)
    print("Ensemble")
    print("=" * 60)

    sorted_m = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    print("\nModels:")
    for m in sorted_m:
        print(f"  {m:20s} Acc={results[m]['acc']:.4f} F1={results[m]['f1']:.4f}")

    best_acc, best_name, best_oof, best_test = -1, None, None, None

    for k in [3, 5, 7, 10, len(sorted_m)]:
        top = sorted_m[:min(k, len(sorted_m))]
        w = np.array([results[m]["acc"] for m in top]) ** 3
        w /= w.sum()
        oof_e = sum(wi * results[m]["oof"] for m, wi in zip(top, w))
        test_e = sum(wi * results[m]["test"] for m, wi in zip(top, w))
        acc = accuracy_score(y_train, oof_e.argmax(1))
        f1 = f1_score(y_train, oof_e.argmax(1), average="macro")
        print(f"\n  top{k}: Acc={acc:.4f} F1={f1:.4f}")
        if acc > best_acc:
            best_acc, best_name = acc, f"top{k}"
            best_oof, best_test = oof_e, test_e

    # Simple avg
    oof_avg = sum(results[m]["oof"] for m in sorted_m) / len(sorted_m)
    test_avg = sum(results[m]["test"] for m in sorted_m) / len(sorted_m)
    acc = accuracy_score(y_train, oof_avg.argmax(1))
    f1 = f1_score(y_train, oof_avg.argmax(1), average="macro")
    print(f"\n  avg: Acc={acc:.4f} F1={f1:.4f}")
    if acc > best_acc:
        best_acc, best_name = acc, "avg"
        best_oof, best_test = oof_avg, test_avg

    final_f1 = f1_score(y_train, best_oof.argmax(1), average="macro")
    print(f"\n*** Best: {best_name} | Acc={best_acc:.4f} F1={final_f1:.4f} ***")
    print(classification_report(y_train, best_oof.argmax(1), target_names=LABELS, digits=4))

    # Save submission
    sub_ids = []
    for _, r in val_df.iterrows():
        if pd.notna(r.get("hs")):
            sub_ids.append(os.path.basename(r["hs"]))
        elif pd.notna(r.get("ms")):
            sub_ids.append(os.path.basename(r["ms"]))
        else:
            sub_ids.append(os.path.basename(r["rgb"]))

    preds = [ID2LBL[p] for p in best_test.argmax(1)]
    sub = pd.DataFrame({"Id": sub_ids, "Category": preds})
    a_s = f"{best_acc:.4f}".replace(".", "p")
    f_s = f"{final_f1:.4f}".replace(".", "p")
    sub_path = os.path.join(OUT_DIR, f"submission_v5_acc_{a_s}_f1_{f_s}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    # Also save best single model
    best_single = sorted_m[0]
    tp = results[best_single]["test"].argmax(1)
    sub_s = pd.DataFrame({"Id": sub_ids, "Category": [ID2LBL[p] for p in tp]})
    a_ss = f"{results[best_single]['acc']:.4f}".replace(".", "p")
    sub_s.to_csv(os.path.join(OUT_DIR, f"submission_v5_{best_single}_acc_{a_ss}.csv"), index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
