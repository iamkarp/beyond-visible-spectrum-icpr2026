"""
Beyond Visible Spectrum: AI for Agriculture 2026
Gradient Boosting approach with rich feature engineering.

For small datasets (600 samples), handcrafted features + gradient boosting
often outperforms deep learning. This extracts ~500+ features from all
three modalities and trains LightGBM + XGBoost ensemble with 5-fold CV.

Features:
- RGB: Color stats, histograms, texture (LBP), spatial gradients
- MS: Band stats, 10+ vegetation indices, band ratios
- HS: PCA components, spectral derivatives, band statistics
"""

import argparse
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import cv2
import tifffile as tiff

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Beyond Visible Spectrum — GBM ensemble classifier")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Root directory containing train/ and val/ subdirectories (default: current directory)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save submission CSVs (default: <data-dir>/output)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

N_FOLDS = 5   # overridden by args at runtime
SEED = 42     # overridden by args at runtime
HS_DROP_FIRST = 10
HS_DROP_LAST = 14

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}


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
# Feature extraction
# ============================================================
def band_statistics(arr, prefix):
    """Basic per-band statistics: mean, std, min, max, median, skew, kurtosis."""
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
        # Skewness
        m = np.mean(band)
        s = np.std(band) + 1e-8
        features[f"{prefix}_b{c}_skew"] = np.mean(((band - m) / s) ** 3)
    return features


def spatial_features(arr_2d, prefix):
    """Spatial gradient and texture features for a single band."""
    features = {}
    arr = arr_2d.astype(np.float32)
    # Gradients
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    features[f"{prefix}_grad_mean"] = np.mean(mag)
    features[f"{prefix}_grad_std"] = np.std(mag)
    features[f"{prefix}_grad_max"] = np.max(mag)

    # Laplacian
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    features[f"{prefix}_lap_mean"] = np.mean(np.abs(lap))
    features[f"{prefix}_lap_std"] = np.std(lap)

    return features


def color_histogram_features(img_rgb, prefix, n_bins=16):
    """Color histogram features."""
    features = {}
    for c, name in enumerate(["r", "g", "b"]):
        hist = np.histogram(img_rgb[:, :, c], bins=n_bins, range=(0, 256))[0]
        hist = hist / (hist.sum() + 1e-8)
        for b in range(n_bins):
            features[f"{prefix}_hist_{name}_{b}"] = hist[b]
        # Entropy
        features[f"{prefix}_entropy_{name}"] = -np.sum(hist * np.log(hist + 1e-8))
    return features


def lbp_features(gray, prefix, radius=1, n_points=8):
    """Simple LBP-like texture features."""
    features = {}
    H, W = gray.shape
    # Use center pixel comparison with neighbors
    padded = np.pad(gray, radius, mode="reflect")
    pattern = np.zeros_like(gray, dtype=np.float32)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dy = int(round(radius * np.sin(angle)))
        dx = int(round(radius * np.cos(angle)))
        neighbor = padded[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
        pattern += (neighbor > gray).astype(np.float32) * (2 ** i)

    # Histogram of LBP patterns
    hist = np.histogram(pattern, bins=32, range=(0, 256))[0]
    hist = hist / (hist.sum() + 1e-8)
    for b in range(32):
        features[f"{prefix}_lbp_{b}"] = hist[b]
    features[f"{prefix}_lbp_entropy"] = -np.sum(hist * np.log(hist + 1e-8))
    return features


def extract_rgb_features(path):
    """Extract features from RGB image."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    features = {}
    # Band statistics
    features.update(band_statistics(img_rgb / 255.0, "rgb"))

    # Color histogram
    features.update(color_histogram_features(img_rgb, "rgb"))

    # HSV features
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    features.update(band_statistics(hsv / np.array([180, 255, 255], dtype=np.float32), "hsv"))

    # Spatial features on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    features.update(spatial_features(gray, "rgb_gray"))

    # LBP on grayscale
    features.update(lbp_features(gray, "rgb"))

    # Green ratio (useful for vegetation)
    total = img_rgb.sum(axis=2) + 1e-8
    features["rgb_green_ratio_mean"] = np.mean(img_rgb[:, :, 1] / total)
    features["rgb_green_ratio_std"] = np.std(img_rgb[:, :, 1] / total)

    # ExG (Excess Green Index)
    r, g, b = img_rgb[:, :, 0] / 255.0, img_rgb[:, :, 1] / 255.0, img_rgb[:, :, 2] / 255.0
    exg = 2 * g - r - b
    features["rgb_exg_mean"] = np.mean(exg)
    features["rgb_exg_std"] = np.std(exg)

    return features


def extract_ms_features(path):
    """Extract features from multispectral data."""
    arr = read_tiff(path)  # (H,W,5): Blue, Green, Red, RedEdge, NIR
    features = {}
    eps = 1e-6

    # Band statistics
    features.update(band_statistics(arr, "ms"))

    blue, green, red, rededge, nir = [arr[:, :, i] for i in range(5)]

    # Vegetation indices (computed on raw values)
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
        return idx

    ndvi = safe_idx("ndvi", nir - red, nir + red)
    ndre = safe_idx("ndre", nir - rededge, nir + rededge)
    gndvi = safe_idx("gndvi", nir - green, nir + green)
    savi = safe_idx("savi", 1.5 * (nir - red), nir + red + 0.5)
    evi_num = 2.5 * (nir - red)
    evi_den = nir + 6 * red - 7.5 * blue + 1
    evi = safe_idx("evi", evi_num, evi_den)
    mcari_val = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))
    features["ms_mcari_mean"] = np.mean(mcari_val)
    features["ms_mcari_std"] = np.std(mcari_val)

    # OSAVI
    safe_idx("osavi", 1.16 * (nir - red), nir + red + 0.16)
    # MSAVI
    msavi = 0.5 * (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red) + eps))
    features["ms_msavi_mean"] = np.mean(msavi)
    features["ms_msavi_std"] = np.std(msavi)
    # CI (Chlorophyll Index)
    safe_idx("ci_green", nir, green)
    safe_idx("ci_rededge", nir, rededge)
    # NDBI (Normalized Difference Blue Index)
    safe_idx("ndbi", blue - red, blue + red)

    # Band ratios
    for i in range(5):
        for j in range(i + 1, 5):
            ratio = arr[:, :, i] / (arr[:, :, j] + eps)
            features[f"ms_ratio_{i}_{j}_mean"] = np.mean(ratio)
            features[f"ms_ratio_{i}_{j}_std"] = np.std(ratio)

    # Spatial features on NDVI
    features.update(spatial_features(ndvi.astype(np.float32), "ms_ndvi_spatial"))

    # Cross-band correlations
    flat = arr.reshape(-1, 5)
    try:
        corr = np.corrcoef(flat.T)
        for i in range(5):
            for j in range(i + 1, 5):
                features[f"ms_corr_{i}_{j}"] = corr[i, j]
    except:
        pass

    return features


def extract_hs_features(path, drop_first=10, drop_last=14, n_pca=20):
    """Extract features from hyperspectral data."""
    arr = read_tiff(path)  # (H,W,125)
    B = arr.shape[2]
    if B > (drop_first + drop_last + 1):
        arr = arr[:, :, drop_first:B - drop_last]

    features = {}
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)  # (H*W, C)

    # Mean spectrum
    mean_spectrum = np.mean(flat, axis=0)
    std_spectrum = np.std(flat, axis=0)

    # Spectral statistics (sample bands evenly to keep feature count manageable)
    n_sample = min(20, C)
    sample_indices = np.linspace(0, C - 1, n_sample, dtype=int)
    for i, idx in enumerate(sample_indices):
        band = flat[:, idx]
        features[f"hs_b{i}_mean"] = np.mean(band)
        features[f"hs_b{i}_std"] = np.std(band)
        features[f"hs_b{i}_min"] = np.min(band)
        features[f"hs_b{i}_max"] = np.max(band)

    # Overall spectral statistics
    features["hs_total_mean"] = np.mean(flat)
    features["hs_total_std"] = np.std(flat)
    features["hs_spectral_range"] = np.mean(mean_spectrum.max() - mean_spectrum.min())

    # Spectral derivatives (1st order)
    deriv1 = np.diff(mean_spectrum)
    features["hs_deriv1_mean"] = np.mean(deriv1)
    features["hs_deriv1_std"] = np.std(deriv1)
    features["hs_deriv1_max"] = np.max(deriv1)
    features["hs_deriv1_min"] = np.min(deriv1)
    features["hs_deriv1_max_pos"] = np.argmax(deriv1) / len(deriv1)

    # Spectral derivatives (2nd order)
    deriv2 = np.diff(deriv1)
    features["hs_deriv2_mean"] = np.mean(deriv2)
    features["hs_deriv2_std"] = np.std(deriv2)
    features["hs_deriv2_max"] = np.max(deriv2)

    # Spectral shape features
    features["hs_peak_band"] = np.argmax(mean_spectrum) / C
    features["hs_trough_band"] = np.argmin(mean_spectrum) / C
    features["hs_peak_value"] = np.max(mean_spectrum)
    features["hs_trough_value"] = np.min(mean_spectrum)

    # Spectral area (integral approximation)
    features["hs_spectral_area"] = np.trapezoid(mean_spectrum)
    # Spectral entropy
    spec_norm = mean_spectrum / (mean_spectrum.sum() + 1e-8)
    features["hs_spectral_entropy"] = -np.sum(spec_norm * np.log(spec_norm + 1e-8))

    # Band ratios at key wavelength regions
    # Approximate band indices for specific wavelengths after trimming
    # Original: 450-950nm, 125 bands, 4nm resolution
    # After trim (10 first, 14 last): bands 10-110, wavelengths ~490-890nm
    n_remaining = C
    # Red edge region
    red_edge_start = int(n_remaining * 0.55)  # ~750nm
    red_edge_end = int(n_remaining * 0.65)    # ~780nm
    red_region = int(n_remaining * 0.4)        # ~650nm
    nir_region = int(n_remaining * 0.85)       # ~850nm
    green_region = int(n_remaining * 0.15)     # ~550nm

    features["hs_red_edge_mean"] = np.mean(mean_spectrum[red_edge_start:red_edge_end])
    features["hs_nir_vs_red"] = mean_spectrum[nir_region] / (mean_spectrum[red_region] + 1e-8)
    features["hs_red_edge_slope"] = (mean_spectrum[red_edge_end] - mean_spectrum[red_edge_start]) / max(1, red_edge_end - red_edge_start)
    features["hs_green_peak"] = mean_spectrum[green_region]

    # PCA on the spectral dimension
    try:
        pca = PCA(n_components=min(n_pca, C, H * W))
        pca_result = pca.fit_transform(flat)  # (H*W, n_pca)
        for i in range(pca_result.shape[1]):
            comp = pca_result[:, i]
            features[f"hs_pca{i}_mean"] = np.mean(comp)
            features[f"hs_pca{i}_std"] = np.std(comp)
        # Explained variance
        for i, ev in enumerate(pca.explained_variance_ratio_[:10]):
            features[f"hs_pca_ev{i}"] = ev
        features["hs_pca_cumev_5"] = np.sum(pca.explained_variance_ratio_[:5])
    except:
        pass

    # Spatial variation per spectral region
    for name, idx in [("blue", 0), ("green", green_region), ("red", red_region),
                       ("rededge", red_edge_start), ("nir", nir_region)]:
        if idx < C:
            band_img = arr[:, :, idx]
            features[f"hs_{name}_spatial_std"] = np.std(band_img)
            features.update(spatial_features(band_img.astype(np.float32), f"hs_{name}"))

    return features


def extract_all_features(row):
    """Extract all features for a single sample."""
    features = {"base_id": row["base_id"]}

    if pd.notna(row.get("rgb")):
        features.update(extract_rgb_features(row["rgb"]))

    if pd.notna(row.get("ms")):
        features.update(extract_ms_features(row["ms"]))

    if pd.notna(row.get("hs")):
        features.update(extract_hs_features(row["hs"], HS_DROP_FIRST, HS_DROP_LAST))

    return features


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    ROOT = os.path.abspath(args.data_dir)
    OUT_DIR = args.output_dir if args.output_dir else os.path.join(ROOT, "output")
    os.makedirs(OUT_DIR, exist_ok=True)

    global N_FOLDS, SEED
    N_FOLDS = args.n_folds
    SEED = args.seed
    np.random.seed(SEED)

    print(f"Data directory : {ROOT}")
    print(f"Output directory: {OUT_DIR}")
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
    print(f"  Train features shape: {train_feat_df.shape}")

    print("\nExtracting validation features...")
    val_features = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        feats = extract_all_features(row)
        val_features.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val_df)}")
    val_feat_df = pd.DataFrame(val_features)
    print(f"  Val features shape: {val_feat_df.shape}")

    # Prepare X, y
    feature_cols = [c for c in train_feat_df.columns if c != "base_id"]
    X_train = train_feat_df[feature_cols].fillna(0).values
    X_test = val_feat_df[feature_cols].fillna(0).values
    y_train = np.array([LBL2ID[l] for l in train_df["label"]])

    print(f"\nFeature matrix: {X_train.shape[1]} features")

    # Replace inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Try importing gradient boosting libraries
    try:
        import lightgbm as lgb
        HAS_LGB = True
        print("LightGBM available")
    except ImportError:
        HAS_LGB = False
        print("LightGBM not available")

    try:
        import xgboost as xgb
        HAS_XGB = True
        print("XGBoost available")
    except ImportError:
        HAS_XGB = False
        print("XGBoost not available")

    from sklearn.ensemble import (
        GradientBoostingClassifier, RandomForestClassifier,
        ExtraTreesClassifier, VotingClassifier
    )
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    # ====== Model 1: LightGBM (if available) ======
    results = {}

    if HAS_LGB:
        print("\n" + "=" * 60)
        print("Training LightGBM...")
        print("=" * 60)

        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_estimators": 1000,
            "verbose": -1,
            "random_state": SEED,
        }

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_probs_lgb = np.zeros((len(X_train), 3))
        test_probs_lgb = np.zeros((len(X_test), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            oof_probs_lgb[va_idx] = model.predict_proba(X_va)
            test_probs_lgb += model.predict_proba(X_test) / N_FOLDS

            fold_f1 = f1_score(y_va, oof_probs_lgb[va_idx].argmax(1), average="macro")
            print(f"  Fold {fold + 1} F1: {fold_f1:.4f}")

        oof_f1_lgb = f1_score(y_train, oof_probs_lgb.argmax(1), average="macro")
        oof_acc_lgb = accuracy_score(y_train, oof_probs_lgb.argmax(1))
        print(f"\nLightGBM OOF F1: {oof_f1_lgb:.4f} | Acc: {oof_acc_lgb:.4f}")
        results["lgb"] = {"oof_probs": oof_probs_lgb, "test_probs": test_probs_lgb, "f1": oof_f1_lgb}

        # Feature importance
        print("\nTop 20 features:")
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[-20:][::-1]
        for idx in top_idx:
            print(f"  {feature_cols[idx]:40s} {imp[idx]:.0f}")

    # ====== Model 2: XGBoost (if available) ======
    if HAS_XGB:
        print("\n" + "=" * 60)
        print("Training XGBoost...")
        print("=" * 60)

        xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 1000,
            "tree_method": "hist",
            "random_state": SEED,
            "verbosity": 0,
        }

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_probs_xgb = np.zeros((len(X_train), 3))
        test_probs_xgb = np.zeros((len(X_test), 3))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            oof_probs_xgb[va_idx] = model.predict_proba(X_va)
            test_probs_xgb += model.predict_proba(X_test) / N_FOLDS

            fold_f1 = f1_score(y_va, oof_probs_xgb[va_idx].argmax(1), average="macro")
            print(f"  Fold {fold + 1} F1: {fold_f1:.4f}")

        oof_f1_xgb = f1_score(y_train, oof_probs_xgb.argmax(1), average="macro")
        oof_acc_xgb = accuracy_score(y_train, oof_probs_xgb.argmax(1))
        print(f"\nXGBoost OOF F1: {oof_f1_xgb:.4f} | Acc: {oof_acc_xgb:.4f}")
        results["xgb"] = {"oof_probs": oof_probs_xgb, "test_probs": test_probs_xgb, "f1": oof_f1_xgb}

    # ====== Model 3: sklearn Gradient Boosting ======
    print("\n" + "=" * 60)
    print("Training sklearn GradientBoosting...")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probs_gb = np.zeros((len(X_train), 3))
    test_probs_gb = np.zeros((len(X_test), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = GradientBoostingClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=SEED,
        )
        model.fit(X_tr, y_tr)

        oof_probs_gb[va_idx] = model.predict_proba(X_va)
        test_probs_gb += model.predict_proba(X_test) / N_FOLDS

        fold_f1 = f1_score(y_va, oof_probs_gb[va_idx].argmax(1), average="macro")
        print(f"  Fold {fold + 1} F1: {fold_f1:.4f}")

    oof_f1_gb = f1_score(y_train, oof_probs_gb.argmax(1), average="macro")
    oof_acc_gb = accuracy_score(y_train, oof_probs_gb.argmax(1))
    print(f"\nsklearn GB OOF F1: {oof_f1_gb:.4f} | Acc: {oof_acc_gb:.4f}")
    results["gb"] = {"oof_probs": oof_probs_gb, "test_probs": test_probs_gb, "f1": oof_f1_gb}

    # ====== Model 4: ExtraTrees ======
    print("\n" + "=" * 60)
    print("Training ExtraTrees...")
    print("=" * 60)

    oof_probs_et = np.zeros((len(X_train), 3))
    test_probs_et = np.zeros((len(X_test), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = ExtraTreesClassifier(
            n_estimators=1000, max_depth=None, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        oof_probs_et[va_idx] = model.predict_proba(X_va)
        test_probs_et += model.predict_proba(X_test) / N_FOLDS

        fold_f1 = f1_score(y_va, oof_probs_et[va_idx].argmax(1), average="macro")
        print(f"  Fold {fold + 1} F1: {fold_f1:.4f}")

    oof_f1_et = f1_score(y_train, oof_probs_et.argmax(1), average="macro")
    oof_acc_et = accuracy_score(y_train, oof_probs_et.argmax(1))
    print(f"\nExtraTrees OOF F1: {oof_f1_et:.4f} | Acc: {oof_acc_et:.4f}")
    results["et"] = {"oof_probs": oof_probs_et, "test_probs": test_probs_et, "f1": oof_f1_et}

    # ====== Ensemble ======
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)

    # Weighted average based on individual OOF F1 scores
    model_names = list(results.keys())
    weights = np.array([results[m]["f1"] for m in model_names])
    weights = weights / weights.sum()

    print("\nModel weights:")
    for name, w in zip(model_names, weights):
        print(f"  {name}: {w:.3f} (F1={results[name]['f1']:.4f})")

    oof_probs_ens = sum(w * results[m]["oof_probs"] for m, w in zip(model_names, weights))
    test_probs_ens = sum(w * results[m]["test_probs"] for m, w in zip(model_names, weights))

    oof_preds_ens = oof_probs_ens.argmax(1)
    oof_f1_ens = f1_score(y_train, oof_preds_ens, average="macro")
    oof_acc_ens = accuracy_score(y_train, oof_preds_ens)

    print(f"\nEnsemble OOF F1: {oof_f1_ens:.4f} | Acc: {oof_acc_ens:.4f}")
    print(classification_report(y_train, oof_preds_ens, target_names=LABELS, digits=4))

    # Also try simple average
    oof_probs_avg = sum(results[m]["oof_probs"] for m in model_names) / len(model_names)
    test_probs_avg = sum(results[m]["test_probs"] for m in model_names) / len(model_names)
    oof_f1_avg = f1_score(y_train, oof_probs_avg.argmax(1), average="macro")
    oof_acc_avg = accuracy_score(y_train, oof_probs_avg.argmax(1))
    print(f"Simple average OOF F1: {oof_f1_avg:.4f} | Acc: {oof_acc_avg:.4f}")

    # Use whichever ensemble is better
    if oof_f1_avg > oof_f1_ens:
        print("Using simple average (better)")
        final_test_probs = test_probs_avg
        final_oof_f1 = oof_f1_avg
        final_oof_acc = oof_acc_avg
    else:
        print("Using weighted average (better)")
        final_test_probs = test_probs_ens
        final_oof_f1 = oof_f1_ens
        final_oof_acc = oof_acc_ens

    # Build submission
    test_preds = final_test_probs.argmax(1)
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
    sub_name = f"submission_gbm_oof_f1_{f1_str}_acc_{acc_str}.csv"
    sub_path = os.path.join(OUT_DIR, sub_name)
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved: {sub_path}")

    # Save individual model submissions too
    for name in model_names:
        probs = results[name]["test_probs"]
        preds = probs.argmax(1)
        labels = [ID2LBL[p] for p in preds]
        f1_val = results[name]["f1"]
        f1_s = f"{f1_val:.4f}".replace(".", "p")
        sub_i = pd.DataFrame({"Id": sub_ids, "Category": labels})
        sub_i.to_csv(os.path.join(OUT_DIR, f"submission_{name}_oof_f1_{f1_s}.csv"), index=False)

    # Save OOF probabilities for potential stacking with deep learning model
    np.save(os.path.join(OUT_DIR, "gbm_oof_probs.npy"), oof_probs_ens)
    np.save(os.path.join(OUT_DIR, "gbm_test_probs.npy"), final_test_probs)

    print("\nDone!")


if __name__ == "__main__":
    main()
