"""
Beyond Visible Spectrum v11 - Add unused CNN features (ResNet50+ConvNeXt+MS-ResNet)

Key insight: v8/v9/v10 only used 100 EfficientNet PCA features.
We have 250 MORE CNN features (ResNet50=100, ConvNeXt=100, MS-ResNet=50) in cnn_features_v2.npz
that were NEVER used. These capture different visual patterns and could break the 0.765 ceiling.

Strategy:
1. Load v8 handcrafted features (807)
2. Add ALL CNN features: EfficientNet(100) + ResNet50(100) + ConvNeXt(100) + MS-ResNet(50) = 350
3. Total: 807 + 350 = 1157 features (vs v8's 907)
4. Use MI-based feature selection to remove noise and stay under dimensionality limit
5. Optuna-tuned individual models + proven configs + seed diversity + Health weighting
6. Optuna ensemble + cross-blend
"""

import os, re, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import rankdata

import lightgbm as lgb
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
# Spectral library (reuse)
# ============================================================
def spectral_angle(s1, s2):
    cos = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))

def build_library(spectra, labels):
    libraries = {}
    for label in LABELS:
        cidx = [i for i, l in enumerate(labels) if l == LBL2ID[label]]
        if not cidx: continue
        cs = np.array([spectra[i] for i in cidx])
        mean_s, std_s = np.mean(cs, axis=0), np.std(cs, axis=0) + 1e-6
        lib = {"mean": mean_s, "std": std_s, "cov_inv": None}
        if len(cidx) > 20:
            try:
                cov = np.cov(cs.T)
                reg = np.eye(cov.shape[0]) * 0.01 * np.trace(cov) / cov.shape[0]
                lib["cov_inv"] = np.linalg.inv(cov + reg)
            except: pass
        libraries[label] = lib
    return libraries

def library_features(spectrum, libraries):
    features = {}
    if spectrum is None: return features
    for label, lib in libraries.items():
        diff = spectrum - lib["mean"]
        features[f"sl_{label}_euclid"] = np.sqrt(np.sum(diff**2))
        features[f"sl_{label}_norm_euclid"] = np.sqrt(np.sum((diff / lib["std"])**2))
        features[f"sl_{label}_angle"] = spectral_angle(spectrum, lib["mean"])
        features[f"sl_{label}_corr"] = np.corrcoef(spectrum, lib["mean"])[0, 1]
        if lib["cov_inv"] is not None:
            try:
                features[f"sl_{label}_mahal"] = np.sqrt(np.clip(diff @ lib["cov_inv"] @ diff, 0, None))
            except: pass
    for l1 in LABELS:
        for l2 in LABELS:
            if l1 < l2:
                for metric in ["euclid", "angle"]:
                    k1, k2 = f"sl_{l1}_{metric}", f"sl_{l2}_{metric}"
                    if k1 in features and k2 in features:
                        features[f"sl_{l1}_{l2}_{metric}_ratio"] = features[k1] / (features[k2] + 1e-8)
    return features


# ============================================================
# Train/predict helpers
# ============================================================
def train_lgb_cv(params, X_oof, X_all, X_test, y_train, skf, X_base_train,
                 sample_weight=None, name="lgb"):
    n_train, n_test = len(y_train), len(X_test)
    oof_p = np.zeros((n_train, 3))
    test_p = np.zeros((n_test, 3))
    for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
        sw = sample_weight[tri] if sample_weight is not None else None
        m = lgb.LGBMClassifier(**params)
        m.fit(X_oof[tri], y_train[tri], sample_weight=sw,
              eval_set=[(X_oof[vai], y_train[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_p[vai] = m.predict_proba(X_oof[vai])
        m2 = lgb.LGBMClassifier(**params)
        sw2 = sample_weight[tri] if sample_weight is not None else None
        m2.fit(X_all[tri], y_train[tri], sample_weight=sw2,
               eval_set=[(X_all[vai], y_train[vai])],
               callbacks=[lgb.early_stopping(100, verbose=False)])
        test_p += m2.predict_proba(X_test) / N_FOLDS
    acc = accuracy_score(y_train, oof_p.argmax(1))
    f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
    return {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

def train_xgb_cv(params, X_oof, X_all, X_test, y_train, skf, X_base_train, name="xgb"):
    n_train, n_test = len(y_train), len(X_test)
    oof_p = np.zeros((n_train, 3))
    test_p = np.zeros((n_test, 3))
    for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
        m = xgb.XGBClassifier(**params)
        m.fit(X_oof[tri], y_train[tri], eval_set=[(X_oof[vai], y_train[vai])], verbose=False)
        oof_p[vai] = m.predict_proba(X_oof[vai])
        m2 = xgb.XGBClassifier(**params)
        m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])], verbose=False)
        test_p += m2.predict_proba(X_test) / N_FOLDS
    acc = accuracy_score(y_train, oof_p.argmax(1))
    f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
    return {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

def train_sklearn_cv(cls, params, X_oof, X_all, X_test, y_train, skf, X_base_train, name="sk"):
    n_train, n_test = len(y_train), len(X_test)
    oof_p = np.zeros((n_train, 3))
    test_p = np.zeros((n_test, 3))
    for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
        m = cls(**params)
        m.fit(X_oof[tri], y_train[tri])
        oof_p[vai] = m.predict_proba(X_oof[vai])
        m2 = cls(**params)
        m2.fit(X_all[tri], y_train[tri])
        test_p += m2.predict_proba(X_test) / N_FOLDS
    acc = accuracy_score(y_train, oof_p.argmax(1))
    f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
    return {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("Beyond Visible Spectrum v11 - Multi-CNN feature fusion")
    print("=" * 60)

    # === Phase 1: Load cached features + new CNN features ===
    print("\nPhase 1: Loading features...")
    feat_data = np.load(os.path.join(OUT_DIR, "v8_features.npz"), allow_pickle=True)
    X_hand_train = feat_data["X_hand_train"]  # 807 handcrafted
    X_hand_test = feat_data["X_hand_test"]
    train_cnn_eff = feat_data["train_cnn"]     # 100 EfficientNet PCA
    val_cnn_eff = feat_data["val_cnn"]

    # Load v2 CNN features (ResNet50 + ConvNeXt + MS-ResNet)
    cnn_v2 = np.load(os.path.join(OUT_DIR, "cnn_features_v2.npz"), allow_pickle=True)
    train_resnet = cnn_v2["train_resnet_pca"]   # 100
    val_resnet = cnn_v2["val_resnet_pca"]
    train_convnext = cnn_v2["train_convnext_pca"]  # 100
    val_convnext = cnn_v2["val_convnext_pca"]
    train_ms_cnn = cnn_v2["train_ms_pca"]       # 50
    val_ms_cnn = cnn_v2["val_ms_pca"]

    # Align v2 features by name
    v2_names_t = {n: i for i, n in enumerate(cnn_v2["train_names"])}
    v2_names_v = {n: i for i, n in enumerate(cnn_v2["val_names"])}
    v2_ms_names_t = {n: i for i, n in enumerate(cnn_v2["train_ms_names"])}
    v2_ms_names_v = {n: i for i, n in enumerate(cnn_v2["val_ms_names"])}

    # Load labels
    train_idx_dir = os.path.join(ROOT, "train")
    train_files = sorted(os.listdir(os.path.join(train_idx_dir, "RGB")))
    train_files = [f for f in train_files if f.lower().endswith((".png", ".jpg"))]
    y_labels = []
    train_bids = []
    for f in train_files:
        bid = os.path.splitext(f)[0]
        m = re.match(r"^(Health|Rust|Other)_", bid)
        if m:
            y_labels.append(m.group(1))
            train_bids.append(bid)
    y_train = np.array([LBL2ID[l] for l in y_labels])

    val_files = sorted(os.listdir(os.path.join(ROOT, "val", "RGB")))
    val_files = [f for f in val_files if f.lower().endswith((".png", ".jpg"))]
    val_base_ids = [os.path.splitext(f)[0] for f in val_files]

    n_train, n_val = len(y_train), len(val_base_ids)

    # Build aligned multi-CNN feature matrices
    train_resnet_aligned = np.zeros((n_train, 100), dtype=np.float32)
    train_convnext_aligned = np.zeros((n_train, 100), dtype=np.float32)
    train_ms_aligned = np.zeros((n_train, 50), dtype=np.float32)
    val_resnet_aligned = np.zeros((n_val, 100), dtype=np.float32)
    val_convnext_aligned = np.zeros((n_val, 100), dtype=np.float32)
    val_ms_aligned = np.zeros((n_val, 50), dtype=np.float32)

    for i, bid in enumerate(train_bids):
        if bid in v2_names_t:
            train_resnet_aligned[i] = train_resnet[v2_names_t[bid]]
            train_convnext_aligned[i] = train_convnext[v2_names_t[bid]]
        if bid in v2_ms_names_t:
            train_ms_aligned[i] = train_ms_cnn[v2_ms_names_t[bid]]

    for i, bid in enumerate(val_base_ids):
        if bid in v2_names_v:
            val_resnet_aligned[i] = val_resnet[v2_names_v[bid]]
            val_convnext_aligned[i] = val_convnext[v2_names_v[bid]]
        if bid in v2_ms_names_v:
            val_ms_aligned[i] = val_ms_cnn[v2_ms_names_v[bid]]

    # Combine ALL CNN features
    train_cnn_all = np.hstack([train_cnn_eff, train_resnet_aligned, train_convnext_aligned, train_ms_aligned])
    val_cnn_all = np.hstack([val_cnn_eff, val_resnet_aligned, val_convnext_aligned, val_ms_aligned])

    X_base_train = np.hstack([X_hand_train, train_cnn_all])
    X_base_test = np.hstack([X_hand_test, val_cnn_all])
    print(f"  Train: {n_train} | Val: {n_val}")
    print(f"  Handcrafted: {X_hand_train.shape[1]} | CNN: {train_cnn_all.shape[1]} "
          f"(Eff={train_cnn_eff.shape[1]}, ResNet={100}, ConvNeXt={100}, MS={50})")
    print(f"  Total base features: {X_base_train.shape[1]}")

    # Also keep v8-compatible features (only EfficientNet CNN) for comparison
    X_base_v8_train = np.hstack([X_hand_train, train_cnn_eff])
    X_base_v8_test = np.hstack([X_hand_test, val_cnn_eff])

    # === Phase 2: Feature selection via MI ===
    print("\nPhase 2: Feature selection via mutual information...")
    mi_scores = mutual_info_classif(X_base_train, y_train, random_state=SEED)
    mi_threshold = np.percentile(mi_scores, 25)  # Keep top 75%
    mi_mask = mi_scores >= mi_threshold
    n_selected = mi_mask.sum()
    print(f"  MI threshold: {mi_threshold:.4f} | Selected: {n_selected}/{X_base_train.shape[1]} features")

    X_sel_train = X_base_train[:, mi_mask]
    X_sel_test = X_base_test[:, mi_mask]

    # === Phase 3: HS spectra and spectral library ===
    print("\nPhase 3: CV spectral library...")
    HS_DROP_FIRST, HS_DROP_LAST, HS_TARGET_CH = 10, 14, 101
    import tifffile as tiff

    def read_tiff(path):
        arr = tiff.imread(path)
        if arr.ndim != 3: raise ValueError(f"Expected 3D, got {arr.shape}")
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)

    train_spectra, val_spectra = [], []
    for bid in train_bids:
        hs_path = os.path.join(train_idx_dir, "HS", bid + ".tif")
        if os.path.exists(hs_path):
            arr = read_tiff(hs_path)
            B = arr.shape[2]
            if B > (HS_DROP_FIRST + HS_DROP_LAST + 1):
                arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST]
            C = arr.shape[2]
            if C > HS_TARGET_CH: arr = arr[:, :, :HS_TARGET_CH]
            spec = np.mean(arr.reshape(-1, min(C, HS_TARGET_CH)), axis=0)
            if len(spec) < HS_TARGET_CH: spec = np.pad(spec, (0, HS_TARGET_CH - len(spec)))
            train_spectra.append(spec)
        else:
            train_spectra.append(np.zeros(HS_TARGET_CH))

    for bid in val_base_ids:
        hs_path = os.path.join(ROOT, "val", "HS", bid + ".tif")
        if os.path.exists(hs_path):
            arr = read_tiff(hs_path)
            B = arr.shape[2]
            if B > (HS_DROP_FIRST + HS_DROP_LAST + 1):
                arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST]
            C = arr.shape[2]
            if C > HS_TARGET_CH: arr = arr[:, :, :HS_TARGET_CH]
            spec = np.mean(arr.reshape(-1, min(C, HS_TARGET_CH)), axis=0)
            if len(spec) < HS_TARGET_CH: spec = np.pad(spec, (0, HS_TARGET_CH - len(spec)))
            val_spectra.append(spec)
        else:
            val_spectra.append(np.zeros(HS_TARGET_CH))

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

    # Full feature sets: base + slib
    X_oof_full = np.hstack([X_base_train, X_slib_oof])
    X_test_full = np.hstack([X_base_test, X_slib_test])
    X_all_full = np.hstack([X_base_train, X_slib_all])

    # MI-selected + slib
    X_oof_sel = np.hstack([X_sel_train, X_slib_oof])
    X_test_sel = np.hstack([X_sel_test, X_slib_test])
    X_all_sel = np.hstack([X_sel_train, X_slib_all])

    # v8-compatible (EfficientNet only) + slib
    X_oof_v8 = np.hstack([X_base_v8_train, X_slib_oof])
    X_test_v8 = np.hstack([X_base_v8_test, X_slib_test])
    X_all_v8 = np.hstack([X_base_v8_train, X_slib_all])

    print(f"  Full features: {X_oof_full.shape[1]} | MI-selected: {X_oof_sel.shape[1]} | v8-compat: {X_oof_v8.shape[1]}")

    # === Phase 4: Train diverse models on different feature sets ===
    print("\n" + "=" * 60)
    print("Phase 4: Training diverse models")
    print("=" * 60)

    results = {}
    lgb_base = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                "verbose": -1, "random_state": SEED}
    xgb_base = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                "tree_method": "hist", "random_state": SEED, "verbosity": 0}

    # --- Models on FULL features (all CNN) ---
    print("\n  Models on full features (all CNN)...")
    lgb_configs = [
        ("f_lgb_a", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                     "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                     "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
        ("f_lgb_b", {**lgb_base, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4,
                     "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
                     "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000}),
        ("f_lgb_c", {**lgb_base, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                     "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
                     "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500}),
        ("f_lgb_d", {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                     "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                     "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
        # Optuna-tuned params from v10
        ("f_lgb_t", {**lgb_base, "learning_rate": 0.011, "num_leaves": 33, "max_depth": 4,
                     "min_child_samples": 10, "subsample": 0.52, "colsample_bytree": 0.40,
                     "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
    ]
    for name, params in lgb_configs:
        r = train_lgb_cv(params, X_oof_full, X_all_full, X_test_full, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    # Health-weighted on full features
    health_sw = np.ones(n_train)
    for i, y in enumerate(y_train):
        if y == 0: health_sw[i] = 1.5
    r = train_lgb_cv(lgb_configs[0][1], X_oof_full, X_all_full, X_test_full, y_train, skf,
                     X_base_train, sample_weight=health_sw, name="f_lgb_hw")
    print(f"  f_lgb_hw: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    results["f_lgb_hw"] = r

    for name, params in [
        ("f_xgb_a", {**xgb_base, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
                     "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
                     "reg_lambda": 1.0, "n_estimators": 2000}),
        ("f_xgb_b", {**xgb_base, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
                     "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
                     "reg_lambda": 2.0, "n_estimators": 3000}),
    ]:
        r = train_xgb_cv(params, X_oof_full, X_all_full, X_test_full, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    for name, cls, params in [
        ("f_et", ExtraTreesClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
        ("f_rf", RandomForestClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
    ]:
        r = train_sklearn_cv(cls, params, X_oof_full, X_all_full, X_test_full, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    # --- Models on MI-selected features ---
    print("\n  Models on MI-selected features...")
    for name, params in [
        ("s_lgb_a", lgb_configs[0][1]),
        ("s_lgb_d", lgb_configs[3][1]),
        ("s_lgb_t", lgb_configs[4][1]),
    ]:
        r = train_lgb_cv(params, X_oof_sel, X_all_sel, X_test_sel, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    r = train_lgb_cv(lgb_configs[0][1], X_oof_sel, X_all_sel, X_test_sel, y_train, skf,
                     X_base_train, sample_weight=health_sw, name="s_lgb_hw")
    print(f"  s_lgb_hw: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    results["s_lgb_hw"] = r

    # --- Models on v8-compatible features (EfficientNet only) ---
    print("\n  Models on v8-compat features (EfficientNet CNN only)...")
    for name_suffix, params in [("lgb_d", lgb_configs[3][1]), ("lgb_t", lgb_configs[4][1])]:
        name = f"v8_{name_suffix}"
        r = train_lgb_cv(params, X_oof_v8, X_all_v8, X_test_v8, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    r = train_lgb_cv(lgb_configs[0][1], X_oof_v8, X_all_v8, X_test_v8, y_train, skf,
                     X_base_train, sample_weight=health_sw, name="v8_lgb_hw")
    print(f"  v8_lgb_hw: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    results["v8_lgb_hw"] = r

    # === Phase 5: Seed diversity ===
    print("\n" + "=" * 60)
    print("Phase 5: Seed diversity")
    print("=" * 60)

    for seed_off in [100, 200, 300, 400, 500]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_off)
        seed_params = {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                       "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                       "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000,
                       "random_state": SEED + seed_off}

        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf_s.split(X_base_train, y_train)):
            lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
            slib_vai = np.array([list(library_features(train_spectra[i], lib).values()) for i in vai]).astype(np.float32)
            slib_tri = np.array([list(library_features(train_spectra[i], lib).values()) for i in tri]).astype(np.float32)
            X_tri = np.hstack([X_base_train[tri], slib_tri])
            X_vai = np.hstack([X_base_train[vai], slib_vai])

            m = lgb.LGBMClassifier(**seed_params)
            m.fit(X_tri, y_train[tri], eval_set=[(X_vai, y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_vai)

            m2 = lgb.LGBMClassifier(**seed_params)
            m2.fit(X_all_full[tri], y_train[tri], eval_set=[(X_all_full[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test_full) / N_FOLDS

        name = f"lgb_fs{seed_off}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Also v8-compat seed diversity (for cross-diversity with full feature models)
    for seed_off in [100, 200, 300]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_off)
        seed_params = {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                       "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                       "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000,
                       "random_state": SEED + seed_off}

        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf_s.split(X_base_train, y_train)):
            lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
            slib_vai = np.array([list(library_features(train_spectra[i], lib).values()) for i in vai]).astype(np.float32)
            slib_tri = np.array([list(library_features(train_spectra[i], lib).values()) for i in tri]).astype(np.float32)
            X_tri_v8 = np.hstack([X_base_v8_train[tri], slib_tri])
            X_vai_v8 = np.hstack([X_base_v8_train[vai], slib_vai])

            m = lgb.LGBMClassifier(**seed_params)
            m.fit(X_tri_v8, y_train[tri], eval_set=[(X_vai_v8, y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_vai_v8)

            m2 = lgb.LGBMClassifier(**seed_params)
            m2.fit(X_all_v8[tri], y_train[tri], eval_set=[(X_all_v8[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test_v8) / N_FOLDS

        name = f"lgb_v8s{seed_off}"
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
    X_oof_pca = pca_stack.fit_transform(X_oof_full)
    X_test_pca = pca_stack.transform(X_test_full)
    stack_oof_aug = np.hstack([stack_oof, X_oof_pca])
    stack_test_aug = np.hstack([stack_test, X_test_pca])

    for lr, nl, name in [(0.05, 15, "stack_lgb_a"), (0.03, 20, "stack_lgb_b"), (0.02, 10, "stack_lgb_c")]:
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
    print("Phase 7: Optuna ensemble (5000 trials)")
    print("=" * 60)

    all_models = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    print(f"\n  Total models: {len(all_models)}")
    print("\n  Top 20 models:")
    for m in all_models[:20]:
        print(f"    {m:20s} Acc={results[m]['acc']:.4f} F1={results[m]['f1']:.4f}")

    ens_oofs = [results[m]["oof"] for m in all_models]
    ens_tests = [results[m]["test"] for m in all_models]

    def objective_ens(trial):
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in all_models]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
        weights /= ws
        oof_e = sum(w * o for w, o in zip(weights, ens_oofs))
        return accuracy_score(y_train, oof_e.argmax(1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_ens, n_trials=5000, show_progress_bar=False)

    best_weights = np.array([study.best_params[f"w_{m}"] for m in all_models])
    best_weights /= best_weights.sum()
    optuna_oof = sum(w * o for w, o in zip(best_weights, ens_oofs))
    optuna_test = sum(w * o for w, o in zip(best_weights, ens_tests))
    acc = accuracy_score(y_train, optuna_oof.argmax(1))
    f1 = f1_score(y_train, optuna_oof.argmax(1), average="macro")
    print(f"\n  Optuna ensemble: Acc={acc:.4f} F1={f1:.4f}")
    sig_w = {m: f"{w:.3f}" for m, w in zip(all_models, best_weights) if w > 0.01}
    print(f"  Significant weights: {sig_w}")

    # === Phase 8: Cross-version blending ===
    print("\n" + "=" * 60)
    print("Phase 8: Cross-version blending")
    print("=" * 60)

    cross_oofs = {"v11": optuna_oof}
    cross_tests = {"v11": optuna_test}

    for vname, oof_path, test_path in [
        ("v4", "v4_oof_probs.npy", "v4_test_probs.npy"),
        ("v8", "v8_best_oof_probs.npy", "v8_best_test_probs.npy"),
        ("v10", "v10_best_oof_probs.npy", "v10_best_test_probs.npy"),
    ]:
        op = os.path.join(OUT_DIR, oof_path)
        tp = os.path.join(OUT_DIR, test_path)
        if os.path.exists(op) and os.path.exists(tp):
            cross_oofs[vname] = np.load(op)
            cross_tests[vname] = np.load(tp)
            print(f"  Loaded {vname} predictions")

    cross_names = sorted(cross_oofs.keys())
    cross_oof_list = [cross_oofs[n] for n in cross_names]
    cross_test_list = [cross_tests[n] for n in cross_names]

    def objective_cross(trial):
        weights = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in cross_names]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
        weights /= ws
        blend = sum(w * o for w, o in zip(weights, cross_oof_list))
        return accuracy_score(y_train, blend.argmax(1))

    study_cross = optuna.create_study(direction="maximize",
                                       sampler=optuna.samplers.TPESampler(seed=SEED + 3))
    study_cross.optimize(objective_cross, n_trials=3000, show_progress_bar=False)

    cross_weights = np.array([study_cross.best_params[f"w_{n}"] for n in cross_names])
    cross_weights /= cross_weights.sum()
    cross_oof = sum(w * o for w, o in zip(cross_weights, cross_oof_list))
    cross_test = sum(w * o for w, o in zip(cross_weights, cross_test_list))
    cross_acc = accuracy_score(y_train, cross_oof.argmax(1))
    cross_f1 = f1_score(y_train, cross_oof.argmax(1), average="macro")
    print(f"  Cross blend: Acc={cross_acc:.4f} F1={cross_f1:.4f}")
    print(f"  Weights: {dict(zip(cross_names, [f'{w:.3f}' for w in cross_weights]))}")

    # === Phase 9: Threshold optimization ===
    print("\n" + "=" * 60)
    print("Phase 9: Threshold optimization")
    print("=" * 60)

    # Choose best starting point
    candidates = {"optuna": (optuna_oof, optuna_test),
                  "cross_blend": (cross_oof, cross_test)}
    best_name = max(candidates, key=lambda n: accuracy_score(y_train, candidates[n][0].argmax(1)))
    best_oof, best_test = candidates[best_name]
    best_acc = accuracy_score(y_train, best_oof.argmax(1))
    print(f"  Starting from: {best_name} (Acc={best_acc:.4f})")

    improved = False
    for bias_h in np.arange(-0.15, 0.16, 0.01):
        for bias_r in np.arange(-0.10, 0.11, 0.01):
            bias_o = -(bias_h + bias_r)
            adjusted = best_oof.copy()
            adjusted[:, 0] += bias_h
            adjusted[:, 1] += bias_r
            adjusted[:, 2] += bias_o
            ta = accuracy_score(y_train, adjusted.argmax(1))
            if ta > best_acc:
                best_acc = ta
                best_oof = adjusted.copy()
                best_test = candidates[best_name][1].copy()
                best_test[:, 0] += bias_h
                best_test[:, 1] += bias_r
                best_test[:, 2] += bias_o
                improved = True
                print(f"  Threshold: H={bias_h:+.2f} R={bias_r:+.2f} O={bias_o:+.2f} -> Acc={ta:.4f}")
                best_name += "+thresh"

    if not improved:
        print("  No threshold improvement.")

    # === Final results ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    final_acc = accuracy_score(y_train, best_oof.argmax(1))
    final_f1 = f1_score(y_train, best_oof.argmax(1), average="macro")
    print(f"\n*** Best: {best_name} | Acc={final_acc:.4f} F1={final_f1:.4f} ***")
    print(classification_report(y_train, best_oof.argmax(1), target_names=LABELS))

    # Save
    np.save(os.path.join(OUT_DIR, "v11_best_oof_probs.npy"), best_oof)
    np.save(os.path.join(OUT_DIR, "v11_best_test_probs.npy"), best_test)

    test_preds = best_test.argmax(1)
    result_csv = os.path.join(ROOT, "result.csv")
    if os.path.exists(result_csv):
        template = pd.read_csv(result_csv)
    else:
        template = pd.DataFrame({"filename": val_base_ids})

    template["predict"] = [ID2LBL[p] for p in test_preds]
    sub_name = f"submission_v11_acc_{final_acc:.4f}_f1_{final_f1:.4f}".replace(".", "p") + ".csv"
    sub_path = os.path.join(OUT_DIR, sub_name)
    template.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    elapsed = time.time() - t0
    print(f"Done! ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
