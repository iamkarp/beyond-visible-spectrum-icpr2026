"""
Beyond Visible Spectrum v12 - Hierarchical classification + feature reduction

The 0.765 ceiling comes from: Health recall at 55.5% (89 misclassified of 200).
All models agree on the same errors → ensemble can't fix them.

New strategies:
1. Hierarchical classification: Stage 1 (Health vs Rest) → Stage 2 (Rust vs Other)
   - A dedicated binary classifier can focus entirely on the Health boundary
   - Binary tasks are easier than 3-class → better per-class accuracy
2. Aggressive feature selection via LGB importance → reduce to 200-300 features
   - Directly addresses curse of dimensionality (600 samples vs 907 features)
3. Combine hierarchical predictions with flat 3-class predictions
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
# Spectral library
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
            try: features[f"sl_{label}_mahal"] = np.sqrt(np.clip(diff @ lib["cov_inv"] @ diff, 0, None))
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
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("Beyond Visible Spectrum v12 - Hierarchical + feature reduction")
    print("=" * 60)

    # === Phase 1: Load features ===
    print("\nPhase 1: Loading features...")
    feat_data = np.load(os.path.join(OUT_DIR, "v8_features.npz"), allow_pickle=True)
    X_hand_train = feat_data["X_hand_train"]
    X_hand_test = feat_data["X_hand_test"]
    train_cnn = feat_data["train_cnn"]
    val_cnn = feat_data["val_cnn"]

    X_base_train = np.hstack([X_hand_train, train_cnn])
    X_base_test = np.hstack([X_hand_test, val_cnn])

    # Labels
    train_files = sorted(os.listdir(os.path.join(ROOT, "train", "RGB")))
    train_files = [f for f in train_files if f.lower().endswith((".png", ".jpg"))]
    y_labels, train_bids = [], []
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
    print(f"  Train: {n_train} | Val: {n_val} | Features: {X_base_train.shape[1]}")

    # HS spectra for spectral library
    print("  Loading HS spectra...")
    HS_DROP_FIRST, HS_DROP_LAST, HS_TARGET_CH = 10, 14, 101
    import tifffile as tiff
    def read_tiff(path):
        arr = tiff.imread(path)
        if arr.ndim != 3: raise ValueError(f"Expected 3D")
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)

    train_spectra, val_spectra = [], []
    for bid in train_bids:
        hs_path = os.path.join(ROOT, "train", "HS", bid + ".tif")
        if os.path.exists(hs_path):
            arr = read_tiff(hs_path)
            B = arr.shape[2]
            arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST] if B > (HS_DROP_FIRST + HS_DROP_LAST + 1) else arr
            C = arr.shape[2]
            arr = arr[:, :, :HS_TARGET_CH] if C > HS_TARGET_CH else arr
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
            arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST] if B > (HS_DROP_FIRST + HS_DROP_LAST + 1) else arr
            C = arr.shape[2]
            arr = arr[:, :, :HS_TARGET_CH] if C > HS_TARGET_CH else arr
            spec = np.mean(arr.reshape(-1, min(C, HS_TARGET_CH)), axis=0)
            if len(spec) < HS_TARGET_CH: spec = np.pad(spec, (0, HS_TARGET_CH - len(spec)))
            val_spectra.append(spec)
        else:
            val_spectra.append(np.zeros(HS_TARGET_CH))

    # === Phase 2: CV spectral library ===
    print("\nPhase 2: CV spectral library...")
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

    # === Phase 3: Feature importance selection ===
    print("\nPhase 3: Feature importance selection...")

    # Train one LGB to get importance
    imp_model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=25,
        max_depth=5, min_child_samples=10, subsample=0.75, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=0.5, n_estimators=500, verbose=-1, random_state=SEED
    )
    imp_model.fit(X_oof, y_train)
    importances = imp_model.feature_importances_

    # Select top K features by importance
    for top_k in [200, 300, 400, 500]:
        top_idx = np.argsort(importances)[::-1][:top_k]
        mask = np.zeros(X_oof.shape[1], dtype=bool)
        mask[top_idx] = True
        print(f"  Top {top_k}: features selected")

    # We'll use multiple reduced feature sets
    feat_sets = {}
    for top_k in [200, 300, 400, 500]:
        top_idx = np.argsort(importances)[::-1][:top_k]
        mask = np.zeros(X_oof.shape[1], dtype=bool)
        mask[top_idx] = True
        feat_sets[top_k] = {
            "X_oof": X_oof[:, mask],
            "X_test": X_test[:, mask],
            "X_all": X_all[:, mask],
        }

    # === Phase 4: Flat 3-class models on different feature sets ===
    print("\n" + "=" * 60)
    print("Phase 4: Flat 3-class models on different feature sets")
    print("=" * 60)

    results = {}
    lgb_base = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                "verbose": -1, "random_state": SEED}

    # Health weight
    health_sw = np.ones(n_train)
    for i, y in enumerate(y_train):
        if y == 0: health_sw[i] = 1.5

    configs = [
        ("lgb_d", {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                   "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                   "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
        ("lgb_a", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                   "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                   "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
        ("lgb_t", {**lgb_base, "learning_rate": 0.011, "num_leaves": 33, "max_depth": 4,
                   "min_child_samples": 10, "subsample": 0.52, "colsample_bytree": 0.40,
                   "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
    ]

    # Full features (928)
    print("\n  Full features (928)...")
    for cname, params in configs:
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
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
        print(f"  full_{cname}: Acc={acc:.4f} F1={f1:.4f}")
        results[f"full_{cname}"] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Health-weighted
    oof_p = np.zeros((n_train, 3))
    test_p = np.zeros((n_val, 3))
    hw_params = configs[1][1]  # lgb_a params
    for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
        m = lgb.LGBMClassifier(**hw_params)
        m.fit(X_oof[tri], y_train[tri], sample_weight=health_sw[tri],
              eval_set=[(X_oof[vai], y_train[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_p[vai] = m.predict_proba(X_oof[vai])
        m2 = lgb.LGBMClassifier(**hw_params)
        m2.fit(X_all[tri], y_train[tri], sample_weight=health_sw[tri],
               eval_set=[(X_all[vai], y_train[vai])],
               callbacks=[lgb.early_stopping(100, verbose=False)])
        test_p += m2.predict_proba(X_test) / N_FOLDS
    acc = accuracy_score(y_train, oof_p.argmax(1))
    f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
    print(f"  full_lgb_hw: Acc={acc:.4f} F1={f1:.4f}")
    results["full_lgb_hw"] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Reduced feature sets
    for top_k in [200, 300, 400, 500]:
        fs = feat_sets[top_k]
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        params = configs[0][1]  # lgb_d
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(**params)
            m.fit(fs["X_oof"][tri], y_train[tri],
                  eval_set=[(fs["X_oof"][vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(fs["X_oof"][vai])
            m2 = lgb.LGBMClassifier(**params)
            m2.fit(fs["X_all"][tri], y_train[tri],
                   eval_set=[(fs["X_all"][vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(fs["X_test"]) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  top{top_k}_lgb_d: Acc={acc:.4f} F1={f1:.4f}")
        results[f"top{top_k}_lgb_d"] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # XGB on full
    xgb_base = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                "tree_method": "hist", "random_state": SEED, "verbosity": 0}
    for cname, params in [
        ("xgb_a", {**xgb_base, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
                   "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
                   "reg_lambda": 1.0, "n_estimators": 2000}),
        ("xgb_b", {**xgb_base, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
                   "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
                   "reg_lambda": 2.0, "n_estimators": 3000}),
    ]:
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = xgb.XGBClassifier(**params)
            m.fit(X_oof[tri], y_train[tri], eval_set=[(X_oof[vai], y_train[vai])], verbose=False)
            oof_p[vai] = m.predict_proba(X_oof[vai])
            m2 = xgb.XGBClassifier(**params)
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])], verbose=False)
            test_p += m2.predict_proba(X_test) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  full_{cname}: Acc={acc:.4f} F1={f1:.4f}")
        results[f"full_{cname}"] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # ET/RF
    for name, cls, params in [
        ("et", ExtraTreesClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
        ("rf", RandomForestClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
    ]:
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = cls(**params)
            m.fit(X_oof[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(X_oof[vai])
            m2 = cls(**params)
            m2.fit(X_all[tri], y_train[tri])
            test_p += m2.predict_proba(X_test) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  full_{name}: Acc={acc:.4f} F1={f1:.4f}")
        results[f"full_{name}"] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 5: Hierarchical classification ===
    print("\n" + "=" * 60)
    print("Phase 5: HIERARCHICAL classification")
    print("=" * 60)

    # Stage 1: Health (0) vs Rest (1)
    y_binary = (y_train > 0).astype(int)  # 0=Health, 1=Rest
    print(f"\n  Stage 1: Health vs Rest (Health={np.sum(y_binary==0)}, Rest={np.sum(y_binary==1)})")

    lgb_bin = {"objective": "binary", "metric": "binary_logloss", "verbose": -1, "random_state": SEED}

    stage1_configs = [
        ("s1_lgb_a", {**lgb_bin, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                      "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                      "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
        ("s1_lgb_d", {**lgb_bin, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                      "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                      "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
        ("s1_lgb_hw", {**lgb_bin, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                       "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                       "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000,
                       "scale_pos_weight": 0.5}),  # Upweight Health (negative class)
    ]

    stage1_results = {}
    for cname, params in stage1_configs:
        oof_p1 = np.zeros(n_train)  # P(Rest)
        test_p1 = np.zeros(n_val)
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_oof[tri], y_binary[tri],
                  eval_set=[(X_oof[vai], y_binary[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p1[vai] = m.predict_proba(X_oof[vai])[:, 1]
            m2 = lgb.LGBMClassifier(**params)
            m2.fit(X_all[tri], y_binary[tri],
                   eval_set=[(X_all[vai], y_binary[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p1 += m2.predict_proba(X_test)[:, 1] / N_FOLDS

        # Binary accuracy
        bin_acc = accuracy_score(y_binary, (oof_p1 > 0.5).astype(int))
        health_recall = np.mean(oof_p1[y_binary == 0] < 0.5)  # Correctly identified as Health
        rest_recall = np.mean(oof_p1[y_binary == 1] >= 0.5)    # Correctly identified as Rest
        print(f"  {cname}: BinAcc={bin_acc:.4f} Health_recall={health_recall:.4f} Rest_recall={rest_recall:.4f}")
        stage1_results[cname] = {"oof": oof_p1, "test": test_p1}

    # Stage 2: Rust (0) vs Other (1) on Rest samples only
    rest_mask = y_train > 0
    y_rustother = (y_train[rest_mask] - 1).astype(int)  # 0=Rust, 1=Other
    print(f"\n  Stage 2: Rust vs Other (Rust={np.sum(y_rustother==0)}, Other={np.sum(y_rustother==1)})")

    stage2_configs = [
        ("s2_lgb_a", {**lgb_bin, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                      "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                      "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
        ("s2_lgb_d", {**lgb_bin, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                      "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                      "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
    ]

    stage2_results = {}
    # Need to handle the fact that stage 2 only trains on Rest samples
    # But we need OOF predictions for ALL samples (for when stage 1 classifies them as Rest)
    for cname, params in stage2_configs:
        oof_p2 = np.zeros(n_train)  # P(Other | Rest)
        test_p2 = np.zeros(n_val)
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            # Only use Rest samples in this fold for training
            tri_rest = np.array([i for i in tri if y_train[i] > 0])
            vai_all = vai  # Predict on ALL validation samples

            y_tr2 = (y_train[tri_rest] - 1).astype(int)  # Rust=0, Other=1
            m = lgb.LGBMClassifier(**params)
            m.fit(X_oof[tri_rest], y_tr2,
                  eval_set=[(X_oof[tri_rest[:20]], y_tr2[:20])],  # minimal eval set
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p2[vai_all] = m.predict_proba(X_oof[vai_all])[:, 1]

            # For test: train on all Rest samples
            all_rest = np.array([i for i in tri if y_train[i] > 0])
            y_all2 = (y_train[all_rest] - 1).astype(int)
            m2 = lgb.LGBMClassifier(**params)
            m2.fit(X_all[all_rest], y_all2,
                   eval_set=[(X_all[all_rest[:20]], y_all2[:20])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p2 += m2.predict_proba(X_test)[:, 1] / N_FOLDS

        # Accuracy on Rest samples only
        rest_preds = (oof_p2[rest_mask] > 0.5).astype(int)
        s2_acc = accuracy_score(y_rustother, rest_preds)
        print(f"  {cname}: RustOther Acc={s2_acc:.4f}")
        stage2_results[cname] = {"oof": oof_p2, "test": test_p2}

    # Combine hierarchical predictions into 3-class probabilities
    print("\n  Combining hierarchical predictions...")
    hier_results = {}

    for s1name in stage1_results:
        for s2name in stage2_results:
            p_rest = stage1_results[s1name]["oof"]  # P(Rest)
            p_other_given_rest = stage2_results[s2name]["oof"]  # P(Other | Rest)

            # 3-class probabilities
            oof_3class = np.zeros((n_train, 3))
            oof_3class[:, 0] = 1 - p_rest           # P(Health)
            oof_3class[:, 1] = p_rest * (1 - p_other_given_rest)  # P(Rust) = P(Rest) * P(Rust|Rest)
            oof_3class[:, 2] = p_rest * p_other_given_rest         # P(Other) = P(Rest) * P(Other|Rest)

            # Test
            p_rest_t = stage1_results[s1name]["test"]
            p_other_t = stage2_results[s2name]["test"]
            test_3class = np.zeros((n_val, 3))
            test_3class[:, 0] = 1 - p_rest_t
            test_3class[:, 1] = p_rest_t * (1 - p_other_t)
            test_3class[:, 2] = p_rest_t * p_other_t

            acc = accuracy_score(y_train, oof_3class.argmax(1))
            f1 = f1_score(y_train, oof_3class.argmax(1), average="macro")
            hname = f"hier_{s1name}_{s2name}"
            print(f"  {hname}: Acc={acc:.4f} F1={f1:.4f}")

            # Per-class
            hr = np.mean(oof_3class.argmax(1)[y_train == 0] == 0)
            rr = np.mean(oof_3class.argmax(1)[y_train == 1] == 1)
            orr = np.mean(oof_3class.argmax(1)[y_train == 2] == 2)
            print(f"    Health recall={hr:.3f} Rust recall={rr:.3f} Other recall={orr:.3f}")

            results[hname] = {"oof": oof_3class, "test": test_3class, "acc": acc, "f1": f1}

    # === Phase 6: Seed diversity ===
    print("\n" + "=" * 60)
    print("Phase 6: Seed diversity")
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
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test) / N_FOLDS

        name = f"lgb_s{seed_off}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 7: Stacking ===
    print("\n" + "=" * 60)
    print("Phase 7: Stacking")
    print("=" * 60)

    model_names = sorted(results.keys())
    stack_oof = np.hstack([results[m]["oof"] for m in model_names])
    stack_test = np.hstack([results[m]["test"] for m in model_names])
    pca_stack = PCA(n_components=50, random_state=SEED)
    X_oof_pca = pca_stack.fit_transform(X_oof)
    X_test_pca = pca_stack.transform(X_test)
    stack_oof_aug = np.hstack([stack_oof, X_oof_pca])
    stack_test_aug = np.hstack([stack_test, X_test_pca])

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

    # === Phase 8: Optuna ensemble ===
    print("\n" + "=" * 60)
    print("Phase 8: Optuna ensemble (5000 trials)")
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

    # === Phase 9: Cross-version blending ===
    print("\n" + "=" * 60)
    print("Phase 9: Cross-version blending")
    print("=" * 60)

    cross_oofs = {"v12": optuna_oof}
    cross_tests = {"v12": optuna_test}
    for vname, oof_path, test_path in [
        ("v4", "v4_oof_probs.npy", "v4_test_probs.npy"),
        ("v8", "v8_best_oof_probs.npy", "v8_best_test_probs.npy"),
        ("v10", "v10_best_oof_probs.npy", "v10_best_test_probs.npy"),
    ]:
        op, tp = os.path.join(OUT_DIR, oof_path), os.path.join(OUT_DIR, test_path)
        if os.path.exists(op) and os.path.exists(tp):
            cross_oofs[vname] = np.load(op)
            cross_tests[vname] = np.load(tp)
            print(f"  Loaded {vname}")

    cross_names = sorted(cross_oofs.keys())
    cross_oof_list = [cross_oofs[n] for n in cross_names]
    cross_test_list = [cross_tests[n] for n in cross_names]

    def objective_cross(trial):
        weights = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in cross_names]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
        weights /= ws
        return accuracy_score(y_train, sum(w * o for w, o in zip(weights, cross_oof_list)).argmax(1))

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

    # === Phase 10: Threshold optimization ===
    print("\n" + "=" * 60)
    print("Phase 10: Threshold optimization")
    print("=" * 60)

    candidates = {"optuna": (optuna_oof, optuna_test), "cross": (cross_oof, cross_test)}
    best_name = max(candidates, key=lambda n: accuracy_score(y_train, candidates[n][0].argmax(1)))
    best_oof, best_test = candidates[best_name]
    best_acc = accuracy_score(y_train, best_oof.argmax(1))
    print(f"  Starting from: {best_name} (Acc={best_acc:.4f})")

    improved = False
    for bias_h in np.arange(-0.15, 0.16, 0.01):
        for bias_r in np.arange(-0.10, 0.11, 0.01):
            bias_o = -(bias_h + bias_r)
            adjusted = best_oof.copy()
            adjusted[:, 0] += bias_h; adjusted[:, 1] += bias_r; adjusted[:, 2] += bias_o
            ta = accuracy_score(y_train, adjusted.argmax(1))
            if ta > best_acc:
                best_acc = ta
                best_oof = adjusted.copy()
                best_test = candidates[best_name][1].copy()
                best_test[:, 0] += bias_h; best_test[:, 1] += bias_r; best_test[:, 2] += bias_o
                improved = True
                print(f"  H={bias_h:+.2f} R={bias_r:+.2f} O={bias_o:+.2f} -> Acc={ta:.4f}")
                best_name += "+thresh"

    if not improved:
        print("  No threshold improvement.")

    # === Final ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    final_acc = accuracy_score(y_train, best_oof.argmax(1))
    final_f1 = f1_score(y_train, best_oof.argmax(1), average="macro")
    print(f"\n*** Best: {best_name} | Acc={final_acc:.4f} F1={final_f1:.4f} ***")
    print(classification_report(y_train, best_oof.argmax(1), target_names=LABELS))

    np.save(os.path.join(OUT_DIR, "v12_best_oof_probs.npy"), best_oof)
    np.save(os.path.join(OUT_DIR, "v12_best_test_probs.npy"), best_test)

    test_preds = best_test.argmax(1)
    result_csv = os.path.join(ROOT, "result.csv")
    template = pd.read_csv(result_csv) if os.path.exists(result_csv) else pd.DataFrame({"filename": val_base_ids})
    template["predict"] = [ID2LBL[p] for p in test_preds]
    sub_name = f"submission_v12_acc_{final_acc:.4f}_f1_{final_f1:.4f}".replace(".", "p") + ".csv"
    sub_path = os.path.join(OUT_DIR, sub_name)
    template.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")
    print(f"Done! ({(time.time() - t0) / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
