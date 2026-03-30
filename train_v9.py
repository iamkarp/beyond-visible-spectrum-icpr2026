"""
Beyond Visible Spectrum v9 - Maximum model diversity

Key insight from v8: Optuna ensemble went from 0.7367 (best individual) to 0.7650 (ensemble).
Seed diversity models got highest weights. More diversity = better ensemble.

Strategy:
1. Load cached features from v8 (skip extraction - same 907 features)
2. Train 30+ diverse models via seed/hyperparameter/fold variations
3. Add SVM models (fast on selected features)
4. Run 5000 Optuna trials
5. Cross-version blend with all available predictions
"""

import os, re, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb
import xgboost as xgb
import optuna
import tifffile as tiff

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = "/Users/macbook/Library/CloudStorage/GoogleDrive-jason.karpeles@pmg.com/My Drive/Projects/Beyond Visible Spectrum"
OUT_DIR = os.path.join(ROOT, "output")
N_FOLDS = 5
SEED = 42
HS_TARGET_CH = 101

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

np.random.seed(SEED)


def spectral_angle(s1, s2):
    cos = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


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


def read_tiff(path):
    arr = tiff.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D, got {arr.shape}")
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def main():
    print("=" * 60)
    print("Beyond Visible Spectrum v9 - Max diversity ensemble")
    print("=" * 60)

    # === Phase 1: Load cached features ===
    print("\nPhase 1: Loading cached features from v8...")
    cache = np.load(os.path.join(OUT_DIR, "v8_features.npz"), allow_pickle=True)
    X_hand_train = cache["X_hand_train"]
    X_hand_test = cache["X_hand_test"]
    train_cnn = cache["train_cnn"]
    val_cnn = cache["val_cnn"]
    base_cols = cache["base_cols"]

    X_base_train = np.hstack([X_hand_train, train_cnn])
    X_base_test = np.hstack([X_hand_test, val_cnn])
    n_train = X_base_train.shape[0]
    n_val = X_base_test.shape[0]
    print(f"  Train: {n_train} | Val: {n_val} | Features: {X_base_train.shape[1]}")

    # Load labels
    def parse_label(bid):
        import re
        m = re.match(r"^(Health|Rust|Other)_", bid)
        return m.group(1) if m else None

    def build_index(root, split):
        split_dir = os.path.join(root, split)
        idx = {}
        for mod, exts in [("rgb", (".png", ".jpg")), ("ms", (".tif", ".tiff")), ("hs", (".tif", ".tiff"))]:
            folder = os.path.join(split_dir, mod.upper())
            if os.path.isdir(folder):
                for f in sorted(os.listdir(folder)):
                    if f.lower().endswith(exts):
                        bid = os.path.splitext(f)[0]
                        idx.setdefault(bid, {})[mod] = os.path.join(folder, f)
        return idx

    train_idx = build_index(ROOT, "train")
    val_idx = build_index(ROOT, "val")

    train_bids = sorted([bid for bid in train_idx.keys() if parse_label(bid)])
    val_bids = sorted(val_idx.keys())
    train_labels = [parse_label(bid) for bid in train_bids]
    y_train = np.array([LBL2ID[l] for l in train_labels])

    # Load train spectra for spectral library
    print("  Loading HS spectra for spectral library...")
    train_spectra = []
    for bid in train_bids:
        hs_path = train_idx[bid].get("hs")
        if hs_path:
            arr = read_tiff(hs_path)
            B = arr.shape[2]
            if B > 25:
                arr = arr[:, :, 10:B - 14]
            C = arr.shape[2]
            if C > HS_TARGET_CH:
                arr = arr[:, :, :HS_TARGET_CH]
            elif C < HS_TARGET_CH:
                pad = np.zeros((arr.shape[0], arr.shape[1], HS_TARGET_CH - C), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=2)
            spec = np.mean(arr.reshape(-1, HS_TARGET_CH), axis=0)
        else:
            spec = None
        train_spectra.append(spec)

    val_spectra = []
    for bid in val_bids:
        hs_path = val_idx[bid].get("hs")
        if hs_path:
            arr = read_tiff(hs_path)
            B = arr.shape[2]
            if B > 25:
                arr = arr[:, :, 10:B - 14]
            C = arr.shape[2]
            if C > HS_TARGET_CH:
                arr = arr[:, :, :HS_TARGET_CH]
            elif C < HS_TARGET_CH:
                pad = np.zeros((arr.shape[0], arr.shape[1], HS_TARGET_CH - C), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=2)
            spec = np.mean(arr.reshape(-1, HS_TARGET_CH), axis=0)
        else:
            spec = None
        val_spectra.append(spec)

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

    # Feature selection for SVM/kNN (MI-based, top 300)
    print("  Feature selection for SVM/kNN...")
    mi = mutual_info_classif(X_all, y_train, random_state=SEED, n_neighbors=5)
    top_300_idx = np.argsort(mi)[::-1][:300]
    X_sel_oof = X_oof[:, top_300_idx]
    X_sel_test = X_test[:, top_300_idx]
    X_sel_all = X_all[:, top_300_idx]

    scaler = StandardScaler()
    X_sel_oof_sc = scaler.fit_transform(X_sel_oof)
    X_sel_test_sc = scaler.transform(X_sel_test)
    scaler_all = StandardScaler()
    X_sel_all_sc = scaler_all.fit_transform(X_sel_all)
    X_sel_test_sc_all = scaler_all.transform(X_sel_test)

    # === Phase 3: Train MANY diverse models ===
    print("\n" + "=" * 60)
    print("Phase 3: Training 30+ diverse models")
    print("=" * 60)

    results = {}
    lgb_base = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                "verbose": -1, "random_state": SEED}

    def train_lgb(name, params, X_tr, X_te, X_al, use_weight=False):
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        sw = np.ones(n_train)
        if use_weight:
            for i, y in enumerate(y_train):
                sw[i] = {0: 1.3, 1: 1.0, 2: 1.0}[y]
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
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
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    def train_xgb(name, params, X_tr, X_te, X_al):
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr[tri], y_train[tri], eval_set=[(X_tr[vai], y_train[vai])], verbose=False)
            oof_p[vai] = m.predict_proba(X_tr[vai])
            m2 = xgb.XGBClassifier(**params)
            m2.fit(X_al[tri], y_train[tri], eval_set=[(X_al[vai], y_train[vai])], verbose=False)
            test_p += m2.predict_proba(X_te) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # --- Core LGB models (5 configs) ---
    print("\n  Core LGB models...")
    train_lgb("lgb_a", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
              "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
              "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}, X_oof, X_test, X_all)

    train_lgb("lgb_b", {**lgb_base, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4,
              "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
              "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000}, X_oof, X_test, X_all)

    train_lgb("lgb_c", {**lgb_base, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
              "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500}, X_oof, X_test, X_all)

    train_lgb("lgb_d", {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
              "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
              "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}, X_oof, X_test, X_all)

    train_lgb("lgb_w", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
              "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
              "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}, X_oof, X_test, X_all, use_weight=True)

    # --- Additional LGB configs for diversity ---
    print("\n  Additional LGB configs...")
    train_lgb("lgb_e", {**lgb_base, "learning_rate": 0.015, "num_leaves": 12, "max_depth": 3,
              "min_child_samples": 25, "subsample": 0.65, "colsample_bytree": 0.35,
              "reg_alpha": 3.0, "reg_lambda": 3.0, "n_estimators": 5000}, X_oof, X_test, X_all)

    train_lgb("lgb_f", {**lgb_base, "learning_rate": 0.04, "num_leaves": 40, "max_depth": 7,
              "min_child_samples": 5, "subsample": 0.85, "colsample_bytree": 0.8,
              "reg_alpha": 0.1, "reg_lambda": 0.1, "n_estimators": 1000}, X_oof, X_test, X_all)

    train_lgb("lgb_g", {**lgb_base, "learning_rate": 0.025, "num_leaves": 18, "max_depth": 5,
              "min_child_samples": 12, "subsample": 0.7, "colsample_bytree": 0.55,
              "reg_alpha": 0.8, "reg_lambda": 0.8, "n_estimators": 2500}, X_oof, X_test, X_all)

    # --- XGBoost models ---
    print("\n  XGBoost models...")
    xgb_base = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                "tree_method": "hist", "random_state": SEED, "verbosity": 0}

    train_xgb("xgb_a", {**xgb_base, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
              "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
              "reg_lambda": 1.0, "n_estimators": 2000}, X_oof, X_test, X_all)

    train_xgb("xgb_b", {**xgb_base, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
              "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
              "reg_lambda": 2.0, "n_estimators": 3000}, X_oof, X_test, X_all)

    train_xgb("xgb_c", {**xgb_base, "learning_rate": 0.01, "max_depth": 3, "min_child_weight": 12,
              "subsample": 0.7, "colsample_bytree": 0.4, "reg_alpha": 2.0,
              "reg_lambda": 3.0, "n_estimators": 5000}, X_oof, X_test, X_all)

    # --- sklearn ensemble ---
    print("\n  Sklearn ensemble models...")
    for name, model_cls, kw in [
        ("et", ExtraTreesClassifier, {"n_estimators": 2000, "min_samples_leaf": 2,
                                      "random_state": SEED, "n_jobs": -1}),
        ("rf", RandomForestClassifier, {"n_estimators": 2000, "min_samples_leaf": 2,
                                        "random_state": SEED, "n_jobs": -1}),
        ("et2", ExtraTreesClassifier, {"n_estimators": 3000, "min_samples_leaf": 3,
                                       "max_features": 0.5, "random_state": SEED + 1, "n_jobs": -1}),
        ("rf2", RandomForestClassifier, {"n_estimators": 3000, "min_samples_leaf": 3,
                                         "max_features": 0.5, "random_state": SEED + 1, "n_jobs": -1}),
    ]:
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
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

    # --- SVM models (fast on 300 selected features) ---
    print("\n  SVM models...")
    for C_val, gamma, name in [(1.0, "scale", "svm_1s"), (5.0, "scale", "svm_5s"),
                                (10.0, "scale", "svm_10s"), (1.0, 0.001, "svm_1g"),
                                (5.0, 0.001, "svm_5g")]:
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = SVC(C=C_val, kernel="rbf", gamma=gamma, probability=True, random_state=SEED)
            m.fit(X_sel_oof_sc[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(X_sel_oof_sc[vai])
            m2 = SVC(C=C_val, kernel="rbf", gamma=gamma, probability=True, random_state=SEED)
            m2.fit(X_sel_all_sc[tri], y_train[tri])
            test_p += m2.predict_proba(X_sel_test_sc_all) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # --- kNN models ---
    print("\n  kNN models...")
    for k, w in [(5, "distance"), (10, "distance"), (15, "distance"), (20, "distance")]:
        name = f"knn_{k}"
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1)
            m.fit(X_sel_oof_sc[tri], y_train[tri])
            oof_p[vai] = m.predict_proba(X_sel_oof_sc[vai])
            m2 = KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1)
            m2.fit(X_sel_all_sc[tri], y_train[tri])
            test_p += m2.predict_proba(X_sel_test_sc_all) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # --- Seed diversity (10 seeds) ---
    print("\n  Seed diversity models (10 seeds)...")
    for seed_off in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
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

    # --- Fold variation diversity (7-fold and 3-fold) ---
    print("\n  Fold variation models...")
    for n_f, seed_f in [(7, 77), (3, 33)]:
        skf_f = StratifiedKFold(n_splits=n_f, shuffle=True, random_state=seed_f)
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf_f.split(X_base_train, y_train)):
            lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
            slib_vai = np.array([list(library_features(train_spectra[i], lib).values()) for i in vai]).astype(np.float32)
            slib_tri = np.array([list(library_features(train_spectra[i], lib).values()) for i in tri]).astype(np.float32)
            X_tri = np.hstack([X_base_train[tri], slib_tri])
            X_vai = np.hstack([X_base_train[vai], slib_vai])
            params = {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                      "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                      "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000,
                      "random_state": seed_f}
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tri, y_train[tri], eval_set=[(X_vai, y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_vai)
            m2 = lgb.LGBMClassifier(**params)
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test) / n_f
        name = f"lgb_{n_f}f"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 4: Stacking ===
    print("\n" + "=" * 60)
    print("Phase 4: Stacking")
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

    for C_val, name in [(0.3, "stack_lr_03"), (1.0, "stack_lr_10"), (5.0, "stack_lr_50")]:
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

    # === Phase 5: Optuna ensemble (5000 trials) ===
    print("\n" + "=" * 60)
    print("Phase 5: Optuna ensemble (5000 trials)")
    print("=" * 60)

    all_models = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    print(f"\n  Total models: {len(all_models)}")
    print("\nTop 15 models:")
    for m in all_models[:15]:
        print(f"  {m:20s} Acc={results[m]['acc']:.4f} F1={results[m]['f1']:.4f}")

    top_n = min(20, len(all_models))
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
    study.optimize(objective, n_trials=5000, show_progress_bar=False)

    best_weights = np.array([study.best_params[f"w_{m}"] for m in top_models])
    best_weights /= best_weights.sum()
    optuna_oof = sum(w * o for w, o in zip(best_weights, top_oofs))
    optuna_test = sum(w * o for w, o in zip(best_weights, top_tests))
    acc = accuracy_score(y_train, optuna_oof.argmax(1))
    f1 = f1_score(y_train, optuna_oof.argmax(1), average="macro")
    print(f"\n  Optuna ensemble: Acc={acc:.4f} F1={f1:.4f}")
    sig_w = {m: f"{w:.3f}" for m, w in zip(top_models, best_weights) if w > 0.01}
    print(f"  Significant weights: {sig_w}")
    results["optuna_ens"] = {"oof": optuna_oof, "test": optuna_test, "acc": acc, "f1": f1}

    # Cross-version blend with v4 and v8
    print("\n  Cross-version blending...")
    blend_sources = {}
    for name, oof_path, test_path in [
        ("v4", "v4_oof_probs.npy", "v4_test_probs.npy"),
        ("v8", "v8_best_oof_probs.npy", "v8_best_test_probs.npy"),
    ]:
        op = os.path.join(OUT_DIR, oof_path)
        tp = os.path.join(OUT_DIR, test_path)
        if os.path.exists(op):
            blend_sources[name] = {"oof": np.load(op), "test": np.load(tp)}
            print(f"  Loaded {name} predictions")

    if blend_sources:
        # Optuna cross-version blend
        all_blend_oofs = [optuna_oof] + [v["oof"] for v in blend_sources.values()]
        all_blend_tests = [optuna_test] + [v["test"] for v in blend_sources.values()]
        blend_names = ["v9"] + list(blend_sources.keys())

        def obj_cross(trial):
            weights = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in blend_names]
            weights = np.array(weights)
            ws = weights.sum()
            if ws < 1e-8: return 0.0
            weights /= ws
            oof_e = sum(w * o for w, o in zip(weights, all_blend_oofs))
            return accuracy_score(y_train, oof_e.argmax(1))

        study_cross = optuna.create_study(direction="maximize",
                                           sampler=optuna.samplers.TPESampler(seed=SEED + 1))
        study_cross.optimize(obj_cross, n_trials=2000, show_progress_bar=False)

        cross_w = np.array([study_cross.best_params[f"w_{n}"] for n in blend_names])
        cross_w /= cross_w.sum()
        cross_oof = sum(w * o for w, o in zip(cross_w, all_blend_oofs))
        cross_test = sum(w * o for w, o in zip(cross_w, all_blend_tests))
        cross_acc = accuracy_score(y_train, cross_oof.argmax(1))
        cross_f1 = f1_score(y_train, cross_oof.argmax(1), average="macro")
        print(f"  Cross blend: Acc={cross_acc:.4f} F1={cross_f1:.4f}")
        print(f"  Weights: {dict(zip(blend_names, [f'{w:.3f}' for w in cross_w]))}")
        results["cross_blend"] = {"oof": cross_oof, "test": cross_test, "acc": cross_acc, "f1": cross_f1}

    # === Phase 6: Threshold optimization ===
    print("\n" + "=" * 60)
    print("Phase 6: Threshold optimization")
    print("=" * 60)

    all_final = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    best = all_final[0]
    best_oof = results[best]["oof"]
    best_test = results[best]["test"]
    print(f"  Starting from: {best} (Acc={results[best]['acc']:.4f})")

    best_thresh_acc = results[best]["acc"]
    best_bias = np.zeros(3)
    for h_bias in np.arange(-0.20, 0.25, 0.005):
        for r_bias in np.arange(-0.15, 0.15, 0.005):
            o_bias = -(h_bias + r_bias)
            bias = np.array([h_bias, r_bias, o_bias])
            a = accuracy_score(y_train, (best_oof + bias).argmax(1))
            if a > best_thresh_acc:
                best_thresh_acc = a
                best_bias = bias.copy()

    if np.any(best_bias != 0):
        adj_oof = best_oof + best_bias
        adj_test = best_test + best_bias
        adj_f1 = f1_score(y_train, adj_oof.argmax(1), average="macro")
        print(f"  Threshold: Acc={best_thresh_acc:.4f} F1={adj_f1:.4f}")
        print(f"  Bias: H={best_bias[0]:.4f} R={best_bias[1]:.4f} O={best_bias[2]:.4f}")
        results["thresh_opt"] = {"oof": adj_oof, "test": adj_test, "acc": best_thresh_acc, "f1": adj_f1}
    else:
        print("  No threshold improvement.")

    # === Final ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    all_final = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    best = all_final[0]
    best_acc = results[best]["acc"]
    best_f1 = results[best]["f1"]
    best_oof = results[best]["oof"]
    best_test = results[best]["test"]

    print(f"\n*** Best: {best} | Acc={best_acc:.4f} F1={best_f1:.4f} ***")
    print(classification_report(y_train, best_oof.argmax(1), target_names=LABELS, digits=4))

    sub_ids = []
    for bid in val_bids:
        paths = val_idx[bid]
        if "hs" in paths:
            sub_ids.append(os.path.basename(paths["hs"]))
        elif "ms" in paths:
            sub_ids.append(os.path.basename(paths["ms"]))
        else:
            sub_ids.append(os.path.basename(paths["rgb"]))

    preds = [ID2LBL[p] for p in best_test.argmax(1)]
    sub = pd.DataFrame({"Id": sub_ids, "Category": preds})
    a_s = f"{best_acc:.4f}".replace(".", "p")
    f_s = f"{best_f1:.4f}".replace(".", "p")
    sub_path = os.path.join(OUT_DIR, f"submission_v9_acc_{a_s}_f1_{f_s}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    np.save(os.path.join(OUT_DIR, "v9_best_oof_probs.npy"), best_oof)
    np.save(os.path.join(OUT_DIR, "v9_best_test_probs.npy"), best_test)

    # Save optuna ensemble separately if not the best
    if best != "optuna_ens":
        opt_preds = [ID2LBL[p] for p in results["optuna_ens"]["test"].argmax(1)]
        opt_sub = pd.DataFrame({"Id": sub_ids, "Category": opt_preds})
        oa = f"{results['optuna_ens']['acc']:.4f}".replace(".", "p")
        opt_sub.to_csv(os.path.join(OUT_DIR, f"submission_v9_optuna_acc_{oa}.csv"), index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
