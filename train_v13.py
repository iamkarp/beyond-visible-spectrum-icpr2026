"""
Beyond Visible Spectrum v13 - Optimal feature reduction

Key insight from v12: Reducing to top 200 features by LGB importance gave
the BEST individual model ever (0.7500 vs 0.7417).

Strategy:
1. Find optimal feature count: test 100, 150, 200, 250, 300
2. Train MANY models on optimal feature set (not just one config)
3. Health-weighted models on reduced features
4. Seed diversity on reduced features
5. Also train on full features for cross-diversity in ensemble
6. Combine reduced-feature and full-feature models in Optuna ensemble
"""

import os, re, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
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


def main():
    t0 = time.time()
    print("=" * 60)
    print("Beyond Visible Spectrum v13 - Optimal feature reduction")
    print("=" * 60)

    # === Load features ===
    print("\nPhase 1: Loading features...")
    feat_data = np.load(os.path.join(OUT_DIR, "v8_features.npz"), allow_pickle=True)
    X_hand_train = feat_data["X_hand_train"]
    X_hand_test = feat_data["X_hand_test"]
    train_cnn = feat_data["train_cnn"]
    val_cnn = feat_data["val_cnn"]
    X_base_train = np.hstack([X_hand_train, train_cnn])
    X_base_test = np.hstack([X_hand_test, val_cnn])

    train_files = sorted([f for f in os.listdir(os.path.join(ROOT, "train", "RGB")) if f.lower().endswith((".png", ".jpg"))])
    y_labels, train_bids = [], []
    for f in train_files:
        bid = os.path.splitext(f)[0]
        m = re.match(r"^(Health|Rust|Other)_", bid)
        if m: y_labels.append(m.group(1)); train_bids.append(bid)
    y_train = np.array([LBL2ID[l] for l in y_labels])

    val_files = sorted([f for f in os.listdir(os.path.join(ROOT, "val", "RGB")) if f.lower().endswith((".png", ".jpg"))])
    val_base_ids = [os.path.splitext(f)[0] for f in val_files]
    n_train, n_val = len(y_train), len(val_base_ids)
    print(f"  Train: {n_train} | Val: {n_val} | Base features: {X_base_train.shape[1]}")

    # HS spectra
    print("  Loading HS spectra...")
    HS_DROP_FIRST, HS_DROP_LAST, HS_TARGET_CH = 10, 14, 101
    import tifffile as tiff
    def read_tiff(path):
        arr = tiff.imread(path)
        if arr.ndim != 3: raise ValueError("3D expected")
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)

    train_spectra, val_spectra = [], []
    for bid in train_bids:
        p = os.path.join(ROOT, "train", "HS", bid + ".tif")
        if os.path.exists(p):
            arr = read_tiff(p)
            B = arr.shape[2]
            arr = arr[:, :, HS_DROP_FIRST:B-HS_DROP_LAST] if B > HS_DROP_FIRST+HS_DROP_LAST+1 else arr
            C = arr.shape[2]
            arr = arr[:, :, :HS_TARGET_CH] if C > HS_TARGET_CH else arr
            spec = np.mean(arr.reshape(-1, min(C, HS_TARGET_CH)), axis=0)
            if len(spec) < HS_TARGET_CH: spec = np.pad(spec, (0, HS_TARGET_CH-len(spec)))
            train_spectra.append(spec)
        else: train_spectra.append(np.zeros(HS_TARGET_CH))

    for bid in val_base_ids:
        p = os.path.join(ROOT, "val", "HS", bid + ".tif")
        if os.path.exists(p):
            arr = read_tiff(p)
            B = arr.shape[2]
            arr = arr[:, :, HS_DROP_FIRST:B-HS_DROP_LAST] if B > HS_DROP_FIRST+HS_DROP_LAST+1 else arr
            C = arr.shape[2]
            arr = arr[:, :, :HS_TARGET_CH] if C > HS_TARGET_CH else arr
            spec = np.mean(arr.reshape(-1, min(C, HS_TARGET_CH)), axis=0)
            if len(spec) < HS_TARGET_CH: spec = np.pad(spec, (0, HS_TARGET_CH-len(spec)))
            val_spectra.append(spec)
        else: val_spectra.append(np.zeros(HS_TARGET_CH))

    # CV spectral library
    print("\nPhase 2: CV spectral library...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_slib = [None] * n_train
    for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
        lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
        for i in vai: oof_slib[i] = library_features(train_spectra[i], lib)
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

    # === Feature importance ===
    print("\nPhase 3: Computing feature importance...")
    imp_model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=25,
        max_depth=5, min_child_samples=10, subsample=0.75, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=0.5, n_estimators=500, verbose=-1, random_state=SEED)
    imp_model.fit(X_oof, y_train)
    importances = imp_model.feature_importances_

    # Build feature masks for different top-K values
    def get_reduced_data(top_k):
        top_idx = np.argsort(importances)[::-1][:top_k]
        mask = np.zeros(X_oof.shape[1], dtype=bool)
        mask[top_idx] = True
        return X_oof[:, mask], X_test[:, mask], X_all[:, mask]

    # === Find optimal feature count ===
    print("\n" + "=" * 60)
    print("Phase 4: Finding optimal feature count")
    print("=" * 60)

    lgb_base = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                "verbose": -1, "random_state": SEED}
    lgb_d = {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
             "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
             "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}

    best_k, best_k_acc = 200, 0
    for k in [100, 150, 175, 200, 225, 250, 300, 400]:
        Xo, Xt, Xa = get_reduced_data(k)
        oof_p = np.zeros((n_train, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(**lgb_d)
            m.fit(Xo[tri], y_train[tri], eval_set=[(Xo[vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(Xo[vai])
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        marker = " ***" if acc > best_k_acc else ""
        if acc > best_k_acc: best_k, best_k_acc = k, acc
        print(f"  Top {k:4d}: Acc={acc:.4f} F1={f1:.4f}{marker}")

    print(f"\n  Optimal: top {best_k} features (Acc={best_k_acc:.4f})")

    # Use best K and also try neighbors
    feature_sets = {}
    for k in set([best_k, max(best_k - 25, 100), best_k + 25, 200, 928]):
        if k > X_oof.shape[1]: k = X_oof.shape[1]
        Xo, Xt, Xa = get_reduced_data(k) if k < X_oof.shape[1] else (X_oof, X_test, X_all)
        feature_sets[k] = {"X_oof": Xo, "X_test": Xt, "X_all": Xa}

    # === Phase 5: Train many models on optimal features ===
    print("\n" + "=" * 60)
    print(f"Phase 5: Training models on top {best_k} features + full features")
    print("=" * 60)

    results = {}
    health_sw = np.ones(n_train)
    for i, y in enumerate(y_train):
        if y == 0: health_sw[i] = 1.5

    health_sw2 = np.ones(n_train)
    for i, y in enumerate(y_train):
        if y == 0: health_sw2[i] = 1.3

    xgb_base = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                "tree_method": "hist", "random_state": SEED, "verbosity": 0}

    # Model configs
    configs = [
        ("lgb_d", "lgb", {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                          "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                          "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
        ("lgb_a", "lgb", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                          "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                          "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
        ("lgb_b", "lgb", {**lgb_base, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4,
                          "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
                          "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000}),
        ("lgb_c", "lgb", {**lgb_base, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                          "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
                          "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500}),
        ("lgb_e", "lgb", {**lgb_base, "learning_rate": 0.015, "num_leaves": 25, "max_depth": 5,
                          "min_child_samples": 12, "subsample": 0.65, "colsample_bytree": 0.5,
                          "reg_alpha": 1.5, "reg_lambda": 1.5, "n_estimators": 3000}),
        ("xgb_a", "xgb", {**xgb_base, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
                          "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
                          "reg_lambda": 1.0, "n_estimators": 2000}),
        ("xgb_b", "xgb", {**xgb_base, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
                          "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
                          "reg_lambda": 2.0, "n_estimators": 3000}),
        ("xgb_c", "xgb", {**xgb_base, "learning_rate": 0.01, "max_depth": 4, "min_child_weight": 10,
                          "subsample": 0.7, "colsample_bytree": 0.4, "reg_alpha": 2.0,
                          "reg_lambda": 2.0, "n_estimators": 5000}),
    ]

    # Train on optimal reduced features AND full features
    for k_feats in sorted(feature_sets.keys()):
        fs = feature_sets[k_feats]
        prefix = f"t{k_feats}" if k_feats < 928 else "full"
        print(f"\n  --- {prefix} features ({k_feats}) ---")

        for cname, mtype, params in configs:
            name = f"{prefix}_{cname}"
            oof_p = np.zeros((n_train, 3))
            test_p = np.zeros((n_val, 3))
            for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
                if mtype == "lgb":
                    m = lgb.LGBMClassifier(**params)
                    m.fit(fs["X_oof"][tri], y_train[tri], eval_set=[(fs["X_oof"][vai], y_train[vai])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                    oof_p[vai] = m.predict_proba(fs["X_oof"][vai])
                    m2 = lgb.LGBMClassifier(**params)
                    m2.fit(fs["X_all"][tri], y_train[tri], eval_set=[(fs["X_all"][vai], y_train[vai])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
                    test_p += m2.predict_proba(fs["X_test"]) / N_FOLDS
                elif mtype == "xgb":
                    m = xgb.XGBClassifier(**params)
                    m.fit(fs["X_oof"][tri], y_train[tri], eval_set=[(fs["X_oof"][vai], y_train[vai])], verbose=False)
                    oof_p[vai] = m.predict_proba(fs["X_oof"][vai])
                    m2 = xgb.XGBClassifier(**params)
                    m2.fit(fs["X_all"][tri], y_train[tri], eval_set=[(fs["X_all"][vai], y_train[vai])], verbose=False)
                    test_p += m2.predict_proba(fs["X_test"]) / N_FOLDS
            acc = accuracy_score(y_train, oof_p.argmax(1))
            f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
            print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
            results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

        # Health-weighted on this feature set (only for top configs)
        for hw, hw_name in [(health_sw2, "hw13"), (health_sw, "hw15")]:
            for cname_hw, _, params_hw in [configs[0], configs[1]]:  # lgb_d and lgb_a
                name = f"{prefix}_{cname_hw}_{hw_name}"
                oof_p = np.zeros((n_train, 3))
                test_p = np.zeros((n_val, 3))
                for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
                    m = lgb.LGBMClassifier(**params_hw)
                    m.fit(fs["X_oof"][tri], y_train[tri], sample_weight=hw[tri],
                          eval_set=[(fs["X_oof"][vai], y_train[vai])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                    oof_p[vai] = m.predict_proba(fs["X_oof"][vai])
                    m2 = lgb.LGBMClassifier(**params_hw)
                    m2.fit(fs["X_all"][tri], y_train[tri], sample_weight=hw[tri],
                           eval_set=[(fs["X_all"][vai], y_train[vai])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
                    test_p += m2.predict_proba(fs["X_test"]) / N_FOLDS
                acc = accuracy_score(y_train, oof_p.argmax(1))
                f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
                print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
                results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

        # ET/RF on this feature set
        for sk_name, cls, params in [
            ("et", ExtraTreesClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
            ("rf", RandomForestClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
        ]:
            name = f"{prefix}_{sk_name}"
            oof_p = np.zeros((n_train, 3))
            test_p = np.zeros((n_val, 3))
            for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
                m = cls(**params)
                m.fit(fs["X_oof"][tri], y_train[tri])
                oof_p[vai] = m.predict_proba(fs["X_oof"][vai])
                m2 = cls(**params)
                m2.fit(fs["X_all"][tri], y_train[tri])
                test_p += m2.predict_proba(fs["X_test"]) / N_FOLDS
            acc = accuracy_score(y_train, oof_p.argmax(1))
            f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
            print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
            results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 6: Seed diversity on optimal features ===
    print("\n" + "=" * 60)
    print(f"Phase 6: Seed diversity on top {best_k} features")
    print("=" * 60)

    fs_opt = feature_sets[best_k]
    for seed_off in [100, 200, 300, 400, 500, 600, 700]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_off)

        # Use lgb_d params on reduced features
        seed_params = {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                       "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                       "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000,
                       "random_state": SEED + seed_off}

        # Need to recompute spectral library for different fold split
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf_s.split(X_base_train, y_train)):
            # Recompute slib for this fold split (to maintain proper OOF)
            lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
            slib_vai = np.array([list(library_features(train_spectra[i], lib).values()) for i in vai]).astype(np.float32)
            slib_tri = np.array([list(library_features(train_spectra[i], lib).values()) for i in tri]).astype(np.float32)

            # Reduced features: need to select from base+slib
            X_tri_full = np.hstack([X_base_train[tri], slib_tri])
            X_vai_full = np.hstack([X_base_train[vai], slib_vai])

            # Apply same feature mask (from importance)
            top_idx = np.argsort(importances)[::-1][:best_k]
            mask = np.zeros(X_oof.shape[1], dtype=bool)
            mask[top_idx] = True
            X_tri_red = X_tri_full[:, mask]
            X_vai_red = X_vai_full[:, mask]

            m = lgb.LGBMClassifier(**seed_params)
            m.fit(X_tri_red, y_train[tri], eval_set=[(X_vai_red, y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_vai_red)

            # Test: use full-train slib
            m2 = lgb.LGBMClassifier(**seed_params)
            m2.fit(fs_opt["X_all"][tri], y_train[tri],
                   eval_set=[(fs_opt["X_all"][vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(fs_opt["X_test"]) / N_FOLDS

        name = f"rs{seed_off}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Also seed diversity on full features
    for seed_off in [100, 200, 300]:
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
        name = f"fs{seed_off}"
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
    pca_stack = PCA(n_components=min(50, stack_oof.shape[1] // 2), random_state=SEED)
    X_oof_pca = pca_stack.fit_transform(fs_opt["X_oof"])
    X_test_pca = pca_stack.transform(fs_opt["X_test"])
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
    print("\n  Top 25:")
    for m in all_models[:25]:
        print(f"    {m:25s} Acc={results[m]['acc']:.4f} F1={results[m]['f1']:.4f}")

    ens_oofs = [results[m]["oof"] for m in all_models]
    ens_tests = [results[m]["test"] for m in all_models]

    def objective_ens(trial):
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in all_models]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
        weights /= ws
        return accuracy_score(y_train, sum(w * o for w, o in zip(weights, ens_oofs)).argmax(1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_ens, n_trials=5000, show_progress_bar=False)

    bw = np.array([study.best_params[f"w_{m}"] for m in all_models])
    bw /= bw.sum()
    optuna_oof = sum(w * o for w, o in zip(bw, ens_oofs))
    optuna_test = sum(w * o for w, o in zip(bw, ens_tests))
    acc = accuracy_score(y_train, optuna_oof.argmax(1))
    f1 = f1_score(y_train, optuna_oof.argmax(1), average="macro")
    print(f"\n  Optuna ensemble: Acc={acc:.4f} F1={f1:.4f}")
    sig_w = {m: f"{w:.3f}" for m, w in zip(all_models, bw) if w > 0.01}
    print(f"  Significant weights: {sig_w}")

    # === Phase 9: Cross-version blending ===
    print("\n" + "=" * 60)
    print("Phase 9: Cross-version blending")
    print("=" * 60)

    cross_oofs = {"v13": optuna_oof}
    cross_tests = {"v13": optuna_test}
    for vn, op, tp in [("v8", "v8_best_oof_probs.npy", "v8_best_test_probs.npy"),
                        ("v10", "v10_best_oof_probs.npy", "v10_best_test_probs.npy")]:
        p1, p2 = os.path.join(OUT_DIR, op), os.path.join(OUT_DIR, tp)
        if os.path.exists(p1) and os.path.exists(p2):
            cross_oofs[vn] = np.load(p1); cross_tests[vn] = np.load(p2)
            print(f"  Loaded {vn}")

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

    cw = np.array([study_cross.best_params[f"w_{n}"] for n in cross_names])
    cw /= cw.sum()
    cross_oof = sum(w * o for w, o in zip(cw, cross_oof_list))
    cross_test = sum(w * o for w, o in zip(cw, cross_test_list))
    cross_acc = accuracy_score(y_train, cross_oof.argmax(1))
    cross_f1 = f1_score(y_train, cross_oof.argmax(1), average="macro")
    print(f"  Cross blend: Acc={cross_acc:.4f} F1={cross_f1:.4f}")
    print(f"  Weights: {dict(zip(cross_names, [f'{w:.3f}' for w in cw]))}")

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
            adj = best_oof.copy()
            adj[:, 0] += bias_h; adj[:, 1] += bias_r; adj[:, 2] += bias_o
            ta = accuracy_score(y_train, adj.argmax(1))
            if ta > best_acc:
                best_acc = ta
                best_oof = adj.copy()
                best_test = candidates[best_name][1].copy()
                best_test[:, 0] += bias_h; best_test[:, 1] += bias_r; best_test[:, 2] += bias_o
                improved = True
                print(f"  H={bias_h:+.2f} R={bias_r:+.2f} O={bias_o:+.2f} -> Acc={ta:.4f}")
                best_name += "+t"

    if not improved: print("  No threshold improvement.")

    # === Final ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    final_acc = accuracy_score(y_train, best_oof.argmax(1))
    final_f1 = f1_score(y_train, best_oof.argmax(1), average="macro")
    print(f"\n*** Best: {best_name} | Acc={final_acc:.4f} F1={final_f1:.4f} ***")
    print(classification_report(y_train, best_oof.argmax(1), target_names=LABELS))

    np.save(os.path.join(OUT_DIR, "v13_best_oof_probs.npy"), best_oof)
    np.save(os.path.join(OUT_DIR, "v13_best_test_probs.npy"), best_test)

    test_preds = best_test.argmax(1)
    result_csv = os.path.join(ROOT, "result.csv")
    template = pd.read_csv(result_csv) if os.path.exists(result_csv) else pd.DataFrame({"filename": val_base_ids})
    template["predict"] = [ID2LBL[p] for p in test_preds]
    sub_name = f"submission_v13_acc_{final_acc:.4f}_f1_{final_f1:.4f}".replace(".", "p") + ".csv"
    sub_path = os.path.join(OUT_DIR, sub_name)
    template.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")
    print(f"Done! ({(time.time()-t0)/60:.1f} minutes)")


if __name__ == "__main__":
    main()
