"""
Beyond Visible Spectrum v10 - Breaking the 0.765 ceiling

Key changes from v8/v9:
1. CatBoost - untried boosting library with ordered boosting (different regularization)
2. Optuna hyperparameter tuning for individual LGB/XGB models (not just ensemble weights)
3. Feature noise augmentation - train on noisy copies for better regularization
4. Better stacking: use OOF probs + rank features + disagreement features
5. Power ensemble: geometric mean + arithmetic mean + rank-based combination
6. Targeted Health class features: analyze errors, add Health-specific augmentation weight
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

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: CatBoost not installed. Install with: pip install catboost")


# ============================================================
# Spectral library (reuse from v8)
# ============================================================
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


# ============================================================
# Train/predict helper with fold-level OOF
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


def train_catboost_cv(params, X_oof, X_all, X_test, y_train, skf, X_base_train, name="cb"):
    n_train, n_test = len(y_train), len(X_test)
    oof_p = np.zeros((n_train, 3))
    test_p = np.zeros((n_test, 3))

    for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
        m = CatBoostClassifier(**params)
        m.fit(X_oof[tri], y_train[tri], eval_set=(X_oof[vai], y_train[vai]),
              verbose=False, early_stopping_rounds=100)
        oof_p[vai] = m.predict_proba(X_oof[vai])

        m2 = CatBoostClassifier(**params)
        m2.fit(X_all[tri], y_train[tri], eval_set=(X_all[vai], y_train[vai]),
               verbose=False, early_stopping_rounds=100)
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
    print("Beyond Visible Spectrum v10 - Breaking the 0.765 ceiling")
    print("=" * 60)

    # === Phase 1: Load cached features ===
    print("\nPhase 1: Loading cached features from v8...")
    feat_data = np.load(os.path.join(OUT_DIR, "v8_features.npz"), allow_pickle=True)
    X_hand_train = feat_data["X_hand_train"]
    X_hand_test = feat_data["X_hand_test"]
    train_cnn = feat_data["train_cnn"]
    val_cnn = feat_data["val_cnn"]
    base_cols = feat_data["base_cols"]

    X_base_train = np.hstack([X_hand_train, train_cnn])
    X_base_test = np.hstack([X_hand_test, val_cnn])

    # Load labels
    train_idx_dir = os.path.join(ROOT, "train")
    train_files = sorted(os.listdir(os.path.join(train_idx_dir, "RGB")))
    train_files = [f for f in train_files if f.lower().endswith((".png", ".jpg"))]
    y_labels = []
    for f in train_files:
        bid = os.path.splitext(f)[0]
        m = re.match(r"^(Health|Rust|Other)_", bid)
        if m:
            y_labels.append(m.group(1))
    y_train = np.array([LBL2ID[l] for l in y_labels])

    # Load val base_ids for submission
    val_files = sorted(os.listdir(os.path.join(ROOT, "val", "RGB")))
    val_files = [f for f in val_files if f.lower().endswith((".png", ".jpg"))]
    val_base_ids = [os.path.splitext(f)[0] for f in val_files]

    n_train, n_val = len(y_train), len(val_base_ids)
    print(f"  Train: {n_train} | Val: {n_val} | Features: {X_base_train.shape[1]}")

    # Load HS spectra for spectral library
    print("  Loading HS spectra...")
    HS_DROP_FIRST, HS_DROP_LAST, HS_TARGET_CH = 10, 14, 101

    import tifffile as tiff
    def read_tiff(path):
        arr = tiff.imread(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D, got {arr.shape}")
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)

    train_spectra = []
    for f in train_files:
        bid = os.path.splitext(f)[0]
        hs_path = os.path.join(train_idx_dir, "HS", bid + ".tif")
        if os.path.exists(hs_path):
            arr = read_tiff(hs_path)
            B = arr.shape[2]
            if B > (HS_DROP_FIRST + HS_DROP_LAST + 1):
                arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST]
            C = arr.shape[2]
            if C > HS_TARGET_CH:
                arr = arr[:, :, :HS_TARGET_CH]
            spec = np.mean(arr.reshape(-1, min(C, HS_TARGET_CH)), axis=0)
            if len(spec) < HS_TARGET_CH:
                spec = np.pad(spec, (0, HS_TARGET_CH - len(spec)))
            train_spectra.append(spec)
        else:
            train_spectra.append(np.zeros(HS_TARGET_CH))

    val_spectra = []
    for bid in val_base_ids:
        hs_path = os.path.join(ROOT, "val", "HS", bid + ".tif")
        if os.path.exists(hs_path):
            arr = read_tiff(hs_path)
            B = arr.shape[2]
            if B > (HS_DROP_FIRST + HS_DROP_LAST + 1):
                arr = arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST]
            C = arr.shape[2]
            if C > HS_TARGET_CH:
                arr = arr[:, :, :HS_TARGET_CH]
            spec = np.mean(arr.reshape(-1, min(C, HS_TARGET_CH)), axis=0)
            if len(spec) < HS_TARGET_CH:
                spec = np.pad(spec, (0, HS_TARGET_CH - len(spec)))
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
    n_features = X_oof.shape[1]
    print(f"  Total features: {n_features}")

    # === Phase 3: Optuna hyperparameter tuning for individual LGB/XGB ===
    print("\n" + "=" * 60)
    print("Phase 3: Optuna hyperparameter tuning (individual models)")
    print("=" * 60)

    results = {}
    lgb_base = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
                "verbose": -1, "random_state": SEED}

    # --- 3a: Tune LightGBM ---
    print("\n  Tuning LightGBM hyperparameters (50 trials)...")

    def lgb_objective(trial):
        params = {
            **lgb_base,
            "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            "n_estimators": 2000,
        }

        oof_p = np.zeros((n_train, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_oof[tri], y_train[tri],
                  eval_set=[(X_oof[vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_p[vai] = m.predict_proba(X_oof[vai])

        return accuracy_score(y_train, oof_p.argmax(1))

    lgb_study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
    lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)

    best_lgb_params = {
        **lgb_base,
        "learning_rate": lgb_study.best_params["lr"],
        "num_leaves": lgb_study.best_params["num_leaves"],
        "max_depth": lgb_study.best_params["max_depth"],
        "min_child_samples": lgb_study.best_params["min_child_samples"],
        "subsample": lgb_study.best_params["subsample"],
        "colsample_bytree": lgb_study.best_params["colsample_bytree"],
        "reg_alpha": lgb_study.best_params["reg_alpha"],
        "reg_lambda": lgb_study.best_params["reg_lambda"],
        "n_estimators": 2000,
    }
    print(f"  Best LGB trial: Acc={lgb_study.best_value:.4f}")
    print(f"  Params: lr={best_lgb_params['learning_rate']:.4f}, leaves={best_lgb_params['num_leaves']}, "
          f"depth={best_lgb_params['max_depth']}, subsample={best_lgb_params['subsample']:.2f}, "
          f"colsample={best_lgb_params['colsample_bytree']:.2f}")

    # Train tuned LGB
    r = train_lgb_cv(best_lgb_params, X_oof, X_all, X_test, y_train, skf, X_base_train, name="lgb_tuned")
    print(f"  lgb_tuned: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    results["lgb_tuned"] = r

    # --- 3b: Tune XGBoost ---
    print("\n  Tuning XGBoost hyperparameters (50 trials)...")

    xgb_base = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                "tree_method": "hist", "random_state": SEED, "verbosity": 0}

    def xgb_objective(trial):
        params = {
            **xgb_base,
            "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "n_estimators": 2000,
        }

        oof_p = np.zeros((n_train, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = xgb.XGBClassifier(**params)
            m.fit(X_oof[tri], y_train[tri],
                  eval_set=[(X_oof[vai], y_train[vai])], verbose=False)
            oof_p[vai] = m.predict_proba(X_oof[vai])

        return accuracy_score(y_train, oof_p.argmax(1))

    xgb_study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED + 1))
    xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

    best_xgb_params = {
        **xgb_base,
        "learning_rate": xgb_study.best_params["lr"],
        "max_depth": xgb_study.best_params["max_depth"],
        "min_child_weight": xgb_study.best_params["min_child_weight"],
        "subsample": xgb_study.best_params["subsample"],
        "colsample_bytree": xgb_study.best_params["colsample_bytree"],
        "reg_alpha": xgb_study.best_params["reg_alpha"],
        "reg_lambda": xgb_study.best_params["reg_lambda"],
        "gamma": xgb_study.best_params["gamma"],
        "n_estimators": 2000,
    }
    print(f"  Best XGB trial: Acc={xgb_study.best_value:.4f}")

    r = train_xgb_cv(best_xgb_params, X_oof, X_all, X_test, y_train, skf, X_base_train, name="xgb_tuned")
    print(f"  xgb_tuned: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    results["xgb_tuned"] = r

    # --- 3c: CatBoost ---
    if HAS_CATBOOST:
        print("\n  Training CatBoost models...")
        cb_configs = [
            ("cb_a", {"iterations": 3000, "learning_rate": 0.03, "depth": 5,
                      "l2_leaf_reg": 3.0, "random_seed": SEED, "task_type": "CPU",
                      "loss_function": "MultiClass", "classes_count": 3,
                      "subsample": 0.8, "colsample_bylevel": 0.6}),
            ("cb_b", {"iterations": 5000, "learning_rate": 0.01, "depth": 4,
                      "l2_leaf_reg": 5.0, "random_seed": SEED, "task_type": "CPU",
                      "loss_function": "MultiClass", "classes_count": 3,
                      "subsample": 0.7, "colsample_bylevel": 0.5}),
            ("cb_c", {"iterations": 3000, "learning_rate": 0.05, "depth": 6,
                      "l2_leaf_reg": 1.0, "random_seed": SEED, "task_type": "CPU",
                      "loss_function": "MultiClass", "classes_count": 3,
                      "subsample": 0.8, "colsample_bylevel": 0.7}),
            ("cb_d", {"iterations": 3000, "learning_rate": 0.03, "depth": 5,
                      "l2_leaf_reg": 3.0, "random_seed": SEED, "task_type": "CPU",
                      "loss_function": "MultiClass", "classes_count": 3,
                      "subsample": 0.8, "colsample_bylevel": 0.6,
                      "class_weights": [1.3, 1.0, 1.0]}),  # Health weighted
        ]
        for name, params in cb_configs:
            r = train_catboost_cv(params, X_oof, X_all, X_test, y_train, skf, X_base_train, name=name)
            print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
            results[name] = r

    # === Phase 4: Hand-tuned models (proven configs from v8) ===
    print("\n" + "=" * 60)
    print("Phase 4: Proven model configs from v8")
    print("=" * 60)

    v8_configs = [
        ("lgb_a", {**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                   "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                   "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000}),
        ("lgb_b", {**lgb_base, "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4,
                   "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.5,
                   "reg_alpha": 1.0, "reg_lambda": 1.5, "n_estimators": 3000}),
        ("lgb_c", {**lgb_base, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                   "min_child_samples": 8, "subsample": 0.8, "colsample_bytree": 0.7,
                   "reg_alpha": 0.3, "reg_lambda": 0.3, "n_estimators": 1500}),
        ("lgb_d", {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                   "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                   "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000}),
    ]

    for name, params in v8_configs:
        r = train_lgb_cv(params, X_oof, X_all, X_test, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    # Health-weighted versions
    health_sw = np.ones(n_train)
    for i, y in enumerate(y_train):
        if y == 0:  # Health
            health_sw[i] = 1.5

    r = train_lgb_cv({**lgb_base, "learning_rate": 0.03, "num_leaves": 25, "max_depth": 5,
                       "min_child_samples": 10, "subsample": 0.75, "colsample_bytree": 0.6,
                       "reg_alpha": 0.5, "reg_lambda": 0.5, "n_estimators": 2000},
                     X_oof, X_all, X_test, y_train, skf, X_base_train,
                     sample_weight=health_sw, name="lgb_hw15")
    print(f"  lgb_hw15: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    results["lgb_hw15"] = r

    health_sw2 = np.ones(n_train)
    for i, y in enumerate(y_train):
        if y == 0: health_sw2[i] = 2.0
    r = train_lgb_cv({**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                       "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                       "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000},
                     X_oof, X_all, X_test, y_train, skf, X_base_train,
                     sample_weight=health_sw2, name="lgb_hw20")
    print(f"  lgb_hw20: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    results["lgb_hw20"] = r

    # XGB configs
    xgb_configs = [
        ("xgb_a", {**xgb_base, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
                   "subsample": 0.75, "colsample_bytree": 0.6, "reg_alpha": 0.5,
                   "reg_lambda": 1.0, "n_estimators": 2000}),
        ("xgb_b", {**xgb_base, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 8,
                   "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 1.0,
                   "reg_lambda": 2.0, "n_estimators": 3000}),
    ]
    for name, params in xgb_configs:
        r = train_xgb_cv(params, X_oof, X_all, X_test, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    # Sklearn ensembles
    for name, cls, params in [
        ("et", ExtraTreesClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
        ("rf", RandomForestClassifier, {"n_estimators": 2000, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}),
    ]:
        r = train_sklearn_cv(cls, params, X_oof, X_all, X_test, y_train, skf, X_base_train, name=name)
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
        results[name] = r

    # === Phase 5: Seed diversity with tuned params ===
    print("\n" + "=" * 60)
    print("Phase 5: Seed diversity (tuned + original params)")
    print("=" * 60)

    for seed_off in [100, 200, 300, 400, 500]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_off)

        # Use tuned params with different seed/fold split
        seed_params = {**best_lgb_params, "random_state": SEED + seed_off}

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

        name = f"lgb_ts{seed_off}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # Also seed diversity with original v8 best params (lgb_d)
    for seed_off in [100, 200, 300]:
        skf_s = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + seed_off)
        orig_params = {**lgb_base, "learning_rate": 0.01, "num_leaves": 20, "max_depth": 4,
                       "min_child_samples": 20, "subsample": 0.7, "colsample_bytree": 0.4,
                       "reg_alpha": 2.0, "reg_lambda": 2.0, "n_estimators": 5000,
                       "random_state": SEED + seed_off}

        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf_s.split(X_base_train, y_train)):
            lib = build_library([train_spectra[i] for i in tri], [y_train[i] for i in tri])
            slib_vai = np.array([list(library_features(train_spectra[i], lib).values()) for i in vai]).astype(np.float32)
            slib_tri = np.array([list(library_features(train_spectra[i], lib).values()) for i in tri]).astype(np.float32)
            X_tri = np.hstack([X_base_train[tri], slib_tri])
            X_vai = np.hstack([X_base_train[vai], slib_vai])

            m = lgb.LGBMClassifier(**orig_params)
            m.fit(X_tri, y_train[tri], eval_set=[(X_vai, y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_vai)

            m2 = lgb.LGBMClassifier(**orig_params)
            m2.fit(X_all[tri], y_train[tri], eval_set=[(X_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test) / N_FOLDS

        name = f"lgb_s{seed_off}"
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 6: Feature noise augmentation ===
    print("\n" + "=" * 60)
    print("Phase 6: Feature noise augmentation")
    print("=" * 60)

    # Add Gaussian noise to features and train on augmented data
    # This is like dropout/regularization at the feature level
    for noise_level, name_suffix in [(0.01, "n01"), (0.02, "n02"), (0.05, "n05")]:
        # Compute feature-wise std for noise scaling
        feat_std = np.std(X_oof, axis=0) + 1e-8

        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))

        noise_params = {**best_lgb_params, "random_state": SEED + 7777}

        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            # Create noisy copies of training data
            n_copies = 3
            X_aug_list = [X_oof[tri]]
            y_aug_list = [y_train[tri]]
            for copy_i in range(n_copies):
                noise = np.random.randn(*X_oof[tri].shape).astype(np.float32) * feat_std * noise_level
                X_aug_list.append(X_oof[tri] + noise)
                y_aug_list.append(y_train[tri])

            X_aug = np.vstack(X_aug_list)
            y_aug = np.concatenate(y_aug_list)

            m = lgb.LGBMClassifier(**noise_params)
            m.fit(X_aug, y_aug,
                  eval_set=[(X_oof[vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_p[vai] = m.predict_proba(X_oof[vai])

            # For test: augment all training data
            X_all_aug_list = [X_all[tri]]
            y_all_aug_list = [y_train[tri]]
            for copy_i in range(n_copies):
                noise = np.random.randn(*X_all[tri].shape).astype(np.float32) * feat_std * noise_level
                X_all_aug_list.append(X_all[tri] + noise)
                y_all_aug_list.append(y_train[tri])
            X_all_aug = np.vstack(X_all_aug_list)
            y_all_aug = np.concatenate(y_all_aug_list)

            m2 = lgb.LGBMClassifier(**noise_params)
            m2.fit(X_all_aug, y_all_aug,
                   eval_set=[(X_all[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            test_p += m2.predict_proba(X_test) / N_FOLDS

        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        name = f"lgb_{name_suffix}"
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 7: Advanced stacking ===
    print("\n" + "=" * 60)
    print("Phase 7: Advanced stacking")
    print("=" * 60)

    model_names_for_stack = sorted(results.keys())
    stack_oof = np.hstack([results[m]["oof"] for m in model_names_for_stack])
    stack_test = np.hstack([results[m]["test"] for m in model_names_for_stack])

    # Add rank features (rank within each sample across models for each class)
    n_models = len(model_names_for_stack)
    rank_oof = np.zeros_like(stack_oof)
    rank_test = np.zeros_like(stack_test)
    for i in range(n_train):
        for c in range(3):
            vals = [results[m]["oof"][i, c] for m in model_names_for_stack]
            rank_oof[i, c * n_models:(c + 1) * n_models] = rankdata(vals)
    for i in range(n_val):
        for c in range(3):
            vals = [results[m]["test"][i, c] for m in model_names_for_stack]
            rank_test[i, c * n_models:(c + 1) * n_models] = rankdata(vals)

    # Add disagreement features
    disagree_oof = np.zeros((n_train, n_models))
    disagree_test = np.zeros((n_val, n_models))

    # For each model, compute how much it disagrees with the ensemble mean
    mean_oof = np.mean([results[m]["oof"] for m in model_names_for_stack], axis=0)
    mean_test = np.mean([results[m]["test"] for m in model_names_for_stack], axis=0)
    for j, m in enumerate(model_names_for_stack):
        disagree_oof[:, j] = np.sum(np.abs(results[m]["oof"] - mean_oof), axis=1)
        disagree_test[:, j] = np.sum(np.abs(results[m]["test"] - mean_test), axis=1)

    # PCA of base features for stacking
    pca_stack = PCA(n_components=50, random_state=SEED)
    X_oof_pca = pca_stack.fit_transform(X_oof)
    X_test_pca = pca_stack.transform(X_test)

    # Combine stack features
    stack_oof_full = np.hstack([stack_oof, rank_oof, disagree_oof, X_oof_pca])
    stack_test_full = np.hstack([stack_test, rank_test, disagree_test, X_test_pca])

    # Scale for LR
    scaler_s = StandardScaler()
    stack_oof_sc = scaler_s.fit_transform(stack_oof_full)
    stack_test_sc = scaler_s.transform(stack_test_full)

    # LR stackers
    for C_val, name in [(0.3, "stack_lr_03"), (1.0, "stack_lr_10"), (3.0, "stack_lr_30")]:
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

    # LGB stackers
    for lr, nl, name in [(0.05, 15, "stack_lgb_a"), (0.03, 20, "stack_lgb_b"), (0.02, 10, "stack_lgb_c")]:
        oof_p = np.zeros((n_train, 3))
        test_p = np.zeros((n_val, 3))
        for fold, (tri, vai) in enumerate(skf.split(X_base_train, y_train)):
            m = lgb.LGBMClassifier(objective="multiclass", num_class=3, learning_rate=lr,
                                    num_leaves=nl, max_depth=4, min_child_samples=15,
                                    subsample=0.8, colsample_bytree=0.5, reg_alpha=1.0,
                                    reg_lambda=1.0, n_estimators=1000, verbose=-1, random_state=SEED)
            m.fit(stack_oof_full[tri], y_train[tri], eval_set=[(stack_oof_full[vai], y_train[vai])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_p[vai] = m.predict_proba(stack_oof_full[vai])
            m2 = lgb.LGBMClassifier(objective="multiclass", num_class=3, learning_rate=lr,
                                     num_leaves=nl, max_depth=4, min_child_samples=15,
                                     subsample=0.8, colsample_bytree=0.5, reg_alpha=1.0,
                                     reg_lambda=1.0, n_estimators=1000, verbose=-1, random_state=SEED)
            m2.fit(stack_oof_full, y_train, eval_set=[(stack_oof_full[vai], y_train[vai])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
            test_p += m2.predict_proba(stack_test_full) / N_FOLDS
        acc = accuracy_score(y_train, oof_p.argmax(1))
        f1 = f1_score(y_train, oof_p.argmax(1), average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
        results[name] = {"oof": oof_p, "test": test_p, "acc": acc, "f1": f1}

    # === Phase 8: Multi-strategy Optuna ensemble ===
    print("\n" + "=" * 60)
    print("Phase 8: Optuna ensemble (5000 trials)")
    print("=" * 60)

    all_models = sorted(results.keys(), key=lambda m: results[m]["acc"], reverse=True)
    print(f"\n  Total models: {len(all_models)}")
    print("\n  Top 20 models:")
    for m in all_models[:20]:
        print(f"    {m:20s} Acc={results[m]['acc']:.4f} F1={results[m]['f1']:.4f}")

    # Use all models for ensemble
    ens_models = all_models
    ens_oofs = [results[m]["oof"] for m in ens_models]
    ens_tests = [results[m]["test"] for m in ens_models]

    def objective_ens(trial):
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in ens_models]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
        weights /= ws
        oof_e = sum(w * o for w, o in zip(weights, ens_oofs))
        return accuracy_score(y_train, oof_e.argmax(1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_ens, n_trials=5000, show_progress_bar=False)

    best_weights = np.array([study.best_params[f"w_{m}"] for m in ens_models])
    best_weights /= best_weights.sum()
    optuna_oof = sum(w * o for w, o in zip(best_weights, ens_oofs))
    optuna_test = sum(w * o for w, o in zip(best_weights, ens_tests))

    acc = accuracy_score(y_train, optuna_oof.argmax(1))
    f1 = f1_score(y_train, optuna_oof.argmax(1), average="macro")
    print(f"\n  Optuna ensemble: Acc={acc:.4f} F1={f1:.4f}")
    sig_w = {m: f"{w:.3f}" for m, w in zip(ens_models, best_weights) if w > 0.01}
    print(f"  Significant weights: {sig_w}")
    results["optuna_ens"] = {"oof": optuna_oof, "test": optuna_test, "acc": acc, "f1": f1}

    # Also try geometric mean ensemble (power ensemble)
    print("\n  Geometric mean ensemble...")
    top_k = min(15, len(all_models))
    top_m = all_models[:top_k]

    def objective_geo(trial):
        weights = [trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in top_m]
        weights = np.array(weights)
        ws = weights.sum()
        if ws < 1e-8: return 0.0
        weights /= ws
        # Geometric mean: exp(sum(w * log(p)))
        log_probs = [np.log(results[m]["oof"] + 1e-8) for m in top_m]
        oof_e = np.exp(sum(w * lp for w, lp in zip(weights, log_probs)))
        # Normalize per sample
        oof_e = oof_e / (oof_e.sum(axis=1, keepdims=True) + 1e-8)
        return accuracy_score(y_train, oof_e.argmax(1))

    study_geo = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED + 2))
    study_geo.optimize(objective_geo, n_trials=3000, show_progress_bar=False)

    geo_weights = np.array([study_geo.best_params[f"w_{m}"] for m in top_m])
    geo_weights /= geo_weights.sum()
    log_probs_oof = [np.log(results[m]["oof"] + 1e-8) for m in top_m]
    log_probs_test = [np.log(results[m]["test"] + 1e-8) for m in top_m]
    geo_oof = np.exp(sum(w * lp for w, lp in zip(geo_weights, log_probs_oof)))
    geo_oof = geo_oof / (geo_oof.sum(axis=1, keepdims=True) + 1e-8)
    geo_test = np.exp(sum(w * lp for w, lp in zip(geo_weights, log_probs_test)))
    geo_test = geo_test / (geo_test.sum(axis=1, keepdims=True) + 1e-8)

    geo_acc = accuracy_score(y_train, geo_oof.argmax(1))
    geo_f1 = f1_score(y_train, geo_oof.argmax(1), average="macro")
    print(f"  Geometric ensemble: Acc={geo_acc:.4f} F1={geo_f1:.4f}")
    results["geo_ens"] = {"oof": geo_oof, "test": geo_test, "acc": geo_acc, "f1": geo_f1}

    # Blend arithmetic and geometric
    print("\n  Blending arithmetic + geometric ensembles...")
    best_blend_acc = max(acc, geo_acc)
    best_blend_oof = optuna_oof if acc >= geo_acc else geo_oof
    best_blend_test = optuna_test if acc >= geo_acc else geo_test
    best_blend_name = "optuna" if acc >= geo_acc else "geo"

    for alpha in np.arange(0.1, 1.0, 0.1):
        blend_oof = alpha * optuna_oof + (1 - alpha) * geo_oof
        ba = accuracy_score(y_train, blend_oof.argmax(1))
        if ba > best_blend_acc:
            best_blend_acc = ba
            best_blend_oof = blend_oof
            best_blend_test = alpha * optuna_test + (1 - alpha) * geo_test
            best_blend_name = f"arith({alpha:.1f})+geo({1-alpha:.1f})"

    print(f"  Best arith+geo blend: {best_blend_name} Acc={best_blend_acc:.4f}")

    # === Phase 9: Cross-version blending ===
    print("\n" + "=" * 60)
    print("Phase 9: Cross-version blending")
    print("=" * 60)

    cross_oofs = {"v10": best_blend_oof}
    cross_tests = {"v10": best_blend_test}

    # Load v4
    v4_path = os.path.join(OUT_DIR, "v4_oof_probs.npy")
    if os.path.exists(v4_path):
        cross_oofs["v4"] = np.load(v4_path)
        cross_tests["v4"] = np.load(os.path.join(OUT_DIR, "v4_test_probs.npy"))
        print(f"  Loaded v4 predictions")

    # Load v8
    v8_path = os.path.join(OUT_DIR, "v8_best_oof_probs.npy")
    if os.path.exists(v8_path):
        cross_oofs["v8"] = np.load(v8_path)
        cross_tests["v8"] = np.load(os.path.join(OUT_DIR, "v8_best_test_probs.npy"))
        print(f"  Loaded v8 predictions")

    # Optuna cross-blend
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
    study_cross.optimize(objective_cross, n_trials=2000, show_progress_bar=False)

    cross_weights = np.array([study_cross.best_params[f"w_{n}"] for n in cross_names])
    cross_weights /= cross_weights.sum()
    cross_oof = sum(w * o for w, o in zip(cross_weights, cross_oof_list))
    cross_test = sum(w * o for w, o in zip(cross_weights, cross_test_list))
    cross_acc = accuracy_score(y_train, cross_oof.argmax(1))
    cross_f1 = f1_score(y_train, cross_oof.argmax(1), average="macro")
    print(f"  Cross blend: Acc={cross_acc:.4f} F1={cross_f1:.4f}")
    cross_w_str = {n: f"{w:.3f}" for n, w in zip(cross_names, cross_weights)}
    print(f"  Weights: {cross_w_str}")

    # === Phase 10: Threshold optimization ===
    print("\n" + "=" * 60)
    print("Phase 10: Threshold optimization")
    print("=" * 60)

    # Try threshold optimization on the best result
    candidates = {
        "optuna_ens": (results["optuna_ens"]["oof"], results["optuna_ens"]["test"]),
        "cross_blend": (cross_oof, cross_test),
        "geo_ens": (geo_oof, geo_test),
    }
    if best_blend_name != "optuna" and best_blend_name != "geo":
        candidates["arith_geo"] = (best_blend_oof, best_blend_test)

    best_overall_acc = 0
    best_overall_oof = None
    best_overall_test = None
    best_overall_name = None

    for cname, (c_oof, c_test) in candidates.items():
        c_acc = accuracy_score(y_train, c_oof.argmax(1))
        if c_acc > best_overall_acc:
            best_overall_acc = c_acc
            best_overall_oof = c_oof
            best_overall_test = c_test
            best_overall_name = cname

    print(f"  Starting from: {best_overall_name} (Acc={best_overall_acc:.4f})")

    # Grid search thresholds
    improved = False
    best_thresh_oof = best_overall_oof.copy()
    best_thresh_test = best_overall_test.copy()

    for bias_h in np.arange(-0.15, 0.16, 0.01):
        for bias_r in np.arange(-0.10, 0.11, 0.01):
            bias_o = -(bias_h + bias_r)  # Zero-sum
            adjusted = best_overall_oof.copy()
            adjusted[:, 0] += bias_h
            adjusted[:, 1] += bias_r
            adjusted[:, 2] += bias_o
            ta = accuracy_score(y_train, adjusted.argmax(1))
            if ta > best_overall_acc:
                best_overall_acc = ta
                best_thresh_oof = adjusted
                adjusted_test = best_overall_test.copy()
                adjusted_test[:, 0] += bias_h
                adjusted_test[:, 1] += bias_r
                adjusted_test[:, 2] += bias_o
                best_thresh_test = adjusted_test
                improved = True
                print(f"  Threshold: H={bias_h:+.2f} R={bias_r:+.2f} O={bias_o:+.2f} -> Acc={ta:.4f}")

    if improved:
        best_overall_oof = best_thresh_oof
        best_overall_test = best_thresh_test
        best_overall_name += "+thresh"
    else:
        print("  No threshold improvement.")

    # === Final results ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    final_acc = accuracy_score(y_train, best_overall_oof.argmax(1))
    final_f1 = f1_score(y_train, best_overall_oof.argmax(1), average="macro")
    print(f"\n*** Best: {best_overall_name} | Acc={final_acc:.4f} F1={final_f1:.4f} ***")
    print(classification_report(y_train, best_overall_oof.argmax(1), target_names=LABELS))

    # Save predictions
    np.save(os.path.join(OUT_DIR, "v10_best_oof_probs.npy"), best_overall_oof)
    np.save(os.path.join(OUT_DIR, "v10_best_test_probs.npy"), best_overall_test)

    # Generate submission
    test_preds = best_overall_test.argmax(1)
    result_csv = os.path.join(ROOT, "result.csv")
    if os.path.exists(result_csv):
        template = pd.read_csv(result_csv)
    else:
        template = pd.DataFrame({"filename": val_base_ids})

    template["predict"] = [ID2LBL[p] for p in test_preds]

    sub_name = f"submission_v10_acc_{final_acc:.4f}_f1_{final_f1:.4f}".replace(".", "p") + ".csv"
    sub_path = os.path.join(OUT_DIR, sub_name)
    template.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    elapsed = time.time() - t0
    print(f"Done! ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
