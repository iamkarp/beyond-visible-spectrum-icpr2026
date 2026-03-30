"""
Fine-tune pretrained CNN on all 3 modalities for crop disease classification.
Blend with tabular model predictions for best results.
"""
import os, re, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import tifffile as tiff
import cv2

warnings.filterwarnings("ignore")

ROOT = "/Users/macbook/Library/CloudStorage/GoogleDrive-jason.karpeles@pmg.com/My Drive/Projects/Beyond Visible Spectrum"
OUT_DIR = os.path.join(ROOT, "output")
SEED = 42
N_FOLDS = 5
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
IMG_SIZE = 128
HS_DROP_FIRST = 10
HS_DROP_LAST = 14
HS_TARGET_CH = 101

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)


def read_tiff(path):
    arr = tiff.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D, got {arr.shape}")
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


class MultimodalDataset(Dataset):
    def __init__(self, df, is_train=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.augment = augment

        self.rgb_transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.rgb_transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # RGB
        rgb_path = row.get("rgb")
        if pd.notna(rgb_path):
            img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.augment:
                rgb_tensor = self.rgb_transform_train(img_rgb)
            else:
                rgb_tensor = self.rgb_transform_val(img_rgb)
        else:
            rgb_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        # MS (5 bands -> normalize to [0,1])
        ms_path = row.get("ms")
        if pd.notna(ms_path):
            ms_arr = read_tiff(ms_path)
            ms_arr = cv2.resize(ms_arr, (IMG_SIZE, IMG_SIZE))
            # Per-band normalization
            for c in range(ms_arr.shape[2]):
                mn, mx = ms_arr[:,:,c].min(), ms_arr[:,:,c].max()
                ms_arr[:,:,c] = (ms_arr[:,:,c] - mn) / (mx - mn + 1e-8)
            ms_tensor = torch.from_numpy(ms_arr.transpose(2, 0, 1))
            if self.augment:
                if np.random.random() > 0.5:
                    ms_tensor = ms_tensor.flip(-1)
                if np.random.random() > 0.5:
                    ms_tensor = ms_tensor.flip(-2)
        else:
            ms_tensor = torch.zeros(5, IMG_SIZE, IMG_SIZE)

        # HS (reduce to manageable channels via PCA-like approach)
        hs_path = row.get("hs")
        if pd.notna(hs_path):
            hs_arr = read_tiff(hs_path)
            B = hs_arr.shape[2]
            if B > (HS_DROP_FIRST + HS_DROP_LAST + 1):
                hs_arr = hs_arr[:, :, HS_DROP_FIRST:B - HS_DROP_LAST]
            C = hs_arr.shape[2]
            if C > HS_TARGET_CH:
                hs_arr = hs_arr[:, :, :HS_TARGET_CH]
            elif C < HS_TARGET_CH:
                pad = np.zeros((hs_arr.shape[0], hs_arr.shape[1], HS_TARGET_CH - C), dtype=np.float32)
                hs_arr = np.concatenate([hs_arr, pad], axis=2)
            # Resize spatial dims
            hs_arr = cv2.resize(hs_arr, (IMG_SIZE, IMG_SIZE))
            # Sample 20 evenly spaced bands to keep model manageable
            band_idx = np.linspace(0, HS_TARGET_CH - 1, 20, dtype=int)
            hs_arr = hs_arr[:, :, band_idx]
            # Normalize
            for c in range(hs_arr.shape[2]):
                mn, mx = hs_arr[:,:,c].min(), hs_arr[:,:,c].max()
                hs_arr[:,:,c] = (hs_arr[:,:,c] - mn) / (mx - mn + 1e-8)
            hs_tensor = torch.from_numpy(hs_arr.transpose(2, 0, 1))
            if self.augment:
                if np.random.random() > 0.5:
                    hs_tensor = hs_tensor.flip(-1)
                if np.random.random() > 0.5:
                    hs_tensor = hs_tensor.flip(-2)
        else:
            hs_tensor = torch.zeros(20, IMG_SIZE, IMG_SIZE)

        label = LBL2ID[row["label"]] if "label" in row and pd.notna(row.get("label")) else -1
        return rgb_tensor, ms_tensor, hs_tensor, label


class MultimodalModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # RGB encoder: EfficientNet-B0 pretrained
        self.rgb_enc = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.rgb_enc.classifier = nn.Identity()
        rgb_dim = 1280

        # MS encoder: Small CNN
        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        ms_dim = 128

        # HS encoder: Small CNN for 20 sampled bands
        self.hs_enc = nn.Sequential(
            nn.Conv2d(20, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        hs_dim = 256

        # Fusion
        total_dim = rgb_dim + ms_dim + hs_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb, ms, hs):
        rgb_feat = self.rgb_enc(rgb)
        ms_feat = self.ms_enc(ms)
        hs_feat = self.hs_enc(hs)
        fused = torch.cat([rgb_feat, ms_feat, hs_feat], dim=1)
        return self.classifier(fused)


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


def make_df(idx, has_labels=True):
    rows = []
    for bid, paths in idx.items():
        row = {"base_id": bid, **paths}
        if has_labels:
            m = re.match(r"^(Health|Rust|Other)_", bid)
            if m:
                row["label"] = m.group(1)
            else:
                continue
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Fine-tune multimodal CNN")
    print("=" * 60)

    train_idx = build_index(ROOT, "train")
    val_idx = build_index(ROOT, "val")
    train_df = make_df(train_idx, has_labels=True)
    val_df = make_df(val_idx, has_labels=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    y_train = np.array([LBL2ID[l] for l in train_df["label"]])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_probs = np.zeros((len(train_df), 3))
    test_probs = np.zeros((len(val_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(np.zeros(len(train_df)), y_train)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        train_fold = train_df.iloc[tri]
        val_fold = train_df.iloc[vai]

        train_ds = MultimodalDataset(train_fold, is_train=True, augment=True)
        val_ds = MultimodalDataset(val_fold, is_train=True, augment=False)
        test_ds = MultimodalDataset(val_df, is_train=False, augment=False)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

        model = MultimodalModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0
        best_state = None

        for epoch in range(N_EPOCHS):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for rgb, ms, hs, labels in train_dl:
                rgb, ms, hs = rgb.to(device), ms.to(device), hs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(rgb, ms, hs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += len(labels)

            scheduler.step()

            # Validate
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for rgb, ms, hs, labels in val_dl:
                    rgb, ms, hs = rgb.to(device), ms.to(device), hs.to(device)
                    logits = model(rgb, ms, hs)
                    val_preds.append(logits.cpu())
                    val_labels.append(labels)

            val_preds = torch.cat(val_preds)
            val_labels = torch.cat(val_labels).numpy()
            val_acc = accuracy_score(val_labels, val_preds.argmax(1).numpy())

            if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
                print(f"  Epoch {epoch+1}: train_loss={train_loss/train_total:.4f} "
                      f"train_acc={train_correct/train_total:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best model and get predictions
        model.load_state_dict(best_state)
        model.eval()

        # OOF predictions
        val_probs_list = []
        with torch.no_grad():
            for rgb, ms, hs, _ in val_dl:
                rgb, ms, hs = rgb.to(device), ms.to(device), hs.to(device)
                logits = model(rgb, ms, hs)
                val_probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        val_probs = np.vstack(val_probs_list)
        oof_probs[vai] = val_probs

        # Test predictions
        test_probs_list = []
        with torch.no_grad():
            for rgb, ms, hs, _ in test_dl:
                rgb, ms, hs = rgb.to(device), ms.to(device), hs.to(device)
                logits = model(rgb, ms, hs)
                test_probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        fold_test = np.vstack(test_probs_list)
        test_probs += fold_test / N_FOLDS

        fold_acc = accuracy_score(y_train[vai], val_probs.argmax(1))
        fold_f1 = f1_score(y_train[vai], val_probs.argmax(1), average="macro")
        print(f"  Fold {fold+1} best: Acc={fold_acc:.4f} F1={fold_f1:.4f}")

    # Overall OOF
    oof_acc = accuracy_score(y_train, oof_probs.argmax(1))
    oof_f1 = f1_score(y_train, oof_probs.argmax(1), average="macro")
    print(f"\n{'='*60}")
    print(f"CNN OOF: Acc={oof_acc:.4f} F1={oof_f1:.4f}")
    print(classification_report(y_train, oof_probs.argmax(1), target_names=LABELS, digits=4))

    # Save CNN predictions
    np.save(os.path.join(OUT_DIR, "cnn_finetune_oof_probs.npy"), oof_probs)
    np.save(os.path.join(OUT_DIR, "cnn_finetune_test_probs.npy"), test_probs)

    # Blend with tabular (v4 + v6)
    v4_oof = np.load(os.path.join(OUT_DIR, "v4_oof_probs.npy"))
    v4_test = np.load(os.path.join(OUT_DIR, "v4_test_probs.npy"))
    v6_oof = np.load(os.path.join(OUT_DIR, "v6_best_oof_probs.npy"))
    v6_test = np.load(os.path.join(OUT_DIR, "v6_best_test_probs.npy"))

    print("\nBlending with tabular models:")
    for w_cnn in [0.2, 0.3, 0.4, 0.5]:
        w_tab = 1 - w_cnn
        for w4_frac in [0.5, 0.6]:
            blend_oof = w_cnn * oof_probs + w_tab * (w4_frac * v4_oof + (1-w4_frac) * v6_oof)
            acc = accuracy_score(y_train, blend_oof.argmax(1))
            f1 = f1_score(y_train, blend_oof.argmax(1), average="macro")
            print(f"  CNN({w_cnn})+v4({w_tab*w4_frac:.1f})+v6({w_tab*(1-w4_frac):.1f}): Acc={acc:.4f} F1={f1:.4f}")

    # Best simple blend: try all combos
    best_acc, best_blend_oof, best_blend_test, best_desc = 0, None, None, ""
    for wc in np.arange(0.0, 0.6, 0.05):
        for w4 in np.arange(0.0, 1.0 - wc, 0.05):
            w6 = 1.0 - wc - w4
            if w6 < 0:
                continue
            blend = wc * oof_probs + w4 * v4_oof + w6 * v6_oof
            acc = accuracy_score(y_train, blend.argmax(1))
            if acc > best_acc:
                best_acc = acc
                best_blend_oof = blend
                best_blend_test = wc * test_probs + w4 * v4_test + w6 * v6_test
                best_desc = f"CNN({wc:.2f})+v4({w4:.2f})+v6({w6:.2f})"

    best_f1 = f1_score(y_train, best_blend_oof.argmax(1), average="macro")
    print(f"\nBest blend: {best_desc}")
    print(f"  Acc={best_acc:.4f} F1={best_f1:.4f}")
    print(classification_report(y_train, best_blend_oof.argmax(1), target_names=LABELS, digits=4))

    # Save
    sub_ids = []
    for _, r in val_df.iterrows():
        if pd.notna(r.get("hs")):
            sub_ids.append(os.path.basename(r["hs"]))
        elif pd.notna(r.get("ms")):
            sub_ids.append(os.path.basename(r["ms"]))
        else:
            sub_ids.append(os.path.basename(r["rgb"]))

    preds = [ID2LBL[p] for p in best_blend_test.argmax(1)]
    sub = pd.DataFrame({"Id": sub_ids, "Category": preds})
    a_s = f"{best_acc:.4f}".replace(".", "p")
    f_s = f"{best_f1:.4f}".replace(".", "p")
    sub_path = os.path.join(OUT_DIR, f"submission_finetune_blend_acc_{a_s}_f1_{f_s}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")

    # Also save CNN-only
    cnn_preds = [ID2LBL[p] for p in test_probs.argmax(1)]
    sub_cnn = pd.DataFrame({"Id": sub_ids, "Category": cnn_preds})
    a_c = f"{oof_acc:.4f}".replace(".", "p")
    sub_cnn.to_csv(os.path.join(OUT_DIR, f"submission_finetune_cnn_acc_{a_c}.csv"), index=False)

    print("Done!")


if __name__ == "__main__":
    main()
