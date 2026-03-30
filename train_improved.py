"""
Beyond Visible Spectrum: AI for Agriculture 2026
Improved Multimodal Crop Disease Classification

Key improvements over baseline:
1. 5-Fold Stratified CV with OOF predictions
2. Pretrained encoders for ALL modalities (not just RGB)
3. Spectral vegetation indices (NDVI, NDRE, GNDVI, etc.)
4. Attention-gated cross-modal fusion
5. MixUp augmentation
6. Cosine annealing LR with warmup + label smoothing
7. 8x TTA (4 rotations x 2 flips)
8. Fold-ensemble averaging probabilities
9. Better HS encoder with spectral attention

Designed for M4 Max (MPS) with ~15-20GB RAM usage.
"""

import os
import re
import random
import time
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

import cv2
import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

import timm

# ============================================================
# Configuration
# ============================================================
@dataclass
class CFG:
    ROOT: str = "/Users/macbook/Library/CloudStorage/GoogleDrive-jason.karpeles@pmg.com/My Drive/Projects/Beyond Visible Spectrum"
    TRAIN_DIR: str = "train"
    VAL_DIR: str = "val"

    USE_RGB: bool = True
    USE_MS: bool = True
    USE_HS: bool = True

    IMG_SIZE: int = 128  # Moderate upscale from 64x64 (not 224 - too much interpolation)
    HS_SIZE: int = 64    # Upscale HS from 32x32 to 64x64

    BATCH_SIZE: int = 16
    EPOCHS: int = 40
    LR: float = 2e-4
    MIN_LR: float = 1e-6
    WD: float = 1e-3
    WARMUP_EPOCHS: int = 3
    LABEL_SMOOTHING: float = 0.1
    MIXUP_ALPHA: float = 0.4
    CUTMIX_ALPHA: float = 1.0
    MIXUP_PROB: float = 0.5  # probability of applying mixup vs cutmix

    N_FOLDS: int = 5
    NUM_WORKERS: int = 4
    SEED: int = 42

    # Backbones
    RGB_BACKBONE: str = "convnext_tiny.fb_in22k_ft_in1k"
    MS_BACKBONE: str = "convnext_tiny.fb_in22k_ft_in1k"

    # HS params
    HS_DROP_FIRST: int = 10
    HS_DROP_LAST: int = 14
    HS_EMBED_DIM: int = 384

    # TTA
    TTA_ENABLED: bool = True

    OUT_DIR: str = "/Users/macbook/Library/CloudStorage/GoogleDrive-jason.karpeles@pmg.com/My Drive/Projects/Beyond Visible Spectrum/output"


LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}


# ============================================================
# Utilities
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# Data indexing
# ============================================================
def list_files(folder: str, exts: Tuple[str, ...]) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(exts)
    ])


def base_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_label(bid: str) -> Optional[str]:
    m = re.match(r"^(Health|Rust|Other)_", bid)
    return m.group(1) if m else None


def build_index(root: str, split: str) -> Dict[str, Dict[str, str]]:
    split_dir = os.path.join(root, split)
    idx: Dict[str, Dict[str, str]] = {}
    for mod, exts in [("rgb", (".png", ".jpg")), ("ms", (".tif", ".tiff")), ("hs", (".tif", ".tiff"))]:
        mod_dir = os.path.join(split_dir, mod.upper())
        for p in list_files(mod_dir, exts):
            idx.setdefault(base_id(p), {})[mod] = p
    return idx


def make_train_df(train_idx: Dict) -> pd.DataFrame:
    rows = []
    for bid, paths in train_idx.items():
        lab = parse_label(bid)
        if lab is None:
            continue
        rows.append({"base_id": bid, "label": lab, **paths})
    return pd.DataFrame(rows)


def make_val_df(val_idx: Dict) -> pd.DataFrame:
    rows = []
    for bid, paths in val_idx.items():
        rows.append({"base_id": bid, **paths})
    return pd.DataFrame(rows)


# ============================================================
# Reading & preprocessing
# ============================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_rgb(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    # Normalize
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)  # (3,H,W)


def read_tiff(path: str) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D TIFF, got {arr.shape}")
    # Ensure (H,W,C) format
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def normalize_bands(arr: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    H, W, C = arr.shape
    flat = arr.reshape(-1, C)
    mn = flat.min(axis=0, keepdims=True)
    mx = flat.max(axis=0, keepdims=True)
    denom = mx - mn
    denom[denom < 1e-6] = 1.0
    arr = (arr - mn.reshape(1, 1, C)) / denom.reshape(1, 1, C)
    return np.clip(arr, 0.0, 1.0)


def compute_vegetation_indices(ms: np.ndarray) -> np.ndarray:
    """
    Compute vegetation indices from multispectral data.
    MS bands: Blue(0), Green(1), Red(2), RedEdge(3), NIR(4)
    Returns (H,W,N_indices) float32 array, values clipped to [-1,1] then scaled to [0,1].
    """
    eps = 1e-6
    blue, green, red, rededge, nir = [ms[:, :, i].astype(np.float32) for i in range(5)]

    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red + eps)
    # NDRE: (NIR - RedEdge) / (NIR + RedEdge)
    ndre = (nir - rededge) / (nir + rededge + eps)
    # GNDVI: (NIR - Green) / (NIR + Green)
    gndvi = (nir - green) / (nir + green + eps)
    # SAVI: 1.5 * (NIR - Red) / (NIR + Red + 0.5)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    # EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
    # MCARI: ((RedEdge - Red) - 0.2*(RedEdge - Green)) * (RedEdge / (Red + eps))
    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + eps))

    indices = np.stack([ndvi, ndre, gndvi, savi, evi, mcari], axis=-1)
    # Clip and scale to [0, 1]
    indices = np.clip(indices, -1.0, 1.0)
    indices = (indices + 1.0) / 2.0
    return indices.astype(np.float32)


def read_ms(path: str, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (5+6, H, W) = MS bands + vegetation indices."""
    arr = read_tiff(path)  # (H,W,5)
    # Compute indices BEFORE normalization (need raw values for ratios)
    indices = compute_vegetation_indices(arr)  # (H,W,6)
    # Now normalize bands
    arr_norm = normalize_bands(arr)  # (H,W,5)
    # Concatenate: 5 bands + 6 indices = 11 channels
    combined = np.concatenate([arr_norm, indices], axis=-1)  # (H,W,11)
    if combined.shape[0] != size or combined.shape[1] != size:
        combined = cv2.resize(combined, (size, size), interpolation=cv2.INTER_LINEAR)
    return combined.transpose(2, 0, 1)  # (11,H,W)


HS_TARGET_CHANNELS = 101  # Fixed output channels for HS after trimming

def read_hs(path: str, drop_first: int, drop_last: int, size: int) -> np.ndarray:
    arr = read_tiff(path)  # (H,W,125 or 126)
    B = arr.shape[2]
    if B > (drop_first + drop_last + 1):
        arr = arr[:, :, drop_first:B - drop_last]
    arr = normalize_bands(arr)
    # Standardize channel count - some images have 125 bands, others 126
    C = arr.shape[2]
    if C > HS_TARGET_CHANNELS:
        arr = arr[:, :, :HS_TARGET_CHANNELS]
    elif C < HS_TARGET_CHANNELS:
        pad = np.zeros((arr.shape[0], arr.shape[1], HS_TARGET_CHANNELS - C), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=2)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    return arr.transpose(2, 0, 1)  # (101,H,W)


# ============================================================
# Augmentations
# ============================================================
def random_flip_rotate(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Apply consistent random flips and rotations to list of (C,H,W) arrays."""
    k = random.randint(0, 3)
    do_h = random.random() < 0.5
    do_v = random.random() < 0.5

    results = []
    for arr in arrays:
        if arr is None:
            results.append(None)
            continue
        # arr: (C,H,W)
        if k:
            arr = np.rot90(arr, k, axes=(1, 2)).copy()
        if do_h:
            arr = np.flip(arr, axis=2).copy()
        if do_v:
            arr = np.flip(arr, axis=1).copy()
        results.append(arr)
    return results


def mixup_data(x_list, y, alpha=0.4):
    """MixUp: linearly interpolate pairs of samples and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = y.size(0)
    index = torch.randperm(batch_size)

    mixed_x = []
    for x in x_list:
        mixed_x.append(lam * x + (1 - lam) * x[index])

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x_list, y, alpha=1.0):
    """CutMix: replace a random patch with another sample's patch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = y.size(0)
    index = torch.randperm(batch_size)

    # Get bounding box for the first tensor in x_list that exists
    ref = x_list[0]
    _, _, H, W = ref.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = []
    for x in x_list:
        x_mixed = x.clone()
        _, _, xH, xW = x.shape
        # Scale box proportionally
        sx1 = int(bbx1 * xW / W)
        sy1 = int(bby1 * xH / H)
        sx2 = int(bbx2 * xW / W)
        sy2 = int(bby2 * xH / H)
        x_mixed[:, :, sy1:sy2, sx1:sx2] = x[index, :, sy1:sy2, sx1:sx2]
        mixed_x.append(x_mixed)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ============================================================
# Dataset
# ============================================================
class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, hs_channels: int, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.hs_channels = hs_channels
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        bid = row["base_id"]

        # Read modalities
        x_rgb = None
        x_ms = None
        x_hs = None

        if self.cfg.USE_RGB and pd.notna(row.get("rgb")):
            x_rgb = read_rgb(row["rgb"], self.cfg.IMG_SIZE)

        if self.cfg.USE_MS and pd.notna(row.get("ms")):
            x_ms = read_ms(row["ms"], self.cfg.IMG_SIZE)

        if self.cfg.USE_HS and pd.notna(row.get("hs")):
            x_hs = read_hs(row["hs"], self.cfg.HS_DROP_FIRST, self.cfg.HS_DROP_LAST, self.cfg.HS_SIZE)

        # Apply augmentations (train only)
        if self.is_train:
            arrays = random_flip_rotate([x_rgb, x_ms, x_hs])
            x_rgb, x_ms, x_hs = arrays

        # Create zero placeholders for missing modalities
        if x_rgb is None:
            x_rgb = np.zeros((3, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE), dtype=np.float32)
        if x_ms is None:
            x_ms = np.zeros((11, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE), dtype=np.float32)
        if x_hs is None:
            x_hs = np.zeros((self.hs_channels, self.cfg.HS_SIZE, self.cfg.HS_SIZE), dtype=np.float32)

        out = {
            "id": bid,
            "rgb": torch.from_numpy(x_rgb.astype(np.float32)),
            "ms": torch.from_numpy(x_ms.astype(np.float32)),
            "hs": torch.from_numpy(x_hs.astype(np.float32)),
        }

        if "label" in row:
            out["y"] = torch.tensor(LBL2ID[row["label"]], dtype=torch.long)

        return out


# ============================================================
# Model Architecture
# ============================================================
class SpectralAttentionEncoder(nn.Module):
    """
    Encoder for hyperspectral data with spectral attention.
    Uses 1x1 convolutions to reduce spectral dimensions,
    then spatial processing with residual blocks.
    """
    def __init__(self, in_ch: int, embed_dim: int = 384):
        super().__init__()
        # Spectral reduction with attention
        self.spectral_squeeze = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Spectral attention: learn which reduced bands are important
        self.spectral_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.Sigmoid(),
        )
        # Spatial processing
        self.spatial = nn.Sequential(
            # Block 1
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.spectral_squeeze(x)  # (B, 32, H, W)
        # Apply spectral attention
        attn = self.spectral_attn(x)  # (B, 32)
        x = x * attn.unsqueeze(-1).unsqueeze(-1)
        x = self.spatial(x)  # (B, 256, 1, 1)
        return self.head(x)  # (B, embed_dim)


class AdaptedBackbone(nn.Module):
    """Pretrained backbone adapted for non-3-channel input."""
    def __init__(self, backbone_name: str, in_channels: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        # Replace first conv layer to accept different number of input channels
        if in_channels != 3:
            old_conv = None
            # Find the first conv layer (different models have different structures)
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    old_conv = module
                    old_conv_name = name
                    break
            if old_conv is not None:
                new_conv = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                # Initialize: repeat pretrained weights across new channels
                with torch.no_grad():
                    # Average the pretrained weights and repeat for new channels
                    old_weight = old_conv.weight.data  # (out, 3, kH, kW)
                    mean_weight = old_weight.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
                    new_conv.weight.data = mean_weight.repeat(1, in_channels, 1, 1)
                    if old_conv.bias is not None:
                        new_conv.bias.data = old_conv.bias.data.clone()

                # Set the new conv layer
                parts = old_conv_name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_conv)

    @property
    def num_features(self):
        return self.model.num_features

    def forward(self, x):
        return self.model(x)


class CrossModalAttention(nn.Module):
    """Lightweight cross-modal attention for fusion."""
    def __init__(self, dims: List[int], hidden: int = 256):
        super().__init__()
        total = sum(dims)
        self.gate = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, len(dims)),
            nn.Softmax(dim=-1),
        )
        self.projs = nn.ModuleList([
            nn.Linear(d, hidden) for d in dims
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features: list of (B, D_i) tensors
        concat = torch.cat(features, dim=-1)  # (B, sum(D_i))
        gate_weights = self.gate(concat)  # (B, num_modalities)

        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projs)):
            projected.append(proj(feat) * gate_weights[:, i:i+1])

        return sum(projected)  # (B, hidden)


class ImprovedMultiModalNet(nn.Module):
    def __init__(self, cfg: CFG, hs_channels: int, n_classes: int = 3):
        super().__init__()

        # RGB encoder: pretrained ConvNeXt-Tiny
        self.rgb_enc = timm.create_model(
            cfg.RGB_BACKBONE, pretrained=True, num_classes=0, global_pool="avg"
        )
        rgb_dim = self.rgb_enc.num_features

        # MS encoder: pretrained ConvNeXt-Tiny adapted for 11 channels (5 bands + 6 indices)
        self.ms_enc = AdaptedBackbone(cfg.MS_BACKBONE, in_channels=11, pretrained=True)
        ms_dim = self.ms_enc.num_features

        # HS encoder: custom spectral attention encoder
        self.hs_enc = SpectralAttentionEncoder(in_ch=hs_channels, embed_dim=cfg.HS_EMBED_DIM)
        hs_dim = cfg.HS_EMBED_DIM

        # Cross-modal attention fusion
        dims = [rgb_dim, ms_dim, hs_dim]
        fusion_hidden = 512
        self.fusion = CrossModalAttention(dims, hidden=fusion_hidden)

        # Also keep direct concat path for residual diversity
        total_dim = sum(dims)
        self.direct_proj = nn.Sequential(
            nn.Linear(total_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, rgb, ms, hs):
        f_rgb = self.rgb_enc(rgb)
        f_ms = self.ms_enc(ms)
        f_hs = self.hs_enc(hs)

        # Cross-modal attention path
        fused = self.fusion([f_rgb, f_ms, f_hs])
        # Direct concat path
        direct = self.direct_proj(torch.cat([f_rgb, f_ms, f_hs], dim=-1))

        # Combine both paths
        combined = torch.cat([fused, direct], dim=-1)
        return self.classifier(combined)


# ============================================================
# Training functions
# ============================================================
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, cfg):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        ms = batch["ms"].to(device)
        hs = batch["hs"].to(device)
        y = batch["y"].to(device)

        # Apply MixUp or CutMix
        use_mix = random.random() < 0.7  # 70% chance of using mixing
        if use_mix:
            if random.random() < cfg.MIXUP_PROB:
                [rgb, ms, hs], y_a, y_b, lam = mixup_data([rgb, ms, hs], y, cfg.MIXUP_ALPHA)
            else:
                [rgb, ms, hs], y_a, y_b, lam = cutmix_data([rgb, ms, hs], y, cfg.CUTMIX_ALPHA)
        else:
            y_a, y_b, lam = y, y, 1.0

        optimizer.zero_grad(set_to_none=True)
        logits = model(rgb, ms, hs)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    all_ids = []

    for batch in loader:
        rgb = batch["rgb"].to(device)
        ms = batch["ms"].to(device)
        hs = batch["hs"].to(device)

        logits = model(rgb, ms, hs)
        probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_ids.extend(batch["id"])

        if "y" in batch:
            all_targets.extend(batch["y"].numpy().tolist())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.array(all_preds)

    metrics = {}
    if all_targets:
        targets = np.array(all_targets)
        metrics["acc"] = accuracy_score(targets, preds)
        metrics["macro_f1"] = f1_score(targets, preds, average="macro")
        metrics["per_class_f1"] = f1_score(targets, preds, average=None)

    return all_ids, probs, preds, metrics


@torch.no_grad()
def predict_tta(model, loader, device, n_rotations=4, n_flips=2):
    """Test-Time Augmentation: 4 rotations x 2 flip states = 8 predictions."""
    model.eval()

    all_probs_accum = None
    all_ids = None
    count = 0

    for rot in range(n_rotations):
        for flip in range(n_flips):
            batch_probs = []
            batch_ids = []

            for batch in loader:
                rgb = batch["rgb"].to(device)
                ms = batch["ms"].to(device)
                hs = batch["hs"].to(device)

                # Apply TTA transforms
                if rot > 0:
                    rgb = torch.rot90(rgb, rot, dims=(2, 3))
                    ms = torch.rot90(ms, rot, dims=(2, 3))
                    hs = torch.rot90(hs, rot, dims=(2, 3))
                if flip:
                    rgb = torch.flip(rgb, dims=(3,))
                    ms = torch.flip(ms, dims=(3,))
                    hs = torch.flip(hs, dims=(3,))

                logits = model(rgb, ms, hs)
                probs = F.softmax(logits, dim=-1)
                batch_probs.append(probs.cpu().numpy())
                batch_ids.extend(batch["id"])

            probs = np.concatenate(batch_probs, axis=0)
            if all_probs_accum is None:
                all_probs_accum = probs
                all_ids = batch_ids
            else:
                all_probs_accum += probs
            count += 1

    all_probs_accum /= count
    preds = all_probs_accum.argmax(axis=1)
    return all_ids, all_probs_accum, preds


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    cfg = CFG()
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"RGB backbone: {cfg.RGB_BACKBONE}")
    print(f"MS backbone: {cfg.MS_BACKBONE}")
    print(f"Folds: {cfg.N_FOLDS}, Epochs: {cfg.EPOCHS}")
    print(f"Image sizes - RGB/MS: {cfg.IMG_SIZE}, HS: {cfg.HS_SIZE}")
    print()

    # Build data
    train_idx = build_index(cfg.ROOT, cfg.TRAIN_DIR)
    val_idx = build_index(cfg.ROOT, cfg.VAL_DIR)
    train_df = make_train_df(train_idx)
    val_df = make_val_df(val_idx)

    print(f"Train samples: {len(train_df)} | Val/Test samples: {len(val_df)}")
    print(f"Class distribution: {train_df['label'].value_counts().to_dict()}")

    # Infer HS channels
    sample_hs_path = train_df["hs"].dropna().iloc[0]
    sample_hs = read_hs(sample_hs_path, cfg.HS_DROP_FIRST, cfg.HS_DROP_LAST, cfg.HS_SIZE)
    hs_channels = sample_hs.shape[0]
    print(f"HS channels after trimming: {hs_channels}")
    print()

    # K-Fold setup
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    oof_preds = np.zeros((len(train_df), 3))
    oof_labels = np.zeros(len(train_df), dtype=np.int64)
    test_probs_all = np.zeros((len(val_df), 3))

    fold_scores = []

    for fold, (train_idx_fold, val_idx_fold) in enumerate(skf.split(train_df, train_df["label"])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{cfg.N_FOLDS}")
        print(f"{'='*60}")

        df_tr = train_df.iloc[train_idx_fold].reset_index(drop=True)
        df_va = train_df.iloc[val_idx_fold].reset_index(drop=True)

        print(f"  Train: {len(df_tr)} | Val: {len(df_va)}")

        # Datasets
        ds_tr = MultiModalDataset(df_tr, cfg, hs_channels, is_train=True)
        ds_va = MultiModalDataset(df_va, cfg, hs_channels, is_train=False)
        ds_te = MultiModalDataset(val_df, cfg, hs_channels, is_train=False)

        pin = device.type == "cuda"  # MPS doesn't support pin_memory
        dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=True,
                           num_workers=cfg.NUM_WORKERS, pin_memory=pin, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=cfg.BATCH_SIZE, shuffle=False,
                           num_workers=cfg.NUM_WORKERS, pin_memory=pin)
        dl_te = DataLoader(ds_te, batch_size=cfg.BATCH_SIZE, shuffle=False,
                           num_workers=cfg.NUM_WORKERS, pin_memory=pin)

        # Model
        model = ImprovedMultiModalNet(cfg, hs_channels=hs_channels).to(device)

        # Optimizer with layer-wise LR decay for pretrained backbones
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if "rgb_enc" in name or "ms_enc" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.LR * 0.1},  # Lower LR for pretrained
            {"params": head_params, "lr": cfg.LR},
        ], weight_decay=cfg.WD)

        # Cosine annealing scheduler with warmup
        total_steps = len(dl_tr) * cfg.EPOCHS
        warmup_steps = len(dl_tr) * cfg.WARMUP_EPOCHS
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[cfg.LR * 0.1, cfg.LR],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1000,
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

        best_f1 = -1.0
        best_epoch = 0
        patience = 10
        no_improve = 0
        best_state = None

        for ep in range(1, cfg.EPOCHS + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, dl_tr, optimizer, scheduler, criterion, device, cfg)
            _, _, _, metrics = evaluate(model, dl_va, device)
            elapsed = time.time() - t0

            f1 = metrics["macro_f1"]
            acc = metrics["acc"]
            per_f1 = metrics["per_class_f1"]

            print(f"  Ep {ep:02d}/{cfg.EPOCHS} | loss={tr_loss:.4f} | "
                  f"acc={acc:.4f} | F1={f1:.4f} "
                  f"[H:{per_f1[0]:.3f} R:{per_f1[1]:.3f} O:{per_f1[2]:.3f}] | "
                  f"{elapsed:.1f}s")

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep} (best: ep {best_epoch}, F1={best_f1:.4f})")
                break

        # Load best model
        model.load_state_dict(best_state)
        model.to(device)
        print(f"  Best epoch: {best_epoch} | Best val F1: {best_f1:.4f}")

        # OOF predictions
        val_ids, val_probs, val_preds, val_metrics = evaluate(model, dl_va, device)
        for i, idx in enumerate(val_idx_fold):
            oof_preds[idx] = val_probs[i]
            oof_labels[idx] = LBL2ID[train_df.iloc[idx]["label"]]

        # Test predictions with TTA
        if cfg.TTA_ENABLED:
            print("  Running TTA (8x)...")
            _, test_probs, _ = predict_tta(model, dl_te, device)
        else:
            _, test_probs, _, _ = evaluate(model, dl_te, device)

        test_probs_all += test_probs
        fold_scores.append(best_f1)

        # Save fold checkpoint
        torch.save(best_state, os.path.join(cfg.OUT_DIR, f"fold{fold}.pt"))

        # Clean up memory
        del model, optimizer, scheduler, best_state
        clear_memory(device)

    # Average test predictions across folds
    test_probs_all /= cfg.N_FOLDS
    test_preds = test_probs_all.argmax(axis=1)

    # Compute OOF score
    oof_pred_labels = oof_preds.argmax(axis=1)
    oof_acc = accuracy_score(oof_labels, oof_pred_labels)
    oof_f1 = f1_score(oof_labels, oof_pred_labels, average="macro")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF Macro F1: {oof_f1:.4f}")
    print(f"Per-fold F1s: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean fold F1:  {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print()
    print(classification_report(
        oof_labels, oof_pred_labels,
        target_names=LABELS,
        digits=4,
    ))

    # Build submission
    sub_ids = []
    for _, r in val_df.iterrows():
        # Use HS filename as the submission ID (or MS or RGB)
        if pd.notna(r.get("hs")):
            sub_ids.append(os.path.basename(r["hs"]))
        elif pd.notna(r.get("ms")):
            sub_ids.append(os.path.basename(r["ms"]))
        else:
            sub_ids.append(os.path.basename(r["rgb"]))

    pred_labels = [ID2LBL[p] for p in test_preds]
    sub = pd.DataFrame({"Id": sub_ids, "Category": pred_labels})

    # Save submission with OOF score in filename
    oof_f1_str = f"{oof_f1:.4f}".replace(".", "p")
    oof_acc_str = f"{oof_acc:.4f}".replace(".", "p")
    sub_name = f"submission_oof_f1_{oof_f1_str}_acc_{oof_acc_str}.csv"
    sub_path = os.path.join(cfg.OUT_DIR, sub_name)
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}")

    # Also save a copy as submission.csv for convenience
    sub.to_csv(os.path.join(cfg.OUT_DIR, "submission.csv"), index=False)

    # Save OOF predictions
    oof_df = train_df.copy()
    oof_df["oof_pred"] = [ID2LBL[p] for p in oof_pred_labels]
    oof_df["oof_prob_health"] = oof_preds[:, 0]
    oof_df["oof_prob_rust"] = oof_preds[:, 1]
    oof_df["oof_prob_other"] = oof_preds[:, 2]
    oof_df.to_csv(os.path.join(cfg.OUT_DIR, "oof_predictions.csv"), index=False)

    print("\nDone!")
    return oof_f1


if __name__ == "__main__":
    main()
