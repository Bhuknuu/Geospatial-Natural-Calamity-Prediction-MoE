#!/usr/bin/env python
"""Multi-Hazard Mixture of Experts (MoE): Heatwave and Hailstorm Prediction.

Real data, real models, no synthetic placeholders.
- Heatwave: NASA POWER daily point API / cached reanalysis grid -> ConvLSTM (hybrid L1/L2 loss).
- Hailstorm: NOAA SWDI NEXRAD Level-III hail signatures -> DAM-EfficientNet (CBAM + ECA).

Usage:
    python src/multihazard_moe.py smoke-test     # verify model architectures
    python src/multihazard_moe.py run-all        # complete end-to-end pipeline with visualizations
    python src/multihazard_moe.py download-hail  # fetch SWDI radar data
"""

from __future__ import annotations

import math
import io
import os
import sys
import json
import random
import platform
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 1. Phase A - Reproducibility & Dependency Checker
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, PyTorch for exact reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

def check_dependencies() -> dict:
    """Validate core libraries and return exact installed versions."""
    deps = {
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
    }
    import torch
    deps["PyTorch"] = torch.__version__
    import torchvision
    deps["Torchvision"] = torchvision.__version__
    import timm
    deps["TIMM"] = timm.__version__
    import numpy as np
    deps["NumPy"] = np.__version__
    import pandas as pd
    deps["Pandas"] = pd.__version__
    import scipy
    deps["SciPy"] = scipy.__version__
    import matplotlib
    deps["Matplotlib"] = matplotlib.__version__
    import requests
    deps["Requests"] = requests.__version__

    print("=" * 65)
    print("  DEPENDENCY CHECKER: EXACT INSTALLED VERSIONS")
    print("=" * 65)
    for lib, ver in deps.items():
        print(f"  {lib:<16}: {ver}")
    print("-" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Compute Device  : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Accelerator : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version    : {torch.version.cuda}")
    else:
        print("  CUDA Status     : CUDA unavailable (running on CPU)")
    print("=" * 65)
    return deps

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive for scripts and background execution
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = device.type == "cuda"
VIS_DIR = "data/visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Phase B - Heatwave Data Ingestion & Visual Verification
# ---------------------------------------------------------------------------
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = ["T2M", "T2MDEW", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "WS10M", "TS"]

def fetch_power_point(lat, lon, start, end, parameters=POWER_PARAMS, community="AG"):
    """Query NASA POWER for one coordinate point."""
    resp = requests.get(
        POWER_URL,
        params={
            "parameters": ",".join(parameters),
            "community": community,
            "longitude": lon,
            "latitude": lat,
            "start": start,
            "end": end,
            "format": "JSON",
        },
        verify=False,
        timeout=25,
    )
    resp.raise_for_status()
    payload = resp.json()["properties"]["parameter"]
    df = pd.DataFrame({p: pd.Series(v) for p, v in payload.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.replace(-999, np.nan)

def load_or_fetch_heatwave_data(cache_path="data/heatwave_delhi_2023.npz"):
    """Load cached NASA POWER grid or fetch live for New Delhi heatwave season."""
    if os.path.exists(cache_path):
        print(f"[Heatwave] Loading cached NASA POWER grid from {cache_path}...")
        npz = np.load(cache_path, allow_pickle=True)
        return npz["grid"], npz["dates"], list(npz["params"])

    print("[Heatwave] Fetching 4x4 spatial grid from NASA POWER API...")
    center_lat, center_lon = 28.6139, 77.2090
    start, end = "20230515", "20230615"
    offsets = np.linspace(-0.3, 0.3, 4)
    dates, grid = None, np.zeros((32, len(POWER_PARAMS), 4, 4), dtype=np.float32)

    for i, dlat in enumerate(offsets):
        for j, dlon in enumerate(offsets):
            df = fetch_power_point(center_lat + dlat, center_lon + dlon, start, end)
            if dates is None:
                dates = df.index.strftime("%Y-%m-%d").values
            grid[:, :, i, j] = df[POWER_PARAMS].values

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, grid=grid, dates=dates, params=POWER_PARAMS)
    print(f"[Heatwave] Cached {grid.shape} grid to {cache_path}")
    return grid, dates, POWER_PARAMS

def visualize_heatwave_dataset(grid, dates, params):
    """Visual representation check of the heatwave dataset before modeling."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for c, param in enumerate(params):
        im = axes[c].imshow(grid[-1, c], cmap="inferno" if "T" in param else "viridis")
        axes[c].set_title(f"{param} (Spatial Map)", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    # Time series of max skin temperature vs air temperature
    ax_ts = axes[7]
    ts_idx = params.index("TS")
    t2m_idx = params.index("T2M")
    ax_ts.plot(grid[:, ts_idx].mean(axis=(1, 2)), label="Skin Temp (TS)", color="crimson", lw=2)
    ax_ts.plot(grid[:, t2m_idx].mean(axis=(1, 2)), label="Air Temp (T2M)", color="darkorange", lw=2)
    ax_ts.set_title("32-Day Temporal Evolution", fontsize=11, fontweight="bold")
    ax_ts.set_xlabel("Day Index")
    ax_ts.set_ylabel("deg C")
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    plt.suptitle("Heatwave Dataset: NASA POWER Multi-Channel Verification", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, "heatwave_data_verification.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visual Check] Heatwave dataset verification saved -> {out_path}")

# ---------------------------------------------------------------------------
# 3. Phase C - Hailstorm Data Ingestion & Visual Verification
# ---------------------------------------------------------------------------
SWDI_COLUMNS = ["ZTIME", "LON", "LAT", "WSR_ID", "CELL_ID", "RANGE", "AZIMUTH", "SEVPROB", "PROB", "MAXSIZE"]
_SWDI_BASE = "https://noaa-swdi-pds.s3.amazonaws.com/hail-{year}.csv"

def load_or_fetch_hail_data(local_path="data/hail-2015.csv", sample_mb=5):
    """Load local SWDI CSV or stream sample directly from NOAA S3."""
    if os.path.exists(local_path):
        print(f"[Hailstorm] Loading SWDI records from {local_path}...")
        df = pd.read_csv(local_path, comment="#", header=None, names=SWDI_COLUMNS)
        df["ZTIME"] = pd.to_datetime(df["ZTIME"], format="%Y%m%d%H%M%S", errors="coerce")
        return df

    print(f"[Hailstorm] Streaming sample of {sample_mb} MB from NOAA S3 bucket...")
    url = _SWDI_BASE.format(year=2015)
    resp = requests.get(url, stream=True, verify=False, timeout=(10, 60))
    resp.raise_for_status()

    target_bytes = int(sample_mb * 1024 * 1024)
    buf = bytearray()
    while len(buf) < target_bytes:
        chunk = resp.raw.read(min(65536, target_bytes - len(buf)))
        if not chunk:
            break
        buf.extend(chunk)

    last_nl = buf.rfind(b"\n")
    if last_nl != -1:
        buf = buf[:last_nl + 1]

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(buf)

    df = pd.read_csv(io.BytesIO(buf), comment="#", header=None, names=SWDI_COLUMNS)
    df["ZTIME"] = pd.to_datetime(df["ZTIME"], format="%Y%m%d%H%M%S", errors="coerce")
    print(f"[Hailstorm] Downloaded and parsed {len(df):,} records -> {local_path}")
    return df

def visualize_hailstorm_dataset(df):
    """Visual representation check of NOAA radar storm cells before modeling."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Geographic radar distribution
    sample = df.head(3000)
    sc = axes[0].scatter(sample["LON"], sample["LAT"], c=sample["MAXSIZE"], cmap="magma", s=12, alpha=0.7)
    axes[0].set_title("NEXRAD Hail Cell Coordinates", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(sc, ax=axes[0], label="Max Hail Size (inches)")

    # 2. Probability and severity distribution
    axes[1].hist(df["PROB"].dropna(), bins=20, color="royalblue", alpha=0.7, label="Hail Prob %")
    axes[1].hist(df["SEVPROB"].dropna(), bins=20, color="crimson", alpha=0.5, label="Severe Prob %")
    axes[1].set_title("Detection Probability Distribution", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Probability (%)")
    axes[1].set_ylabel("Storm Cell Count")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # 3. Top active radar stations
    top_stations = df["WSR_ID"].value_counts().head(8)
    axes[2].barh(top_stations.index, top_stations.values, color="teal", alpha=0.8)
    axes[2].set_title("Top 8 Active NEXRAD Radars", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Recorded Detections")
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Hailstorm Dataset: NOAA SWDI Radar Verification", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, "hailstorm_data_verification.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visual Check] Hailstorm dataset verification saved -> {out_path}")

# ---------------------------------------------------------------------------
# 4. Phase D - Preprocessing & Rasterization
# ---------------------------------------------------------------------------
def climatology_normalize(grid):
    """Normalize atmospheric channels via climatology anomaly."""
    t, c, h, w = grid.shape
    flat = grid.reshape(t, -1)
    flat = pd.DataFrame(flat).ffill().bfill().values
    grid = flat.reshape(t, c, h, w)
    mean = grid.mean(axis=0, keepdims=True)
    std = grid.std(axis=0, keepdims=True) + 1e-6
    anomaly = (grid - mean) / std
    return anomaly.astype(np.float32), mean, std

def make_sequences(grid, seq_len=3):
    """Extract temporal sequences (N, seq_len, C, H, W) and targets (N, 1, H, W)."""
    ts_idx = POWER_PARAMS.index("TS")
    xs, ys = [], []
    for t in range(grid.shape[0] - seq_len):
        xs.append(grid[t : t + seq_len])
        ys.append(grid[t + seq_len, ts_idx : ts_idx + 1])
    return np.stack(xs), np.stack(ys)

def rasterize_scene(cells_df, extent_nmi=100, size=224, splat_sigma=2.0):
    """Vectorized scatter-max projection from polar radar detections to (3, size, size)."""
    grid = np.zeros((3, size, size), dtype=np.float32)
    az_rad = np.deg2rad(cells_df["AZIMUTH"].values)
    x = cells_df["RANGE"].values * np.sin(az_rad)
    y = cells_df["RANGE"].values * np.cos(az_rad)
    px = ((x + extent_nmi) / (2 * extent_nmi) * (size - 1)).astype(int).clip(0, size - 1)
    py = ((extent_nmi - y) / (2 * extent_nmi) * (size - 1)).astype(int).clip(0, size - 1)

    for ch, col in enumerate(["PROB", "SEVPROB", "MAXSIZE"]):
        np.maximum.at(grid[ch], (py, px), cells_df[col].values.astype(np.float32))
        grid[ch] = gaussian_filter(grid[ch], sigma=splat_sigma)
    return grid

def build_hail_dataset(df, time_bucket="5min", max_scenes=100):
    """Group SWDI detections into radar volume scenes and generate binary labels."""
    df = df.dropna(subset=["ZTIME"]).copy()
    df["scene_id"] = df["WSR_ID"] + "_" + df["ZTIME"].dt.floor(time_bucket).astype(str)
    scenes, labels = [], []
    for _, group in df.groupby("scene_id"):
        scenes.append(rasterize_scene(group))
        is_hail = ((group["MAXSIZE"] > 0) & (group["PROB"] == 100)).any()
        labels.append(int(is_hail))
        if len(scenes) >= max_scenes:
            break
    return np.stack(scenes), np.array(labels)

def visualize_preprocessed_raster(scene, label):
    """Visual check of preprocessed 3-channel radar raster tensor."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ["Channel 0: PROB", "Channel 1: SEVPROB", "Channel 2: MAXSIZE"]
    for c in range(3):
        im = axes[c].imshow(scene[c], cmap="viridis")
        axes[c].set_title(names[c], fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    plt.suptitle(f"Rasterized Radar Scene (Label: {'HAIL' if label==1 else 'NO-HAIL'})", fontsize=13, y=0.98)
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, "rasterized_radar_scene.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visual Check] Rasterized radar scene saved -> {out_path}")

# ---------------------------------------------------------------------------
# 5. Phase E - Model Architectures
# ---------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, height, width, device):
        shape = (batch, self.hidden_channels, height, width)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

class HeatwaveConvLSTM(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=(32, 64)):
        super().__init__()
        self.cells = nn.ModuleList()
        prev = in_channels
        for h in hidden_channels:
            self.cells.append(ConvLSTMCell(prev, h))
            prev = h
        self.project = nn.Conv2d(prev, 1, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        states = [cell.init_state(b, h, w, x.device) for cell in self.cells]
        hidden_history = []
        for step in range(t):
            inp = x[:, step]
            for i, cell in enumerate(self.cells):
                hs, cs = states[i]
                hs, cs = cell(inp, hs, cs)
                states[i] = (hs, cs)
                inp = hs
            hidden_history.append(states[-1][0])
        return self.project(states[-1][0]), hidden_history

class ChannelAttention(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        return x * torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, k=7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 7, padding=3, bias=False)
        self.combine = nn.Conv2d(3 * in_channels, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        c1, c2, c3 = self.conv1(x), self.conv2(x), self.conv3(x)
        return x * torch.sigmoid(self.combine(torch.cat([c1, c2, c3], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.ca = ChannelAttention(channels, r=r)
        self.sa = SpatialAttention(channels)

    def forward(self, x):
        return self.sa(self.ca(x))

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = k if k % 2 else k + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)

def replace_se_with_eca(module):
    """Swap every SqueezeExcite block for ECA in-place."""
    for name, child in module.named_children():
        if child.__class__.__name__ == "SqueezeExcite":
            channels = child.conv_reduce.in_channels
            setattr(module, name, ECA(channels))
        else:
            replace_se_with_eca(child)

class DAMEfficientNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        backbone = timm.create_model("efficientnet_b1", pretrained=pretrained, num_classes=num_classes)
        stem_channels = backbone.conv_stem.out_channels
        self.stem = nn.Sequential(backbone.conv_stem, backbone.bn1)
        self.cbam = CBAM(stem_channels)
        replace_se_with_eca(backbone.blocks)
        self.blocks = backbone.blocks
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.global_pool = backbone.global_pool
        self.classifier = backbone.classifier

    def forward(self, x, return_features=False):
        x = self.stem(x)
        feat = self.cbam(x)
        x = self.blocks(feat)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.global_pool(x)
        out = self.classifier(x)
        if return_features:
            return out, feat
        return out

# ---------------------------------------------------------------------------
# 6. Phase F - Executable Training Loops with Loss Curves
# ---------------------------------------------------------------------------
class ArrayDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.from_numpy(xs).float()
        self.ys = torch.from_numpy(ys)
        if self.ys.dtype != torch.long:
            self.ys = self.ys.float()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

def train_heatwave_pipeline(model, xs, ys, epochs=10, lr=1e-3):
    """Train ConvLSTM on real meteorology sequences with loss visualization."""
    model.to(device)
    split = int(0.8 * len(xs))
    train_loader = DataLoader(ArrayDataset(xs[:split], ys[:split]), batch_size=4, shuffle=True)
    val_loader = DataLoader(ArrayDataset(xs[split:], ys[split:]), batch_size=4, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    history = {"train": [], "val": []}

    print(f"\n[Training] Training ConvLSTM on {len(xs)} temporal sequences for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                pred, _ = model(xb)
                loss = 0.7 * F.l1_loss(pred, yb) + 0.3 * F.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * len(xb)

        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                    pred, _ = model(xb)
                    loss = 0.7 * F.l1_loss(pred, yb) + 0.3 * F.mse_loss(pred, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= max(len(val_loader.dataset), 1)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train"], label="Train Hybrid Loss", color="royalblue", lw=2)
    ax.plot(history["val"], label="Val Hybrid Loss", color="darkorange", lw=2, linestyle="--")
    ax.set_title("Heatwave ConvLSTM Training & Validation Curves", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hybrid Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(VIS_DIR, "heatwave_loss_curve.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visual Check] Heatwave loss curve saved -> {out_path}")
    return history

def train_hailstorm_pipeline(model, scenes, labels, epochs=8, lr=3e-4):
    """Train DAM-EfficientNet on real rasterized radar scenes."""
    model.to(device)
    split = int(0.8 * len(scenes))
    train_loader = DataLoader(ArrayDataset(scenes[:split], labels[:split].astype(np.int64)), batch_size=8, shuffle=True)
    val_loader = DataLoader(ArrayDataset(scenes[split:], labels[split:].astype(np.int64)), batch_size=8, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    history = {"train": [], "val": [], "acc": []}

    print(f"\n[Training] Training DAM-EfficientNet on {len(scenes)} radar scenes for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()

        train_loss /= len(train_loader.dataset)
        acc = correct / len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                    out = model(xb)
                    loss = crit(out, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= max(len(val_loader.dataset), 1)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["acc"].append(acc)
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {acc*100:.1f}%")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history["train"], label="Train Loss", color="crimson", lw=2)
    ax1.plot(history["val"], label="Val Loss", color="salmon", linestyle="--", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax2 = ax1.twinx()
    ax2.plot(history["acc"], label="Accuracy", color="teal", lw=2)
    ax2.set_ylabel("Train Accuracy")
    ax1.set_title("Hailstorm DAM-EfficientNet Convergence", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    out_path = os.path.join(VIS_DIR, "hailstorm_loss_curve.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visual Check] Hailstorm loss curve saved -> {out_path}")
    return history

# ---------------------------------------------------------------------------
# 7. Phase G - Visual Interpretability & Attention
# ---------------------------------------------------------------------------
def visualize_heatwave_predictions(model, xs, ys):
    """Plot ConvLSTM spatiotemporal prediction vs ground truth and hidden states."""
    model.eval()
    xb = torch.from_numpy(xs[:1]).float().to(device)
    with torch.no_grad():
        pred, hidden_states = model(xb)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    im0 = axes[0].imshow(xs[0, -1, 6], cmap="inferno")
    axes[0].set_title("Input TS (t-1)", fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(ys[0, 0], cmap="inferno")
    axes[1].set_title("Target TS (t)", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(pred[0, 0].cpu().numpy(), cmap="inferno")
    axes[2].set_title("ConvLSTM Predicted (t)", fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(hidden_states[-1][0].mean(dim=0).cpu().numpy(), cmap="magma")
    axes[3].set_title("Final Hidden State Activation", fontsize=10, fontweight="bold")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.suptitle("ConvLSTM Next-Day Heatwave Anomaly Forecast", fontsize=12, y=0.98)
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, "heatwave_forecast_eval.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visual Check] Heatwave prediction evaluation saved -> {out_path}")

def visualize_hailstorm_attention(model, scene):
    """Plot CBAM attention feature maps and gradient saliency."""
    model.eval()
    xb = torch.from_numpy(scene).float().unsqueeze(0).to(device)
    xb.requires_grad_(True)
    out, feat = model(xb, return_features=True)
    pred_cls = out.argmax(1).item()

    out[0, pred_cls].backward()
    grad = xb.grad[0].abs().max(dim=0)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].imshow(scene[0], cmap="viridis")
    axes[0].set_title("Input Radar PROB Channel", fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(feat[0].mean(dim=0).detach().cpu().numpy(), cmap="inferno")
    axes[1].set_title("Stem CBAM Attention Map", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(grad, cmap="cividis")
    axes[2].set_title(f"Gradient Saliency (Class {pred_cls})", fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle("DAM-EfficientNet Radar Scene Attention & Saliency", fontsize=12, y=0.98)
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, "hailstorm_attention_eval.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[Visual Check] Hailstorm attention evaluation saved -> {out_path}")

# ---------------------------------------------------------------------------
# 8. Phase H - Shared Mixture of Experts (MoE) Interface
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class HazardAlert:
    hazard_type: str
    severity_score: float
    confidence: float
    spatial_extent: tuple
    valid_time: str
    evidence_map: np.ndarray

class HazardExpert(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def predict(self, raw_input) -> HazardAlert:
        ...

class HeatwaveHazardExpert(HazardExpert):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @property
    def name(self):
        return "heatwave"

    def predict(self, raw_input):
        xb = torch.from_numpy(raw_input).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = self.model(xb)
        pred_map = pred.squeeze().cpu().numpy()
        return HazardAlert(
            hazard_type=self.name,
            severity_score=float(pred_map.max()),
            confidence=float(np.clip(1.0 - (pred_map.std() / (pred_map.mean() + 1e-6)), 0.0, 1.0)),
            spatial_extent=pred_map.shape,
            valid_time="next_day",
            evidence_map=pred_map,
        )

class HailstormHazardExpert(HazardExpert):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @property
    def name(self):
        return "hailstorm"

    def predict(self, raw_input):
        xb = torch.from_numpy(raw_input).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.model(xb)
            prob = F.softmax(out, dim=1)[0, 1].item()
        return HazardAlert(
            hazard_type=self.name,
            severity_score=prob,
            confidence=abs(prob - 0.5) * 2,
            spatial_extent=raw_input.shape[1:],
            valid_time="nowcast_15min",
            evidence_map=raw_input[0],
        )

# ---------------------------------------------------------------------------
# 9. Main Execution Sequence
# ---------------------------------------------------------------------------
def run_all():
    print("\n" + "=" * 65)
    print("  MULTI-HAZARD MIXTURE OF EXPERTS: COMPLETE PIPELINE")
    print("=" * 65)

    check_dependencies()

    # Step 1: Heatwave Data & Visual Check
    grid, dates, params = load_or_fetch_heatwave_data()
    visualize_heatwave_dataset(grid, dates, params)

    # Step 2: Hailstorm Data & Visual Check
    df_hail = load_or_fetch_hail_data()
    visualize_hailstorm_dataset(df_hail)

    # Step 3: Preprocessing & Visual Check
    anomaly_grid, _, _ = climatology_normalize(grid)
    xs, ys = make_sequences(anomaly_grid, seq_len=3)
    print(f"[Preprocessing] Heatwave sequences: {xs.shape}, Targets: {ys.shape}")

    scenes, labels = build_hail_dataset(df_hail, max_scenes=120)
    print(f"[Preprocessing] Hailstorm scenes: {scenes.shape}, Positive hail count: {labels.sum()}/{len(labels)}")
    visualize_preprocessed_raster(scenes[0], labels[0])

    # Step 4: Model Training with Loss Visualization
    hw_model = HeatwaveConvLSTM(in_channels=7, hidden_channels=(32, 64))
    train_heatwave_pipeline(hw_model, xs, ys, epochs=6)

    hail_model = DAMEfficientNet(num_classes=2, pretrained=False)
    train_hailstorm_pipeline(hail_model, scenes, labels, epochs=6)

    # Step 5: Visual Interpretability & Attention
    visualize_heatwave_predictions(hw_model, xs, ys)
    visualize_hailstorm_attention(hail_model, scenes[0])

    # Step 6: MoE Dispatcher Inference
    hw_expert = HeatwaveHazardExpert(hw_model)
    alert_hw = hw_expert.predict(xs[0])

    hail_expert = HailstormHazardExpert(hail_model)
    alert_hail = hail_expert.predict(scenes[0])

    print("\n" + "=" * 65)
    print("  MOE HAZARD DISPATCHER: ACTIVE ALERTS")
    print("=" * 65)
    print(f"  [Alert 1] {alert_hw.hazard_type.upper():<10} | Severity: {alert_hw.severity_score:+.3f} | Confidence: {alert_hw.confidence:.1%} | Horizon: {alert_hw.valid_time}")
    print(f"  [Alert 2] {alert_hail.hazard_type.upper():<10} | Severity: {alert_hail.severity_score:+.3f} | Confidence: {alert_hail.confidence:.1%} | Horizon: {alert_hail.valid_time}")
    print("=" * 65)
    print(f"[Complete] All visual checks saved under {VIS_DIR}/")

def smoke_test():
    print("--- Running Fast Architecture Smoke Tests ---")
    hw_model = HeatwaveConvLSTM(in_channels=7, hidden_channels=(32, 64)).to(device)
    dummy_hw = torch.randn(2, 3, 7, 32, 32, device=device)
    hw_out, hw_hist = hw_model(dummy_hw)
    assert hw_out.shape == (2, 1, 32, 32), f"Unexpected shape {hw_out.shape}"

    hail_model = DAMEfficientNet(num_classes=2, pretrained=False).to(device)
    dummy_hail = torch.randn(4, 3, 224, 224, device=device)
    hail_out, hail_feat = hail_model(dummy_hail, return_features=True)
    assert hail_out.shape == (4, 2), f"Unexpected shape {hail_out.shape}"
    print("[OK] Smoke tests passed cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-hazard MoE Pipeline")
    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.add_parser("smoke-test", help="Sanity check architectures")
    subparsers.add_parser("run-all", help="Execute complete pipeline with visual checks")
    p_dl = subparsers.add_parser("download-hail", help="Fetch NOAA SWDI hail data")
    p_dl.add_argument("--sample-mb", type=float, default=10.0)

    args = parser.parse_args()
    if args.cmd == "run-all" or args.cmd is None:
        run_all()
    elif args.cmd == "smoke-test":
        smoke_test()
    elif args.cmd == "download-hail":
        load_or_fetch_hail_data(sample_mb=args.sample_mb)
