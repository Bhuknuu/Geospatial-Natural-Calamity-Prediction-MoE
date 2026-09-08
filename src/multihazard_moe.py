#!/usr/bin/env python
"""Multi-Hazard Mixture of Experts (MoE): Production Engine & Performance Benchmark.

Real data, real models, production-grade throughput:
- Heatwave: NASA POWER reanalysis grid -> ConvLSTM (hybrid L1/L2 loss).
- Hailstorm: NOAA SWDI NEXRAD radar detections -> DAM-EfficientNet (CBAM + ECA).

Usage:
    python src/multihazard_moe.py benchmark      # measure GPU throughput & memory
    python src/multihazard_moe.py run-all        # execute end-to-end pipeline
    python src/multihazard_moe.py smoke-test     # verify model architectures
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import platform
import argparse
import warnings

import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 1. Reproducibility & Environment
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = device.type == "cuda"

def print_env_specs() -> None:
    print("=" * 65)
    print("  SYSTEM & HARDWARE RUNTIME CONFIGURATION")
    print("=" * 65)
    print(f"  Python / OS     : {sys.version.split()[0]} ({platform.system()})")
    print(f"  PyTorch / CUDA  : {torch.__version__} (CUDA: {torch.version.cuda or 'N/A'})")
    print(f"  Compute Device  : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Device      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 65)

# ---------------------------------------------------------------------------
# 2. Data Ingestion & Preprocessing
# ---------------------------------------------------------------------------
POWER_PARAMS = ["T2M", "T2MDEW", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "WS10M", "TS"]
SWDI_COLUMNS = ["ZTIME", "LON", "LAT", "WSR_ID", "CELL_ID", "RANGE", "AZIMUTH", "SEVPROB", "PROB", "MAXSIZE"]

def load_heatwave_data(path="data/heatwave_delhi_2023.npz"):
    if not os.path.exists(path) and os.path.exists("../data/heatwave_delhi_2023.npz"):
        path = "../data/heatwave_delhi_2023.npz"
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return data["grid"], data["dates"], list(data["params"])
    raise FileNotFoundError(f"Missing {path}. Run notebook or data fetch first.")

def load_hail_data(path="data/hail-2015.csv"):
    if not os.path.exists(path) and os.path.exists("../data/hail-2015.csv"):
        path = "../data/hail-2015.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, comment="#", header=None, names=SWDI_COLUMNS)
        df["ZTIME"] = pd.to_datetime(df["ZTIME"], format="%Y%m%d%H%M%S", errors="coerce")
        return df
    raise FileNotFoundError(f"Missing {path}. Run data download first.")

def climatology_normalize(grid):
    t, c, h, w = grid.shape
    flat = pd.DataFrame(grid.reshape(t, -1)).ffill().bfill().values.reshape(t, c, h, w)
    mean, std = flat.mean(axis=0, keepdims=True), flat.std(axis=0, keepdims=True) + 1e-6
    return ((flat - mean) / std).astype(np.float32)

def make_sequences(grid, seq_len=3):
    ts_idx = POWER_PARAMS.index("TS")
    xs, ys = [], []
    for t in range(grid.shape[0] - seq_len):
        xs.append(grid[t : t + seq_len])
        ys.append(grid[t + seq_len, ts_idx : ts_idx + 1])
    return np.stack(xs), np.stack(ys)

def rasterize_scene(cells_df, extent_nmi=100, size=224, splat_sigma=2.0):
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
    df = df.dropna(subset=["ZTIME"]).copy()
    # Balance positive confirmed hail scenes with regular scans
    pos_df = df[(df["MAXSIZE"] > 0) & (df["PROB"] == 100)].head(3000)
    neg_df = df.head(3000)
    sample_df = pd.concat([pos_df, neg_df]).drop_duplicates().copy()
    sample_df["scene_id"] = sample_df["WSR_ID"] + "_" + sample_df["ZTIME"].dt.floor(time_bucket).astype(str)
    scenes, labels = [], []
    for _, group in sample_df.groupby("scene_id"):
        scenes.append(rasterize_scene(group))
        labels.append(int(((group["MAXSIZE"] > 0) & (group["PROB"] == 100)).any()))
        if len(scenes) >= max_scenes:
            break
    return np.stack(scenes), np.array(labels, dtype=np.int64)

# ---------------------------------------------------------------------------
# 3. Model Architectures
# ---------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_state(self, b, h, w, device):
        shape = (b, self.hidden_channels, h, w)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

class HeatwaveConvLSTM(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=(32, 64)):
        super().__init__()
        self.cells = nn.ModuleList([ConvLSTMCell(in_channels if i == 0 else hidden_channels[i - 1], hc) for i, hc in enumerate(hidden_channels)])
        self.project = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        states = [cell.init_state(b, h, w, x.device) for cell in self.cells]
        for step in range(t):
            inp = x[:, step]
            for i, cell in enumerate(self.cells):
                hs, cs = cell(inp, states[i][0], states[i][1])
                states[i] = (hs, cs)
                inp = hs
        return self.project(states[-1][0])

class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels // r, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // r, channels, bias=False))
        self.spatial = nn.Conv2d(3 * channels, 1, kernel_size=7, padding=3, bias=False)
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(channels, channels, 5, padding=2, bias=False)
        self.c3 = nn.Conv2d(channels, channels, 7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        ca = torch.sigmoid(self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3)))).view(b, c, 1, 1)
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([self.c1(x), self.c2(x), self.c3(x)], dim=1)))
        return x * sa

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = max(int(abs((math.log2(channels) / gamma) + (b / gamma))), 3)
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        y = self.conv(x.mean(dim=(2, 3), keepdim=True).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)

def replace_se_with_eca(module):
    for name, child in module.named_children():
        if child.__class__.__name__ == "SqueezeExcite":
            setattr(module, name, ECA(child.conv_reduce.in_channels))
        else:
            replace_se_with_eca(child)

class DAMEfficientNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        bb = timm.create_model("efficientnet_b1", pretrained=pretrained, num_classes=num_classes)
        self.stem = nn.Sequential(bb.conv_stem, bb.bn1)
        self.cbam = CBAM(bb.conv_stem.out_channels)
        replace_se_with_eca(bb.blocks)
        self.blocks, self.conv_head, self.bn2, self.global_pool, self.classifier = bb.blocks, bb.conv_head, bb.bn2, bb.global_pool, bb.classifier

    def forward(self, x):
        return self.classifier(self.global_pool(self.bn2(self.conv_head(self.blocks(self.cbam(self.stem(x)))))))

# ---------------------------------------------------------------------------
# 4. Production Benchmarking & Execution
# ---------------------------------------------------------------------------
class ArrayDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.from_numpy(xs).float()
        self.ys = torch.from_numpy(ys).long() if np.issubdtype(ys.dtype, np.integer) else torch.from_numpy(ys).float()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

def benchmark_models():
    print_env_specs()
    print("\n[BENCHMARK] Measuring forward-pass throughput and GPU memory efficiency...")

    # ConvLSTM Benchmark
    hw_model = HeatwaveConvLSTM().to(device).eval()
    dummy_hw = torch.randn(16, 3, 7, 32, 32, device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(50):
            _ = hw_model(dummy_hw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    hw_time = (time.perf_counter() - t0) / 50
    hw_fps = 16 / hw_time
    hw_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    # DAM-EfficientNet Benchmark
    hail_model = DAMEfficientNet().to(device).eval()
    dummy_hail = torch.randn(16, 3, 224, 224, device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(50):
            _ = hail_model(dummy_hail)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    hail_time = (time.perf_counter() - t0) / 50
    hail_fps = 16 / hail_time
    hail_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print("=" * 65)
    print("  BENCHMARK SUMMARY RESULTS")
    print("=" * 65)
    print(f"  ConvLSTM (Heatwave)       : {hw_fps:6.1f} samples/sec | Latency: {hw_time*1000:5.1f} ms | VRAM: {hw_vram:5.1f} MB")
    print(f"  DAM-EfficientNet (Hail)   : {hail_fps:6.1f} samples/sec | Latency: {hail_time*1000:5.1f} ms | VRAM: {hail_vram:5.1f} MB")
    print("=" * 65)

def run_pipeline():
    print_env_specs()
    grid, _, _ = load_heatwave_data()
    xs, ys = make_sequences(climatology_normalize(grid))
    df = load_hail_data()
    scenes, labels = build_hail_dataset(df, max_scenes=120)

    print(f"\n[DATA] Heatwave: {len(xs)} seqs | Hailstorm: {len(scenes)} scenes (Pos: {labels.sum()}/{len(labels)})")

    # Heatwave training
    hw = HeatwaveConvLSTM().to(device)
    opt_hw = torch.optim.Adam(hw.parameters(), lr=1e-3)
    loader_hw = DataLoader(ArrayDataset(xs, ys), batch_size=4, shuffle=True)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    t0 = time.perf_counter()
    hw.train()
    for ep in range(5):
        loss_acc = 0.0
        for xb, yb in loader_hw:
            xb, yb = xb.to(device), yb.to(device)
            opt_hw.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                loss = 0.7 * F.l1_loss(hw(xb), yb) + 0.3 * F.mse_loss(hw(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt_hw)
            scaler.update()
            loss_acc += loss.item() * len(xb)
    hw_dur = time.perf_counter() - t0
    print(f"[TRAIN] ConvLSTM 5 epochs finished in {hw_dur*1000:.1f} ms ({hw_dur/5*1000:.1f} ms/epoch)")

    # Hailstorm training
    hail = DAMEfficientNet().to(device)
    opt_hail = torch.optim.AdamW(hail.parameters(), lr=3e-4, weight_decay=1e-2)
    loader_hail = DataLoader(ArrayDataset(scenes, labels), batch_size=8, shuffle=True)
    crit = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    hail.train()
    for ep in range(5):
        loss_acc, corr = 0.0, 0
        for xb, yb in loader_hail:
            xb, yb = xb.to(device), yb.to(device)
            opt_hail.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                out = hail(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt_hail)
            scaler.update()
            loss_acc += loss.item() * len(xb)
            corr += (out.argmax(1) == yb).sum().item()
    hail_dur = time.perf_counter() - t0
    print(f"[TRAIN] DAM-EfficientNet 5 epochs finished in {hail_dur*1000:.1f} ms ({hail_dur/5*1000:.1f} ms/epoch) | Final Acc: {corr/len(scenes):.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-hazard MoE Production Engine")
    parser.add_argument("cmd", nargs="?", default="run-all", choices=["run-all", "benchmark", "smoke-test"])
    args = parser.parse_args()

    if args.cmd == "benchmark":
        benchmark_models()
    elif args.cmd == "run-all":
        run_pipeline()
    elif args.cmd == "smoke-test":
        benchmark_models()
