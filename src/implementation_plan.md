<div align="center" style="padding-top: 20px; padding-bottom: 20px;">

## <span style="font-family:'Playfair Display', Georgia, serif; font-weight:500; font-size:1.6em;"> PoC for Natural Calamity Detection</span>
#### <span style="font-family:'Playfair Display', Georgia, serif; font-weight:200; font-size:1.0em;"> _Multi-Hazard MoE: Heatwave and Hailstorm Experts_</span>

<p style="font-family:'Playfair Display', Georgia, serif; font-weight:250; font-size:1.1em; max-width: 800px; margin: 0 auto; text-align: center;">
Real data, real models, no synthetic placeholders.<br><br>
<b>Heatwave</b>: NASA POWER daily point API &rarr; ConvLSTM (hybrid L1/L2 loss).<br>
<b>Hailstorm</b>: NOAA SWDI hail signatures (NEXRAD radar) &rarr; DAM-EfficientNet (EfficientNet-B1 + CBAM + ECA).
</p>

</div>

---

This supersedes the earlier hackathon draft. No synthetic data, no surrogate models, no time-box. Two experts: heatwave and hailstorm.

## 1. Data sources

**Heatwave: NASA POWER API.** Free, no signup, no key. Verified endpoint:
`https://power.larc.nasa.gov/api/temporal/daily/point?parameters=...&community=AG&longitude=...&latitude=...&start=YYYYMMDD&end=YYYYMMDD&format=JSON`
Parameters used: `T2M`, `T2MDEW`, `RH2M`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`, `WS10M`, `TS` (Earth Skin Temperature: our real proxy for MODIS LST, since true MODIS rasters need Earthdata/GEE credentials you would rather not set up).
**Caveat, stated plainly:** POWER's underlying reanalysis is native ~0.5 deg (~55 km). Querying a fine grid of points inside that cell returns real numbers, but nearby points will be heavily correlated, not independently sensed. We are trading spatial independence for zero setup friction. If that trade stops being worth it, the swap point is exactly `preprocess()`: nothing downstream changes.

**Hailstorm: NOAA Severe Weather Data Inventory (SWDI), hail layer.** The dataset originally cited by the DAM-EfficientNet paper (`github.com/dhn9132/hail_images`) is dead (verified directly: GitHub API + codeload both 404, and the account's other repos are unrelated). Real replacement: NOAA's own NEXRAD Level-III hail product, public S3 bucket `noaa-swdi-pds`, file pattern `hail-{year}.csv`. Verified columns: `ZTIME, LON, LAT, WSR_ID, CELL_ID, RANGE, AZIMUTH, SEVPROB, PROB, MAXSIZE`. This is point/attribute data (radar-derived storm-cell detections), not imagery: so we rasterize it ourselves (Section 3).

## 2. Rasterization (hailstorm only: our design, not the paper's)

Group detections by `WSR_ID` + time bucket (one radar volume scan). Convert each cell's `(RANGE, AZIMUTH)` to a local Cartesian offset from the radar site, bin onto a 224x224 grid, one channel per field (`PROB`, `SEVPROB`, `MAXSIZE`), light Gaussian splat per cell so a point has visible extent. Label a scene "hail" if any cell has `MAXSIZE > 0` and `PROB == 100`, the same filter NOAA and the paper both use for "confirmed" hail signatures.

Why this instead of synthetic imagery: NEXRAD Level-III data is real observations from real radars detecting real storms. Rasterizing it preserves the physical signals the network needs to learn (core intensity via `MAXSIZE`, certainty via `PROB`), and produces standard tensors without scraping dead links or inventing pixels.

## 3. Models

**Heatwave: ConvLSTM (from scratch).** Input `(B, T, C, H, W)` where `T=3` days, `C=7` (the POWER parameters above), `H=W=32`. Two stacked ConvLSTM cells (hidden channels 32, 64), 1x1 conv projection to `(B, 1, H, W)` predicting next-day `TS`. Loss is hybrid L1 + L2 (70% L1 for edge sharpness, 30% L2 for mean calibration). Verified forward pass: 263K parameters, outputs exact shape, zero external dependencies beyond PyTorch.

**Hailstorm: DAM-EfficientNet (reconstructed from Liu et al. 2024, Sci. Rep. 14:3505).** EfficientNet-B1 backbone with two modifications:
1. CBAM inserted immediately after the stem convolution (before stage 1). Channel attention via shared MLP on avg/max pooled descriptors; spatial attention via concatenated 1x1, 3x3, 5x5 convs fed to a 7x7 conv. (The paper says "two weight vectors" from three convs, which is a contradiction: implemented as 3-conv concat -> 7x7, which is the self-consistent reading).
2. Squeeze-and-Excitation blocks inside all MBConv stages replaced by ECA (Efficient Channel Attention: adaptive 1D conv, kernel size k = |log2(C)/2 + 1/2| odd, no channel reduction).
Verified forward pass: 5.4M parameters, outputs `(B, 2)` hail probability, forward pass confirmed working.

## 4. Shared interface

Both experts implement one minimal interface:
- `preprocess(raw) -> torch.Tensor`
- `predict(x) -> raw_pred`
- `postprocess(raw_pred) -> HazardAlert(hazard_type, severity_score, raw_metric)`

No router here: each expert is trained and run independently. The MoE structure is in the contract, not a routing layer. Deliberate simplification.

## 5. Deliverable

One standalone Jupyter notebook (`multihazard_moe.ipynb`) that:
1. Downloads real data (NASA POWER point query for heatwave, NOAA SWDI S3 CSV for hailstorm).
2. Rasterizes / preprocesses each into tensors.
3. Defines both model architectures with verified forward passes.
4. Trains each on real data (CPU-feasible mini-splits).
5. Visualizes predictions: heatwave predicted vs actual LST map; hailstorm CAM (Class Activation Map) showing which radar cells triggered the detection.
6. Evaluates both via the shared interface.
