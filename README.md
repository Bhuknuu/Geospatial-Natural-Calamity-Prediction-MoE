<div align="center" style="padding-top: 20px; padding-bottom: 20px;">

## <span style="font-family:'Playfair Display', Georgia, serif; font-weight:500; font-size:1.8em;"> Natural Calamity Prediction Using Geospatial Image Analysis</span>
#### <span style="font-family:'Playfair Display', Georgia, serif; font-weight:200; font-size:1.1em;"> _Multi-Hazard Mixture of Experts (MoE) Architecture_</span>

<p style="font-family:'Playfair Display', Georgia, serif; font-weight:250; font-size:1.1em; max-width: 800px; margin: 0 auto; text-align: center;">
End-to-end multi-hazard prediction pipelines trained on authentic geospatial imagery: Heatwave thermal sequences (NASA MODIS Terra Satellite) and Severe Convective Hailstorms (NOAA NEXRAD Level-III Doppler Radar).
</p>

</div>

---

## 1. Overview & Dual Execution Modes

This repository provides two coordinated execution engines for multi-hazard disaster prediction:

| Component | Target Audience | Primary Function | Key Feature |
|---|---|---|---|
| `src/multihazard_moe.ipynb` | **Judges & Evaluators** | Interactive presentation & visual audits | Real satellite scenes, Doppler radar reflectivity sweeps, live epoch-by-epoch loss & accuracy curves, attention overlays |
| `src/multihazard_moe.py` | **Production & CI/CD** | High-throughput GPU execution & benchmarking | 1200+ samples/sec forward pass, minimal dependencies, sub-250MB VRAM footprint |

---

## 2. Multi-Hazard Expert Specifications

| Hazard | Primary Data Source | Physical Ingestion | Spatiotemporal Formulation | Model Architecture | Metric / Horizon |
|---|---|---|---|---|---|
| **Heatwave** | NASA MODIS Terra Satellite | 32 daily Land Surface Temperature (LST) scenes (Delhi 2023) | Spatiotemporal sliding sequences ($T=3 \rightarrow 1$) | **ConvLSTM** (263K params) | Hybrid 0.7 L1 / 0.3 L2 next-day thermal forecast |
| **Severe Hail / Storm** | NOAA NEXRAD Doppler Radar | Calibrated Base Reflectivity ($Z$ in dBZ, 224×224 RGB) | Balanced severe vs. benign convective sweeps | **DAM-EfficientNet** (CBAM + ECA) | CrossEntropy, ~85.4% Val Acc (Zero data leakage) |

---

## 3. Production GPU Benchmarks (RTX 4060 Laptop GPU)

```text
=================================================================
  BENCHMARK SUMMARY RESULTS (src/multihazard_moe.py benchmark)
=================================================================
  ConvLSTM (Satellite Seq)  : 1219.1 samples/sec | Latency: 13.1 ms | VRAM:  45.6 MB
  DAM-EfficientNet (Radar)  :  680.8 samples/sec | Latency: 23.5 ms | VRAM: 228.8 MB
=================================================================
```

---

## 4. Quickstart

### Environment Setup
```bash
pip install -r requirements.txt
```

### Production Benchmark
Measure inference throughput and VRAM efficiency:
```bash
python src/multihazard_moe.py benchmark
```

### Fast End-to-End Execution
Run complete training and hazard alert dispatch:
```bash
python src/multihazard_moe.py run-all
```

---

## 5. Repository Structure

```text
├── src/
│   ├── multihazard_moe.ipynb    # Interactive notebook with dynamic training visuals & judge audit blocks
│   ├── multihazard_moe.py       # Production engine & GPU benchmarking CLI
│   └── implementation_plan.md   # Architectural formulations & provenance
├── docs/
│   ├── Dataset_Collection.md    # Curated Indian subcontinent disaster datasets
│   ├── Literature_Report.md     # Deep learning research synthesis
│   ├── Knowledge_Base.md        # Regional vulnerability reference
│   └── Prithvi_LoRA_PoC.plan.md # Burn scar ViT PoC plan
├── requirements.txt
├── .gitignore
└── README.md
```
