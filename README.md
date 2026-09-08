<div align="center" style="padding-top: 20px; padding-bottom: 20px;">

## <span style="font-family:'Playfair Display', Georgia, serif; font-weight:500; font-size:1.8em;"> Natural Calamity Prediction Using Geospatial Analysis</span>
#### <span style="font-family:'Playfair Display', Georgia, serif; font-weight:200; font-size:1.1em;"> _Multi-Hazard Mixture of Experts (MoE) Architecture_</span>

<p style="font-family:'Playfair Display', Georgia, serif; font-weight:250; font-size:1.1em; max-width: 800px; margin: 0 auto; text-align: center;">
End-to-end multi-hazard prediction pipelines trained on real data sources: Heatwave (NASA POWER) and Hailstorm (NOAA SWDI NEXRAD Radar).
</p>

</div>

---

## 1. Overview & Dual Execution Modes

This repository provides two coordinated execution engines for multi-hazard disaster prediction:

| Component | Target Audience | Primary Function | Key Feature |
|---|---|---|---|
| `src/multihazard_moe.ipynb` | **Judges & Evaluators** | Interactive presentation & visual audits | Initial dataset tables, post-reprocessing checks, live epoch-by-epoch training curves |
| `src/multihazard_moe.py` | **Production & CI/CD** | High-throughput GPU execution & benchmarking | 1200+ samples/sec forward pass, minimal dependencies, sub-250MB VRAM footprint |

---

## 2. Multi-Hazard Expert Specifications

| Hazard | Primary Data Source | Initial Ingestion | Reprocessed State | Model Architecture | Metric / Horizon |
|---|---|---|---|---|---|
| **Heatwave** | NASA POWER Daily Point API | 7 atmospheric channels (T2M, TS, RH2M, etc.) | Climatology-anomaly (Z-score) sequences (T=3) | **ConvLSTM** (263K params) | Hybrid 0.7 L1 / 0.3 L2 next-day TS anomaly |
| **Hailstorm** | NOAA SWDI NEXRAD Radar | 88,941 polar point detections (Range/Azimuth) | Dense (3, 224, 224) Cartesian radar raster | **DAM-EfficientNet** (CBAM + ECA) | CrossEntropy loss nowcasting severe hail |

---

## 3. Production GPU Benchmarks (RTX 4060 Laptop GPU)

```text
=================================================================
  BENCHMARK SUMMARY RESULTS (src/multihazard_moe.py benchmark)
=================================================================
  ConvLSTM (Heatwave)       : 1200.4 samples/sec | Latency: 13.3 ms | VRAM:  47.0 MB
  DAM-EfficientNet (Hail)   :  544.1 samples/sec | Latency: 29.4 ms | VRAM: 237.6 MB
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
