<div align="center" style="padding-top: 20px; padding-bottom: 20px;">

## <span style="font-family:'Playfair Display', Georgia, serif; font-weight:500; font-size:1.8em;"> Natural Calamity Prediction Using Geospatial Analysis</span>
#### <span style="font-family:'Playfair Display', Georgia, serif; font-weight:200; font-size:1.1em;"> _Multi-Hazard Mixture of Experts (MoE) Architecture_</span>

<p style="font-family:'Playfair Display', Georgia, serif; font-weight:250; font-size:1.1em; max-width: 800px; margin: 0 auto; text-align: center;">
End-to-end multi-hazard prediction pipelines trained on real data sources: Heatwave (NASA POWER) and Hailstorm (NOAA SWDI NEXRAD Radar).
</p>

</div>

---

## 1. Overview

This repository implements a modular Multi-Hazard Mixture of Experts (MoE) architecture for predicting natural calamities:

| Hazard | Primary Data Source | Resolution / Format | Architecture | Task |
|---|---|---|---|---|
| **Heatwave** | NASA POWER Daily Point API | Reanalysis (7 atmospheric channels) | **ConvLSTM** (from scratch) | Next-day surface temperature anomaly map |
| **Hailstorm** | NOAA SWDI NEXRAD Level-III | Radar detections rasterized to (3, 224, 224) | **DAM-EfficientNet** (CBAM + ECA) | Nowcasting hail occurrence & severity |

---

## 2. Project Structure

```text
├── src/
│   ├── multihazard_moe.py       # Full executable pipeline (ConvLSTM + DAM-EfficientNet + CLI)
│   ├── multihazard_moe.ipynb    # Interactive notebook counterpart
│   └── implementation_plan.md   # Architectural decisions & mathematical formulations
├── docs/
│   ├── Dataset_Collection.md    # Curated geospatial hazard datasets (2020–2025)
│   ├── Literature_Report.md     # Domain research synthesis for Indian terrain
│   ├── Knowledge_Base.md        # Hazard knowledge base & statistical baselines
│   └── Prithvi_LoRA_PoC.plan.md # Architectural plan for ViT burn scar detection
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignores __pycache__, checkpoints, large CSV datasets
└── README.md                    # Project documentation
```

---

## 3. Quickstart

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Model Smoke Tests
Verify both model architectures, training loops, and expert alert interfaces:
```bash
python src/multihazard_moe.py smoke-test
```

### Fetch Real SWDI Radar Hail Data
Download a sample slice of the NOAA SWDI hail detection dataset:
```bash
python src/multihazard_moe.py download-hail --year 2015 --sample-mb 10
```

---

## 4. Shared Expert Interface

All hazard experts subclass the minimal `HazardExpert` interface returning standardized `HazardAlert` objects:

```python
from src.multihazard_moe import HeatwaveHazardExpert, HailstormHazardExpert

# Initialize expert
hail_expert = HailstormHazardExpert()

# Predict on a rasterized radar scene (3, 224, 224)
alert = hail_expert.predict(radar_scene)
print(f"Hazard: {alert.hazard_type} | Severity: {alert.severity_score:.2f} | Confidence: {alert.confidence:.2f}")
```
