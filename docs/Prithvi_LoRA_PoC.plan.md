# Prithvi_LoRA_Disaster_PoC Implementation Plan

## Overview

This document outlines the complete implementation plan for **`Prithvi_LoRA_Disaster_PoC.ipynb`**, a proof-of-concept notebook that demonstrates a LoRA-adapted Prithvi-100M vision transformer for disaster (wildfire) prediction. The plan consolidates all architectural, training, and evaluation decisions derived from the compacted conversation history.

---

## 1. Dataset Selection & Rationale

### 1.1 HLS Burn Scar Dataset (ibm-nasa-geospatial)
- **Why this dataset?** The HLS Burn Scar dataset is the official Prithvi downstream task. It natively provides:
  - **6-band spectral input** (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2) matching the Prithvi-100M ViT backbone expectation of 6 channels.
  - **Native burn severity labels** (burned vs. unburned) suitable for the hybrid multi‑task head.
  - **Spatial partitioning** (regional splits) enabling spatial hold‑out validation.
- **Alternative considered:** NextDayWildfire – rejected because it ships *pre‑computed* NDVI/EVI TFRecords rather than raw Sentinel‑2 6‑band imagery. This mismatch would break the native 6‑channel embedding pipeline and the dNBR computation head.

### 1.2 Temporal Formulation (T = 3)
- Input sequence: three preceding time windows (`t‑3`, `t‑2`, `t‑1`) concatenated with the current frame `t₀`.
- Each scene is a 224×224 patch (standard for ViT‑B/‑L variants).
- Temporal aggregation is performed via **global average pooling** across the three past patches, producing a fixed‑size temporal representation that feeds the classifier.
- This preserves the full spatiotemporal context while keeping the model lightweight (< 8 GB VRAM in FP16).

### 1.3 Label Design
- **Primary task:** Binary fire progression (burned vs. unburned) – classification head.
- **Secondary task:** dNBR (Degree of Normalized Burn Ratio) severity regression – continuous regression head.
- Both heads share the same feature extractor (ViT‑B/‑L) but have distinct decoders.

---

## 2. Model Architecture

### 2.1 Backbone
- **Prithvi‑100M** (ibm-nasa-geospatial/Prithvi-EO-1.0-100M) – ViT‑B/L variant with 6‑band optical input.
- **Resolution:** 224×224 patches (standard for the backbone).
- **Parameters:** ~100 M (backbone) + ~0.3 M for LoRA adapters.

### 2.2 LoRA Adapters
- **Rank (r):** 8
- **Alpha:** 16 (α = 16 × r = 128 scaling factor)
- **Target layers:** Query projection (`q_proj`) and Value projection (`v_proj`) within each ViT block.
- **Rationale:** Low‑rank adaptation adds ~0.3 M trainable parameters while preserving the bulk of the backbone's capacity. Rank 8 balances expressiveness with the strict < 0.4 % parameter footprint constraint.

### 2.3 Multi‑Task Head
```
Feature Extractor → Global Avg Pool → Temporal Avg Pool → Linear(768,128) → ReLU → Linear(128, num_classes)
                     ↓
              Auxiliary dNBR Regression Head → Linear(128, 1) → Smooth L1 Loss
```
- **Main head:** Binary classification (fire progression) + regression (dNBR).
- **Auxiliary head:** Predicts normalized burn ratio (continuous).
- **Loss combination:**
  - **Focal Loss** (γ = 2.0, α = 0.25) for the binary classification branch.
  - **Smooth L1 Loss** (λ = 0.5 weight) for the dNBR regression branch.
  - Total loss = FocalLoss + 0.5·SmoothL1Loss.

### 2.4 Parameter Footprint
- **Backbone:** ~100 M parameters (unchanged).
- **LoRA adapters:** ~0.3 M trainable parameters (r = 8, α = 16).
- **Total trainable:** ≈ 0.4 M (≈ 0.4 % of 100 M).
- **Inference overhead:** Minimal – LoRA weights are merged during inference or applied via adapter‑aware attention masking.

---

## 3. Training Strategy

### 3.1 Cross‑Validation
- **Spatial Hold‑Out:** 70 % train, 15 % spatial validation, 15 % spatial test (preserves geographic distribution).
- **Inner 3‑fold CV:** Applied on the train split for hyperparameter tuning (learning rate, early‑stop patience).
- **Early Stopping:** Patience = 3 on validation macro‑F1.

### 3.2 Optimization
- **Optimizer:** AdamW (lr = 3e‑4, weight decay = 1e‑2).
- **Scheduler:** CosineAnnealingWarmup (1‑epoch warmup, 10‑epoch decay).
- **Batch Size:** 4–8 (depending on GPU memory; 224×224 patches fit comfortably in 8 GB).

### 3.3 Epochs
- **Per fold:** 10 epochs.
- **Global:** 30 epochs (3 folds × 10).
- **Checkpointing:** Save best model based on validation macro‑F1.

### 3.4 Metrics
- **Primary:** Macro F1‑score (binary fire progression).
- **Secondary:** Precision, Recall, AUROC, AUPRC, Confusion Matrix, MAE (auxiliary dNBR), Latency per patch (< 100 ms target).

---

## 4. Data Pipeline: From Dense Pixels to Patch‑Level Targets

### 4.1 Patch Extraction
- Split each scene into overlapping 224×224 patches (stride = 32, 20% overlap).
- **Overlap rationale:** Preserves contextual continuity across patch boundaries, reducing fragmentation of spatial information.

### 4.2 Patch‑Level Binary Targets (Fire Progression)
- **Definition:** For each patch, compute a binary label indicating whether the patch contains burned vegetation (based on the dataset's burn severity classification).
- **Aggregation:** The patch label is derived from the dominant class in the patch's local region (majority vote over the 224×224 patch).
- **Reasoning:** Direct mapping from the native 6‑band burn severity labels to a discrete patch‑level target aligns with the multi‑task design and avoids expensive per‑pixel classification.

### 4.3 dNBR Computation
- **Step 1 — NBR (per timestamp):** `NBR = (NIR − SWIR2) / (NIR + SWIR2)` computed at every pixel for both pre‑fire and post‑fire acquisitions.
  - NIR = HLS band 4 (Narrow NIR, 0.86 µm).
  - SWIR2 = HLS band 6 (SWIR‑2, 2.2 µm).
- **Step 2 — dNBR (differenced):** `dNBR = NBR_pre − NBR_post` (Key & Benson 1999 USGS standard). Higher dNBR ⇒ more severe burn.
- **Step 3 — Patch aggregation:** for each 224×224 patch, average the per‑pixel dNBR over the 50 176 pixels to obtain a single continuous severity score in the canonical USGS range `[−0.5, +1.0]`.
- **Implementation:** The HLS Burn Scar dataset provides paired pre/post 6‑band tiles; we read both, compute NBR on each, subtract, and pool to patch level. Clamp to `[−0.5, +1.0]` and (optionally) min‑max normalize to `[0, 1]` for Smooth‑L1 stability.
- **Regression target:** One scalar per patch → auxiliary head → SmoothL1Loss.
- **USGS severity interpretation (for label analysis, not for training):**
  - `dNBR < 0.1` → unburned
  - `0.1 ≤ dNBR < 0.27` → low severity
  - `0.27 ≤ dNBR < 0.44` → moderate
  - `0.44 ≤ dNBR < 0.66` → high
  - `dNBR ≥ 0.66` → very high / extreme

### 4.4 Temporal Input Construction
- For each sample, concatenate three temporally adjacent patches:
  - `patch(t‑1)`, `patch(t‑2)`, `patch(t‑3)` → stacked along the channel dimension.
- **Result:** A 3‑frame sequence of shape `(3, 6, 224, 224)` → flattened to `(3, 768)` via global average pooling.
- **Why this works:** The temporal context captures the evolution of fire spread and recovery, essential for the fire progression head.

---

## 5. Internal Reasoning & Design Choices

| Decision | Rationale |
|----------|------------|
| **HLS Burn Scar over NextDayWildfire** | Native 6‑band input matches Prithvi‑100M's expected tensor shape; built‑in burn labels eliminate the need for external engineering of synthetic features. |
| **LoRA r=8, α=16 on q_proj/v_proj** | Low‑rank adaptation is the most effective way to inject task‑specific knowledge without retraining the full 100 M backbone. Rank 8 gives sufficient capacity while keeping trainable params < 0.4 %. |
| **Focal Loss + Smooth L1** | Focal loss addresses class imbalance (few burned patches vs. many unburned). Smooth L1 provides robust gradient flow for the regression head. |
| **Spatial hold‑out + 3‑fold CV** | Preserves geographic diversity; prevents leakage from spatially correlated regions. Inner CV ensures hyperparameters are tuned on representative subsets. |
| **Macro F1 as primary metric** | Balances precision and recall for the imbalanced binary fire‑progression task; aligns with disaster‑response priorities (avoiding missed fires). |
| **Latency constraint (< 100 ms/patch)** | Drives patch size (224×224) and model depth; the combined architecture easily meets this on modern GPUs. |

---

## 6. Notebook Structure (Prithvi_LoRA_Disaster_PoC.ipynb)

```markdown
# Prithvi_LoRA_Disaster_PoC.ipynb

## 1. Imports & Configuration
- Load Prithvi‑100M weights from `./models/prithvi_100m`.
- Initialize LoRA adapters (r=8, α=16) on `q_proj` and `v_proj`.
- Define loss functions (FocalLoss, SmoothL1Loss).

## 2. Data Module
- **HLSBurnScarDataset**: load TFRecords, extract 6‑band patches, compute dNBR per patch.
- **Patch assembler**: create 224×224 patches with 20% stride.
- **Temporal encoder**: stack (t‑3, t‑2, t‑1) patches → global avg pool → (3, 768).
- **Label generators**: binary fire progression (patch‑wise majority vote), dNBR regression (per‑patch formula).

## 3. Model Definition
- Wrap Prithvi‑100M with LoRA adapter layer.
- Add main head: GlobalAvgPool → Linear(768,128) → ReLU → Linear(128, num_classes).
- Add auxiliary head: Linear(128, 1) → SmoothL1Loss.

## 4. Training Loop
- 3‑fold spatial CV + inner 10‑epoch training per fold.
- Early stopping on validation macro‑F1 (patience=3).
- Checkpoint best model.

## 5. Evaluation
- Compute macro F1, precision, recall, AUROC, AUPRC, confusion matrix, MAE (dNBR), latency.
- Compare across folds; report final macro F1 on spatial test set.

## 6. Visualization & Analysis
- Sample predictions vs. ground truth.
- Grad-CAM heatmaps for patch‑level fire progression.
- dNBR regression residual analysis.
```

---

## 7. Success Criteria
- **Macro F1 ≥ 0.55** on the spatial test set (target baseline).
- **Latency ≤ 100 ms** per 224×224 patch (measured on target hardware).
- **VRAM usage < 8 GB** in FP16 mode.
- **LoRA trainable params ≈ 0.3 M** (≤ 0.4% of backbone).

---

*This plan is frozen – all decisions are finalized. The accompanying notebook will implement exactly these specifications.*