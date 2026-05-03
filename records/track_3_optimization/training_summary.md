# Training Summary — `train_gpt_simple.py`

_Last updated: 2026-05-03_

---

## 1. Completed Run: Muon Optimizer (baseline)

| Field | Value |
|---|---|
| **Script** | `train_gpt_simple.py` |
| **Log** | `logs/21b4b9af-07a4-4f65-be44-c2d08dd8c58d.txt` |
| **Date** | May 1, 2026 |
| **Hardware** | 2× NVIDIA H100 NVL |
| **PyTorch** | 2.11.0+cu128 |
| **Total Steps** | 3,375 |
| **Batch Size** | 8 × 64 × 1024 = 524,288 tokens |
| **Val Tokens** | 10,485,760 (20 × 524,288) |

### Optimizer Setup

- **AdamW** (embed, proj head, 1D params): lr ∈ {0.3, 1/320, 0.01}, β = (0.8, 0.95), no weight decay
- **Muon** (all 2D block params): lr = 0.025, weight_decay = 0.025, momentum = 0.95
- **LR schedule**: stable → linear cooldown over final 70% of training

### Validation Loss Curve

| Step | Val Loss | Wall Time (s) | Step Avg (ms) |
|-----:|--------:|-------------:|-----------:|
| 0 | 10.82585 | 0.0 | — |
| 125 | 4.63407 | 127.4 | 1018.90 |
| 250 | 4.10881 | 223.6 | 769.67 |
| 375 | 3.92574 | 319.1 | 764.17 |
| 500 | 3.82126 | 415.0 | 767.27 |
| 625 | 3.75147 | 511.0 | 767.88 |
| 750 | 3.70761 | 606.3 | 762.62 |
| 875 | 3.66462 | 702.1 | 766.36 |
| 1000 | 3.62893 | 797.9 | 766.27 |
| 1125 | 3.60085 | 893.2 | 762.75 |
| 1250 | 3.56831 | 989.1 | 766.57 |
| 1375 | 3.54267 | 1084.7 | 765.50 |
| 1500 | 3.51593 | 1179.9 | 761.14 |
| 1625 | 3.49499 | 1275.5 | 764.99 |
| 1750 | 3.47198 | 1371.1 | 765.03 |
| 1875 | 3.45268 | 1466.2 | 760.33 |
| 2000 | 3.43366 | 1561.8 | 764.80 |
| 2125 | 3.41741 | 1657.4 | 764.66 |
| 2250 | 3.40000 | 1752.4 | 760.27 |
| 2375 | 3.38483 | 1848.0 | 764.56 |
| 2500 | 3.36910 | 1943.6 | 764.83 |
| 2625 | 3.35341 | 2038.6 | 760.30 |
| 2750 | 3.33870 | 2134.2 | 764.58 |
| 2875 | 3.32412 | 2229.8 | 764.69 |
| 3000 | 3.31005 | 2324.8 | 760.16 |
| 3125 | 3.29636 | 2420.4 | 764.61 |
| 3250 | 3.28502 | 2515.9 | 764.47 |
| 3375 | **3.27898** | **2611.0** | 760.32 |

### Key Takeaways

- **Final val loss: 3.27898** — beats the 3.28 target
- **Total training time: ~43.5 min** on 2× H100 NVL
- **Steady-state throughput: ~764 ms/step** (after compilation warmup in the first 125 steps)
- Loss drops rapidly early (10.83 → 4.63 in first 125 steps), then converges smoothly
- Zero-init on projection weights (`attn.proj`, `mlp.proj`) with standard Muon + weight decay

---

## 2. Completed Run: MuonH Optimizer (experimental)

| Field | Value |
|---|---|
| **Script** | `train_gpt_simple_muonh.py` |
| **Log** | `logs/572180fd-60f5-4b5e-b0cc-90bb71a67cf8.txt` |
| **Date** | May 2–3, 2026 |
| **Hardware** | 4× NVIDIA H100 NVL |
| **PyTorch** | 2.11.0+cu128 |
| **Total Steps** | 3,325 |
| **Batch Size** | 8 × 64 × 1024 = 524,288 tokens |
| **Val Tokens** | 10,485,760 (20 × 524,288) |

### Changes vs. Baseline

- **Optimizer**: MuonH replaces Muon — uses hyperball projection (Frobenius-norm-preserving) instead of vanilla SGD + orthogonalised direction
- **No weight decay** on hidden 2D matrices (the hyperball constraint prevents norm growth)
- **Non-zero init** for proj weights via per-module Kaiming multipliers (×1.25 attn.proj, ×3.0 mlp.proj, ×1.5 mlp.fc)
- **Per-group cooldown**: MuonH uses full linear cooldown (`cooldown_frac=1.0`), AdamW uses 40% (`cooldown_frac=0.4`)
- **Fewer steps**: 3,325 vs. 3,375 (target remains val_loss ≤ 3.28)
- **MuonH lr**: 0.018 (vs. Muon 0.025)

### Validation Loss Curve

| Step | Val Loss | Wall Time (s) | Step Avg (ms) |
|-----:|--------:|-------------:|-----------:|
| 0 | 11.00799 | 0.0 | — |
| 125 | 4.84212 | 84.4 | 675.30 |
| 250 | 4.20182 | 136.8 | 419.49 |
| 375 | 3.99625 | 189.3 | 419.53 |
| 500 | 3.88944 | 241.5 | 417.90 |
| 625 | 3.82028 | 293.7 | 417.33 |
| 750 | 3.77485 | 345.8 | 417.05 |
| 875 | 3.72478 | 398.0 | 417.20 |
| 1000 | 3.68867 | 450.1 | 417.09 |
| 1125 | 3.66533 | 502.2 | 416.84 |
| 1250 | 3.63288 | 554.4 | 417.13 |
| 1375 | 3.60763 | 606.5 | 416.79 |
| 1500 | 3.57938 | 658.6 | 416.93 |
| 1625 | 3.55860 | 710.7 | 417.13 |
| 1750 | 3.53753 | 762.9 | 417.20 |
| 1875 | 3.51498 | 815.0 | 417.08 |
| 2000 | 3.49355 | 867.2 | 417.26 |
| 2125 | 3.47005 | 919.3 | 417.18 |
| 2250 | 3.44674 | 971.4 | 416.92 |
| 2375 | 3.42508 | 1023.5 | 417.02 |
| 2500 | 3.40353 | 1075.7 | 417.20 |
| 2625 | 3.38028 | 1127.9 | 417.30 |
| 2750 | 3.35936 | 1180.0 | 417.12 |
| 2875 | 3.33915 | 1232.2 | 417.42 |
| 3000 | 3.31924 | 1284.3 | 416.84 |
| 3125 | 3.30076 | 1336.4 | 417.20 |
| 3250 | 3.28634 | 1388.6 | 417.08 |
| 3325 | **3.28228** | **1419.8** | 417.03 |

### Key Takeaways

- **Final val loss: 3.28228** — narrowly misses the 3.28 target (off by 0.002)
- **Total training time: ~23.7 min** on 4× H100 NVL — **1.84× faster** wall-clock than the Muon baseline
- **Steady-state throughput: ~417 ms/step** (vs. 764 ms for Muon on 2 GPUs)
- MuonH starts with higher loss (11.008 vs. 10.826) and runs slightly behind Muon at matched steps, but closes the gap through the cooldown phase
- 50 fewer steps than Muon (3,325 vs. 3,375) while reaching comparable final loss
- The hyperball constraint (norm-preserving projection) eliminates the need for weight decay on 2D matrices

---

## 3. Summary

| Metric | Muon (2× H100) | MuonH (4× H100) |
|---|---|---|
| Steps | 3,375 | 3,325 |
| Final val loss | **3.279** | **3.282** |
| Target (≤ 3.28) | ✅ Met | ❌ Missed by 0.002 |
| Step avg (steady) | 764 ms | 417 ms |
| Total time | 43.5 min | 23.7 min |
| Weight decay (2D) | 0.025 | 0 (norm-constrained) |
| Init (proj weights) | Zero | Scaled Kaiming |
