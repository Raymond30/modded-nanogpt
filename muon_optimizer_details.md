# Muon / Leon Optimizer Implementation Details

This document outlines the specialized architectural choices and hyperparameters driving the optimizers in `train_gpt_muon.py` and `train_gpt_leon.py`.
Sections 7 and 8 (below) cover the LR and momentum schedules specific to **`train_gpt_leon.py`**.

## 1. Application Scope (Muon vs. Adam)
*(Defined in param table: [train_gpt_muon.py:L1646-L1662](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L1646-L1662))*
- **Muon**: Applied exclusively to the large 2D projection matrices (`attn_bank` and `mlp_bank`). Muon is designed specifically to constrain matrix transformations dynamically through orthogonalization without affecting their spectral scales.
- **Adam**: Applied to all remaining parameters, including dense 1D vectors (embeddings, router gate weights, and scalar multipliers).

## 2. Core Hyperparameters for Muon
*(Configuration dict: [train_gpt_muon.py:L1679-L1683](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L1679-L1683))*
The primary mathematical properties of the Muon update step are driven by three main hyperparameter parameters defined inside `muon_defaults`:
- **`lr: 0.023`** 
  - The learning rate step size. Exceptionally large compared to standard Adam learning rates, making it heavily sample-efficient.
- **`momentum: 0.95`** 
  - Standard Nesterov SGD momentum tracking coefficient. Gradients are accumulated into this buffer prior to passing through the Polar Express algorithm.
- **`weight_decay: 1.2`** 
  - A massive parameter penalty utilized exclusively via a *Cautious Weight Decay* mask. Weight decay is artificially bypassed whenever it directly opposes a parameter's gradient update. This massive parameter gravity anchors the network space and reliably suppresses training divergence over time.

## 3. Aspect-Ratio Scaling
*(Scaling logic: [train_gpt_muon.py:L501-L502](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L501-L502))*
Muon handles structurally different matrices automatically by observing the aspect ratio of the underlying dense parameter chunk:
- Calculated automatically as: `max(1.0, rows/columns) ** 0.5`
- This dynamic multiplier natively bolsters the learning rate on "taller" matrices, creating aligned convergence curves across diverse weight classes without extensive custom tuning.

## 4. Alternating MLP Projections (`c_proj` Scaling)
*(Setup logic: [train_gpt_muon.py:L507-L513](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L507-L513), Application: [train_gpt_muon.py:L871-L873](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L871-L873))*
Feed-forward network projections (`c_fc` expanding, `c_proj` compressing) perform separate geometrical manipulations and require independent constraints to learn optimally.
- In `train_gpt_muon.py`, the `mlp_bank` array targets individual matrices via sequential index checks (`mat_idx % 2 == 1`).
- All mapped down-projections (`c_proj`) natively receive a **2.0x learning rate multiplier**. 
- This enables the dense compression sweeps of the MLP blocks to scale exactly twice as forcefully as the preceding up-projections, which accelerates baseline feature generation drastically.

## 5. Learning Rate Scheduler
*(Schedule lookup logic: [train_gpt_muon.py:L1588-L1596](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L1588-L1596))*
The overall learning rate schedule (`get_lr(step)`) evaluates custom step schedules decoupled from typical PyTorch generic schedulers. It is directly coupled to the model's dynamic batch size curriculum defined in `TRAINING_STAGES`:
- **Step-wise Batch Scaling**: As the batch sizes artificially step up across stages (e.g., from `8` to `16` to `24` blocks), the global learning rate multiplier steps up proportionally via fractional power rules (e.g., `(16/8)**0.6 -> 1.52x LR`). 
- **Linear Cooldown**: A simple linear cooldown phase is enforced over the trailing `60%` of scheduled iterations (`cooldown_frac=0.60`). During this duration, the learning rate linear decays downwards to a terminal fraction of `0.15` by the final training step to ensure smooth convergence.

## 6. Parameter Bank Shapes and Sharding
*(Bank instantiation: [train_gpt_muon.py:L1154-L1160](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L1154-L1160))*
To optimize distributed gradient communication and efficiently accelerate the Polar Express calculation, Muon groups independent parameters into contiguous blocks called "banks". By analyzing the script, we document the specific bank sharding sizes driving the efficiency.
- **`attn_bank`**: Instantiated with shape `(10, 3072, 768)` where `10` is the number of attention layers and `3072` embeds 4 individual queries of standard model dimension `768`. Before reducing scattered gradients over the network, it is reshaped strictly to `(40, 768, 768)`. This flattens the chunk allocations so that it cleanly shards along the leading dimension into 40 distinct square blocks, which perfectly divides by the parallel processing size of `8` GPUs (allowing 5 full chunk operations per rank).
- **`mlp_bank`**: Instantiated natively with shape `(12, 2, 3072, 768)` mapping to 12 feed-forward layers, 2 projections (`c_fc` expanding layer, and `c_proj` contracting layer), a raw hidden dimension of `3072` and input model dimension of `768`. This is algorithmically reshaped onto a flat scale of `(24, 3072, 768)`, cleanly slicing standard bounds along the number of overall matrices (24 total matrices). This enables each of the 8 participating GPUs to cleanly and independently update exactly 3 dense rectangle matrices per iteration without requiring any ragged padded bounds.

### Gradients and Orthogonalization Reshaping
The architecture parameters are inherently 3-dimensional. To seamlessly feed the gradients into the `polar_express` optimization logic (which expects sequences of 2D matrices), the gradient tracking layer dynamically reshapes and slices them prior to application:

1. **Dimension Parsing**: During optimizer initialization in `_build_param_cfg` ([train_gpt_muon.py:L491-L499](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L491-L499)), the optimizer reads the defined `.reshape` attribute. It statically calculates a `chunk_shape` per GPU partition (e.g. `(5, 768, 768)` for the `attn_bank` on an 8-GPU scale).
2. **Gradient Flattening**: Given that backpropagation creates gradients mirroring the `(num_layers, inner_dims, outer_dims)` model shape, `_launch_reduce` uses `grad_reshaped = grad.view(p_cfg.reshape)` ([train_gpt_muon.py:L583](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L583)) to reshape the complex gradient arrays into the flat sequence of matrices format during the gradient syncing process.
3. **Polar Express Processing**: In `_muon_update` ([train_gpt_muon.py:L861-L864](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_muon.py#L861-L864)), the effectively sliced gradients are fed into `polar_express` as `v_chunk`. Because `chunk_shape[-2]` (the middle dimension) operates natively as the effective matrix row-count constraint, these contiguous 3D arrays represent an exact batched sequence of 2D matrices. This sequence gracefully utilizes `torch.baddbmm` within the Polar Express procedure to perform orthogonalization sweeps cleanly in parallel.

---

## 7. Learning Rate Schedule (`train_gpt_leon.py`)
*(Schedule definition: [train_gpt_leon.py:L1599-L1623](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_leon.py#L1599-L1623))*

The LR schedule in `train_gpt_leon.py` is controlled by `TrainingSchedule.get_lr(step)`, which ties the LR multiplier to the batch-size curriculum defined in `TRAINING_STAGES`.

### 7.1 Stage-wise LR multipliers

Each training stage raises the global LR proportionally to the increased batch size (a power-law scaling rule):

| Stage | Steps (approx) | Batch size | Seq len | LR multiplier | Derivation |
|-------|---------------|------------|---------|---------------|------------|
| Stage 1 | 0 – 463 | 8 × 2048 | 896 | **×1.00** | baseline |
| Stage 2 | 464 – 926 | 16 × 2048 | 2048 | **×1.52** | `(16/8)^0.6 ≈ 1.52` |
| Stage 3 | 927 – 1390 | 24 × 2048 | 2048 | **×1.73** | `(24/8)^0.5 ≈ 1.73` |
| Extension | 1391 – 1500 | 24 × 2048 | 2048 | *(unused/1.0)* | window 6,13 only |

### 7.2 Linear Cooldown

A **linear cooldown** is applied over the trailing **60 %** of `scheduled_iterations` (i.e., from step ≈ 556 to step 1390). During cooldown the effective multiplier linearly interpolates to **15 %** of the current stage's LR multiplier:

```python
cd_start = int(scheduled_iterations * (1 - cooldown_frac))  # cooldown_frac = 0.60
t = min(1.0, (step - cd_start) / (scheduled_iterations - cd_start))
lr = stage_lr_mul * (1 - t) + 0.15 * t
```

The final LR is applied by scaling each parameter's `initial_lr` at every optimizer step:
```python
p_cfg.lr = p_cfg.initial_lr * step_lr   # step_lr = get_lr(step)
```

### 7.3 Leon defaults (base LR before multiplier)
*(Config dict: [train_gpt_leon.py:L1690-L1694](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_leon.py#L1690-L1694))*
```python
leon_defaults = dict(
    lr           = float(os.environ.get("LEON_LR",   "0.046")),
    momentum     = 0.95,
    weight_decay = float(os.environ.get("LEON_WD",   "0.6")),
    beta2        = float(os.environ.get("LEON_BETA2", "0.8")),
)
```

- **`lr = 0.046`** (env-overridable): base Leon learning rate, higher than typical Muon `0.023` to account for the second-momentum dampening from the Gram matrix.
- **`beta2 = 0.8`**: EMA decay for the second-momentum (Gram matrix) buffer inside `polar_express`. A lower β₂ means the Gram matrix tracks shorter history, responding faster to gradient changes.
- **`weight_decay = 0.6`**: Applied as [cautious weight decay](https://arxiv.org/abs/2411.16085) — gated by sign-alignment between gradient and parameter so it never actively opposes the update direction.

---

## 8. Momentum Schedule (`train_gpt_leon.py`)
*(Schedule function: [train_gpt_leon.py:L1626-L1638](file:///mnt/nas/rj23424/modded-nanogpt/train_gpt_leon.py#L1626-L1638))*

The Nesterov momentum coefficient β₁ is **not fixed** — it follows a three-phase schedule via `get_muon_momentum(step)`:

```python
def get_muon_momentum(step, muon_warmup_steps=300, muon_cooldown_steps=50,
                      momentum_min=0.85, momentum_max=0.95):
    momentum_cd_start = training_schedule.total_steps - muon_cooldown_steps
    if step < muon_warmup_steps:                   # Phase 1: Linear warmup
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:                 # Phase 3: Linear cooldown
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:                                          # Phase 2: Flat
        momentum = momentum_max
    return momentum
```

| Phase | Steps | Behaviour | β₁ range |
|-------|-------|-----------|----------|
| **Warmup** | 0 → 300 | Linear increase | 0.85 → 0.95 |
| **Flat** | 300 → 1450 | Constant | 0.95 |
| **Cooldown** | 1450 → 1500 | Linear decrease | 0.95 → 0.85 |

**Rationale:** Starting with lower momentum prevents the EMA buffer from over-committing to very noisy early gradients. The brief cooldown at the end mirrors the LR decay and prevents the heavy momentum buffer from carrying stale history into the final convergence phase.

This schedule is applied inside `step_optimizers` by writing into `p_cfg.momentum` before every optimizer step:
```python
if p_cfg.optim == "leon":
    p_cfg.momentum = muon_momentum   # overrides the static default at each step
```

---

## 9. Combined Schedule Visualization

The chart below overlays both schedules across the full training run (1390 scheduled + 110 extension steps). Stage background bands correspond to the batch-size curriculum.

![LR and Momentum Schedule](file:///mnt/nas/rj23424/modded-nanogpt/lr_momentum_schedule.png)
