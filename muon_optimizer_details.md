# Muon Optimizer Implementation Details

This document outlines the specialized architectural choices and hyperparameters driving the `MuonAndAdam` optimizer in `train_gpt_muon.py`.

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
