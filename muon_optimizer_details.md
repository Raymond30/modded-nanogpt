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
