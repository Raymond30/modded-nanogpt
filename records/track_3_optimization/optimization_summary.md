# Neural Network Optimization Summary

This document summarizes the neural network architecture, training hyperparameters, optimization strategies, and hyperparameter tuning efforts found in the `track_3_optimization` directory.

## 1. Key Neural Network Architecture

The models are based on a lightweight GPT-style causal language model, descended from the `modded-nanogpt` speedrun.

*   **Dimensions:** 12 layers (`num_layers=12`), model dimension of 768 (`model_dim=768`).
*   **Vocabulary:** 50,304 vocabulary size.
*   **Normalization:** `RMSNorm` with learned gain parameters (no biases) applied before attention and MLP blocks, as well as on queries and keys.
*   **Positional Embeddings:** Rotary Positional Embeddings (`RoPE`) applied to half of the query and key dimensions. Base frequency is scaled.
*   **Attention:** Causal scaled dot-product attention with 6 heads (`head_dim=128`), scaled by a constant `0.12`.
*   **MLP:** Expansion factor of 4 (hidden dimension 3072). Activation function is a squared ReLU (`F.relu(x).square()`).
*   **Output Scaling:** The final output logits are normalized across the vocabulary dimension to maintain a constant L2 norm via: `logits = 15 * logits * (logits.square() + 15**2).rsqrt()`.
*   **Biases:** Excluded from `RMSNorm` but kept in all `Linear` projections.

## 2. Training Hyperparameters

*   **Batch & Sequence Size:** Batch size of 524,288 tokens per step (8 microbatches * 64 batch size * 1024 sequence length).
*   **Dataset:** `FineWeb10B`. Validation loss is calculated over ~10.4 million tokens.
*   **Initialization:** 
    *   Embeddings and biases use default initialization (normal/zero respectively).
    *   Linear weights use standard deviation scaled by `0.33**0.5 / fan_in**0.5`.
    *   Vocab projection (`proj.weight`) and all projection biases are initialized to strictly `0` so initial logits are `0`.
    *   *Scale-Invariant Specifics:* For models using hyperball projection optimizers (like `MuonH` and `AdamH`), hidden block weights (`attn.proj`, `mlp.proj`, `mlp.fc`) are explicitly scaled up by factors of 1.25, 3.0, and 1.5 respectively to ensure they have non-zero initial matrix norms to operate on.
*   **Learning Rate Schedule:** Models employ varying scheduling strategies:
    *   **AdamW Aux Parameters (1D):** No warmup. Flat learning rate for the first 60% of training, followed by a linear decay over the final 40% (`cooldown_frac=0.4`).
    *   **AdamW Block Matrices (Baseline):** 250 steps of linear warmup, followed by a flat learning rate, then linear decay.
    *   **AdamH Block Matrices:** 250 steps of linear warmup (`h_warmup_steps=250`), followed immediately by a linear decay over the rest of the run (`cooldown_frac=1.0`).
    *   **Muon Block Matrices:** Flat learning rate for the first 30% of training, followed by linear decay over the final 70% (`cooldown_frac=0.7`).
    *   **MuonH Block Matrices:** Immediate linear decay from step 0 across the entire training run (`cooldown_frac=1.0`).

## 3. Optimizers

The scripts implement specialized optimization strategies for the 2D hidden weight matrices inside the Transformer blocks, separating them from the 1D parameters (like embeddings, norms, and biases) which are always optimized via standard **AdamW** (lr=0.3 to 0.01). 

### Baseline: AdamW
The standard AdamW baseline (`a63a68d1.txt`) optimizes block matrices using a standard AdamW setup with `lr=0.0015`, `weight_decay=0.10`, `betas=(0.9, 0.95)`, `eps=1e-8`, and a warmup phase of 250 steps. Auxiliary 1D parameters use AdamW with `lr=0.01` to `0.3`, `betas=(0.8, 0.95)`, `eps=1e-10`, and `weight_decay=0`.

### Muon (`train_gpt_simple.py`)
`Muon` is an optimizer specifically designed for deep learning weight matrices. Instead of standard momentum updates, it applies **Newton-Schulz iterations** (`zeropower_via_newtonschulz5`) to the momentum of the gradients. It executes exactly **12 iterations** of the Newton-Schulz loop to perfectly orthogonalize the update, bounding its spectral norm to 1 and thereby optimizing the update geometry for matrices. After orthogonalization, it applies a scaling factor based on the matrix dimensions and subtracts weight decay. It operates with a Nesterov momentum of `mu=0.95`, learning rates ranging from `0.02` to `0.025`, and weight decay from `0.01` to `0.025`.

### MuonH (`train_gpt_simple_muonh.py`)
`MuonH` extends Muon by replacing weight decay with a **Frobenius-norm-preserving hyperball projection**. It takes the same Newton-Schulz orthogonalized direction (also using **12 iterations**) but applies it via a step (`scale_invariant_update_`) that exactly renormalizes the parameter back onto the Frobenius sphere of its original initialized radius. This invariant completely prevents norm growth, allowing weight decay to be dropped entirely for hidden matrices. It operates with a Nesterov momentum of `mu=0.95`, `lr=0.018`, and `weight_decay=0`.

### AdamH (`20260430_adamh/`)
Similar to `MuonH`, `AdamH` applies the standard Adam-preconditioned gradient update but projects it using the exact same Frobenius-norm-preserving hyperball projection, allowing zero weight decay. It operates with `lr=0.018`, `betas=(0.9, 0.95)`, `eps=1e-8`, and `weight_decay=0`.

## 4. Hyperparameter Tuning Efforts

The optimization goal across all logs was to reach a validation loss of **3.28** as fast as possible (minimizing total training steps) using the FineWeb10B dataset. By extracting final step counts and validation losses from the logged trials, the following progression is observed:

1.  **AdamW Baseline (`a63a68d1.txt`):** 
    *   **Configuration:** `train_steps=5625`, `block_adamw_lr=0.0015`, `block_adamw_weight_decay=0.10`, `warmup_steps=250`.
    *   **Result:** Reached a final `val_loss` of **3.27903** in ~833s.

2.  **AdamH (`20260430_adamh/`):** 
    *   **Configuration:** `train_steps=4875`, `matrix_lr=0.018`, `h_warmup_steps=250`.
    *   **Result:** 5 independent trials converged to `val_loss` ranging between **3.272** and **3.276** (e.g. 3.27637, 3.27246, 3.27425), accelerating convergence relative to standard AdamW.

3.  **Muon (`311d7833.txt`, `7b8270c5.txt`, `51ece938.txt`):** 
    *   **Configuration A:** `train_steps=3600`, `muon_lr=0.02`, `weight_decay=0.01` -> final `val_loss` **3.27765**.
    *   **Configuration B:** `train_steps=3500`, `muon_lr=0.025`, `weight_decay=0.0125` -> final `val_loss` **3.27673**.
    *   **Configuration C:** `train_steps=3375`, `muon_lr=0.025`, `weight_decay=0.025` -> Ran 10 consecutive trials with final `val_loss` values hovering around **3.2769** to **3.2810** (mostly hitting the <3.28 target).

4.  **MuonH (`20260430_muonh/`):** 
    *   **Configuration:** `train_steps=3325`, `matrix_lr=0.018` (zero weight decay due to hyperball projection).
    *   **Result:** 10 distinct trials robustly reached final `val_loss` values between **3.275** and **3.279** (e.g., 3.27906, 3.27796, 3.27528) in just **3325 steps** (averaging ~505s wall-clock time).

**Conclusion:** The progression of tuning shows that matrix-specific optimization (Muon) offers a massive reduction in training steps over AdamW (from 5625 down to 3375). Further confining those matrix updates to a scale-invariant hyperball surface (MuonH) pushes the efficiency limit even further down to 3325 steps, removing the need for weight decay tuning while improving convergence speed and stability.
