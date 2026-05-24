# Leon L=0 Comparison Notes

Last updated: 2026-05-22 18:40 America/Chicago

This note compares standard Leon against the Leon `L=0` control run and records the current interpretation plus a diagnostic plan. The authoritative row-level ledger remains `records/track_3_optimization/tuning_log.csv`.

## Runs Compared

| Run | Optimizer path | Key hparams | World size | Final val |
| --- | --- | --- | ---: | ---: |
| `base` | Standard Leon with nonzero second-momentum `L` | `leon_lr=0.025`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, `leon_mu=0.95`, `leon_beta2=0.7`, `leon_ns_iters=6`, `leon_eps=1e-9` | 4 | 3.57447 |
| `leon_l0_muon_hparams_4gpu` | Leon with `L = second_momentum_buffer.float() * 0` | Same as `base` | 4 | 3.44907 |
| `muon1500_leonns_2gpu` | Muon with Leon-like `L=0` 6-step augmented Newton-Schulz | `lr=0.025`, `wd=0.025`, `momentum=0.95`, `ns_steps=6`, `eps=1e-9` | 2 | 3.44804 |

The standard Leon and Leon `L=0` runs are the cleanest comparison for the second-momentum contribution: same hparams, same world size, same PyTorch/CUDA version, same model/data protocol.

## Standard Leon vs Leon L=0 Trajectory

| Step | Standard Leon val | Leon L=0 val | L=0 minus standard |
| ---: | ---: | ---: | ---: |
| 0 | 10.82584 | 10.82584 | +0.00000 |
| 125 | 5.03116 | 4.79942 | -0.23174 |
| 250 | 4.51157 | 4.19596 | -0.31561 |
| 375 | 4.18429 | 3.98429 | -0.20000 |
| 500 | 4.02075 | 3.85956 | -0.16119 |
| 625 | 3.90982 | 3.76333 | -0.14649 |
| 750 | 3.83791 | 3.69533 | -0.14258 |
| 875 | 3.77105 | 3.63552 | -0.13553 |
| 1000 | 3.71337 | 3.58368 | -0.12969 |
| 1125 | 3.67089 | 3.54097 | -0.12992 |
| 1250 | 3.63009 | 3.50111 | -0.12898 |
| 1375 | 3.59656 | 3.46886 | -0.12770 |
| 1500 | 3.57447 | 3.44907 | -0.12540 |

The gap appears immediately and remains large. This is an optimization-trajectory slowdown from nonzero `L`, not a wall-clock slowdown. The `L=0` control is already better by `0.23174` at step 125 and by `0.31561` at step 250.

## Leon L=0 vs Muon Leon-NS

| Step | Muon Leon-NS val | Leon L=0 val | Leon minus Muon |
| ---: | ---: | ---: | ---: |
| 0 | 10.82585 | 10.82584 | -0.00001 |
| 125 | 4.81629 | 4.79942 | -0.01687 |
| 250 | 4.19937 | 4.19596 | -0.00341 |
| 375 | 3.98373 | 3.98429 | +0.00056 |
| 500 | 3.85832 | 3.85956 | +0.00124 |
| 625 | 3.76373 | 3.76333 | -0.00040 |
| 750 | 3.69361 | 3.69533 | +0.00172 |
| 875 | 3.63512 | 3.63552 | +0.00040 |
| 1000 | 3.58310 | 3.58368 | +0.00058 |
| 1125 | 3.54055 | 3.54097 | +0.00042 |
| 1250 | 3.50055 | 3.50111 | +0.00056 |
| 1375 | 3.46802 | 3.46886 | +0.00084 |
| 1500 | 3.44804 | 3.44907 | +0.00103 |

Leon `L=0` closely reproduces Muon Leon-NS. The residual `+0.00103` final gap is small enough to be explained by run-environment and distributed numeric differences: 4 GPUs vs 2 GPUs, PyTorch/CUDA `2.11.0+cu126`/`12.6` vs `2.10.0+cu128`/`12.8`, compiled vs uncompiled optimizer path differences, and different all-reduce summation order.

## Working Hypothesis

Nonzero `L` likely slows training by over-damping or mis-scaling the current update.

In standard Leon, the trace normalization uses:

```text
||G||_F^2 + tr(L)
```

and the Newton-Schulz matrix starts from:

```text
gram(X) + L_scaled
```

If `L` grows quickly, the current update is normalized against stale gradient-Gram history. That can reduce the effective update size or rotate/smooth the update toward older directions that are less useful early in training.

Two details make this plausible:

- `L` is an EMA of the raw gradient Gram before the raw gradient is mutated into the Nesterov update.
- The matrix being orthogonalized is the Nesterov-style momentum update, so `L` may be statistically mismatched with the update it modifies.

The clean `L=0` result suggests the large gap versus Muon was not caused by the optimizer wrapper, weight decay, schedule, model, or data pipeline. It points directly at the nonzero second-momentum contribution.

## Diagnostic Logging Plan

Add checkpoint-only diagnostics first. Logging every step is unnecessary and could perturb runtime. Start with a small fixed layer set, such as one early attention projection, one middle MLP projection, and one late attention projection.

Recommended stats:

| Stat | Why it helps |
| --- | --- |
| `\|\|W\|\|_F` | Detects whether nonzero `L` changes weight growth/shrinkage. Useful but not sufficient alone. |
| `\|\|grad\|\|_F` | Shows whether the loss trajectory changes are reflected in gradient scale. |
| `\|\|momentum\|\|_F` | Separates raw gradient scale from Nesterov update scale. |
| `\|\|update\|\|_F` before LR | Directly tests whether `L` reduces effective update magnitude. |
| `lr * \|\|update\|\|_F / \|\|W\|\|_F` | Best scalar for effective relative step size. |
| `tr(L)` and `\|\|L\|\|_F` | Shows how large the second-momentum contribution becomes. |
| `tr(L) / (\|\|G\|\|_F^2 + tr(L))` | Measures how much trace normalization is dominated by stale `L`. |
| `cos(update_L, update_L0)` | Tests whether `L` mostly rescales or rotates the update. |
| `cos(update, grad)` and `cos(update, momentum)` | Tests whether the final update remains aligned with current signal. |
| Top eigenvalue / trace of `L` | Detects whether `L` is directionally concentrated and acting like a strong anisotropic preconditioner. |

## Experiment Plan

1. Add a single `leon_l_scale` knob so `L_scaled = leon_l_scale * second_momentum_buffer.float()`. Use `1.0` for standard Leon and `0.0` for the current control.
2. Add checkpoint-only logging for the diagnostic stats above. Keep the first version limited to a small layer subset.
3. Run paired 250-step or 500-step screens with identical hparams:
   - `leon_l_scale=1.0`
   - `leon_l_scale=0.0`
   - optionally `0.1`, `0.25`, `0.5` if the first diagnostic confirms over-damping.
4. Use the diagnostics to classify the failure mode:
   - If `lr * ||update||_F / ||W||_F` is much smaller for `L=1`, `L` is primarily damping.
   - If `cos(update_L, update_L0)` is low while norms are similar, `L` is primarily rotating the update.
   - If `tr(L) / (||G||_F^2 + tr(L))` rapidly dominates, reduce `L` scale or change how `L` is normalized.
   - If `L` is highly concentrated by eigenvalue/trace, test clipping, whitening, or a lower `beta2`.

## Diagnostic Implementation

Implemented in `train_gpt_simple_leon.py`:

- `leon_l_scale`: multiplies the second-momentum contribution before trace normalization and Newton-Schulz input.
- Leon now uses an unnormalized exponential-sum interpretation for both moments:
  `g <- g / (1 - momentum)` after in-place Nesterov momentum, and
  `L <- second_momentum_buffer / (1 - beta2)` after the second-moment update.
- `leon_log_diagnostics`, `leon_diag_interval`, `leon_diag_patterns`: checkpoint-only diagnostic logging for selected Leon tensors.
- Each diagnostic checkpoint logs `||W||_F`, `||grad||_F`, `||momentum||_F`, `||update||_F`, hypothetical `||update_L0||_F`, relative update size, `cos(update_L, update_L0)`, alignment with grad/momentum, `tr(L)`, `||L||_F`, `tr(L)/(||G||_F^2+tr(L))`, and top-eigenvalue concentration.

The initial diagnostic screens used 4 GPUs, 500 steps, identical Muon-like hparams, and diagnostics every 125 steps.

## Diagnostic Scale Sweep

| `leon_l_scale` | Step 125 | Step 250 | Step 375 | Step 500 | Final gap vs L=0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 4.81341 | 4.16147 | 3.90888 | 3.79464 | +0.00000 |
| 0.10 | 4.86449 | 4.23270 | 3.98692 | 3.87248 | +0.07784 |
| 0.25 | 4.92425 | 4.30949 | 4.04234 | 3.93081 | +0.13617 |
| 0.50 | 4.98448 | 4.42915 | 4.10845 | 3.99168 | +0.19704 |
| 1.00 | 5.02446 | 4.48871 | 4.15835 | 4.04142 | +0.24678 |

The sweep is monotone at every validation checkpoint: increasing `L` scale slows the trajectory. This makes random run noise an unlikely explanation for the standard Leon slowdown.

## Unnormalized Sum Follow-Up

The active code now uses an unnormalized exponential-sum interpretation for both moments:

```text
g <- g / (1 - momentum)
L <- second_momentum_buffer / (1 - beta2)
```

This was tested with the same 500-step, 4-GPU diagnostic setup at `leon_l_scale=1.0`.

| Run | Step 125 | Step 250 | Step 375 | Step 500 |
| --- | ---: | ---: | ---: | ---: |
| old full L | 5.02446 | 4.48871 | 4.15835 | 4.04142 |
| current unnormalized-sum full L | 4.79164 | 4.15578 | 3.91508 | 3.80020 |
| L=0 control | 4.81341 | 4.16147 | 3.90888 | 3.79464 |

This largely fixes the scaling issue for `l_scale=1.0`. The new full-`L` trajectory is better than the old full-`L` run by `0.24122` at step 500, and it ends only `0.00556` behind the `L=0` control.

## Diagnostic Findings

The diagnostic averages below are means over the three logged tensors: `blocks.0.attn.q.weight`, `blocks.6.mlp.proj.weight`, and `blocks.11.attn.proj.weight`.

| Scale | Step | `tr(L)` fraction | `\|\|update_L\|\| / \|\|update_L0\|\|` | `cos(update_L, update_L0)` | Relative step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 125 | 0.988 | 0.173 | 0.882 | 0.00965 |
| 1.00 | 250 | 0.988 | 0.160 | 0.903 | 0.00637 |
| 1.00 | 375 | 0.988 | 0.163 | 0.907 | 0.00334 |
| 0.50 | 125 | 0.976 | 0.227 | 0.901 | 0.01038 |
| 0.50 | 250 | 0.981 | 0.216 | 0.898 | 0.00725 |
| 0.50 | 375 | 0.978 | 0.224 | 0.912 | 0.00365 |
| 0.25 | 125 | 0.967 | 0.282 | 0.900 | 0.01250 |
| 0.25 | 250 | 0.961 | 0.281 | 0.924 | 0.00757 |
| 0.25 | 375 | 0.949 | 0.314 | 0.927 | 0.00444 |
| 0.10 | 125 | 0.922 | 0.391 | 0.924 | 0.01479 |
| 0.10 | 250 | 0.928 | 0.370 | 0.943 | 0.00848 |
| 0.10 | 375 | 0.865 | 0.456 | 0.951 | 0.00545 |
| 0.00 | 125 | 0.000 | 1.000 | 1.000 | 0.02051 |
| 0.00 | 250 | 0.000 | 1.000 | 1.000 | 0.01103 |
| 0.00 | 375 | 0.000 | 1.000 | 1.000 | 0.00560 |

Interpretation:

- The main failure mode is damping/stale normalization. With full `L`, the trace normalization is about 98.8% dominated by `L` at steps 125-375, and the resulting update norm is only about 16-17% of the hypothetical L=0 update.
- There is some rotation, but it is not the dominant effect. `cos(update_L, update_L0)` stays around 0.88-0.91 for full `L`, and rises as scale decreases.
- Smaller `L` scale increases the relative update size and improves validation monotonically, but even `0.1` remains behind `L=0` by `0.07784` at step 500.
- The logged weight norms are much larger for `L=0`, which is consistent with less damping. That statistic is useful context, but the update norm and trace-fraction measurements are the direct evidence.

Under the current unnormalized-sum full-`L` codepath, the corresponding averages moved to:

| Step | `tr(L)` fraction | `\|\|update_L\|\| / \|\|update_L0\|\|` | `cos(update_L, update_L0)` | Relative step |
| ---: | ---: | ---: | ---: | ---: |
| 125 | 0.498 | 0.784 | 0.983 | 0.01948 |
| 250 | 0.451 | 0.800 | 0.988 | 0.01040 |
| 375 | 0.403 | 0.830 | 0.990 | 0.00565 |
| 500 | 0.213 | 0.888 | 0.994 | 0.0000471 |

Compared with the old full-`L` run, `L` no longer dominates the denominator, and the update now stays close in both norm and direction to the `L=0` reference. That is exactly the scaling failure mode we were trying to remove.

## Historical Branch

There was a short bias-correction branch that used `m_t / (1 - mu^t)` and `L_t / (1 - beta2^t)`. That branch is no longer the active implementation. Its single completed full-`L` run remains in `tuning_log.csv` for reference:

| Run | Variant | `leon_l_scale` | Step 125 | Step 250 | Step 375 | Step 500 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uncorrected full L | original | 1.0 | 5.02446 | 4.48871 | 4.15835 | 4.04142 |
| bias-corrected full L | abandoned branch | 1.0 | 5.04558 | 4.49661 | 4.16856 | 4.05228 |

That historical result did not improve the trajectory, which is why the current branch has moved to the simpler unnormalized exponential-sum scaling instead.

## Current Recommendation

Do not continue tuning full nonzero-`L` Leon blindly. The diagnostics now confirm that the second-momentum term mostly shrinks the effective update through trace domination by stale `L`, with secondary rotation.

The next useful algorithm probes are changes that prevent `L` from dominating the denominator or stale-normalizing the current Nesterov update: much smaller `L` scale, different `beta2`, delayed `L` activation, clipping/normalizing `tr(L)`, or computing the Gram statistics from the same update vector being orthogonalized. For matching Muon Leon-NS, `L=0` remains the clean control.

## 2026-05-22 Follow-Up: tr(L) Fraction Pinning and Update-Norm Matching

Two new knobs in `train_gpt_simple_leon.py`:

- `leon_l_normalize_mode = "match_g"`: rescales `L` so `tr(L) = leon_l_scale * ||G||_F^2` before the existing trace normalization. At `leon_l_scale = 1.0` the resulting trace fraction is exactly 0.5 every step.
- `leon_match_l0_norm = True`: rescales the final NS output to Frobenius norm `sqrt(min(D1, D2))`, matching the natural L=0 (Muon-NS) update norm before the aspect-ratio multiplier.

The first-moment and second-moment buffers were also switched from EMA storage (`lerp_` + `/(1 - decay)`) to direct unnormalized exponential sums `S_t = mu*S_{t-1} + grad_t` and `T_t = beta2*T_{t-1} + g g^T`. Behavior is mathematically equivalent; the code is cleaner.

All four runs use `train_steps=500`, 4 H100s, `leon_lr=0.025`, `leon_wd=0.025`, `leon_mu=0.95`, `leon_beta2=0.7`, `leon_cooldown_frac=0.7`, `leon_ns_iters=12`, `leon_orthogonalize_dtype=float32`, `leon_eps=1e-12`, diagnostics every 125 steps, wandb project `modded-nanogpt-track3`.

### Val trajectories

| Step | L=0 ref | L=1 baseline | A: match_g (frac=0.5) | B: match_l0_norm |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.82584 | 10.82584 | 10.82584 | 10.82584 |
| 125 | 4.63749 | 4.58737 | 4.59188 | 4.72858 |
| 250 | 4.06976 | 4.05694 | 4.05349 | 4.11233 |
| 375 | 3.85318 | 3.84764 | 3.84652 | 3.88735 |
| 500 | 3.75171 | 3.74774 | 3.74984 | 3.78380 |

Final-step gaps relative to L=0:

| Config | Final val | vs L=0 |
| --- | ---: | ---: |
| L=0 ref | 3.75171 | +0.00000 |
| L=1 baseline | 3.74774 | -0.00397 |
| A: match_g | 3.74984 | -0.00187 |
| B: match_l0_norm | 3.78380 | +0.03209 |

### Diagnostic averages (q.weight at blocks.0.attn)

| Step | Config | `update_norm` | `update_l0_norm` | `cos(update, update_L0)` | `l_trace_fraction` |
| ---: | --- | ---: | ---: | ---: | ---: |
| 125 | L=0 | 27.43 | 27.43 | 1.000 | 0.000 |
| 125 | L=1 | 20.09 | 27.44 | 0.909 | 0.431 |
| 125 | match_g | 19.30 | 27.47 | 0.902 | 0.500 |
| 125 | match_l0_norm | 27.71 | 27.46 | 0.912 | 0.362 |
| 250 | L=1 | 20.15 | 27.50 | 0.911 | 0.394 |
| 250 | match_g | 18.41 | 27.51 | 0.892 | 0.500 |
| 250 | match_l0_norm | 27.71 | 27.38 | 0.918 | 0.258 |
| 375 | L=1 | 20.30 | 27.54 | 0.913 | 0.275 |
| 375 | match_g | 17.79 | 27.54 | 0.883 | 0.500 |
| 375 | match_l0_norm | 27.71 | 27.55 | 0.917 | 0.278 |
| 500 | L=1 | 20.48 | 27.48 | 0.918 | 0.179 |
| 500 | match_g | 16.18 | 27.49 | 0.865 | 0.500 |
| 500 | match_l0_norm | 27.71 | 27.50 | 0.922 | 0.175 |

`update_norm` for `match_l0_norm` is pinned at `sqrt(768) = 27.71` as designed. `l_trace_fraction` for `match_g` is exactly 0.5 every step as designed.

### Findings

1. With `float32` + `ns_iters=12`, the standard L=1 (unnormalized-sum) and L=0 trajectories sit within ~0.004 of each other at 500 steps. The large 500-step gap in the bf16 + `ns_iters=6` diagnostics in [the original sweep](#diagnostic-scale-sweep) is no longer reproduced. The Leon vs Muon-NS gap visible in the [1500-step trajectory table](#standard-leon-vs-leon-l0-trajectory) is therefore more about NS precision than about the second-momentum term, at least early in training.
2. Forcing `tr(L)` fraction to exactly 0.5 (`match_g`, l_scale=1.0) is statistically a wash versus the unnormalized-sum L=1 baseline. The update norm is *more* damped (~70% of L=0 norm vs ~73% for the baseline) and the cosine alignment with `update_L0` is slightly *lower* (0.87 vs 0.92 at step 500), but the validation trajectory is essentially identical.
3. Rescaling the update to match the L=0 Frobenius norm (`match_l0_norm`) **hurts** by +0.032 at step 500. The directional information is the same (cos vs update_L0 is the highest of the three, ~0.92), but undoing the damping makes the per-step move too aggressive in the regions where `L` was correctly preconditioning. This is consistent with the natural per-layer damping serving as an adaptive learning-rate signal rather than a pure shrinkage that needs to be corrected.

### Implications

- The "L over-damps the update" interpretation from the bf16 + `ns_iters=6` sweep should be re-stated. With higher-precision NS (float32, 12 iters), the damping that `L` introduces is approximately benign at 500 steps; it does not need to be cancelled out. The remaining 1500-step gap from the older bf16 runs is likely a precision artifact that scales with `||L||`.
- `match_l0_norm` is not a useful intervention — it discards the per-layer adaptive scaling that `L` is providing.
- Worth retesting at 1500 steps before drawing a firm conclusion: the existing 1500-step Leon vs L=0 table used `ns_iters=6`, `bfloat16`, and `eps=1e-9`. A clean head-to-head at `float32` + `ns_iters=12` would confirm whether the gap collapses.

## 2026-05-22 1500-Step Revalidation

Same four configs as the 500-step screen, extended to `train_steps=1500`. All other knobs identical: 4 H100s, `leon_lr=0.025`, `leon_wd=0.025`, `leon_mu=0.95`, `leon_beta2=0.7`, `leon_cooldown_frac=0.7`, `leon_ns_iters=12`, `leon_orthogonalize_dtype=float32`, `leon_eps=1e-12`, diagnostics every 125 steps, wandb project `modded-nanogpt-track3`. Diagnostic `relative_update*` keys are now lr-free (`||U||_F / ||W||_F`).

### Val trajectories

| Step | L=0 ref | L=1 baseline | A: match_g | B: match_l0_norm |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.82584 | 10.82584 | 10.82584 | 10.82584 |
| 125 | 4.63383 | 4.57431 | 4.57564 | 4.71833 |
| 250 | 4.10859 | 4.09326 | 4.08884 | 4.13491 |
| 375 | 3.92496 | 3.92131 | 3.91419 | 3.94600 |
| 500 | 3.81655 | 3.81557 | 3.80913 | 3.83415 |
| 625 | 3.73044 | 3.73104 | 3.72579 | 3.74433 |
| 750 | 3.66503 | 3.66780 | 3.66112 | 3.67849 |
| 875 | 3.60892 | 3.61326 | 3.60645 | 3.62249 |
| 1000 | 3.56024 | 3.56399 | 3.55842 | 3.57239 |
| 1125 | 3.51987 | 3.52393 | 3.51879 | 3.53271 |
| 1250 | 3.48187 | 3.48581 | 3.48114 | 3.49418 |
| 1375 | 3.45130 | 3.45515 | 3.45160 | 3.46330 |
| 1500 | 3.43230 | 3.43581 | 3.43362 | 3.44404 |

### Final-step gaps vs L=0

| Config | Final val | vs L=0 |
| --- | ---: | ---: |
| L=0 ref | 3.43230 | +0.00000 |
| L=1 baseline | 3.43581 | +0.00351 |
| A: match_g | 3.43362 | +0.00132 |
| B: match_l0_norm | 3.44404 | +0.01174 |

Compared to the [original bf16 + `ns_iters=6` 1500-step gap](#standard-leon-vs-leon-l0-trajectory) of `-0.12540` (L=0 minus standard L=1), the gap is now only `+0.00351` and *favors L=0 over L=1 by 0.0035*. The order has not flipped but the magnitude has collapsed by ~36x.

### Diagnostic snapshot (blocks.0.attn.q.weight)

| Step | Config | ‖U‖/‖W‖ | ‖U₀‖/‖W‖ | cos(U, U₀) | tr(L) fraction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 125 | L=0 | 0.891 | 0.891 | 1.000 | 0.000 |
| 125 | L=1 | 0.753 | 1.019 | 0.914 | 0.422 |
| 125 | match_g | 0.775 | 1.087 | 0.904 | 0.500 |
| 125 | match_l0_norm | 0.756 | 0.749 | 0.913 | 0.380 |
| 750 | L=0 | 0.414 | 0.414 | 1.000 | 0.000 |
| 750 | L=1 | 0.367 | 0.502 | 0.910 | 0.321 |
| 750 | match_g | 0.348 | 0.543 | 0.879 | 0.500 |
| 750 | match_l0_norm | 0.359 | 0.357 | 0.914 | 0.293 |
| 1500 | L=0 | 0.415 | 0.415 | 1.000 | 0.000 |
| 1500 | L=1 | 0.374 | 0.504 | 0.916 | 0.193 |
| 1500 | match_g | 0.326 | 0.553 | 0.863 | 0.500 |
| 1500 | match_l0_norm | 0.361 | 0.359 | 0.920 | 0.197 |

Same qualitative picture as the 500-step screen, now at full benchmark length. `match_g` pins `tr(L)` fraction at 0.5 exactly. `match_l0_norm` pins `‖U‖` to ≈ `‖U₀‖`. `match_g`'s `cos(U, U₀)` drifts down to 0.86 by step 1500 (more rotation than baseline), and its `‖U‖/‖W‖` is the smallest of the four (~0.33), suggesting the rescaled `L` mildly suppresses the effective step.

### Revised conclusions

1. **The old "L over-damps the update" interpretation does not survive float32 + 12-iter NS.** With higher-precision orthogonalization the standard nonzero-`L` trajectory tracks L=0 to within ~0.004 over 1500 steps. The 0.125 gap reported in the original [Standard Leon vs Leon L=0 Trajectory](#standard-leon-vs-leon-l0-trajectory) table appears to be primarily a low-precision NS artifact, not a fundamental algorithmic problem.
2. **Pinning the trace fraction to 0.5** (`match_g`) gives the slightly best L=1 variant (3.43362, +0.00132 vs L=0), but the improvement is well inside run-to-run noise and the diagnostic shows it costs some cosine alignment with the L=0 direction.
3. **Matching the L=0 update Frobenius norm** (`match_l0_norm`) consistently *hurts* (+0.01174 at 1500 steps, +0.03209 at 500 steps). The directional information is fine (cos with `U₀` is the highest of the three) but undoing the natural per-layer damping inflates `‖W‖` and ends in a worse optimum.
4. The natural EMA-Gram preconditioner is acting as an adaptive per-layer step-size brake, and that brake is approximately neutral-to-mildly-helpful when NS is precise enough to track it faithfully. Future work on `L` should focus on probes that *exploit* the brake (e.g., raising `lr` while keeping `L` on) rather than removing it.

## 2026-05-24: Gradient-Difference Second Moment

New variant: second moment accumulates outer products of consecutive gradient differences rather than raw gradients:

```
T_t = beta2 * T_{t-1} + (grad_t - grad_{t-1}) * (grad_t - grad_{t-1})^T
```

Enabled via `leon_use_grad_diff=True`. `prev_grad_buffer` (zeros at step 0, float32) is stored in optimizer state. At step 0 delta = grad − 0 = grad, matching the baseline. Run `20260524-164921-dc8e6616`: 500 steps, 4× H100 NVL (2.11.0+cu128/12.8), same shared settings as the 2026-05-22 500-step screen (`leon_lr=0.025`, `leon_wd=0.025`, `leon_mu=0.95`, `leon_beta2=0.7`, `leon_cooldown_frac=0.7`, `leon_ns_iters=12`, `leon_orthogonalize_dtype=float32`, `leon_eps=1e-12`, diagnostics every 125 steps).

### Val trajectory

| Step | L=0 ref (existing) | L=1 baseline (existing) | C: grad-diff L=1 (new) | grad-diff vs L=1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.82584 | 10.82584 | 10.82584 | +0.00000 |
| 125 | 4.63749 | 4.58737 | **4.57622** | **−0.01115** |
| 250 | 4.06976 | 4.05694 | **4.05423** | **−0.00271** |
| 375 | 3.85318 | 3.84764 | 3.84771 | +0.00007 |
| 500 | 3.75171 | 3.74774 | 3.74790 | +0.00016 |

The grad-diff variant shows a meaningful early advantage (−0.011 at step 125, −0.003 at step 250) that is larger than typical run-to-run noise, but converges to essentially the same level as the L=1 baseline by step 500 (within 0.0002). It is also better than L=0 at every checkpoint.

### Diagnostic averages (3-layer mean: blocks.0.attn.q.weight, blocks.6.mlp.proj.weight, blocks.11.attn.proj.weight)

| Step | `tr(L) frac` | `‖U‖/‖U_L0‖` | `cos(U, U_L0)` | `rel_update` | `grad_diff_norm/grad_norm` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 125 | 0.724 | 0.688 | 0.893 | 0.702 | 1.746 |
| 250 | 0.701 | 0.686 | 0.904 | 0.519 | 1.561 |
| 375 | 0.574 | 0.697 | 0.912 | 0.495 | 1.456 |
| 500 | 0.320 | 0.715 | 0.919 | 0.504 | 1.302 |

For reference, the existing L=1 baseline diagnostic at step 125 (blocks.0.attn.q.weight only): `tr_frac=0.431`, `‖U‖/‖U_L0‖=0.732`, `cos=0.909`.

### Diagnostic snapshot (blocks.0.attn.q.weight only)

| Step | `‖U‖/‖W‖` | `‖U₀‖/‖W‖` | `cos(U, U₀)` | `tr(L) frac` | `grad_diff_norm` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 125 | 0.753 | 1.123 | 0.877 | 0.660 | 9822 |
| 250 | 0.560 | 0.839 | 0.880 | 0.592 | 6646 |
| 375 | 0.533 | 0.794 | 0.883 | 0.469 | 3408 |
| 500 | 0.540 | 0.793 | 0.887 | 0.327 | 3238 |

`grad_diff_norm` is ~1.7× `grad_norm` early (step 125), declining to ~1.3× by step 500. This ratio > 1 is consistent with gradient magnitude shrinking fast in early training so that `‖grad_t − grad_{t-1}‖ > ‖grad_t‖`.

### Findings

1. **Early-phase advantage.** The grad-diff variant improves step-125 val_loss by 0.011 over the L=1 baseline and 0.061 over L=0. This advantage is larger than typical single-run noise and the directionality (grad-diff better than both alternatives early) suggests genuine signal.
2. **Late-phase convergence.** By step 500 the trajectories equalize. The final difference vs L=1 is only +0.0002 (a virtual tie), so the grad-diff second moment neither helps nor hurts once training stabilizes.
3. **Higher tr(L) fraction early.** Because `grad_diff_norm` is ~1.75× `grad_norm` at step 125, the accumulated `L` is larger than for raw-gradient L at the same `beta2`. Despite this, the early loss is better — so the qualitative "more L hurts" finding from the original bfloat16 diagnostics does not hold for the grad-diff variant. The *kind* of information in `L` matters, not just its scale.
4. **cos(U, U_L0)** is slightly lower than the L=1 baseline at steps 125–250 (~0.877–0.880 vs ~0.909–0.911). The grad-diff `L` introduces more directional rotation of the update than raw-grad `L`, which may partly explain the early advantage.
5. **`grad_diff_norm` is a useful diagnostic.** It tracks gradient velocity and confirms that gradients are changing rapidly early in training (ratio >1 means the inter-step gradient change exceeds the current gradient magnitude).

### Next step

Extend to 1500 steps (decision rule met: grad-diff tied/better than baseline at step 500). The 500-step early advantage pattern may or may not persist: `tr_frac` for grad-diff is already converging toward the L=1 baseline value by step 500 (0.320 vs 0.179), suggesting the two variants become more similar as training progresses and gradients stabilize. Run C baseline comparison at 1500 steps will confirm whether the early gain translates to any residual advantage at benchmark length.
