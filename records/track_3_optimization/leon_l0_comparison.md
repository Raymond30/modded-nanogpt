# Leon L=0 Comparison Notes

Last updated: 2026-05-05 14:20 America/Chicago

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

| Scale | Step | `tr(L)` fraction | `||update_L|| / ||update_L0||` | `cos(update_L, update_L0)` | Relative step |
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

| Step | `tr(L)` fraction | `||update_L|| / ||update_L0||` | `cos(update_L, update_L0)` | Relative step |
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
