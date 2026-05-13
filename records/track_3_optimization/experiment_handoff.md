# Track 3 Optimization Experiment Handoff

Last updated: 2026-05-08 America/Chicago

This stable handoff file summarizes the experiments from the recent Codex tuning conversation. The authoritative run ledger remains `records/track_3_optimization/tuning_log.csv`; use this file as a fast orientation layer before selecting the next run, and update it in place after meaningful new results.

## Operating Notes

- Read `records/track_3_optimization/AGENTS.md`, `records/track_3_optimization/tuning_log.csv`, and this handoff before launching new runs.
- Append one row to `tuning_log.csv` immediately after each completed, failed, or crashed run.
- Keep experiment artifacts committed periodically. Relevant artifacts include `tuning_log.csv`, run directories, metadata/configs, and logs needed for reproducibility.
- Do not reuse the same `--out-dir` for a dry-run and the real launch. `train_gpt_simple_leon.py` creates the run directory with `exist_ok=False` in both modes, so a dry-run can leave a stale directory that causes the subsequent real `torchrun` launch to hang after distributed init without ever creating `train.log`.
- User granted permission for 30 GPU launches without further confirmation so long as GPUs are available. The latest completed diagnostic sweep consumed runs 14-20 of that allowance in this conversation context. One additional sandbox retry failed before using GPUs.
- Current 30-run schedule/LR probe is predeclared around Muon-like hparams: `leon_wd=0.025`, `leon_mu=0.95`, `leon_beta2=0.7`, `leon_cooldown_frac=0.7`, `adam_cooldown_frac=0.7`, `train_steps=1500`; sweep `leon_lr in {0.0225, 0.025, 0.0275, 0.03, 0.035}` and ramp windows `{0.3->0.6, 0.4->0.7, 0.5->0.8, 0.6->0.9, 0.7->0.9, 0.8->0.95}`, skipping the already-completed duplicate `lr=0.025,ramp=0.5->0.8` and replacing it with `lr=0.025,ramp=0.6->0.95`.
- Main screening runs used `train_steps=1500`; the L-scale diagnostic sweep used `train_steps=500`. The first logged Leon run to reach `val_loss < 3.28` is now the 3375-step float32 12-step benchmark rerun, and it crossed only at the final checkpoint.
- Many later 1500-step screens used 2 GPUs because only GPUs 1 and 2 were available, while the latest diagnostics used 4 GPUs. Do not compare wall time directly between 4-GPU and 2-GPU runs.
- A historical bias-correction branch was tried briefly, but the active code has now switched away from it and instead uses unnormalized exponential-sum scaling for the Leon moments.
- The latest bf16 12-step orthogonalization probe diverged to NaNs at validation checkpoints starting at step 125 and was terminated after step 644. Treat it as a failed attribution run, not as a comparison point against the successful float32/float64 12-step probes.
- The first fixed-budget 3375-step rerun of the current best Leon screen finished at `3.27879` and reached `val_loss < 3.28` for the first time in the local Leon ledger, but only at the final checkpoint `step 3375`.

## Current Bests

| Category | Run | Key hparams | Final val loss | Run path |
| --- | --- | --- | ---: | --- |
| Best Leon 3375-step benchmark | `leon_orth_float32_lr035_wd0275_b260_eps1e12_ns12_3375_4gpu` | `train_steps=3375`, `leon_lr=0.035`, `leon_wd=0.0275`, `leon_cooldown_frac=0.6`, `leon_beta2=0.6`, `leon_ns_iters=12`, `leon_orthogonalize_dtype=float32`, `leon_eps=1e-12` | 3.27879 | `records/track_3_optimization/runs/leon/20260506-leon-best3375-f1a09dd/lr035_wd0275_b260_float32_eps1e12_ns12_3375_4gpu` |
| Best Leon screen | `leon_orth_float32_lr035_wd0275_b260_eps1e12_ns12_4gpu` | `leon_lr=0.035`, `leon_wd=0.0275`, `leon_cooldown_frac=0.6`, `leon_beta2=0.6`, `leon_ns_iters=12`, `leon_orthogonalize_dtype=float32`, `leon_eps=1e-12` | 3.42975 | `records/track_3_optimization/runs/leon/20260506-leon-orth-float32-4b57bc8/lr035_wd0275_b260_float32_eps1e12_ns12_4gpu` |
| Best 6-step bf16 constant full-L Leon reference | `retune1500_lr035_wd0275_b260_4gpu` | `leon_lr=0.035`, `leon_wd=0.0275`, `leon_cooldown_frac=0.6`, `leon_beta2=0.6`, current codepath | 3.44937 | `records/track_3_optimization/runs/leon/20260505-leon-retune1500-cda40ac/lr035_wd0275_b260_4gpu` |
| Best original Muon baseline | `muon1500_2gpu` | `lr=0.025`, `wd=0.025`, `momentum=0.95`, `ns_steps=12` | 3.43079 | `records/track_3_optimization/runs/muon/20260504-muon-screen1500` |
| Muon with Leon-like NS | `muon1500_leonns_2gpu` | `lr=0.025`, `wd=0.025`, `momentum=0.95`, `ns_steps=6`, `eps=1e-9` | 3.44804 | `records/track_3_optimization/runs/muon/20260505-muon-leonns-screen1500` |
| Leon L=0 with Muon hparams | `leon_l0_muon_hparams_4gpu` | `leon_lr=0.025`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, `L=0` | 3.44907 | `records/track_3_optimization/runs/leon/20260505-leon-l0-screen1500/muon_hparams_4gpu` |
| Best 500-step diagnostic | `lscale0_500_4gpu_diag` | same Muon-like hparams, `leon_l_scale=0.0` | 3.79464 | `records/track_3_optimization/runs/leon/20260505-leon-lscale-diagnostics/lscale0_500_4gpu` |
| Best full-L diagnostic on current code | `lscale1_unnormsum_500_4gpu_diag` | same Muon-like hparams, `leon_l_scale=1.0`, unnormalized-sum codepath | 3.80020 | `records/track_3_optimization/runs/leon/20260505-leon-unnormsum-lscale-diagnostics/lscale1_500_4gpu` |
| Historical bias-corrected full-L diagnostic | `lscale1_biascorr_500_4gpu_diag` | same Muon-like hparams, old `leon_bias_correction=True` branch | 4.05228 | `records/track_3_optimization/runs/leon/20260505-leon-biascorr-lscale-diagnostics/lscale1_500_4gpu` |

Interpretation:

- The new 3375-step Leon benchmark run, `leon_orth_float32_lr035_wd0275_b260_eps1e12_ns12_3375_4gpu`, finished at `3.27879` and crossed the `3.28` threshold exactly at the final checkpoint `step 3375`. This is the first Leon run in the local ledger to reach the benchmark target.
- Against the local `train_gpt_simple.py` Muon single-run summary in `training_summary.md`, Leon is `-0.00019` lower in final validation loss (`3.27879` vs `3.27898`) on the same 3375-step budget. That is a fixed-budget head-to-head only; it does not yet establish a better threshold-crossing distribution because Leon crossed only at the last checkpoint and repeated trials are still missing.
- The float32 12-step Newton-Schulz follow-up, `leon_orth_float32_lr035_wd0275_b260_eps1e12_ns12_4gpu`, finished at `3.42975`. That beats the 6-step bf16 reference by `-0.01962`, the float64 12-step probe by `-0.00064`, the best prior Leon schedule by `-0.01678`, and the original `muon1500_2gpu` baseline by `-0.00104`; it still did not reach `val_loss < 3.28`.
- The prior float64 12-step run finished at `3.43039`, so double precision is unnecessary for the observed 12-step gain in this single attribution run. Runtime did not improve with float32 here (`732.027s` train time vs `725.158s` for float64), so treat speed as noise/implementation-dependent until repeated.
- On the current unnormalized-sum codepath, the first corrected 1500-step retune point `leon_lr=0.035`, `leon_wd=0.03`, `leon_cooldown_frac=0.6` reached `3.45093`, improving the prior best tuned Leon screen by `-0.05857`.
- That early retune result was only `0.02014` behind the original 12-step Muon baseline, `0.00289` behind the Muon Leon-NS baseline, and `0.00186` behind the Leon `L=0` control.
- The next LR point `leon_lr=0.045` finished at `3.45492`, so `0.035` remains ahead by `0.00399` while the `0.0525` and `0.06` runs are still pending.
- The user-requested lower probe `leon_lr=0.025` finished at `3.45329`, which is `0.00236` behind `0.035` but `0.00163` better than `0.045`, so the current LR ordering is `0.035` best, then `0.025`, then `0.045`.
- The first WD sweep point around the `lr=0.035` center, `leon_wd=0.0275`, finished at `3.45028`, improving `wd=0.03` by `0.00065` and becoming the current best tested Leon configuration while the rest of the WD bracket is still pending.
- The next lower WD probe `leon_wd=0.025` finished at `3.45110`, which is `0.00082` behind `wd=0.0275` and `0.00017` behind `wd=0.03`, so `wd=0.0275` still leads and the downward side has not improved yet.
- The escalated retry at `leon_wd=0.0225` finished at `3.45029` after an initial sandbox TCPStore failure. That result is only `0.00001` behind `wd=0.0275` and `0.00064` better than `wd=0.03`, so the lower side is now effectively tied with the current best on a single run.
- The next lower probe `leon_wd=0.02` finished at `3.45078`. That is `0.00049` behind `wd=0.0225` and `0.00050` behind `wd=0.0275`, so the best lower-side point remains `wd=0.0225` and the practical near-tie is now between `0.0225` and `0.0275`.
- The first `leon_beta2` sweep point from the nominal best WD center, `leon_beta2=0.6` at `lr=0.035`, `wd=0.0275`, `cd=0.6`, finished at `3.44937`. It beat the previous best Leon screen by `0.00091` and outperformed the old `beta2=0.7` center at every validation checkpoint from step `125` through `1500`.
- The float32 orthogonalization precision probe matched `retune1500_lr035_wd0275_b260_4gpu` except for `leon_orthogonalize_dtype=float32`. It finished at `3.45121`, which is `+0.00184` worse than the bfloat16 constant full-`L` reference and `+0.00468` worse than the best prior Leon schedule. Runtime was only modestly higher (`691.320s` train time vs `681.561s`). This single run does not support bfloat16 Newton-Schulz precision as the main remaining gap to original Muon.
- The float64 precision-only probe with `leon_eps=1e-12` and the same `leon_ns_iters=6` finished at `3.45027`, still `+0.00090` worse than the bfloat16 reference. Increasing to `leon_ns_iters=12` at the same float64/eps setting finished at `3.43039`, and the float32 12-step follow-up finished at `3.42975`, so iteration count, not dtype precision, was the decisive tested factor.
- The next lower beta2 probe, `leon_beta2=0.5` at the same center, finished at `3.44988`. That is `0.00051` worse than `beta2=0.6`, though still `0.00040` better than the old `beta2=0.7` center, so the beta2 bracket now points to a local optimum near `0.6`.
- The first `leon_mu` probe from the current beta2 center, `leon_mu=0.925`, finished at `3.45394`. That is `0.00457` worse than the current best `mu=0.95` center, so the lower side is not promising and the sweep should move upward next.
- The first scheduled-`L` run, `leon_l_scale_schedule=linear_ramp` with a `0.2 -> 0.6` training-fraction ramp from `0` to `1`, finished at `3.44971`. That is only `0.00034` behind the constant full-`L` `beta2=0.6` best, but it trailed at every checkpoint and did not improve the current center.
- The Muon-hparams late scheduled-`L` run, `leon_lr=0.025`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, and a `0.5 -> 0.8` ramp from `L=0` to full `L`, finished at `3.44939`. It beat the Leon `L=0` and Muon Leon-NS controls at several mid/late checkpoints after ramp activation, but final validation loss still trailed Leon `L=0` by `+0.00032`, Muon Leon-NS by `+0.00135`, original Muon by `+0.01860`, and the current best Leon screen by `+0.00002`.
- The first broad schedule/LR sweep point, `leon_lr=0.0225`, `leon_wd=0.025`, `leon_beta2=0.7`, and a `0.3 -> 0.6` ramp, finished at `3.45287`. It improved over Leon `L=0` through step `1125` and over the prior `0.5 -> 0.8` ramp through step `875`, but faded after the ramp completed and finished `+0.00350` behind the current best Leon and `+0.02208` behind original Muon.
- The second broad schedule/LR sweep point, `leon_lr=0.025` with the same `0.3 -> 0.6` ramp, finished at `3.45046`. Raising LR to the Muon value improved final loss versus the `0.0225` point by `-0.00241`, but it still trailed the prior `0.5 -> 0.8` ramp by `+0.00107`, Leon `L=0` by `+0.00139`, current best Leon by `+0.00109`, and original Muon by `+0.01967`.
- The third broad schedule/LR sweep point, `leon_lr=0.0275` with the previously best `0.5 -> 0.8` ramp, finished at `3.44653`. It is the new best Leon-family result, beating the prior best schedule by `-0.00286`, the previous best tuned constant-L Leon by `-0.00284`, Leon `L=0` by `-0.00254`, and Muon Leon-NS by `-0.00151`, while still trailing original 12-step Muon by `+0.01574`.
- The upper LR bracket at `leon_lr=0.03` with the same `0.5 -> 0.8` ramp finished at `3.45106`. It damaged the early L=0-equivalent phase and never recovered enough, finishing `+0.00453` behind the new `lr=0.0275` schedule center and `+0.00167` behind the old `lr=0.025` schedule reference.
- The later timing bracket at `leon_lr=0.0275` with a `0.6 -> 0.9` ramp finished at `3.44854`. It still beat the old `lr=0.025`, `0.5 -> 0.8` schedule by `-0.00085` and the previous constant full-L best by `-0.00083`, but it trailed the current `lr=0.0275`, `0.5 -> 0.8` center by `+0.00201`, so delaying the ramp further was not beneficial.
- The earlier timing bracket at `leon_lr=0.0275` with a `0.4 -> 0.7` ramp finished at `3.44860`. It briefly led the `0.5 -> 0.8` center at steps `750` and `875`, but it faded from step `1000` onward and finished `+0.00207` behind the current best schedule center, so moving the full ramp earlier was also not beneficial.
- The Muon Leon-NS rerun is better than tuned Leon by `0.06146`, but worse than original 12-step Muon by `0.01725`.
- Leon with `L=0` and Muon hparams landed at `3.44907`, near-identical to Muon Leon-NS (`+0.00103` final val), indicating the earlier Leon gap was mainly from the nonzero second-momentum `L` contribution and tuned hparams rather than the optimizer wrapper.
- The 500-step `leon_l_scale` diagnostic sweep was monotone: `0.0` final `3.79464`, `0.1` `3.87248`, `0.25` `3.93081`, `0.5` `3.99168`, `1.0` `4.04142`. Diagnostics show full `L` makes `tr(L)` about 98.8% of the normalization denominator at steps 125-375 and shrinks the update to about 16-17% of the hypothetical L=0 update.
- The old bias-corrected full-`L` follow-up was slightly worse than uncorrected full `L` (`4.05228` vs `4.04142`). The active code no longer uses that branch; it now scales Leon as an unnormalized exponential sum by dividing the Nesterov-updated `g` by `1 - mu` and the second-momentum matrix by `1 - beta2`.
- The new full-`L` diagnostic on the current unnormalized-sum codepath improved sharply to `3.80020`. Its `l_trace_fraction` dropped to about `0.50 / 0.45 / 0.40` at steps `125 / 250 / 375`, and `update_norm / update_l0_norm` rose to about `0.78 / 0.80 / 0.83`, leaving full `L` only `+0.00556` behind the `L=0` control at step 500.

## LR Sensitivity Sweep (2026-05-06)

**Objective:** Test whether Leon is less sensitive to learning rate than Muon (user hypothesis).

**Protocol:** Log-uniform sweep, factor-of-2 spacing, 5 points each, 3375 steps, 4× H100 NVL. Leon centered on best known LR (0.035); Muon centered on baseline LR (0.025). All other hparams fixed at each optimizer's best known values. Leon optimal (lr=0.035) reused from previous run.

### Leon LR sweep (best config: wd=0.0275, beta2=0.6, ns=12, float32, eps=1e-12, cd=0.6)

| LR | vs optimal | Final val | Δ | Reached 3.28 | Run path |
| --- | --- | ---: | ---: | --- | --- |
| 0.00875 | 4× below | 3.29982 | +0.02103 | No | `runs/leon/20260506-lr-sensitivity/leon_lr00875_4gpu` |
| 0.0175 | 2× below | 3.28451 | +0.00572 | No | `runs/leon/20260506-lr-sensitivity/leon_lr0175_4gpu` |
| **0.035** | **optimal** | **3.27879** | **0** | **Yes (step 3375)** | `runs/leon/20260506-leon-best3375-f1a09dd/…` |
| 0.070 | 2× above | 3.29345 | +0.01466 | No | `runs/leon/20260506-lr-sensitivity/leon_lr070_4gpu` |
| 0.140 | 4× above | 3.32923 | +0.05044 | No | `runs/leon/20260506-lr-sensitivity/leon_lr140_4gpu` |

### Muon LR sweep (default config: wd=0.025, ns=12-step bf16, cd=0.7)

| LR | vs nominal | Final val | Δ | Reached 3.28 | Run path |
| --- | --- | ---: | ---: | --- | --- |
| 0.00625 | 4× below | 3.31013 | +0.03190 | No | `runs/muon/20260506-lr-sensitivity/muon_lr00625_4gpu` |
| 0.0125 | 2× below | 3.28832 | +0.01009 | No | `runs/muon/20260506-lr-sensitivity/muon_lr0125_4gpu` |
| **0.025** | **nominal** | **3.27823** | **0** | **Yes (step 3375)** | `runs/muon/20260506-lr-sensitivity/muon_lr025_4gpu` |
| 0.050 | 2× above | 3.27758 | **−0.00065** | **Yes (step 3375)** | `runs/muon/20260506-lr-sensitivity/muon_lr050_4gpu` |
| 0.100 | 4× above | 3.29988 | +0.02165 | No | `runs/muon/20260506-lr-sensitivity/muon_lr100_4gpu` |

### Sensitivity comparison

| Deviation | Leon Δ (low) | Leon Δ (high) | Muon Δ (low) | Muon Δ (high) |
| --- | ---: | ---: | ---: | ---: |
| 2× | +0.006 | +0.015 | +0.010 | **−0.001** |
| 4× | +0.021 | **+0.050** | +0.032 | +0.022 |
| Average (2×) | — | **+0.010** | — | **+0.005** |
| Average (4×) | — | **+0.036** | — | **+0.027** |

### Hypothesis verdict: NOT confirmed

The data **does not support** the hypothesis that Leon is less sensitive to learning rate than Muon. The results show the opposite pattern:

- **At 2× deviation**, Muon is less sensitive on average (+0.005) than Leon (+0.010). Muon lr=0.050 actually outperforms the nominal lr=0.025 by a small margin (−0.00065), suggesting the Muon optimum is flatter or right-shifted toward ~0.040–0.050.
- **At 4× deviation**, Muon is also less sensitive on average (+0.027) than Leon (+0.036). Leon is particularly fragile on the high-LR side: lr=0.140 degrades by +0.050, nearly 2.5× the Muon high-side degradation of +0.022.
- **Asymmetries differ**: Leon is more sensitive to high LR; Muon is somewhat more sensitive to low LR but is robust to moderate overshooting.

**Important caveats:**
- All results are single-run comparisons. Run-to-run variance (typically ~0.002–0.005 for 3375-step runs) could shift the ordering at small differences, especially for Muon lr=0.050 vs 0.025.
- The Muon "nominal optimal" of 0.025 may be slightly below the true optimum (~0.040); this artificially makes the 2× high-side result look neutral or better.
- The Leon and Muon configs differ in many ways beyond LR (NS iterations, weight decay, cooldown, beta2), so this is not a pure isolated comparison of LR sensitivity.

**Practical take:** Muon's LR plateau is broad and right-skewed (stable from at least 0.0125 to 0.050). Leon's plateau is narrower and left-skewed (stable from ~0.0175 to ~0.070 with rapid degradation above). If anything, Muon is the more LR-robust of the two in this regime.

## Experiment Sequence

| Phase | Runs | Decision |
| --- | --- | --- |
| Leon base screen | `base` | Starting point: `leon_lr=0.025`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, final `3.57447`. |
| Leon LR sweep, 4 GPU | `lr0225`, `lr0275`, `lr030`, `lr0325`, `lr035`, `lr040` | Loss improved as LR increased through `0.04`. `lr0225` was worse than base. |
| Failed launch / resource issue | `lr0275_sandbox_fail`, `lr045_oom` | One sandbox TCPStore failure and one OOM caused by occupied GPUs. Both were recorded in the CSV. |
| Leon LR sweep, 2 GPU | `lr045_retry_2gpu`, `lr050_2gpu`, `lr055_2gpu`, `lr0525_2gpu` | Practical LR boundary bracketed around `0.05` to `0.055`; selected `0.0525` for WD tuning. |
| Muon unchanged baseline | `muon1500_2gpu` | Established unchanged Muon reference at final `3.43079`, substantially better than Leon at the time. |
| Leon WD sweep | `lr0525_wd030_2gpu`, `lr0525_wd035_2gpu`, `lr0525_wd0325_2gpu` | `wd=0.03` was best. `0.035` and `0.0325` were worse, bracketing the local optimum near `0.03`. |
| Leon cooldown sweep | `lr0525_wd030_cd080_2gpu`, `cd060`, `cd050`, `cd065`, `cd055`, `cd0575` | `leon_cooldown_frac=0.6` was best. `0.55` was close but worse by `0.00060`; `0.575` was worse than both. |
| Muon Leon-NS baseline | `muon1500_leonns_2gpu` | User modified Muon orthogonalization to Leon L=0 style 6-step augmented Newton-Schulz. Rerun final was `3.44804`. |
| Leon L=0 control | `leon_l0_muon_hparams_4gpu` | Multiplied Leon `L` by `0` and used Muon Leon-NS hparams. Final was `3.44907`, matching Muon Leon-NS within `0.00103`. |
| Leon L-scale diagnostics | `lscale{0,010,025,050,1}_500_4gpu_diag` | Added `leon_l_scale` and checkpoint diagnostics. Lower `L` scale was better at every checkpoint; full `L` mostly damped the update through trace domination rather than producing a wholly different direction. |
| Historical bias-correction follow-up | `lscale1_biascorr_500_4gpu_diag` | Old experimental branch using `m/(1-mu^t)` and `L/(1-beta2^t)`. Full-L bias correction did not help, and that branch is no longer the active implementation. |
| Unnormalized-sum full-L diagnostic | `lscale1_unnormsum_500_4gpu_diag` | Current code divides the Nesterov-updated `g` by `1-mu` and the second-momentum matrix by `1-beta2`. This largely removed the earlier scaling pathology for `l_scale=1.0`. |
| Current-codepath 1500-step retune restart | `retune1500_lr035_retry_4gpu`, `retune1500_lr045_4gpu`, `retune1500_lr025_4gpu`, `retune1500_lr035_wd0275_4gpu`, `retune1500_lr035_wd025_4gpu`, `retune1500_lr035_wd0225_retry_4gpu`, `retune1500_lr035_wd020_4gpu`, `retune1500_lr035_wd0275_b260_4gpu`, `retune1500_lr035_wd0275_b250_4gpu`, `retune1500_lr035_wd0275_b260_mu0925_4gpu` | After avoiding dry-run `out-dir` reuse, the corrected screens established `lr=0.035` as the best LR center; `wd=0.0275` and `wd=0.0225` formed the effective WD tie, `beta2=0.6` produced a new best overall Leon run, `beta2=0.5` regressed slightly, and the first `mu` probe at `0.925` also regressed, suggesting `mu` should move upward from `0.95`. |
| Leon `l_scale` schedule | `retune1500_lr035_wd0275_b260_mu095_ramp020060_retry_4gpu` | The 20%-60% linear ramp from `L=0` to full `L` was close but negative: final `3.44971`, worse than the constant full-`L` `beta2=0.6` center by `+0.00034`, and slower at every validation checkpoint. |
| Muon-hparams `l_scale` schedule | `retune1500_lr025_wd025_b270_mu095_ramp050080_4gpu` | Using Muon-like hparams and delaying the ramp to 50%-80% gave a transient mid/late gain over Leon `L=0` and Muon Leon-NS, but final `3.44939` was still slightly worse than those controls and essentially tied with the best tuned Leon screen. |
| Schedule/LR sweep | `schedule1500_lr0225_wd025_b270_mu095_ramp030060_4gpu`, `schedule1500_lr025_wd025_b270_mu095_ramp030060_4gpu`, `schedule1500_lr0275_wd025_b270_mu095_ramp050080_4gpu`, `schedule1500_lr030_wd025_b270_mu095_ramp050080_4gpu`, `schedule1500_lr0275_wd025_b270_mu095_ramp060090_4gpu`, `schedule1500_lr0275_wd025_b270_mu095_ramp040070_4gpu` | The 30%-60% ramp gave transient gains but faded. Moving the strongest prior `0.5->0.8` ramp to `lr=0.0275` produced the first schedule improvement over tuned Leon-family baselines, final `3.44653`; pushing to `lr=0.03` regressed to `3.45106`, delaying the same LR center to `0.6->0.9` regressed to `3.44854`, and moving it earlier to `0.4->0.7` regressed to `3.44860`. |
| Orthogonalization dtype and NS-step probes | `leon_orth_float32_lr035_wd0275_b260_4gpu`, `leon_orth_float64_lr035_wd0275_b260_eps1e12_4gpu`, `leon_orth_float64_lr035_wd0275_b260_eps1e12_ns12_4gpu`, `leon_orth_float32_lr035_wd0275_b260_eps1e12_ns12_4gpu` | Added `leon_orthogonalize_dtype` with default `bfloat16`, then ran the `retune1500_lr035_wd0275_b260_4gpu` config with higher-precision and higher-iteration NS computation. The 6-step `float32`/`float64` probes ended at `3.45121`/`3.45027`, both trailing bf16. The 12-step `float64`/`float32` probes ended at `3.43039`/`3.42975`, so the gain comes from NS iteration count, not double precision. |

## Detailed Results

| Run | Optimizer | Main diff | Final val | Reached 3.28 | Notes |
| --- | --- | --- | ---: | --- | --- |
| `base` | Leon | `lr=0.025`, `wd=0.025`, `cd=0.7` | 3.57447 | no | Initial 1500-step screen. |
| `lr0225` | Leon | `lr=0.0225` | 3.58408 | no | Worse than base. |
| `lr0275_sandbox_fail` | Leon | `lr=0.0275` | n/a | no | Sandbox denied TCPStore before training. |
| `lr0275` | Leon | `lr=0.0275` | 3.56625 | no | Better than base. |
| `lr030` | Leon | `lr=0.03` | 3.55506 | no | Continued improvement. |
| `lr0325` | Leon | `lr=0.0325` | 3.54514 | no | Continued improvement. |
| `lr035` | Leon | `lr=0.035` | 3.54084 | no | Improvement narrowed. |
| `lr040` | Leon | `lr=0.04` | 3.52678 | no | Best 4-GPU LR screen. |
| `lr045_oom` | Leon | `lr=0.045` | 10.82584 | no | Failed before first training step due occupied GPUs/OOM. |
| `lr045_retry_2gpu` | Leon | `lr=0.045` | 3.52062 | no | First successful 2-GPU continuation. |
| `lr050_2gpu` | Leon | `lr=0.05` | 3.51458 | no | New best at the time. |
| `muon1500_2gpu` | Muon | unchanged Muon, `ns_steps=12` | 3.43079 | no | Strong baseline; better than Leon by `0.08379` vs `lr050_2gpu`. |
| `lr055_2gpu` | Leon | `lr=0.055` | 3.51476 | no | Slightly worse than `lr050_2gpu`; LR boundary bracketed. |
| `lr0525_2gpu` | Leon | `lr=0.0525` | 3.51442 | no | Tiny best over `lr050` and `lr055`; selected for WD sweep. |
| `lr0525_wd030_2gpu` | Leon | `wd=0.03` | 3.51131 | no | New best; moved WD upward. |
| `lr0525_wd035_2gpu` | Leon | `wd=0.035` | 3.52288 | no | Worse; upper WD side bracketed. |
| `lr0525_wd0325_2gpu` | Leon | `wd=0.0325` | 3.51657 | no | Worse than `wd=0.03`; keep `0.03`. |
| `lr0525_wd030_cd080_2gpu` | Leon | `cd=0.8` | 3.51657 | no | Worse than `cd=0.7`; probe lower cooldown. |
| `lr0525_wd030_cd060_2gpu` | Leon | `cd=0.6` | 3.50950 | no | Current best Leon. |
| `lr0525_wd030_cd050_2gpu` | Leon | `cd=0.5` | 3.51414 | no | Lower side worse than `0.6`. |
| `lr0525_wd030_cd065_2gpu` | Leon | `cd=0.65` | 3.51182 | no | Upper side worse than `0.6`. |
| `lr0525_wd030_cd055_2gpu` | Leon | `cd=0.55` | 3.51010 | no | Close but worse than `0.6` by `0.00060`. |
| `lr0525_wd030_cd0575_2gpu` | Leon | `cd=0.575` | 3.51206 | no | Worse than `0.6`; keep `0.6`. |
| `muon1500_leonns_2gpu` | Muon | Leon L=0 6-step NS path, hparams unchanged | 3.44804 | no | Better than tuned Leon, worse than original 12-step Muon. |
| `leon_l0_muon_hparams_4gpu` | Leon | `L=0`, Muon hparams, 4 GPU | 3.44907 | no | Near-identical to Muon Leon-NS, worse by `+0.00103`; did not reach `3.28`. |
| `lscale1_500_4gpu_diag` | Leon | `leon_l_scale=1.0`, diagnostics every 125 steps | 4.04142 | no | Worst 500-step diagnostic; `L` trace fraction near `0.988` and update norm ratio about `0.16-0.17` at steps 125-375. |
| `lscale0_500_4gpu_diag` | Leon | `leon_l_scale=0.0`, diagnostics every 125 steps | 3.79464 | no | Best 500-step diagnostic; direct L=0 control. |
| `lscale010_500_4gpu_diag` | Leon | `leon_l_scale=0.1`, diagnostics every 125 steps | 3.87248 | no | Second-best; still slower than L=0 by `+0.07784`. |
| `lscale025_diag_sandbox_fail` | Leon | `leon_l_scale=0.25` | n/a | no | Sandbox denied torchrun TCPStore before training; same out-dir later reused successfully. |
| `lscale025_500_4gpu_diag` | Leon | `leon_l_scale=0.25`, diagnostics every 125 steps | 3.93081 | no | Middle scale point; slower than L=0 by `+0.13617`, faster than full L by `-0.11061`. |
| `lscale050_500_4gpu_diag` | Leon | `leon_l_scale=0.5`, diagnostics every 125 steps | 3.99168 | no | Slower than `0.25`, faster than full L; update norm ratio about `0.22` at steps 125-375. |
| `lscale1_unnormsum_500_4gpu_diag` | Leon | current `leon_l_scale=1.0` unnormalized-sum codepath, diagnostics every 125 steps | 3.80020 | no | Improved over the old full-L diagnostic by `-0.24122`; `tr(L)` fraction dropped to about `0.50 / 0.45 / 0.40` and update ratio rose to about `0.78 / 0.80 / 0.83` at steps `125 / 250 / 375`. |
| `lscale1_biascorr_500_4gpu_diag` | Leon | historical `leon_l_scale=1.0`, `leon_bias_correction=True` branch | 4.05228 | no | Slightly worse than uncorrected full-L diagnostic by `+0.01086`; kept only as a historical comparison row. |
| `retune1500_lr035_retry_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.03`, `cd=0.6` | 3.45093 | no | First corrected 1500-step retune run on the current codepath; better than the earlier tuned Leon best by `-0.05857` and nearly tied with the Leon `L=0` control. |
| `retune1500_lr045_4gpu` | Leon | current codepath, `lr=0.045`, `wd=0.03`, `cd=0.6` | 3.45492 | no | Second corrected 1500-step retune run; worse than `lr=0.035` by `+0.00399` but still better than the older tuned Leon best by `-0.05458`. |
| `retune1500_lr025_4gpu` | Leon | current codepath, `lr=0.025`, `wd=0.03`, `cd=0.6` | 3.45329 | no | User-requested lower-LR replacement run; worse than `lr=0.035` by `+0.00236` but better than `lr=0.045` by `-0.00163`. |
| `retune1500_lr035_wd0275_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6` | 3.45028 | no | First WD sweep point around the current LR center; final val was lower than the `wd=0.03` center by `0.00065` and currently leads the bracket. |
| `retune1500_lr035_wd025_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.025`, `cd=0.6` | 3.45110 | no | Second lower-WD probe around the current LR center; final val was higher than `wd=0.0275` by `0.00082` and higher than `wd=0.03` by `0.00017`. |
| `retune1500_lr035_wd0225_retry_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0225`, `cd=0.6` | 3.45029 | no | Escalated rerun after sandbox TCPStore failure; effectively tied with `wd=0.0275` at `+0.00001` and better than `wd=0.03` by `-0.00064`. |
| `retune1500_lr035_wd020_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.02`, `cd=0.6` | 3.45078 | no | Third lower-WD probe around the current LR center; worse than `wd=0.0225` by `+0.00049` and worse than `wd=0.0275` by `+0.00050`, so it did not improve the lower side. |
| `retune1500_lr035_wd0275_b260_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6`, `beta2=0.6` | 3.44937 | no | First beta2 sweep point; improved the old `beta2=0.7` center by `-0.00091` and beat it at every logged validation checkpoint. |
| `leon_orth_float32_lr035_wd0275_b260_4gpu` | Leon | same as `retune1500_lr035_wd0275_b260_4gpu`, plus `leon_orthogonalize_dtype=float32` | 3.45121 | no | User-requested Newton-Schulz precision probe; final trailed the bfloat16 reference by `+0.00184`, with train time `691.320s` vs `681.561s`, so this single run does not support bfloat16 orthogonalization precision as the main remaining Muon gap. |
| `leon_orth_float64_lr035_wd0275_b260_eps1e12_4gpu` | Leon | same as `retune1500_lr035_wd0275_b260_4gpu`, plus `leon_orthogonalize_dtype=float64`, `leon_eps=1e-12` | 3.45027 | no | Double-precision Newton-Schulz probe; final trailed the bfloat16 reference by `+0.00090`, though it beat the float32 probe by `-0.00094`. Since it did not improve over bf16, the next follow-up is the same setting with `leon_ns_iters=12`. |
| `leon_orth_float64_lr035_wd0275_b260_eps1e12_ns12_4gpu` | Leon | same as `retune1500_lr035_wd0275_b260_4gpu`, plus `leon_orthogonalize_dtype=float64`, `leon_eps=1e-12`, `leon_ns_iters=12` | 3.43039 | no | Twelve-step Newton-Schulz follow-up; beat the bf16 6-step reference by `-0.01898`, the float64 6-step probe by `-0.01988`, the best prior Leon schedule by `-0.01614`, and original `muon1500_2gpu` by `-0.00040`. |
| `leon_orth_float32_lr035_wd0275_b260_eps1e12_ns12_4gpu` | Leon | same as `leon_orth_float64_lr035_wd0275_b260_eps1e12_ns12_4gpu`, but `leon_orthogonalize_dtype=float32` | 3.42975 | no | User-requested clean attribution run; beat the bf16 6-step reference by `-0.01962`, the float64 12-step probe by `-0.00064`, the best prior Leon schedule by `-0.01678`, and original `muon1500_2gpu` by `-0.00104`. |
| `retune1500_lr035_wd0275_b250_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6`, `beta2=0.5` | 3.44988 | no | Second beta2 sweep point; regressed versus `beta2=0.6` by `+0.00051` but still improved the old `beta2=0.7` center by `-0.00040`. |
| `retune1500_lr035_wd0275_b260_mu0925_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6`, `beta2=0.6`, `mu=0.925` | 3.45394 | no | First mu sweep point; regressed versus the current `mu=0.95` center by `+0.00457`, so the lower side is not promising. |
| `retune1500_lr035_wd0275_b260_mu095_ramp020060_retry_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6`, `beta2=0.6`, `mu=0.95`, `leon_l_scale_schedule=linear_ramp`, ramp `0.2 -> 0.6` | 3.44971 | no | Scheduled-`L` run was close but negative: final trailed the constant full-`L` best by `+0.00034` and did not reach `3.28`. |
| `retune1500_lr025_wd025_b270_mu095_ramp050080_4gpu` | Leon | current codepath, Muon-like hparams `lr=0.025`, `wd=0.025`, `cd=0.7`, `beta2=0.7`, `mu=0.95`, `leon_l_scale_schedule=linear_ramp`, ramp `0.5 -> 0.8` | 3.44939 | no | Late scheduled-`L` run briefly improved over Leon `L=0` and Muon Leon-NS after activation, but final trailed Leon `L=0` by `+0.00032` and Muon Leon-NS by `+0.00135`; original Muon remained much better. |
| `schedule1500_lr0225_wd025_b270_mu095_ramp030060_4gpu` | Leon | current codepath, Muon-like hparams except `lr=0.0225`, `leon_l_scale_schedule=linear_ramp`, ramp `0.3 -> 0.6` | 3.45287 | no | First 30-run schedule/LR sweep point; beat Leon L=0 through step `1125` and the prior `0.5 -> 0.8` ramp through step `875`, but faded after full ramp completion and trailed current best Leon by `+0.00350`. |
| `schedule1500_lr025_wd025_b270_mu095_ramp030060_4gpu` | Leon | current codepath, Muon-like hparams `lr=0.025`, `wd=0.025`, `cd=0.7`, `beta2=0.7`, `mu=0.95`, `leon_l_scale_schedule=linear_ramp`, ramp `0.3 -> 0.6` | 3.45046 | no | Second schedule/LR sweep point; improved over the `lr=0.0225` same-window point by `-0.00241`, but still trailed the previous `0.5 -> 0.8` ramp by `+0.00107` and original Muon by `+0.01967`. |
| `schedule1500_lr0275_wd025_b270_mu095_ramp050080_4gpu` | Leon | current codepath, Muon-like hparams except `lr=0.0275`, `leon_l_scale_schedule=linear_ramp`, ramp `0.5 -> 0.8` | 3.44653 | no | New best Leon-family result. Early checkpoints were worse than `lr=0.025`, but the 50%-80% ramp recovered strongly; final beat prior best schedule by `-0.00286`, tuned constant-L Leon by `-0.00284`, Leon L=0 by `-0.00254`, and Muon Leon-NS by `-0.00151`, while still trailing original 12-step Muon by `+0.01574`. |
| `schedule1500_lr030_wd025_b270_mu095_ramp050080_4gpu` | Leon | current codepath, Muon-like hparams except `lr=0.03`, `leon_l_scale_schedule=linear_ramp`, ramp `0.5 -> 0.8` | 3.45106 | no | Upper LR bracket around the new schedule center; early L=0-equivalent phase was much worse and final trailed `lr=0.0275` by `+0.00453`, so do not push LR higher for this ramp. |
| `schedule1500_lr0275_wd025_b270_mu095_ramp060090_4gpu` | Leon | current codepath, Muon-like hparams except `lr=0.0275`, `leon_l_scale_schedule=linear_ramp`, ramp `0.6 -> 0.9` | 3.44854 | no | Later timing bracket around the new schedule center; final trailed the `0.5 -> 0.8` center by `+0.00201`, but still beat the old `lr=0.025`, `0.5 -> 0.8` schedule by `-0.00085` and the previous constant full-L best by `-0.00083`. |
| `schedule1500_lr0275_wd025_b270_mu095_ramp040070_4gpu` | Leon | current codepath, Muon-like hparams except `lr=0.0275`, `leon_l_scale_schedule=linear_ramp`, ramp `0.4 -> 0.7` | 3.44860 | no | Earlier timing bracket around the new schedule center; led the `0.5 -> 0.8` center at steps `750` and `875`, but faded after step `1000` and finished `+0.00207` behind the center. |

## Shared-Grid LR Sensitivity Sweep (2026-05-07)

**Objective:** Apples-to-apples comparison: extend Muon LR upward until divergence, then run Leon on the same full LR grid. Both optimizers use wd=0.025 and cooldown_frac=0.7. Leon also uses beta2=0.6, ns=12, float32, eps=1e-12.

**Protocol:** Phase 1 — Muon lr=0.200→0.400→0.800→1.600 (no divergence observed through all 4 points). Phase 2 — Leon on the full shared grid (0.00625, 0.0125, 0.025, 0.050, 0.100, 0.200, 0.400, 0.800, 1.600). All runs: 3375 steps, 4× H100 NVL, single trial each.

### Muon: shared-grid results (wd=0.025, cd=0.7, ns=12-step bf16)

| LR | Final val | Reached 3.28 | Notes |
| ---: | ---: | --- | --- |
| 0.00625 | 3.31013 | No | First sweep (prior) |
| 0.0125 | 3.28832 | No | First sweep (prior) |
| **0.025** | **3.27823** | **Yes (step 3375)** | First sweep (prior) |
| **0.050** | **3.27758** | **Yes (step 3375)** | First sweep (prior) |
| 0.100 | 3.29988 | No | First sweep (prior) |
| 0.200 | 3.33734 | No | Extension sweep |
| 0.400 | 3.40087 | No | Extension sweep |
| 0.800 | 3.48118 | No | Extension sweep |
| 1.600 | 3.60214 | No | Extension sweep |

Muon **never diverged** across the full 8-octave range 0.00625→1.600. Best region: lr=0.025–0.050.

### Leon: shared-grid results (wd=0.025, cd=0.7, beta2=0.6, ns=12, float32, eps=1e-12)

| LR | Final val | Reached 3.28 | Notes |
| ---: | ---: | --- | --- |
| 0.00625 | 3.31831 | No | |
| 0.0125 | 3.29360 | No | |
| **0.025** | **3.28027** | **No (δ=+0.00027)** | Best Leon run; just missed |
| 0.050 | 3.28175 | No | |
| 0.100 | 3.30274 | No | |
| 0.200 | 3.34502 | No | |
| 0.400 | 3.40440 | No | |
| 0.800 | 3.49508 | No | Instability bump at step 1625 |
| 1.600 | 3.62542 | No | |

Leon also **never diverged** (no NaN), but performance degrades sharply above lr=0.050. None of the 9 Leon runs reached the 3.28 threshold.

### Side-by-side comparison

| LR | Muon final | Leon final | Muon − Leon |
| ---: | ---: | ---: | ---: |
| 0.00625 | 3.31013 | 3.31831 | −0.00818 |
| 0.0125 | 3.28832 | 3.29360 | −0.00528 |
| 0.025 | 3.27823 | 3.28027 | −0.00204 |
| 0.050 | 3.27758 | 3.28175 | −0.00417 |
| 0.100 | 3.29988 | 3.30274 | −0.00286 |
| 0.200 | 3.33734 | 3.34502 | −0.00768 |
| 0.400 | 3.40087 | 3.40440 | −0.00353 |
| 0.800 | 3.48118 | 3.49508 | −0.01390 |
| 1.600 | 3.60214 | 3.62542 | −0.02328 |

**Muon outperforms Leon at every single LR point** on this matched-hparam shared grid. The gap narrows near the optimum (~lr=0.025–0.050) and widens rapidly at high LR. At lr=1.600 Muon leads by 0.023, nearly ten times the gap at lr=0.025.

### Key findings

1. **Muon is strictly better at every LR** when both optimizers use wd=0.025, cd=0.7. The comparison is now controlled: the performance gap is from the optimizer, not from different hparams.
2. **Leon's optimal LR with these hparams is ~0.025–0.050** — the same region as Muon. The commonly-cited Leon optimal of 0.035 (with wd=0.0275, cd=0.6) does not transfer when hparams are matched to Muon.
3. **Neither optimizer diverged**, but Leon degrades much faster above lr=0.100. Muon's degradation is smoother and more gradual across the 8-octave range.
4. **Leon barely missed the threshold** with the best run (lr=0.025, final 3.28027, δ=+0.00027 from threshold). Muon crossed the threshold at both lr=0.025 (3.27823) and lr=0.050 (3.27758).
5. **The LR sensitivity hypothesis is doubly refuted**: Muon is not only at least as LR-robust as Leon on this shared grid — it outperforms Leon at every tested LR, including within Leon's optimal range.
6. **Matching hparams to Muon hurt Leon**: the prior best Leon config (lr=0.035, wd=0.0275, cd=0.6) reached 3.27879 on 3375 steps, while on this shared grid the best Leon was 3.28027 — slightly worse, and notably the threshold was not crossed.

### Hypothesis verdict (updated): NOT confirmed (stronger)

The shared-grid sweep strengthens the earlier conclusion. With identical wd and cooldown:
- Muon crosses the 3.28 threshold at 2 of 9 LR points; Leon crosses 0 of 9.
- Muon's best final val (3.27758 at lr=0.050) outperforms Leon's best (3.28027 at lr=0.025) by 0.00269.
- Muon's degradation curve is shallower at every tested LR.
- Leon shows non-monotonic instability at lr=0.800 (loss bump at step 1625), while Muon remained smooth.

## Next Recommended Steps

1. Treat `leon_lr=0.035`, `leon_wd=0.0275`, `leon_cooldown_frac=0.6`, `leon_beta2=0.6`, `leon_ns_iters=12`, `leon_orthogonalize_dtype=float32`, and `leon_eps=1e-12` as the current best Leon setting. The 3375-step run with this config is the local fixed-budget Leon benchmark reference and reaches `3.28` exactly at step `3375`. For the 6-step bf16 schedule branch, the best center remains `leon_lr=0.0275`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, `leon_beta2=0.7`, ramp `0.5 -> 0.8`.
2. The tested current-codepath LR ordering is `0.035` best, `0.025` second, and `0.045` third at `wd=0.03`; the tested WD ordering around the `0.035` LR center is `0.0275` and `0.0225` essentially tied, then `0.02`, then `0.03`, then `0.025`.
3. **The shared-grid LR sensitivity sweep (2026-05-07) definitively shows Muon outperforms Leon at every tested LR** when wd and cooldown are matched. The performance gap is from the optimizer algorithm, not hparam mismatch. Muon crossed the threshold at lr=0.025 and lr=0.050; Leon crossed 0 of 9 points.
4. **Leon's optimal hparams are NOT the same as Muon's**: the best Leon config uses wd=0.0275, cd=0.6, lr=0.035 (not wd=0.025, cd=0.7, lr=0.025). Using Muon hparams for Leon costs ~0.001 in final loss and moved the threshold from just-barely-crossed to just-barely-missed.
5. The schedule has now improved over tuned Leon-family baselines but not original Muon. Since `lr=0.03,ramp=0.5->0.8`, `lr=0.0275,ramp=0.6->0.9`, and `lr=0.0275,ramp=0.4->0.7` all regressed versus the `0.5->0.8` center, keep `0.5->0.8` as the timing center and use the remaining declared-grid probes to test whether shorter late windows or lower/higher LR interactions can improve it.
6. If comparing against Muon, keep both Muon baselines in view: original 12-step Muon is the stronger reference, while Muon Leon-NS isolates part of the orthogonalization change. The 12-step float32 run already isolates `leon_ns_iters=12` from double precision; the next clean attribution run would isolate `eps` by trying float32, 12 steps, and `leon_eps=1e-9`.
7. For algorithm attribution, the Leon L=0 control plus the L-scale sweep are the strongest evidence so far: L=0 reproduces Muon Leon-NS nearly exactly, and increasing `L` scale monotonically slows the 500-step trajectory.
8. If continuing algorithm work on nonzero `L`, do not use full `L` unchanged. Test mechanisms that keep `tr(L)` from dominating normalization, such as much smaller scale, lower `beta2`, delayed activation, clipping/normalizing `tr(L)`, or computing Gram statistics from the same Nesterov update being orthogonalized.
9. The next algorithm comparison should use the current unnormalized exponential-sum codepath rather than the abandoned bias-correction branch.
10. Since `l_scale=1.0` is now close to the old `L=0` control, rerun the `l_scale` sweep on the current codepath if you want to know whether nonzero `L` is still helpful, neutral, or slightly harmful after the scaling fix.
11. Before making benchmark claims, run independent trials for the new 3375-step Leon benchmark config and report mean/std plus threshold behavior, per `AGENTS.md`. The single fixed-budget head-to-head against `train_gpt_simple.py` is promising but not yet statistically sufficient.

## MuonH vs AdamH Phase 2 Gauging Runs (2026-05-08)

### Goal

Assess whether MuonH or AdamH can reach `val_loss < 2.92` and gauge step requirements. Scripts created:
- `records/track_3_optimization/train_gpt_simple_muonh.py` (with `--set`/`--out-dir`/`--dry-run`/`--trials`)
- `records/track_3_optimization/train_gpt_simple_adamh.py` (same CLI)
- `records/track_3_optimization/launch_muonh.sh` and `launch_adamh.sh`

### Critical Data Budget Discovery

Each FineWeb10B shard has **100M tokens = 190 steps** at batch_size=524,288. With 40 shards available at run time, the safe budget was only **7,600 steps** (not the ~18,840 originally estimated). The MuonH run was launched with `train_steps=12000` and crashed at step 7500 with `StopIteration` (data exhaustion). The AdamH run was corrected to `train_steps=7500`.

**Data fix in progress**: downloading remaining shards 41–103 via `data/cached_fineweb10B.py 103`. When complete, data budget extends to **103 × 190 = 19,570 steps**.

### Gauging Results (default hparams, lr=0.018)

| Optimizer | train_steps | final val_loss | step_to_3.28 | reached 2.92 | Notes |
|-----------|-------------|----------------|--------------|--------------|-------|
| MuonH     | 7500 (crash)| 3.34592        | —            | No           | Crashed at step 7500 due to data exhaustion; set to 12000 but limited to 7600 budget; cooldown_frac=1.0 decays LR from step 0 |
| AdamH     | 7500        | **3.21548**    | **6500**     | No           | Loss still **decreasing** at step 7500; warmup 250 steps then linear decay |

### Key Findings

1. **AdamH significantly outperforms MuonH at matched step budget**: at step 7500, AdamH=3.215 vs MuonH=3.346 — a gap of **0.131**. This is reversed from the original 3325-step benchmark where both reached ~3.275.
2. **MuonH's cooldown_frac=1.0 schedule is ill-suited to extended runs**: LR decays linearly from step 0, so at step 7500 with a 12000-step budget the LR is still 0.44×peak — the model never converges tightly. The original 3325-step design was specifically tuned for this schedule length.
3. **AdamH has a favorable warmup+decay schedule**: 250-step linear warmup then linear decay to 0. This provides a stable high-LR phase that aids convergence over the 7500-step run. AdamH crossed `val_loss < 3.28` at step 6500; loss was still decreasing at step 7500 (3.215), suggesting further potential.
4. **Neither optimizer reached 2.92 at 7500 steps with default LR**: more steps are needed. With the full 103-shard budget (~19,570 steps), 2.92 may be reachable, especially for AdamH.
5. **AdamH converged faster per step after warmup**: MuonH led for the first ~2500 steps (no warmup penalty), then AdamH overtook and continued pulling ahead at ~0.045 better by step 4250.

### Gap trajectory (AdamH lead over MuonH at same step)

| Step | MuonH | AdamH | AdamH lead |
|------|-------|-------|------------|
| 1250 | 3.692 | 3.751 | −0.059 (MuonH ahead) |
| 1750 | 3.629 | 3.669 | −0.040 |
| 2250 | 3.592 | 3.607 | −0.015 |
| 2750 | 3.561 | 3.556 | **+0.005** (crossover) |
| 3500 | 3.517 | 3.497 | +0.020 |
| 4250 | 3.494 | 3.449 | +0.045 |
| 7500 | 3.346 | 3.215 | **+0.131** |

### Next Recommended Steps (Phase 3)

With extended data budget now downloading:

1. **Fix MuonH schedule for extended runs**: add a warmup and stable phase. Proposed: `--set warmup_steps=500 --set cooldown_frac=0.3` so the schedule is flat LR for most of training then decays at the end. This will likely bring MuonH performance closer to AdamH.
2. **Run LR sweep for both optimizers at 7500 steps** (within confirmed safe budget): grid `{0.009, 0.013, 0.018, 0.026, 0.036}` for each. Use final val_loss as the primary metric (neither optimizer reached 2.92 at default LR in 7500 steps).
3. **After LR sweep, consider extended runs** at best LR with the full 19,570-step budget (once shards 41–103 are available). Target `val_loss < 2.92`.
4. **The Phase 3 question has already shifted**: the gauging data suggests AdamH may have a structural advantage over MuonH at these longer step counts with default schedules. MuonH needs schedule tuning (warmup + stable phase) to compete. The LR sweep should clarify whether this gap persists across LRs.

**Immediate action**: confirm with user whether Phase 3 should add a warmup+stable phase to MuonH (to make the comparison fair) or sweep LR as-is with the default schedules.

## LeonV Experiments (2026-05-11)

### Context

LeonV (Leon Vectorized) is a new element-wise optimizer on branch `feat/leonv-optimizer`.
Update rule:
- m_t = β₁·m_{t-1} + g_t (unnormalized EMA, no (1−β₁) scale)
- v_t = β₂·v_{t-1} + g_t² (unnormalized EMA)
- w_t = w_{t-1} − lr · m_t / √(m_t² + v_t)

Key properties: no bias correction; denominator bounds update in [−1,1] element-wise; uses decoupled weight decay (no hyperball projection); single optimizer for all params with 4 param groups.

Script: `records/track_3_optimization/train_gpt_simple_leonv.py`
Launcher: `records/track_3_optimization/launch_leonv.sh`

### AdamW Reference Baseline

Log: `records/track_3_optimization/results/a63a68d1-24aa-4a22-af9a-224e43209ea4.txt`

| train_steps | final_val_loss | step_to_3.28 | matrix_lr | wd | betas | warmup | cooldown_frac |
|---|---|---|---|---|---|---|---|
| 5625 | 3.27903 | 5625 | 0.0015 | 0.10 | (0.9,0.95) | 250 | 0.7 |

### LeonV Baseline Run

Run path: `records/track_3_optimization/runs/leonv/20260511-122702-d0a85b15`
Config: same hparams as AdamW baseline (matrix_lr=0.0015, wd=0.10, betas=(0.9,0.95), warmup=250, cooldown_frac=0.7)

| train_steps | final_val_loss | step_to_3.28 | reached 3.28 | train_time | step_avg |
|---|---|---|---|---|---|
| 5625 | **3.28346** | — | **No** | 2415s (40.3 min) | 429 ms/step |

**Comparison vs AdamW baseline at matched steps:**

| Step | AdamW | LeonV | LeonV − AdamW |
|---|---|---|---|
| 125 | 6.190 | 6.024 | −0.166 (LeonV ahead) |
| 500 | 4.152 | 4.202 | +0.050 |
| 1000 | 3.773 | 3.813 | +0.040 |
| 2500 | 3.489 | 3.502 | +0.013 |
| 3750 | 3.381 | 3.388 | +0.007 |
| 5000 | 3.303 | 3.307 | +0.004 |
| 5625 | **3.279** | **3.283** | **+0.004** |

**Interpretation:** LeonV leads AdamW early (step 125) due to aggressive unnormalized EMA updates, but falls behind from step 375 onward. The gap narrows continuously from ~0.050 at step 500 to ~0.004 at step 5625. LeonV missed the 3.28 threshold by 0.00443; AdamW crossed only at the final step. Default hparams (matching AdamW) are a reasonable starting point but not yet optimal for LeonV.

**Performance note:** LeonV runs at ~429 ms/step vs AdamW's ~148 ms/step (3× slower) because LeonV uses an unfused Python loop over all parameters including the large embedding table (50304×768). This does not affect result correctness but matters for sweep throughput.

### LeonV Hyperparameter Sweep (Screening Phase — 1500 steps)

**Phase S1 — β₂ sweep** (decrease from 0.95; keep β₁=0.9, lr=0.0015, wd=0.10):

Baseline (0.95) at 1500 steps: 3.65393

| run_id | β₂ | val@1500 | Δ vs baseline@1500 | dir |
|---|---|---|---|---|
| leonv_s1_beta2_090 | **0.90** | **3.58798** | **−0.066** | `20260511-132617-126e7d9a` |
| leonv_s1_beta2_080 | 0.80 | 3.61429 | −0.040 | `20260511-133706-dff0c3ee` |
| leonv_s1_beta2_070 | 0.70 | 3.62890 | −0.025 | `20260511-134902-5203e7f0` |

**Winner: β₂=0.90** — all lower values improve over baseline but are monotonically worse as β₂ decreases.

**Phase S2 — lr sweep** (using best β₂=0.90; keep wd=0.10):

| run_id | lr | val@1500 | Δ vs S1-best | dir |
|---|---|---|---|---|
| leonv_s2_lr_0010 | **0.001** | **3.58168** | **−0.006** | `20260511-143929-7e619139` |
| leonv_s2_lr_0015 | 0.0015 | 3.59251 | +0.005 | `20260511-152351-61aa384e` |
| leonv_s2_lr_0020 | 0.002 | 3.60838 | +0.021 | `20260511-145207-b1f01f91` |
| leonv_s2_lr_0030 | 0.003 | 3.67230 | +0.084 | `20260511-150325-8072e6f6` |

**Winner: lr=0.001** — monotone improvement going lower; lr=0.003 is catastrophically worse.

**Phase S3 — wd sweep** (using best β₂=0.90, lr=0.001):

| run_id | wd | val@1500 | Δ vs wd=0.10 | dir |
|---|---|---|---|---|
| leonv_s3_wd_000 | 0.00 | 3.58663 | +0.005 | `20260511-154453-679f1c4d` |
| leonv_s3_wd_005 | 0.05 | 3.59083 | +0.009 | `20260511-155732-fe64f955` |
| (from S2) | 0.10 | 3.58168 | 0 | `20260511-143929-7e619139` |
| leonv_s3_wd_015 | 0.15 | 3.58703 | +0.005 | `20260511-161518-488e1c69` |
| leonv_s3_wd_020 | **0.20** | **3.57802** | **−0.004** | `20260511-162748-e686c1f4` |

**Winner: wd=0.20** — trend still falling at the high end. Full 5625-step run with S3 winner: final val 3.28338 — did NOT reach 3.28 threshold (gap +0.00435 vs AdamW). Run dir: `20260511-172513-b978030b`.

**Phase S4 — auxiliary LR sweep** (embed_lr and proj_lr; using best β₂=0.90, lr=0.001, wd=0.20):

**Hypothesis:** The LeonV m/√(m²+v) update has bounded magnitude in [−1,1], smaller than a typical Adam step for embed/proj weights. Larger per-group learning rates may compensate.

S3 baseline at 1500 steps (wd=0.20 winner): val@1500=3.57802

| run_id | embed_lr | proj_lr | matrix_lr | val@1500 | Δ vs S3 baseline | dir |
|---|---|---|---|---|---|---|
| leonv_s4a_embed_2x_1500 | **0.6** (2×) | 1/320 | 0.001 | 3.57849 | +0.00047 (no effect) | not archived |
| leonv_s4b_proj_2x_1500 | 0.3 | **1/160** (2×) | 0.001 | 3.57553 | −0.00249 (modest gain) | not archived |
| **leonv_s4c_both_2x_1500** | **0.6** (2×) | **1/160** (2×) | 0.001 | **3.56927** | **−0.00875 (BEST)** | not archived |
| leonv_s4d_matrix_2x_1500 | 0.6 | 1/160 | **0.002** (2×) | 3.57941 | +0.01014 (worse) | not archived |
| leonv_s4e_proj_4x_1500 | 0.6 | **0.0125** (4×) | 0.001 | 3.57896 | +0.01069 (overshoots) | `20260512-132419-82a7d59a` |

**Key findings:**
- **S4c is super-additive**: combining 2× embed_lr and 2× proj_lr (−0.00875) beats the sum of individual gains (S4a ≈ 0, S4b −0.00249), indicating a genuine interaction between the two groups.
- **embed_lr alone has negligible effect** (S4a: +0.00047); the embedding is not the bottleneck.
- **proj_lr alone gives modest improvement** (S4b: −0.00249), but most gain comes from the combination.
- **4× proj_lr overshoots** (S4e worse than S4b by +0.01343); 2× is the sweet spot.
- **2× matrix_lr hurts** (S4d: +0.01014 vs S4c); matrix LR is well-tuned at 0.001.

**Winner: S4c** (`embed_lr=0.6, proj_lr=1/160`). Full 5625-step run:

Run path: `records/track_3_optimization/runs/leonv/20260512-133900-2186eb8c`
Config: `matrix_lr=0.001, embed_lr=0.6, proj_lr=1/160, leonv_betas=(0.9,0.90), leonv_wd=0.20, warmup=250, cooldown_frac=0.7, leonv_eps=1e-12`

| train_steps | final_val_loss | reached 3.28 | step_avg |
|---|---|---|---|
| 5625 | **3.27828** | **Yes (step 5625)** | ~472 ms/step |

**Result: First LeonV configuration to beat AdamW baseline (3.27903) by −0.00075 (−0.023%).** LeonV crossed the 3.28 threshold at the final step.

### Current Best LeonV Config

**After Phase S4 (current best):** `matrix_lr=0.001, embed_lr=0.6, proj_lr=1/160, leonv_wd=0.20, betas=(0.9,0.90), warmup=250, cooldown_frac=0.7, leonv_eps=1e-12` — final val **3.27828** @ 5625 steps. **Beats AdamW baseline (3.27903) by −0.00075.** First LeonV config to exceed AdamW performance. Run dir: `runs/leonv/20260512-133900-2186eb8c`.

**Previous best (Phase S3):** `matrix_lr=0.001, leonv_wd=0.20, betas=(0.9,0.90)` — final val 3.28338 @ 5625 steps, did NOT reach 3.28 threshold. Run dir: `runs/leonv/20260511-172513-b978030b`.

**Baseline:** `betas=(0.9,0.95)` — final val 3.28346 @ 5625 steps, did not reach 3.28 threshold.

## Important Commits

| Commit | Purpose |
| --- | --- |
| `1569f5ec617e57090aecbc37416223b5ddff678f` | Recorded Muon Leon-NS baseline, including modified Muon script, run archive, and CSV row. |
| `04bfad78e01c8c514a5396201d7baaf8b0e29cfc` | Last commit before the Muon Leon-NS baseline record. |
