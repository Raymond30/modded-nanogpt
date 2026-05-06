# Track 3 Optimization Experiment Handoff

Last updated: 2026-05-06 00:49 America/Chicago

This stable handoff file summarizes the experiments from the recent Codex tuning conversation. The authoritative run ledger remains `records/track_3_optimization/tuning_log.csv`; use this file as a fast orientation layer before selecting the next run, and update it in place after meaningful new results.

## Operating Notes

- Read `records/track_3_optimization/AGENTS.md`, `records/track_3_optimization/tuning_log.csv`, and this handoff before launching new runs.
- Append one row to `tuning_log.csv` immediately after each completed, failed, or crashed run.
- Keep experiment artifacts committed periodically. Relevant artifacts include `tuning_log.csv`, run directories, metadata/configs, and logs needed for reproducibility.
- Do not reuse the same `--out-dir` for a dry-run and the real launch. `train_gpt_simple_leon.py` creates the run directory with `exist_ok=False` in both modes, so a dry-run can leave a stale directory that causes the subsequent real `torchrun` launch to hang after distributed init without ever creating `train.log`.
- User granted permission for 30 GPU launches without further confirmation so long as GPUs are available. The latest completed diagnostic sweep consumed runs 14-18 of that allowance in this conversation context. One additional sandbox retry failed before using GPUs.
- Current 30-run schedule/LR probe is predeclared around Muon-like hparams: `leon_wd=0.025`, `leon_mu=0.95`, `leon_beta2=0.7`, `leon_cooldown_frac=0.7`, `adam_cooldown_frac=0.7`, `train_steps=1500`; sweep `leon_lr in {0.0225, 0.025, 0.0275, 0.03, 0.035}` and ramp windows `{0.3->0.6, 0.4->0.7, 0.5->0.8, 0.6->0.9, 0.7->0.9, 0.8->0.95}`, skipping the already-completed duplicate `lr=0.025,ramp=0.5->0.8` and replacing it with `lr=0.025,ramp=0.6->0.95`.
- Main screening runs used `train_steps=1500`; the L-scale diagnostic sweep used `train_steps=500`. No logged run reached `val_loss < 3.28`.
- Many later 1500-step screens used 2 GPUs because only GPUs 1 and 2 were available, while the latest diagnostics used 4 GPUs. Do not compare wall time directly between 4-GPU and 2-GPU runs.
- A historical bias-correction branch was tried briefly, but the active code has now switched away from it and instead uses unnormalized exponential-sum scaling for the Leon moments.

## Current Bests

| Category | Run | Key hparams | Final val loss | Run path |
| --- | --- | --- | ---: | --- |
| Best Leon screen | `schedule1500_lr0275_wd025_b270_mu095_ramp050080_4gpu` | `leon_lr=0.0275`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, `leon_beta2=0.7`, `leon_l_scale_schedule=linear_ramp`, ramp `0.5->0.8` | 3.44653 | `records/track_3_optimization/runs/leon/20260506-leon-schedule-sweep-8dfc948/lr0275_ramp050_080_4gpu` |
| Best constant full-L Leon screen | `retune1500_lr035_wd0275_b260_4gpu` | `leon_lr=0.035`, `leon_wd=0.0275`, `leon_cooldown_frac=0.6`, `leon_beta2=0.6`, current codepath | 3.44937 | `records/track_3_optimization/runs/leon/20260505-leon-retune1500-cda40ac/lr035_wd0275_b260_4gpu` |
| Best original Muon baseline | `muon1500_2gpu` | `lr=0.025`, `wd=0.025`, `momentum=0.95`, `ns_steps=12` | 3.43079 | `records/track_3_optimization/runs/muon/20260504-muon-screen1500` |
| Muon with Leon-like NS | `muon1500_leonns_2gpu` | `lr=0.025`, `wd=0.025`, `momentum=0.95`, `ns_steps=6`, `eps=1e-9` | 3.44804 | `records/track_3_optimization/runs/muon/20260505-muon-leonns-screen1500` |
| Leon L=0 with Muon hparams | `leon_l0_muon_hparams_4gpu` | `leon_lr=0.025`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, `L=0` | 3.44907 | `records/track_3_optimization/runs/leon/20260505-leon-l0-screen1500/muon_hparams_4gpu` |
| Best 500-step diagnostic | `lscale0_500_4gpu_diag` | same Muon-like hparams, `leon_l_scale=0.0` | 3.79464 | `records/track_3_optimization/runs/leon/20260505-leon-lscale-diagnostics/lscale0_500_4gpu` |
| Best full-L diagnostic on current code | `lscale1_unnormsum_500_4gpu_diag` | same Muon-like hparams, `leon_l_scale=1.0`, unnormalized-sum codepath | 3.80020 | `records/track_3_optimization/runs/leon/20260505-leon-unnormsum-lscale-diagnostics/lscale1_500_4gpu` |
| Historical bias-corrected full-L diagnostic | `lscale1_biascorr_500_4gpu_diag` | same Muon-like hparams, old `leon_bias_correction=True` branch | 4.05228 | `records/track_3_optimization/runs/leon/20260505-leon-biascorr-lscale-diagnostics/lscale1_500_4gpu` |

Interpretation:

- On the current unnormalized-sum codepath, the first corrected 1500-step retune point `leon_lr=0.035`, `leon_wd=0.03`, `leon_cooldown_frac=0.6` reached `3.45093`, improving the prior best tuned Leon screen by `-0.05857`.
- That new Leon best is now only `0.02014` behind the original 12-step Muon baseline, `0.00289` behind the Muon Leon-NS baseline, and `0.00186` behind the Leon `L=0` control.
- The next LR point `leon_lr=0.045` finished at `3.45492`, so `0.035` remains ahead by `0.00399` while the `0.0525` and `0.06` runs are still pending.
- The user-requested lower probe `leon_lr=0.025` finished at `3.45329`, which is `0.00236` behind `0.035` but `0.00163` better than `0.045`, so the current LR ordering is `0.035` best, then `0.025`, then `0.045`.
- The first WD sweep point around the `lr=0.035` center, `leon_wd=0.0275`, finished at `3.45028`, improving `wd=0.03` by `0.00065` and becoming the current best tested Leon configuration while the rest of the WD bracket is still pending.
- The next lower WD probe `leon_wd=0.025` finished at `3.45110`, which is `0.00082` behind `wd=0.0275` and `0.00017` behind `wd=0.03`, so `wd=0.0275` still leads and the downward side has not improved yet.
- The escalated retry at `leon_wd=0.0225` finished at `3.45029` after an initial sandbox TCPStore failure. That result is only `0.00001` behind `wd=0.0275` and `0.00064` better than `wd=0.03`, so the lower side is now effectively tied with the current best on a single run.
- The next lower probe `leon_wd=0.02` finished at `3.45078`. That is `0.00049` behind `wd=0.0225` and `0.00050` behind `wd=0.0275`, so the best lower-side point remains `wd=0.0225` and the practical near-tie is now between `0.0225` and `0.0275`.
- The first `leon_beta2` sweep point from the nominal best WD center, `leon_beta2=0.6` at `lr=0.035`, `wd=0.0275`, `cd=0.6`, finished at `3.44937`. It beat the previous best Leon screen by `0.00091` and outperformed the old `beta2=0.7` center at every validation checkpoint from step `125` through `1500`.
- The next lower beta2 probe, `leon_beta2=0.5` at the same center, finished at `3.44988`. That is `0.00051` worse than `beta2=0.6`, though still `0.00040` better than the old `beta2=0.7` center, so the beta2 bracket now points to a local optimum near `0.6`.
- The first `leon_mu` probe from the current beta2 center, `leon_mu=0.925`, finished at `3.45394`. That is `0.00457` worse than the current best `mu=0.95` center, so the lower side is not promising and the sweep should move upward next.
- The first scheduled-`L` run, `leon_l_scale_schedule=linear_ramp` with a `0.2 -> 0.6` training-fraction ramp from `0` to `1`, finished at `3.44971`. That is only `0.00034` behind the constant full-`L` `beta2=0.6` best, but it trailed at every checkpoint and did not improve the current center.
- The Muon-hparams late scheduled-`L` run, `leon_lr=0.025`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, and a `0.5 -> 0.8` ramp from `L=0` to full `L`, finished at `3.44939`. It beat the Leon `L=0` and Muon Leon-NS controls at several mid/late checkpoints after ramp activation, but final validation loss still trailed Leon `L=0` by `+0.00032`, Muon Leon-NS by `+0.00135`, original Muon by `+0.01860`, and the current best Leon screen by `+0.00002`.
- The first broad schedule/LR sweep point, `leon_lr=0.0225`, `leon_wd=0.025`, `leon_beta2=0.7`, and a `0.3 -> 0.6` ramp, finished at `3.45287`. It improved over Leon `L=0` through step `1125` and over the prior `0.5 -> 0.8` ramp through step `875`, but faded after the ramp completed and finished `+0.00350` behind the current best Leon and `+0.02208` behind original Muon.
- The second broad schedule/LR sweep point, `leon_lr=0.025` with the same `0.3 -> 0.6` ramp, finished at `3.45046`. Raising LR to the Muon value improved final loss versus the `0.0225` point by `-0.00241`, but it still trailed the prior `0.5 -> 0.8` ramp by `+0.00107`, Leon `L=0` by `+0.00139`, current best Leon by `+0.00109`, and original Muon by `+0.01967`.
- The third broad schedule/LR sweep point, `leon_lr=0.0275` with the previously best `0.5 -> 0.8` ramp, finished at `3.44653`. It is the new best Leon-family result, beating the prior best schedule by `-0.00286`, the previous best tuned constant-L Leon by `-0.00284`, Leon `L=0` by `-0.00254`, and Muon Leon-NS by `-0.00151`, while still trailing original 12-step Muon by `+0.01574`.
- The Muon Leon-NS rerun is better than tuned Leon by `0.06146`, but worse than original 12-step Muon by `0.01725`.
- Leon with `L=0` and Muon hparams landed at `3.44907`, near-identical to Muon Leon-NS (`+0.00103` final val), indicating the earlier Leon gap was mainly from the nonzero second-momentum `L` contribution and tuned hparams rather than the optimizer wrapper.
- The 500-step `leon_l_scale` diagnostic sweep was monotone: `0.0` final `3.79464`, `0.1` `3.87248`, `0.25` `3.93081`, `0.5` `3.99168`, `1.0` `4.04142`. Diagnostics show full `L` makes `tr(L)` about 98.8% of the normalization denominator at steps 125-375 and shrinks the update to about 16-17% of the hypothetical L=0 update.
- The old bias-corrected full-`L` follow-up was slightly worse than uncorrected full `L` (`4.05228` vs `4.04142`). The active code no longer uses that branch; it now scales Leon as an unnormalized exponential sum by dividing the Nesterov-updated `g` by `1 - mu` and the second-momentum matrix by `1 - beta2`.
- The new full-`L` diagnostic on the current unnormalized-sum codepath improved sharply to `3.80020`. Its `l_trace_fraction` dropped to about `0.50 / 0.45 / 0.40` at steps `125 / 250 / 375`, and `update_norm / update_l0_norm` rose to about `0.78 / 0.80 / 0.83`, leaving full `L` only `+0.00556` behind the `L=0` control at step 500.

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
| Schedule/LR sweep | `schedule1500_lr0225_wd025_b270_mu095_ramp030060_4gpu`, `schedule1500_lr025_wd025_b270_mu095_ramp030060_4gpu`, `schedule1500_lr0275_wd025_b270_mu095_ramp050080_4gpu` | The 30%-60% ramp gave transient gains but faded. Moving the strongest prior `0.5->0.8` ramp to `lr=0.0275` produced the first schedule improvement over tuned Leon-family baselines, final `3.44653`, though original 12-step Muon remains better. |

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
| `retune1500_lr035_wd0275_b250_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6`, `beta2=0.5` | 3.44988 | no | Second beta2 sweep point; regressed versus `beta2=0.6` by `+0.00051` but still improved the old `beta2=0.7` center by `-0.00040`. |
| `retune1500_lr035_wd0275_b260_mu0925_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6`, `beta2=0.6`, `mu=0.925` | 3.45394 | no | First mu sweep point; regressed versus the current `mu=0.95` center by `+0.00457`, so the lower side is not promising. |
| `retune1500_lr035_wd0275_b260_mu095_ramp020060_retry_4gpu` | Leon | current codepath, `lr=0.035`, `wd=0.0275`, `cd=0.6`, `beta2=0.6`, `mu=0.95`, `leon_l_scale_schedule=linear_ramp`, ramp `0.2 -> 0.6` | 3.44971 | no | Scheduled-`L` run was close but negative: final trailed the constant full-`L` best by `+0.00034` and did not reach `3.28`. |
| `retune1500_lr025_wd025_b270_mu095_ramp050080_4gpu` | Leon | current codepath, Muon-like hparams `lr=0.025`, `wd=0.025`, `cd=0.7`, `beta2=0.7`, `mu=0.95`, `leon_l_scale_schedule=linear_ramp`, ramp `0.5 -> 0.8` | 3.44939 | no | Late scheduled-`L` run briefly improved over Leon `L=0` and Muon Leon-NS after activation, but final trailed Leon `L=0` by `+0.00032` and Muon Leon-NS by `+0.00135`; original Muon remained much better. |
| `schedule1500_lr0225_wd025_b270_mu095_ramp030060_4gpu` | Leon | current codepath, Muon-like hparams except `lr=0.0225`, `leon_l_scale_schedule=linear_ramp`, ramp `0.3 -> 0.6` | 3.45287 | no | First 30-run schedule/LR sweep point; beat Leon L=0 through step `1125` and the prior `0.5 -> 0.8` ramp through step `875`, but faded after full ramp completion and trailed current best Leon by `+0.00350`. |
| `schedule1500_lr025_wd025_b270_mu095_ramp030060_4gpu` | Leon | current codepath, Muon-like hparams `lr=0.025`, `wd=0.025`, `cd=0.7`, `beta2=0.7`, `mu=0.95`, `leon_l_scale_schedule=linear_ramp`, ramp `0.3 -> 0.6` | 3.45046 | no | Second schedule/LR sweep point; improved over the `lr=0.0225` same-window point by `-0.00241`, but still trailed the previous `0.5 -> 0.8` ramp by `+0.00107` and original Muon by `+0.01967`. |
| `schedule1500_lr0275_wd025_b270_mu095_ramp050080_4gpu` | Leon | current codepath, Muon-like hparams except `lr=0.0275`, `leon_l_scale_schedule=linear_ramp`, ramp `0.5 -> 0.8` | 3.44653 | no | New best Leon-family result. Early checkpoints were worse than `lr=0.025`, but the 50%-80% ramp recovered strongly; final beat prior best schedule by `-0.00286`, tuned constant-L Leon by `-0.00284`, Leon L=0 by `-0.00254`, and Muon Leon-NS by `-0.00151`, while still trailing original 12-step Muon by `+0.01574`. |

## Next Recommended Steps

1. Treat `leon_lr=0.0275`, `leon_wd=0.025`, `leon_cooldown_frac=0.7`, `leon_beta2=0.7`, and `leon_l_scale_schedule=linear_ramp` with ramp `0.5 -> 0.8` as the current best-tested Leon schedule center. The best constant full-L Leon center remains `leon_lr=0.035`, `leon_wd=0.0275`, `leon_cooldown_frac=0.6`, `leon_beta2=0.6`.
2. The tested current-codepath LR ordering is `0.035` best, `0.025` second, and `0.045` third at `wd=0.03`; the tested WD ordering around the `0.035` LR center is `0.0275` and `0.0225` essentially tied, then `0.02`, then `0.03`, then `0.025`.
3. The schedule has now improved over tuned Leon-family baselines but not original Muon. Continue bracketing the `lr=0.0275,ramp=0.5->0.8` center: next high-value probes are `lr=0.03,ramp=0.5->0.8`, `lr=0.0275,ramp=0.6->0.9`, and `lr=0.025,ramp=0.6->0.9` or `0.6->0.95`.
4. If comparing against Muon, keep both Muon baselines in view: original 12-step Muon is the stronger reference, while Muon Leon-NS isolates part of the orthogonalization change.
5. For algorithm attribution, the Leon L=0 control plus the L-scale sweep are the strongest evidence so far: L=0 reproduces Muon Leon-NS nearly exactly, and increasing `L` scale monotonically slows the 500-step trajectory.
6. If continuing algorithm work on nonzero `L`, do not use full `L` unchanged. Test mechanisms that keep `tr(L)` from dominating normalization, such as much smaller scale, lower `beta2`, delayed activation, clipping/normalizing `tr(L)`, or computing Gram statistics from the same Nesterov update being orthogonalized.
7. The next algorithm comparison should use the current unnormalized exponential-sum codepath rather than the abandoned bias-correction branch.
8. Since `l_scale=1.0` is now close to the old `L=0` control, rerun the `l_scale` sweep on the current codepath if you want to know whether nonzero `L` is still helpful, neutral, or slightly harmful after the scaling fix.
9. Before making benchmark claims, run independent trials for promising configs and report mean/std plus threshold behavior, per `AGENTS.md`.

## Important Commits

| Commit | Purpose |
| --- | --- |
| `1569f5ec617e57090aecbc37416223b5ddff678f` | Recorded Muon Leon-NS baseline, including modified Muon script, run archive, and CSV row. |
| `04bfad78e01c8c514a5396201d7baaf8b0e29cfc` | Last commit before the Muon Leon-NS baseline record. |
