# Track 3 Optimization Experiment Handoff

Last updated: 2026-05-05 10:15 America/Chicago

This stable handoff file summarizes the experiments from the recent Codex tuning conversation. The authoritative run ledger remains `records/track_3_optimization/tuning_log.csv`; use this file as a fast orientation layer before selecting the next run, and update it in place after meaningful new results.

## Operating Notes

- Read `records/track_3_optimization/AGENTS.md`, `records/track_3_optimization/tuning_log.csv`, and this handoff before launching new runs.
- Append one row to `tuning_log.csv` immediately after each completed, failed, or crashed run.
- Keep experiment artifacts committed periodically. Relevant artifacts include `tuning_log.csv`, run directories, metadata/configs, and logs needed for reproducibility.
- User granted permission for 30 GPU launches without further confirmation so long as GPUs are available. The latest Muon Leon-NS rerun was tracked as run 12 of that allowance in this conversation context.
- All logged screening runs here used `train_steps=1500` and did not reach `val_loss < 3.28`.
- Most later runs used 2 GPUs because only GPUs 1 and 2 were available. Do not compare wall time directly between 4-GPU and 2-GPU runs.

## Current Bests

| Category | Run | Key hparams | Final val loss | Run path |
| --- | --- | --- | ---: | --- |
| Best Leon screen | `lr0525_wd030_cd060_2gpu` | `leon_lr=0.0525`, `leon_wd=0.03`, `leon_cooldown_frac=0.6` | 3.50950 | `records/track_3_optimization/runs/leon/20260504-leon-screen1500/lr0525_wd030_cd060_2gpu` |
| Best original Muon baseline | `muon1500_2gpu` | `lr=0.025`, `wd=0.025`, `momentum=0.95`, `ns_steps=12` | 3.43079 | `records/track_3_optimization/runs/muon/20260504-muon-screen1500` |
| Muon with Leon-like NS | `muon1500_leonns_2gpu` | `lr=0.025`, `wd=0.025`, `momentum=0.95`, `ns_steps=6`, `eps=1e-9` | 3.44804 | `records/track_3_optimization/runs/muon/20260505-muon-leonns-screen1500` |

Interpretation:

- Leon tuning improved from `3.57447` at the starting config to `3.50950` at the current best.
- The original Muon baseline remains stronger than the tuned Leon screens by `0.07871` final val loss.
- The Muon Leon-NS rerun is better than tuned Leon by `0.06146`, but worse than original 12-step Muon by `0.01725`.

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

## Next Recommended Steps

1. Treat `leon_lr=0.0525`, `leon_wd=0.03`, `leon_cooldown_frac=0.6` as the current Leon center for 1500-step screens.
2. If continuing pure hyperparameter tuning, consider small reruns around the current best to estimate noise before chasing tiny deltas, because recent cooldown differences are in the `0.0006` to `0.0026` range.
3. Reasonable next Leon probes, if still using single-run 1500-step screens, are `leon_lr=0.0525, leon_wd=0.03, leon_cooldown_frac=0.6` with adjacent parameters such as `adam_cooldown_frac`, `leon_beta2`, `leon_mu`, `leon_eps`, or `leon_ns_iters`.
4. If comparing against Muon, keep both Muon baselines in view: original 12-step Muon is the stronger reference, while Muon Leon-NS isolates part of the orthogonalization change.
5. Before making benchmark claims, run independent trials for promising configs and report mean/std plus threshold behavior, per `AGENTS.md`.

## Important Commits

| Commit | Purpose |
| --- | --- |
| `1569f5ec617e57090aecbc37416223b5ddff678f` | Recorded Muon Leon-NS baseline, including modified Muon script, run archive, and CSV row. |
| `04bfad78e01c8c514a5396201d7baaf8b0e29cfc` | Last commit before the Muon Leon-NS baseline record. |
