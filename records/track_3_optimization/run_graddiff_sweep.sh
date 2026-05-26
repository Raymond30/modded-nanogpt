#!/usr/bin/env bash
# One-at-a-time hyperparameter sweep for leon_use_grad_diff=True, 1500 steps.
# Baseline: lr=0.025, wd=0.025, mu=0.95, beta2=0.7  → final val 3.43389
# Shared flags for every run:
set -e
SHARED="--set train_steps=1500
  --set leon_ns_iters=12 --set leon_orthogonalize_dtype=float32 --set leon_eps=1e-12
  --set leon_use_grad_diff=True
  --set leon_log_diagnostics=True --set leon_diag_interval=125
  --set wandb_project=modded-nanogpt-track3"

run() {
  local run_id="$1"; shift
  echo "===== START $run_id ====="
  CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
    records/track_3_optimization/launch_leon.sh --trials 1 \
    $SHARED --set wandb_run_name="$run_id" "$@"
  echo "===== END $run_id ====="
}

# --- LR sweep (wd=0.025, mu=0.95, beta2=0.7) ---
run gd_lr0175_1500 --set leon_lr=0.0175 --set leon_wd=0.025 --set leon_mu=0.95 --set leon_beta2=0.7 --set leon_cooldown_frac=0.7
run gd_lr035_1500  --set leon_lr=0.035  --set leon_wd=0.025 --set leon_mu=0.95 --set leon_beta2=0.7 --set leon_cooldown_frac=0.7

# --- WD sweep (lr=0.025, mu=0.95, beta2=0.7) ---
run gd_wd0125_1500 --set leon_lr=0.025 --set leon_wd=0.0125 --set leon_mu=0.95 --set leon_beta2=0.7 --set leon_cooldown_frac=0.7
run gd_wd050_1500  --set leon_lr=0.025 --set leon_wd=0.05   --set leon_mu=0.95 --set leon_beta2=0.7 --set leon_cooldown_frac=0.7

# --- mu (beta1) sweep (lr=0.025, wd=0.025, beta2=0.7) ---
run gd_mu085_1500  --set leon_lr=0.025 --set leon_wd=0.025 --set leon_mu=0.85  --set leon_beta2=0.7 --set leon_cooldown_frac=0.7
run gd_mu098_1500  --set leon_lr=0.025 --set leon_wd=0.025 --set leon_mu=0.98  --set leon_beta2=0.7 --set leon_cooldown_frac=0.7

# --- beta2 sweep (lr=0.025, wd=0.025, mu=0.95) ---
run gd_b250_1500   --set leon_lr=0.025 --set leon_wd=0.025 --set leon_mu=0.95 --set leon_beta2=0.5 --set leon_cooldown_frac=0.7
run gd_b290_1500   --set leon_lr=0.025 --set leon_wd=0.025 --set leon_mu=0.95 --set leon_beta2=0.9 --set leon_cooldown_frac=0.7

echo "===== ALL RUNS COMPLETE ====="
