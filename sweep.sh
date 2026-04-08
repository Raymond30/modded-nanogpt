#!/bin/bash

# ==============================================================================
# Hyperparameter Sweep for Leon (Round 2)
# Tunes: LR, Momentum (max), Beta2, Weight Decay
#
# Best from previous sweep: LR=0.023, Beta2=0.4, WD=1.2, Momentum=0.95 (default)
# This sweep probes around that optimum while adding momentum as a new axis.
# ==============================================================================

# Search Space
LEARNING_RATES=(0.046)
MOMENTUM_VALS=(0.95)
BETA2_VALS=(0.4)
WEIGHT_DECAYS=(0.3)

TOTAL_RUNS=$(( ${#LEARNING_RATES[@]} * ${#MOMENTUM_VALS[@]} * ${#BETA2_VALS[@]} * ${#WEIGHT_DECAYS[@]} ))
CURRENT_RUN=1

echo "Starting Leon Hyperparameter Sweep (Round 2)!"
echo "Total Configurations to Test: $TOTAL_RUNS"
echo "----------------------------------------------------------------------"

# Loop over hyperparameter combinations
for lr in "${LEARNING_RATES[@]}"; do
    for momentum in "${MOMENTUM_VALS[@]}"; do
        for beta2 in "${BETA2_VALS[@]}"; do
            for wd in "${WEIGHT_DECAYS[@]}"; do
                echo ""
                echo "======================================================================"
                echo "Run ($CURRENT_RUN / $TOTAL_RUNS) - Configuration:"
                echo "  LR       : $lr"
                echo "  MOMENTUM : $momentum"
                echo "  BETA2    : $beta2"
                echo "  WD       : $wd"
                echo "======================================================================"

                # Export environments to be ingested by train_gpt_leon.py
                export LEON_LR=$lr
                export LEON_MOMENTUM_MAX=$momentum
                export LEON_BETA2=$beta2
                export LEON_WD=$wd

                # Run training using torchrun - standard command for modded-nanogpt
                torchrun --standalone --nproc_per_node=4 train_gpt_leon.py

                # Retrieve the most recent log file. The script dumps to logs/UUID.txt
                LATEST_LOG=$(ls -t logs/*.txt | head -n 1)

                if [ -n "$LATEST_LOG" ]; then
                    echo "Finished combination: LR=$lr, MOM=$momentum, BETA2=$beta2, WD=$wd. Tagging $LATEST_LOG"
                    # Insert hyperparameter header at the top of the log file
                    sed -i "1i # HYPERPARAMETERS: LEON_LR=$lr, LEON_MOMENTUM_MAX=$momentum, LEON_BETA2=$beta2, LEON_WD=$wd" "$LATEST_LOG"
                fi

                CURRENT_RUN=$((CURRENT_RUN + 1))
            done
        done
    done
done

echo "======================================================================"
echo "Sweep Complete! ($TOTAL_RUNS runs)"
echo "You can check your logs against training_summary.md"
