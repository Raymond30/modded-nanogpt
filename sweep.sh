#!/bin/bash

# ==============================================================================
# Hyperparameter Sweep for Leon
# ==============================================================================

# Search Space
LEARNING_RATES=(0.023)
BETA2_VALS=(0.4)
WEIGHT_DECAYS=(0.6 1.2)

TOTAL_RUNS=$(( ${#LEARNING_RATES[@]} * ${#BETA2_VALS[@]} * ${#WEIGHT_DECAYS[@]} ))
CURRENT_RUN=1

echo "Starting Hyperparameter Sweep!"
echo "Total Configurations to Test: $TOTAL_RUNS"
echo "----------------------------------------------------------------------"

# Loop over hyperparameter combinations
for lr in "${LEARNING_RATES[@]}"; do
    for beta2 in "${BETA2_VALS[@]}"; do
        for wd in "${WEIGHT_DECAYS[@]}"; do
            echo ""
            echo "======================================================================"
            echo "Run ($CURRENT_RUN / $TOTAL_RUNS) - Configuration:"
            echo "  LR        : $lr"
            echo "  BETA2     : $beta2"
            echo "  WD        : $wd"
            echo "======================================================================"
            
            # Export environments to be injested by train_gpt_leon.py
            export LEON_LR=$lr
            export LEON_BETA2=$beta2
            export LEON_WD=$wd

            # Run training using torchrun - standard command for modded-nanogpt
            # You can pipe output to tee if you wish, but modded-nanogpt tracks 
            # its own logging gracefully in the logs/ folder.
            torchrun --standalone --nproc_per_node=4 train_gpt_leon.py
            
            # Retrieve the most recent log file. The script dumps to logs/UUID.txt
            # It's highly recommended to append hyperparams to the top of logs 
            # to verify sweeps later.
            LATEST_LOG=$(ls -t logs/*.txt | head -n 1)
            
            if [ -n "$LATEST_LOG" ]; then
                echo "Finished combination: $lr, $beta2, $wd. Appending tags to $LATEST_LOG"
                # Insert details cleanly at the top of the logging file
                sed -i "1i # HYPERPARAMETERS: LEON_LR=$lr, LEON_BETA2=$beta2, LEON_WD=$wd" "$LATEST_LOG"
            fi
            
            CURRENT_RUN=$((CURRENT_RUN + 1))
        done
    done
done

echo "======================================================================"
echo "Sweep Complete!"
echo "You can check your logs against training_summary.md"
