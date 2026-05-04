# AGENTS.md

## Scope

This project is a NanoGPT-style optimizer tuning workspace centered on `records/track_3_optimization/`. 

## Benchmark Objective

The primary experimental goal is not only to minimize final validation loss, but to reach the target validation loss threshold as quickly as possible.

Primary objective:

- Attain `val_loss < 3.28` using as few training steps as possible.

Primary metric:

- `step_to_threshold`: the first validation step at which `val_loss < 3.28`. Because validation is periodic, `step_to_threshold` means the first validation checkpoint where `val_loss < 3.28`.

Secondary metrics:

- final validation loss;
- best validation loss;
- training time to threshold;
- total training time;
- stability, including NaNs, divergence, crashes, or large final-vs-best gaps;
- number of trials and mean/std across independent trials.

When summarizing runs, always report whether the run reached `val_loss < 3.28`. If it did, report the first step where the threshold was crossed. If it did not, report the best validation loss and final validation loss.

Do not claim that one optimizer is better solely because it reaches the threshold earlier in a single run. Promising configurations should be rerun with enough independent trials to estimate reliability.

## GPU Execution Policy

GPU jobs may be expensive. By default, Codex should not launch training, `torchrun`, multi-GPU jobs, or data downloads.

Codex may launch training only when the user's request explicitly asks to run, launch, execute, or sweep experiments.

Before launching any training job, Codex must:

1. show the exact command;
2. run or provide a dry-run command first, if available;
3. state the expected number of runs, GPUs used, and output/log directory;
4. avoid overwriting existing logs or results;
5. use the smallest reasonable test first unless the user explicitly asks for a full sweep.

Allowed without extra confirmation:
- reading code;
- editing scripts;
- creating sweep configs;
- parsing existing logs;
- running non-GPU static checks;
- running dry-run commands.

Allowed only with explicit user instruction:
- launching single-GPU training;
- launching `torchrun`;
- launching multi-GPU training;
- running hyperparameter sweeps;
- downloading datasets;
- archiving or deleting results.

## Main Script

The main experiment target is `train_gpt_simple_leon.py`.

It is self-contained. The base `hparams` dict in `train_gpt_simple_leon.py` controls training steps, AdamW auxiliary hyperparameters, AdamW cooldown, Leon hyperparameters, Leon cooldown, Leon Newton-Schulz iteration count, and Leon diagonal stability epsilon.

Important constraints:
- The script expects `torchrun`/distributed CUDA environment variables.
- Plain `python3 records/track_3_optimization/train_gpt_simple_leon.py --dry-run ...` is safe and does not initialize CUDA/NCCL.
- Do not run it directly with plain `python` for training.
- Do not change Leon optimizer math unless explicitly requested.
- Preserve the existing validation log format.

## Running and Dry-Run

- Safe dry-run, no GPU training:
  `python3 records/track_3_optimization/train_gpt_simple_leon.py --dry-run`
- Equivalent dry-run through the launcher:
  `records/track_3_optimization/launch_leon.sh --dry-run`
- Real Leon training uses the launcher, which prints and then executes a `torchrun` command:
  `records/track_3_optimization/launch_leon.sh --trials 1`
- The launcher sources `.venv/bin/activate` from the repo root when present before resolving `python` or `torchrun`.
- Override hparams with repeatable `--set key=value`, for example:
  `records/track_3_optimization/launch_leon.sh --dry-run --set leon_lr=0.02 --set leon_wd=0.03`
- Unknown `--set` keys are recorded but may have no effect unless the script uses them.
- There is no smoke-test mode at this time.

Each run writes a unique output directory by default under:

- `records/track_3_optimization/runs/leon/<timestamp>-<uuid>/`

Expected files:

- `train.log` for real training runs;
- `config.json` for resolved hparams;
- `config_diff.json` for exact differences from the base hparams;
- `metadata.json` for argv, cwd, script path, dry-run flag, trial count, git commit, PyTorch/CUDA version, and data shard counts.

## Hyperparameter Sweep Rules

- Preserve the base `hparams` dict. Use per-run overrides or copied configs rather than destructively rewriting the baseline values.
- Keep architecture, dataset, batch size, validation protocol, and benchmark rules unchanged unless explicitly requested.
- Do not change Leon optimizer math, Newton-Schulz coefficients, state buffers, distributed sync, or update scaling unless the task explicitly asks for optimizer algorithm changes.
- Prefer conservative sweeps: change one or a small number of related hyperparameters at a time, especially `leon_wd`, `leon_lr`, then secondary values such as `adam_cooldown_frac`, `leon_cooldown_frac`, `leon_mu`, `leon_beta2`, `leon_ns_iters`, and `leon_eps`.
- Predeclare the sweep grid before running jobs. Do not cherry-pick successful runs or omit failed runs from summaries.
- Early stopping is allowed only for exploratory screening or time-to-threshold measurement. Final benchmark claims should use clearly reported fixed-budget or threshold-based protocols and must not omit failed or non-threshold-crossing runs.
- For promising settings, run enough independent trials to summarize mean and variance; use the README significance rule when claiming benchmark validity.

## Early-Stopping and Budget Rules

Early stopping is allowed for exploratory screening and for measuring time-to-threshold.

Allowed:

- Stop runs that produce NaNs, divergence, CUDA errors, or clearly invalid logs.
- Stop exploratory runs after they reach `val_loss < 3.28`, if the purpose is to measure `step_to_threshold`.
- Use fixed-budget proxy runs, such as 250, 500, 1000, or 1500 steps.
- Use successive-halving style selection to decide which configs receive more budget.
- Clearly label these as screening or time-to-threshold runs.

For final benchmark claims:

- Report `step_to_threshold` for each run that reaches `val_loss < 3.28`.
- Also report full-budget final validation loss when available.
- Do not compare an early-stopped run against a full-budget run without clearly labeling the difference.
- Do not omit runs that failed to reach the threshold.
- Do not cherry-pick only successful threshold-crossing runs.

## Logging Expectations

- The current Leon script writes `train.log`, `config.json`, `config_diff.json`, and `metadata.json` in a unique run directory.
- `train.log` includes the full source code at the top for reproducibility.
- Preserve the existing validation log format:
  `step:<step>/<train_steps> val_loss:<loss> train_time:<seconds>s step_avg:<ms>ms`
- Sweep records should include script name, exact config diff from base, number of trials, GPU count/model, PyTorch/CUDA version from logs, log paths, final loss per run, mean/std final loss, and total training time.
- Archive important completed logs under `records/track_3_optimization/results/` only when explicitly asked.


## Summarizing Sweep Results

Use a compact table with:

- run label or log path;
- config diff from base;
- trials `n`;
- whether `val_loss < 3.28` was reached;
- `step_to_threshold`, if reached;
- best validation loss;
- final validation losses;
- mean final validation loss;
- README validity statistic `(3.28 - mean) * sqrt(n)`;
- steps, hardware, train time, and notes.

Always report all attempted runs in the sweep, not just the best result.

## Repo Layout

- `records/track_3_optimization/README.md` - benchmark rules, quickstart for the baseline script, notable results, tuning guidance.
- `records/track_3_optimization/train_gpt_simple.py` - Muon baseline training script.
- `records/track_3_optimization/train_gpt_simple_leon.py` - current Leon optimizer experiment target.
- `records/track_3_optimization/training_summary.md` - local run summaries.
- `records/track_3_optimization/optimization_summary.md` - architecture/optimizer/tuning summary.
- `records/track_3_optimization/make_figure.py` and `figure.png` - plots selected result logs.
- `records/track_3_optimization/results/` - archived run logs, including `20260430_adamh/` and `20260430_muonh/`.
- `data/cached_fineweb10B.py` and `data/fineweb10B/` - dataset helper and local FineWeb10B shards referenced by training scripts.
- `requirements.txt` and `data/requirements.txt` - dependency files. Note: README quickstart mentions `torch==2.11`; root `requirements.txt` currently pins `torch==2.10`.

## Done Criteria

- If training or GPU jobs were launched, the user explicitly requested execution, the exact commands were shown, dry-run behavior was used when available, and all log paths were reported.
- Base config values and optimizer math are preserved unless the user requested changes.
- Any edits are scoped to the requested experiment workflow.
- Unsupported commands are marked `TODO` rather than invented.
- Existing logs/results are not overwritten.
- The final response states what changed, what was not run, and any remaining TODOs.
