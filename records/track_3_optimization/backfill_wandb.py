"""
backfill_wandb.py

Replays finished training runs into wandb by parsing the on-disk artifacts
under records/track_3_optimization/runs/<optimizer>/<run_dir>/:
  - train.log       -> step / val_loss / train_time / step_avg lines
  - config.json     -> wandb run config
  - metadata.json   -> argv, git_commit, device, world_size (added to config)

Logged keys match the live integration in train_gpt_simple_leonh.py:
    val_loss, train_time_s, step_avg_ms      (per validation step)

Usage:
    python records/track_3_optimization/backfill_wandb.py \
        --project modded-nanogpt-backfill \
        --optimizer leonh
    # or specific dirs:
    python records/track_3_optimization/backfill_wandb.py \
        --project modded-nanogpt-backfill \
        records/track_3_optimization/runs/leonh/20260513-122755-73a2fb7d
"""

import argparse
import json
import re
import sys
from pathlib import Path

import wandb


METRIC_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+"
    r"val_loss:(?P<val_loss>[-\d.eE+]+)\s+"
    r"train_time:(?P<train_time>[-\d.eE+]+)s\s+"
    r"step_avg:(?P<step_avg>nan|[-\d.eE+]+)ms"
)


def parse_train_log(log_path: Path):
    rows = []
    for line in log_path.read_text().splitlines():
        m = METRIC_RE.search(line)
        if not m:
            continue
        step_avg = m.group("step_avg")
        rows.append({
            "step": int(m.group("step")),
            "val_loss": float(m.group("val_loss")),
            "train_time_s": float(m.group("train_time")),
            "step_avg_ms": float("nan") if step_avg == "nan" else float(step_avg),
        })
    return rows


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def backfill_run(run_dir: Path, project: str, optimizer: str, opt_root: Path) -> bool:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        print(f"[skip] no train.log in {run_dir}")
        return False
    rows = parse_train_log(log_path)
    if not rows:
        print(f"[skip] no metric lines in {log_path}")
        return False

    config = load_json(run_dir / "config.json")
    metadata = load_json(run_dir / "metadata.json")
    if metadata:
        config["_metadata"] = metadata
    config["_optimizer"] = optimizer
    config["_run_dir"] = str(run_dir)

    # Use the relative path from runs/<optimizer>/ as the wandb display name,
    # so nested layouts (e.g. <batch>/<run>) stay collision-free.
    try:
        rel = run_dir.relative_to(opt_root).as_posix()
    except ValueError:
        rel = run_dir.name
    config["_rel_run_path"] = rel

    final = rows[-1]
    print(f"[upload] {rel}  steps={len(rows)}  "
          f"final_val_loss={final['val_loss']:.5f}  "
          f"final_step={final['step']}/{final['step']}")

    run = wandb.init(
        project=project,
        name=rel,
        config=config,
        tags=[optimizer, "backfill"],
        group=optimizer,
        reinit=True,
    )
    try:
        for row in rows:
            run.log(
                {
                    "val_loss": row["val_loss"],
                    "train_time_s": row["train_time_s"],
                    "step_avg_ms": row["step_avg_ms"],
                },
                step=row["step"],
            )
    finally:
        run.finish()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="wandb project name")
    ap.add_argument(
        "--optimizer",
        default="leonh",
        help="Subfolder under records/track_3_optimization/runs/ to walk "
             "when no explicit run dirs are passed (default: leonh).",
    )
    ap.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Optional explicit run directories. If omitted, walks the "
             "--optimizer subfolder.",
    )
    args = ap.parse_args()

    opt_root = Path(__file__).parent / "runs" / args.optimizer
    if args.run_dirs:
        targets = [d for d in args.run_dirs if d.is_dir()]
    else:
        if not opt_root.is_dir():
            print(f"[error] {opt_root} does not exist", file=sys.stderr)
            sys.exit(1)
        # Recursively find every directory that contains a train.log.
        targets = sorted(p.parent for p in opt_root.rglob("train.log"))

    if not targets:
        print("[error] no run directories found", file=sys.stderr)
        sys.exit(1)

    uploaded = 0
    for run_dir in targets:
        if backfill_run(run_dir, project=args.project, optimizer=args.optimizer, opt_root=opt_root):
            uploaded += 1
    print(f"\n[done] uploaded {uploaded}/{len(targets)} runs to project '{args.project}'")


if __name__ == "__main__":
    main()
