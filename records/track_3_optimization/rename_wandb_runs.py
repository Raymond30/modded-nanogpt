"""
rename_wandb_runs.py

Renames already-uploaded wandb runs so their display names match the
`run_id` column in records/track_3_optimization/tuning_log.csv.

How the mapping works:
  - For every CSV row whose `run_path` lives under records/track_3_optimization/runs/<optimizer>/,
    we use basename(run_path) as the lookup key and run_id as the new name.
  - We iterate runs in the wandb project and rename any whose current name matches a key.
  - The original timestamp+uuid name is preserved in config under `_original_run_name`
    and tags get a `renamed` marker, so the link back to the on-disk run dir is not lost.

Usage:
    python records/track_3_optimization/rename_wandb_runs.py \
        --project modded-nanogpt-backfill \
        --optimizer leonh
    # dry run (print mapping + matches, no API writes):
    python records/track_3_optimization/rename_wandb_runs.py \
        --project modded-nanogpt-backfill --optimizer leonh --dry-run
"""

import argparse
import csv
import sys
from pathlib import Path

import wandb


CSV_PATH = Path(__file__).parent / "tuning_log.csv"


def build_mapping(optimizer: str) -> dict[str, str]:
    """rel_path_under_runs_<opt> -> run_id, for CSV rows belonging to <optimizer>.

    When multiple CSV rows share the same run_path (e.g. a failed launch + a
    successful retry pointing at the same out-dir), prefer the row that did
    not diverge so the successful run wins.
    """
    mapping: dict[str, str] = {}
    mapping_diverged: dict[str, bool] = {}
    needle = f"/runs/{optimizer}/"
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = (row.get("run_path") or "").rstrip("/")
            if needle not in path:
                continue
            rel = path.split(needle, 1)[1]
            run_id = (row.get("run_id") or "").strip()
            if not rel or not run_id:
                continue
            diverged = (row.get("diverged") or "").strip().lower() == "true"

            if rel not in mapping:
                mapping[rel] = run_id
                mapping_diverged[rel] = diverged
                continue
            # Conflict: prefer the non-diverged row; otherwise keep latest.
            if mapping_diverged[rel] and not diverged:
                print(f"[info] {rel}: replacing diverged '{mapping[rel]}' "
                      f"with '{run_id}'", file=sys.stderr)
                mapping[rel] = run_id
                mapping_diverged[rel] = diverged
            elif diverged and not mapping_diverged[rel]:
                print(f"[info] {rel}: keeping non-diverged '{mapping[rel]}', "
                      f"skipping '{run_id}'", file=sys.stderr)
            else:
                print(f"[warn] {rel}: duplicate rows both diverged="
                      f"{diverged}; keeping latest '{run_id}' over "
                      f"'{mapping[rel]}'", file=sys.stderr)
                mapping[rel] = run_id
                mapping_diverged[rel] = diverged
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    help="wandb project name (with optional entity/, e.g. myteam/modded-nanogpt-backfill)")
    ap.add_argument("--optimizer", default="leonh",
                    help="Filter CSV rows by /runs/<optimizer>/ in run_path (default: leonh).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be renamed without calling the API.")
    args = ap.parse_args()

    mapping = build_mapping(args.optimizer)
    print(f"[info] built {len(mapping)} basename -> run_id mappings from CSV")
    if not mapping:
        print("[error] no mappings found; check --optimizer", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    runs = api.runs(args.project)

    renamed = 0
    skipped_no_match = 0
    skipped_already = 0
    for run in runs:
        current = run.display_name
        new_name = mapping.get(current)
        if new_name is None:
            skipped_no_match += 1
            continue
        if current == new_name:
            skipped_already += 1
            continue

        action = "DRY-RUN " if args.dry_run else ""
        print(f"[{action}rename] {current}  ->  {new_name}  ({run.id})")
        if args.dry_run:
            renamed += 1
            continue

        # wandb 0.27.x: only display_name's setter persists via update();
        # run.name's setter is a no-op on the public API.
        run.display_name = new_name
        if run.config.get("_original_run_name") is None:
            run.config["_original_run_name"] = current
        tags = list(run.tags or [])
        if "renamed" not in tags:
            tags.append("renamed")
        run.tags = tags
        run.update()
        renamed += 1

    print(
        f"\n[done] {'(dry-run) ' if args.dry_run else ''}"
        f"renamed {renamed} runs; "
        f"{skipped_no_match} had no CSV mapping; "
        f"{skipped_already} already had the target name"
    )


if __name__ == "__main__":
    main()
