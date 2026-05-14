#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

script="records/track_3_optimization/train_gpt_simple_leonh.py"
venv_activate="${VENV_ACTIVATE:-.venv/bin/activate}"
if [[ -f "${venv_activate}" ]]; then
  source "${venv_activate}"
fi

python_bin="${PYTHON:-python}"

dry_run=0
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    dry_run=1
    break
  fi
done

if [[ "${dry_run}" == "1" ]]; then
  cmd=("${python_bin}" "${script}" "$@")
else
  nproc="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
  cmd=(torchrun --standalone --nproc_per_node="${nproc}" "${script}" "$@")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
