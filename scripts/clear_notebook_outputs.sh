#!/usr/bin/env bash
# Clear all outputs and execution counts from the tutorial notebooks so we only
# ever commit empty notebooks (outputs bloat the repo and create noisy diffs).
#
# Usage:
#   pixi run clear-nb            # via the pixi task (preferred)
#   scripts/clear_notebook_outputs.sh
#
# NOTE: this edits the notebook files in place, so it also clears outputs from
# your *working copy* — re-run a notebook to see results again. If you'd rather
# keep local outputs and only strip them in git, use nbstripout as a git filter
# instead (see README / SETUP for the one-line install).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

shopt -s nullglob
notebooks=(notebooks/*.ipynb)
if [ ${#notebooks[@]} -eq 0 ]; then
  echo "no notebooks found under notebooks/"
  exit 0
fi

jupyter nbconvert --clear-output --inplace "${notebooks[@]}"
echo "cleared outputs from ${#notebooks[@]} notebook(s)"
