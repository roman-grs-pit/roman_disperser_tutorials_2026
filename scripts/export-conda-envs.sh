#!/usr/bin/env bash
# Regenerate the user-facing conda environment files from the pixi manifest.
# Run from the repo root after changing pixi.toml (and `pixi install`):
#
#     bash scripts/export-conda-envs.sh
#
# Produces environment-cpu.yml (pixi env `default`) and environment-gpu.yml
# (pixi env `gpu`). These are what NERSC/RRN/laptop users build from; the pixi
# manifest is the single source of truth — edit it, not the generated files.
set -euo pipefail
cd "$(dirname "$0")/.."

gen() {  # <pixi-env> <out.yml> <conda-name> <blurb>
  local env="$1" out="$2" name="$3" blurb="$4"
  pixi workspace export conda-environment -e "$env" "$out"
  # Users authenticate with `gh auth login` (HTTPS); the dev manifest pins the
  # private disperser via SSH (maintainer's key). Rewrite for users.
  sed -i.bak 's#git+ssh://git@github.com/#git+https://github.com/#' "$out"
  # Give the conda env a meaningful name (pixi exports the env name verbatim).
  sed -i.bak "s/^name: .*/name: $name/" "$out"
  rm -f "$out.bak"
  # Prepend a header (GENERATED — do not edit by hand).
  printf '%s\n%s' "$blurb" "$(cat "$out")" > "$out"
}

gen default environment-cpu.yml roman-disperser-tutorials \
'# GENERATED from pixi.toml by scripts/export-conda-envs.sh — do not edit by hand.
# Roman Disperser tutorials, CPU (laptop / RRN / NERSC-CPU). Includes romanisim.
# Laptop users: run `gh auth login` first (the disperser repo is private).
# romanisim wrapping also needs CRDS/STPSF data (heavy) — see docs/SETUP.md.'

gen gpu environment-gpu.yml roman-disperser-tutorials-gpu \
'# GENERATED from pixi.toml by scripts/export-conda-envs.sh — do not edit by hand.
# Roman Disperser tutorials, GPU/CUDA (NERSC-GPU / GPU box). Includes romanisim.
# Build on a linux-64 + CUDA host. See docs/SETUP.md.'

echo "Wrote environment-cpu.yml and environment-gpu.yml"
