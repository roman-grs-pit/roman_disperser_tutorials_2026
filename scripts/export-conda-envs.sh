#!/usr/bin/env bash
# Regenerate the user-facing conda environment files from the pixi manifest.
# Run from the repo root after changing pixi.toml (and `pixi install`):
#
#     bash scripts/export-conda-envs.sh
#
# Produces environment-cpu.yml (pixi env `default`) and environment-gpu.yml
# (pixi env `gpu`). These are what conda users build from; the pixi
# manifest is the single source of truth — edit it, not the generated files.
set -euo pipefail
cd "$(dirname "$0")/.."

gen() {  # <pixi-env> <out.yml> <conda-name> <blurb>
  local env="$1" out="$2" name="$3" blurb="$4"
  pixi workspace export conda-environment -e "$env" "$out"
  # The manifest pins the (public) disperser over HTTPS, so the export needs no
  # URL rewrite; keep the sed as a no-op safety net if the pin ever changes.
  sed -i.bak 's#git+ssh://git@github.com/#git+https://github.com/#' "$out"
  # Give the conda env a meaningful name (pixi exports the env name verbatim).
  sed -i.bak "s/^name: .*/name: $name/" "$out"
  rm -f "$out.bak"
  # Prepend a header (GENERATED — do not edit by hand).
  printf '%s\n%s' "$blurb" "$(cat "$out")" > "$out"
}

gen default environment-cpu.yml roman-disperser-tutorials \
'# GENERATED from pixi.toml by scripts/export-conda-envs.sh — do not edit by hand.
# Roman Disperser tutorials, CPU. Includes romanisim.
# romanisim wrapping also needs CRDS/STPSF data (heavy) — see docs/SETUP.md.'

gen gpu environment-gpu.yml roman-disperser-tutorials-gpu \
'# GENERATED from pixi.toml by scripts/export-conda-envs.sh — do not edit by hand.
# Roman Disperser tutorials, GPU/CUDA. Includes romanisim.
# Build on a linux-64 + CUDA host. See docs/SETUP.md.'

echo "Wrote environment-cpu.yml and environment-gpu.yml"
