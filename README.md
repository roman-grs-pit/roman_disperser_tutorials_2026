# roman_disperser_tutorials_2026

Tutorial notebooks, markdowns, and supporting material for the Roman GRS PIT's
[Roman Disperser](https://github.com/roman-grs-pit/roman_disperser) tool.

These tutorials are meant to be **self-contained**: you can work through them
from a NERSC allocation, the Roman Research Nexus (RRN), or your own laptop,
whether or not you attended a live session.

## Getting started

See **[docs/SETUP.md](docs/SETUP.md)** for environment setup. In short:

- **NERSC** (GPU) — activate the curated shared `$roman` conda environment.
- **RRN** (CPU) — activate the shared RRN environment (common install path).
- **Laptop** (CPU) — pip-install the disperser into a venv/conda (the only
  path that needs GitHub access to the private repo).

The same notebooks run on CPU or GPU; only the JAX install differs (SETUP.md
explains the one-step GPU overlay).

## Contents

Tutorial notebooks live under `notebooks/` (added incrementally).

## For maintainers

Development uses [pixi](https://pixi.sh) — see [CLAUDE.md](CLAUDE.md) for the
environment strategy and the JAX/CPU/GPU rationale.
