# roman_disperser_tutorials_2026

Tutorial notebooks, markdowns, and supporting material for the Roman GRS PIT's
[Roman Disperser](https://github.com/roman-grs-pit/roman_disperser) tool.

These tutorials are meant to be **self-contained**: you can work through them
from a NERSC allocation, the Roman Research Nexus (RRN), or your own laptop,
whether or not you attended a live session.

## Getting started

See **[docs/SETUP.md](docs/SETUP.md)** for environment setup. In short:

- **NERSC / RRN** — activate the curated shared conda environment (built from
  `environment-gpu.yml` / `environment-cpu.yml`).
- **Laptop** (CPU) — `conda env create -f environment-cpu.yml` (needs GitHub
  access to the private disperser repo — `gh auth login` first).

Both env files bundle romanisim, so the full disperse→wrap pipeline runs in one
kernel. The same notebooks run on CPU or GPU; only the JAX build differs.

## Contents

Tutorial notebooks live under `notebooks/` (added incrementally).

## For maintainers

Development uses [pixi](https://pixi.sh) — see [CLAUDE.md](CLAUDE.md) for the
environment strategy and the JAX/CPU/GPU rationale.
