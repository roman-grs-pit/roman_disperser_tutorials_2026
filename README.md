# roman_disperser_tutorials_2026

Tutorial notebooks, markdowns, and supporting material for the Roman GRS PIT's
[Roman Disperser](https://github.com/roman-grs-pit/roman_disperser) tool.

These tutorials are meant to be **self-contained**: you can work through them
from a NERSC allocation or your own laptop, whether or not you attended a live
session.

## Getting started

See **[docs/SETUP.md](docs/SETUP.md)** for environment setup. In short:

- **NERSC** — activate the curated shared conda environment (built from
  `environment-gpu.yml` / `environment-cpu.yml`).
- **Laptop** (CPU) — a `venv` + `pip install` of the disperser (conda optional);
  needs GitHub access to the private disperser repo.

The same notebooks run on CPU or GPU; only the JAX build differs. Then run
`notebooks/00_environment_check.ipynb` to confirm your setup.

## Contents

Work through `notebooks/` in order:

| #  | Notebook                  | Covers |
|----|---------------------------|--------|
| 00 | `00_environment_check`    | Smoke test: imports, JAX backend, data, a tiny dispersion |
| 01 | `01_spectra_to_counts`    | Templates, bandpasses, sensitivities → count-rate spectra |
| 02 | `02_disperse_a_star`      | Optical-model + PSF payloads; dispersing one star, all orders |
| 03 | `03_stars_galaxies_roll`  | Sérsic galaxies, a mixed field, and a roll |
| 04 | `04_simple_extraction`    | Trace-based 1D extraction; zeroth-order contamination |
| 05 | `05_pa_line_profiles`     | How a galaxy's position angle broadens an emission line |
| 06 | `06_catalogs_scaling`     | Catalog-driven fields and batched dispersion |
| 07 | `07_jax_optical_model`    | The differentiable optical model under the hood |
| 08 | `08_gpu_scale_out`        | A full-density, all-orders field at GPU scale |

## For maintainers

Development uses [pixi](https://pixi.sh) — see [CLAUDE.md](CLAUDE.md) for the
environment strategy and the JAX/CPU/GPU rationale.

Notebooks are committed **without outputs**. After cloning, run once:

```
pixi run setup-nbstripout
```

This installs an `nbstripout` git filter (in the maintainer-only `dev` env) that
strips outputs from what git stores while leaving your working-copy outputs
intact. The filter config lives in `.git/config`, so it's per-clone; the
`.gitattributes` mapping is committed. `pixi run clear-nb` is a manual fallback
that strips outputs from the files in place.
