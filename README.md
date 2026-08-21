# roman_disperser_tutorials_2026

[![smoke](https://github.com/roman-grs-pit/roman_disperser_tutorials_2026/actions/workflows/smoke.yml/badge.svg)](https://github.com/roman-grs-pit/roman_disperser_tutorials_2026/actions/workflows/smoke.yml)

**[Setup](docs/SETUP.md) · [Contents](#contents) · [Migration from v0.10](MIGRATION.md) · [Changelog](#changelog)**

Tutorial notebooks, markdowns, and supporting material for the Roman GRS PIT's
[Roman Disperser](https://github.com/roman-grs-pit/roman_disperser) tool.

These tutorials are meant to be **standalone**: you can work through them on
your own laptop or workstation, whether or not you attended a live session.

The tutorials track `roman_disperser` **v0.14.2**, which supports both WFI
dispersing elements (G150 grism and P127 prism) through the elements API. If
you last ran them against v0.10, read **[MIGRATION.md](MIGRATION.md)** — both
the API and some simulated numbers changed.

## Getting started

See **[docs/SETUP.md](docs/SETUP.md)** for environment setup. In short: a
`venv` + `pip install` of the disperser (conda optional; the disperser repo is
public, so no GitHub auth is needed), then hydrate the reference data.

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
| 09 | `09_prism`                | The P127 prism: same pipeline, second dispersing element |

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

## Changelog

Releases are date-tagged (`vYYYY.MM`), since the tutorials track the evolving
`roman_disperser` rather than versioning themselves. Newest first.

### Unreleased (on `main`)

- Notebook 08 now **warns when a GPU's steady-state rate is far off** the
  expected few ms per source-order — any modern GPU should be faster, so a
  large number means the JAX build has a performance bug (e.g. jax 0.11.0's
  ~15–20× GPU scatter regression, fixed in 0.11.1). This catches performance
  regressions on the user's own hardware, where CI cannot see them.
- **Weekly smoke CI** (badge above): executes notebooks 00–02 via the
  SETUP.md user path (venv-style pip install + hydration) and checks the
  notebook-00 smoke-dispersion total against its validated reference —
  catches crashes and numeric drift from the unpinned dependency stack.
- README: CI badge and navigation links (including this changelog).

### v2026.08 — track disperser v0.14.2 (2026-08-21)

- Tutorials updated from disperser v0.10.0 to **v0.14.2**: the elements API
  (G150 grism + P127 prism) threads through every notebook, and the stale
  "sky→FPA only correct at Dec = 0" caveats are gone (fixed upstream in
  v0.12.0; notebook 07 now demonstrates the removed flat-sky error).
- **New notebook 09 — the prism**: same pipeline with `element=PRISM`; trace,
  dispersion, and resolving power (per-pixel R vs the mission R_GRISM = 461,
  reconciled via the ~3-px resolution element).
- **[MIGRATION.md](MIGRATION.md)**: what changed for tutorial users moving
  from v0.10, including the results-changing placement/RNG fixes upstream.
- Notebook 08 reworked to separate **one-time compile from steady-state
  throughput**: it now times the same pass twice and projects from the steady
  rate (a10g: 441 ms/source-order first pass vs 4.1 ms steady).
- Notebook 02: callout explaining the benign `cpu_aot_loader` /
  "+prefer-no-gather … SIGILL" XLA message seen on JAX compilation-cache hits.
- Setup: repo is standalone (NERSC deployment dropped); the disperser repo is
  now public, so installs need no GitHub auth; SETUP.md states the
  Python ≥ 3.12 floor.
- Validation for this release: all 12 notebooks executed green in the dev pixi
  env (cold JAX cache, CPU; notebook 08 on an a10g GPU) **and** in a clean
  user-style venv built from SETUP.md alone. Rendered HTML of the validated
  run is attached to the GitHub release.

### 2026-06 — initial tutorials (untagged)

- Notebooks 00–08 against disperser v0.10.0, dress-rehearsed on a laptop and
  NERSC (the NERSC deployment has since been dropped).
