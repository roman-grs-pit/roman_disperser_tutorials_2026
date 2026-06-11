# roman_disperser_tutorials_2026
Tutorial notebooks, markdowns, etc for the Roman GRS PIT's Roman Disperser tool.

## Tutorials

### 1. [Single Star and Galaxy Dispersion](single_star_and_galaxy_dispersion_tutorial_2026.ipynb)
Demonstrates Roman grism dispersion for a single star (PSF deposition) and a Sérsic galaxy (Jacobian warping + PSF convolution) on SCA 01. Covers:
- Optical model loading and SCA payload setup
- Star disperser: PSF deposition across 3 orders with relative efficiencies
- Galaxy disperser: Sérsic morphology, Jacobian warping, PSF convolution
- JIT compilation timing (first call vs cached) and flux conservation checks
- Visualization with AsinhNorm (full detector) and LogNorm (zoomed spectral traces)
- Workshop discussion questions with collapsible answers

### 2. [Source Catalog & SED Formats](roman_catalog_seds_tutorial_2026.ipynb)
Explains the `metadata.parquet` schema and `seds.zarr` format, the two input components required by the Roman disperser pipeline. Covers:
- Parquet metadata schema: all 10 required columns, their types and roles
- Zarr v3 store structure: wavelength grid, star SEDs, galaxy SED partitions
- How `sed_index` + `flux_scale` connect metadata to spectral data
- SED units (FLAM, erg/s/cm²/Å), wavelength convention, and trimming
- Per-SCA loading strategy and memory management
- Catalog validation checks and SED scrubbing (pathological spike protection)
- Creating minimal stand-in catalogs for testing without the full Galacticus dataset
- Workshop discussion questions with collapsible answers

## Supporting Files

- [`activating_conda_environment.md`](activating_conda_environment.md) — Setup instructions for the conda environment
- [`check_env.sh`](check_env.sh) — Quick environment validation script
