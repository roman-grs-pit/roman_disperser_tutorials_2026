# What changed: tutorials at v0.10.0 → v0.14.2

The tutorials were written against `roman_disperser` **v0.10.0** and now track
**v0.14.2**. This note summarizes what changed *for tutorial users* — both the
science-level shifts (results computed at v0.10 are not pixel-identical at
v0.14) and the API you will see in the notebooks. The authoritative, complete
story lives in the disperser repo:

- [`docs/migrating-v0.10-to-v0.14.md`](https://github.com/roman-grs-pit/roman_disperser/blob/main/docs/migrating-v0.10-to-v0.14.md) — the full migration guide;
- [`CHANGELOG.md`](https://github.com/roman-grs-pit/roman_disperser/blob/main/CHANGELOG.md) — per-release detail;
- [`docs/element_support.md`](https://github.com/roman-grs-pit/roman_disperser/blob/main/docs/element_support.md) — per-module status of grism/prism support.

## Science-level changes (your numbers move)

Three releases change *results*, not just interfaces. If you have outputs
produced with v0.10-era tutorials, expect these differences:

1. **v0.11.0 — GPU/TF32 and float32-RA precision fixes.** On GPUs, matrix
   ops now force full float32 precision (no TF32), and the sky→FPA right
   ascension is differenced at float64 on the host. GPU-rendered source
   positions move by a median of ~1.8 px (up to ~7 px). As part of this,
   `get_fpa_pos` now **raises a `TypeError`** if you hand it float32 or JAX
   arrays — pass float64 NumPy (what pandas `.values` gives you).
2. **v0.12.0 — exact gnomonic (TAN) sky→FPA projection.** The flat-sky
   approximation (Δα·cos δ, Δδ) is gone. Off-equator placements move by up to
   tens of pixels. **Notebook 07 reproduces the removed flat-sky error
   directly** (§2): for a ±0.4° square field, median 0.12 px at Dec 0, ~7 px
   at Dec 30, and ~20 px at Dec 60 (max ~80 px at the field corners). The old
   tutorials' "only correct at Dec = 0" warnings and workarounds are deleted —
   sky placement is now valid anywhere, including across RA = 0.
3. **v0.13.0 — per-SCA RNG re-keying + provenance.** Noise (`ISIM`)
   realisations are keyed per SCA via `jax.random.fold_in`, so they differ
   from ≤v0.12 runs at identical seeds; the noiseless `MODEL` images are
   untouched. Every FITS product now carries `CODEVER`/`GITSHA` cards.

## The elements API (v0.14.0): grism and prism

The disperser now supports both WFI dispersing elements — the **G150 grism**
and the **P127 prism** — through one abstraction:

```python
from roman_disperser.elements import GRISM, PRISM   # DispersingElement records

element = GRISM          # orders ("0","1","2"), band 0.9–2.0 µm
element = PRISM          # single order ("1",),  band 0.75–1.85 µm
```

Everything element-specific hangs off the record: `element.orders`,
`element.lam_min`/`lam_max`, `element.stpsf_filters` (order → STPSF filter,
including "order 2 shares the order-1 PSF"), `element.sensitivities_subdir`.
The notebooks pass `element=` explicitly at every API touchpoint:

- `paths.optical_model_path(element=...)`, `paths.sensitivity_dir(element=...)`;
- `psf_model.get_or_make_psf_payload(..., element=...)` — derives the STPSF
  filter *and* the PSF wavelength grid together, so a wrong-band cache cannot
  be selected (passing a non-grism `stpsf_filter=` by hand without matching
  `wavelengths=` raises);
- `pipeline.load_sensitivities(dir, sca, wavelengths, orders)` and
  `pipeline.select_sources_per_order(payloads, xfpa, yfpa, orders, wl_min,
  wl_max)` — the orders and band are now **required arguments** (fed from the
  element), where v0.10 baked in grism constants (`pipeline.ORDERS` etc. were
  removed in v0.14.0 and now fail loudly on import).

The production driver `scripts/build_grism_image.py` was renamed
`build_dispersed_image.py` (deprecated alias kept) and takes `--element
grism|prism`; outputs are prefixed `grism_*`/`prism_*` and stamped with an
`OPTELEM` header card.

**New notebook: [09 · The prism](notebooks/09_prism.ipynb)** runs the series'
pipeline with `element=PRISM` and measures the trace/dispersion/resolving-power
differences from the optical model.

## Reference data (v0.14.2): lock-resolved, re-hydrate once

Optical-model resolution is now driven by `data-versions.lock`, written by
`roman-disperser-hydrate`. A data directory assembled by hand or hydrated by an
old release **fails loudly** (`FileNotFoundError` naming the lock). Fix:

```bash
pixi run hydrate            # developers (tutorials repo)
roman-disperser-hydrate     # users
```

Hydration is pinned by default (`--update` upgrades deliberately) and now
fetches the prism assets too (`optical-model-prism-v0.8`,
`sensitivities-prism-v1`, `psf-prism-v1`; ~6.4 GB total for both elements —
`--only`/`--sca` subset if space matters). Notebook 00 checks both elements'
assets from inside the kernel.

## What changed in each notebook

| notebook | change |
|---|---|
| 00 | data check is lock-aware and covers both elements; smoke test passes `element=` |
| 01 | new "dispersing element" section; sensitivity/grid calls take `element=` |
| 02 | orders come from `element.orders`; PSF payloads via `element=`; FITS gains `OPTELEM` |
| 03–05 | `element=GRISM` threaded through setup; helpers (`tutorial_helpers`) are element-aware |
| 06 | new `load_sensitivities`/`select_sources_per_order` signatures; float64 NumPy into `get_fpa_pos`; Dec = 0 caveats deleted |
| 07 | gnomonic-projection discussion + a new demo reproducing the pre-v0.12 flat-sky error; band from the element |
| 08 | per-order PSF mapping now from `element.stpsf_filters`; Dec ≠ 0 workaround language deleted |
| 09 | **new** — the prism |

The environment pin moved to `v0.14.2` in `pixi.toml` (dev) and the generated
`environment-*.yml` / `docs/SETUP.md` (users).
