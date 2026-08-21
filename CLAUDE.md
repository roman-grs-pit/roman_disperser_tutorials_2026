# CLAUDE.md — roman_disperser_tutorials_2026

Guidance for working in this repo. This is the **tutorials** repo, separate
from the `roman_disperser` library (one directory up locally; canonical home
`github.com/roman-grs-pit/roman_disperser`).

## What this repo is

Tutorial notebooks + docs teaching the Roman GRS PIT's disperser tool. The
notebooks `import roman_disperser`; this repo adds the teaching material, the
environment glue, and a self-contained setup path.

**Standalone goal:** someone who finds this repo *without* attending the live
tutorial should be able to set up an environment and run everything from the
docs alone. Don't assume the reader was in the room.

## Status (2026-08-08)

**Tutorial notebooks 00–09 are written and merged to `main`, tracking
`roman_disperser` v0.14.2** (the elements API: G150 grism + P127 prism). The
v0.10→v0.14.2 update rewrote the API touchpoints (elements threading, new
`load_sensitivities`/`select_sources_per_order` signatures, float64-NumPy into
`get_fpa_pos`), deleted the stale "sky→FPA only correct at Dec = 0" caveats
(fixed upstream in v0.12.0 by the exact gnomonic projection — notebook 07 now
*demonstrates* the removed flat-sky error), and added notebook 09 (the prism)
plus `MIGRATION.md` (what changed for tutorial users; links to the disperser's
own migration guide).

- Disperser pinned to **v0.14.2** (elements API, prism support, lock-resolved
  reference data via `roman-disperser-hydrate`).
- Notebooks: 00 env check · 01 spectra→counts · 02 disperse a star · 03 mixed
  field + roll · 04 extraction + 0th-order contamination · 05 PA→line profiles ·
  06 catalogs + batched dispersion · 07 JAX optical-model tools · 08 GPU
  scale-out · 09 the prism. Committed **without outputs** via an nbstripout git
  filter (`pixi run setup-nbstripout` once per clone).
- **NERSC deployment dropped (2026-08-08).** The shared-env recipes existed for
  a one-off onboarding session and are no longer maintained; the repo is
  standalone (own-machine setup only). The historical NERSC build notes live in
  git history (pre-`update-v0.14.2` CLAUDE.md/SETUP.md) if ever needed again.
- **romanisim wrap dropped from the tutorials** (the once-planned disperse→wrap
  step). romanisim stays bundled in the envs for when wrap content is added back,
  but is unused by 00–09.
- **RRN and a standalone GPU notebook dropped:** GPU scale-out is notebook 08,
  run by re-executing it on a GPU node.

## The audience (drives every env decision)

One **user** runtime context: your own machine (laptop or workstation,
osx/linux, GPU optional). The default is a **venv + `pip install`** of the
disperser (`roman_disperser[full]` pulls everything the tutorials use);
`environment-cpu.yml`/conda is the option for when the romanisim wrap returns
(it needs conda for `fftw`). The repo is deliberately **standalone** — no
shared-facility deployment is assumed or maintained.

Two **developer** contexts (Nikhil): laptop (osx-arm64, CPU) and a separate
linux-64 GPU box (CUDA 12).

## Environment strategy

- **Developers use pixi; users do not.** Pixi is more than a tutorial reader
  should have to learn — users get a plain venv (or conda) path in
  `docs/SETUP.md`. Pixi is a dev-only tool here (it also carries a
  maintainer-only `dev` env with `nbstripout` for the output-stripping git
  filter).
- **`pixi.toml`** (dev) is the single source of truth: platforms
  `osx-arm64` + `linux-64`, environments `default` (CPU) and `gpu` (CUDA,
  linux-64). **Both still bundle romanisim** — kept for future disperse→wrap
  content, though the current notebooks (00–08) don't use it. One file covers
  laptop + GPU box.
- The disperser is pulled **from git**, not the `../roman_disperser` sibling
  path, so this repo is reproducible off a laptop (GPU box, CI).
- **The disperser repo is PUBLIC** (since 2026-08; it was private before).
  Clones and pip/pixi installs over HTTPS need no authentication — the
  manifest and all documented install URLs use the `https://` form. (Historic
  note: the pre-2026-08 docs described `gh auth login` / SSH-key setup; that
  requirement is gone.)
- **`environment-cpu.yml` / `environment-gpu.yml`** (user) are the conda specs —
  **generated** from the pixi envs by `scripts/export-conda-envs.sh` (which
  names the conda env). Don't hand-edit; edit `pixi.toml` and rerun. Users
  default to a plain venv + pip; the ymls serve the conda fallback (needed once
  the romanisim wrap returns, for `fftw`).

## The JAX seam (important, easy to get wrong)

`roman_disperser` is JAX-based. **The same notebook runs CPU or GPU — only the
install differs.** So CPU/GPU is an *install* concern, not a *content* concern;
don't write parallel CPU/GPU notebooks. Reserve a dedicated GPU notebook only
where throughput/scale is the actual lesson.

Consequences, and the reason JAX is **never pinned** in this repo's env files:

- In pixi, JAX flavor is automatic: the `cuda` feature overrides
  `jaxlib build=cuda12*`.
- In the pip/conda world it's a deliberate **overlay**: a CPU `jax` floor
  arrives via `roman_disperser`; GPU users then run
  `pip install jax[cuda12-local]` (or `[cuda12]`) themselves.
- Each env yml pins its backend build explicitly: `-gpu` carries
  `jaxlib cuda12*`, `-cpu` carries `jaxlib cpu*`. The `-cpu` pin matters
  whenever the env is *built* on a GPU host: conda-forge resolves the CUDA
  jaxlib there (the `__cuda` virtual package), so without the pin the "CPU"
  env grabs the GPU.
- **Single source of truth for the JAX-flavor choice is
  `roman_disperser/INSTALL.md`.** Tutorials *link* to it; do not duplicate the
  `cuda12-local` vs `cuda12` decision here — a second copy will drift.

## Reference data (hydration)

The disperser's reference data is **vendored** (disperser ≥ 0.10.0) — fetched
with `roman-disperser-hydrate`, not shipped in the package. The disperser is
pinned to **`v0.14.2`** in `pixi.toml`; the env ymls inherit it via the export
script — bump it in `pixi.toml` only. Since v0.14.2 the optical-model delivery
inside a data dir is resolved from `data-versions.lock` (written by hydrate), so
a pre-lock or hand-assembled data dir fails loudly — re-hydrate to fix. A full
hydrate now fetches **both elements'** assets (grism + prism, ~6.4 GB).

- **Data resolution** (disperser side): `$ROMAN_DISPERSER_DATA` →
  `$PIXI_PROJECT_ROOT/data` → `./data`. So the **dev pixi env** lands data in
  `tutorials/data` automatically (`pixi run hydrate`); **users** must set
  `ROMAN_DISPERSER_DATA`.
- **Jupyter kernels do not inherit your shell env.** `ROMAN_DISPERSER_DATA` must
  be set *for the kernel* — via the `kernel.json` `"env"` block. Documented in
  `docs/SETUP.md`. This is the most common "works in the terminal, not in the
  notebook" trap, and notebook 00 checks it from inside the kernel.
- Hydration mechanics (manifest/lock, `--only`/`--sca`) live in the disperser's
  `INSTALL.md`; tutorials link to it rather than duplicating.

## Validation gate

`pixi run check-jax` and a headless `nbconvert --execute` in the dev pixi env
catch gross breakage, but the dev env is not what users build. The gate before
publishing is executing every notebook in a **clean user-style environment**
(venv + pip, or conda from the generated ymls) — the user path is what has to
work.

## Notebook authoring

- **Audience: graduate students and professional astronomers.** Pitch the prose
  accordingly — assume fluency in the physics and astronomy. Don't define basic
  quantities (flux, magnitude, redshift, PSF, FWHM, dispersion, …) or belabour
  standard reasoning; explain what's specific to the disperser, the GRS setup, or
  a non-obvious numerical/code choice. Keep it concise and unpatronising; depth
  belongs on the parts a peer wouldn't already know.
- **Layout.** The intro sequence `00–08` stays flat in `notebooks/`. More
  advanced / validation material goes in `notebooks/advanced/` and
  `notebooks/validation/` (created lazily, as content arrives). A subdir notebook
  reaches the shared `tutorial_helpers.py` (which lives in `notebooks/`) via a
  small walk-up-one-level `sys.path` shim at the top of the notebook.
- **Markdown style: one line per paragraph.** Do **not** hard-wrap prose inside a
  markdown cell — the JupyterLab renderer turns mid-paragraph newlines into line
  breaks, producing false paragraphs. Write each paragraph as a single line,
  separate paragraphs with a blank line, put each list item on its own line, and
  keep `$$…$$` display math on its own line. The `00–08` notebooks are the
  reference for this style.
- **Outputs are stripped in git** by the `*.ipynb` nbstripout filter
  (`pixi run setup-nbstripout` once per clone); your working copy keeps run
  outputs. Author by executing locally (`nbconvert --execute --inplace`), then
  let the filter strip on commit.
- **The `.ipynb` is the source of truth.** It's fine to author a notebook from a
  throwaway Python/nbformat builder, but don't commit the builder — it's scratch.
  If you do build programmatically, route markdown cells through a `reflow()`-style
  pass so they obey the one-line-per-paragraph rule above, and re-execute the
  notebook afterwards to repopulate the (working-copy) outputs.

## Files

- `pixi.toml` — dev environment + single source of truth (CPU + gpu, both with
  romanisim). Root (not `docs/`).
- `environment-cpu.yml` / `environment-gpu.yml` — user conda envs, GENERATED
  from `pixi.toml` by `scripts/export-conda-envs.sh`. Root.
- `docs/SETUP.md` — canonical, standalone user setup (own machine: venv or
  conda). The environment check is `notebooks/00_environment_check.ipynb`
  (replaced the old `check_env.sh`).
- `README.md`, `CLAUDE.md` — stay at root (README renders on GitHub; CLAUDE.md
  must be at root to be auto-loaded). User prose lives in `docs/`.
