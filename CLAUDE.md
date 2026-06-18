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

## Status (2026-06-16)

**Tutorial notebooks 00–08 are written, executed, and merged to `main`.** They
run from `docs/SETUP.md` alone (standalone goal met) and were dress-rehearsed on
a laptop. A NERSC run-through — the real shared-env gate, and the only place the
GPU notebook runs at scale — is still pending.

- Disperser pinned to **v0.10.0** (vendored reference data + `roman-disperser-hydrate`).
- Notebooks: 00 env check · 01 spectra→counts · 02 disperse a star · 03 mixed
  field + roll · 04 extraction + 0th-order contamination · 05 PA→line profiles ·
  06 catalogs + batched dispersion · 07 JAX optical-model tools · 08 GPU
  scale-out. Committed **without outputs** via an nbstripout git filter
  (`pixi run setup-nbstripout` once per clone).
- **NERSC env is built** (see "NERSC deployment"): shared CPU+GPU conda envs,
  data hydrated to CFS, auto-named kernels that resolve `ROMAN_DISPERSER_DATA`.
  Confirm the shared envs carry `roman_disperser ≥ 0.10.0` before a NERSC run.
- **romanisim wrap dropped from the tutorials** (the once-planned disperse→wrap
  step). romanisim stays bundled in the envs for when wrap content is added back,
  but is unused by 00–08.
- **RRN and a standalone GPU notebook dropped:** GPU scale-out is notebook 08,
  run by re-executing it on a GPU node. Known issue flagged in 06/07: the
  sky→FPA conversion (`get_fpa_pos`) is currently only correct at Dec = 0.

## The audience matrix (drives every env decision)

Two **user** runtime contexts:

| Context     | Manager                | Platform   | GPU | Installs disperser? |
|-------------|------------------------|------------|-----|---------------------|
| NERSC       | conda (shared `$roman`)| linux-64   | yes | no — shared env     |
| Own laptop  | venv + pip (or conda)  | osx/linux  | no  | yes — git install   |

**Only the laptop path installs the disperser.** NERSC provides a curated shared
environment with it pre-installed, so it never authenticates. The laptop default
is a **venv + `pip install`** of the disperser (`roman_disperser[full]` pulls
everything the tutorials use); `environment-cpu.yml`/conda is the option for when
the romanisim wrap returns (it needs conda for `fftw`).

Two **developer** contexts (Nikhil): laptop (osx-arm64, CPU) and a separate
linux-64 GPU box (CUDA 12).

## Environment strategy

- **Developers use pixi; users do not.** Pixi fights the conda module system
  and `$HOME` quotas on NERSC, and is more than a tutorial reader should have to
  learn. Pixi is a dev-only tool here (it also carries a maintainer-only `dev`
  env with `nbstripout` for the output-stripping git filter).
- **`pixi.toml`** (dev) is the single source of truth: platforms
  `osx-arm64` + `linux-64`, environments `default` (CPU) and `gpu` (CUDA,
  linux-64). **Both still bundle romanisim** — kept for future disperse→wrap
  content, though the current notebooks (00–08) don't use it. One file covers
  laptop + GPU box.
- The disperser is pulled **from git**, not the `../roman_disperser` sibling
  path, so this repo is reproducible off a laptop (GPU box, CI).
- **The disperser repo is PRIVATE.** Org membership is authorization, not
  authentication — a clone still needs `gh auth login` (HTTPS credential
  helper) or a registered SSH key. This affects only the **laptop** path and
  the **maintainer pixi solve**; NERSC uses a pre-installed shared env.
  `docs/SETUP.md` documents both SSH and `gh auth login` (HTTPS) for the laptop.
- **`environment-cpu.yml` / `environment-gpu.yml`** (user) are the conda specs —
  **generated** from the pixi envs by `scripts/export-conda-envs.sh` (which
  rewrites the disperser URL ssh→https and names the conda env). Don't hand-edit;
  edit `pixi.toml` and rerun. The **NERSC shared envs** are built from these;
  laptop users now default to a plain venv + pip (conda is the fallback for the
  romanisim wrap), so the ymls mainly serve NERSC.

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
  `jaxlib cuda12*`, `-cpu` carries `jaxlib cpu*`. The `-cpu` pin matters on
  NERSC: conda-forge resolves the CUDA jaxlib if the env is *built* on a GPU
  host (the `__cuda` virtual package), so without it the "CPU" env grabs the
  GPU and OOMs on a shared login node.
- **Single source of truth for the JAX-flavor choice is
  `roman_disperser/INSTALL.md`.** Tutorials *link* to it; do not duplicate the
  `cuda12-local` vs `cuda12` decision here — a second copy will drift.

## Reference data (hydration)

The disperser's reference data is **vendored** (disperser ≥ 0.10.0) — fetched
with `roman-disperser-hydrate`, not shipped in the package. The disperser is
pinned to **`v0.10.0`** (first release with the command) in `pixi.toml`; the
env ymls inherit it via the export script — bump it in `pixi.toml` only.

- **Data resolution** (disperser side): `$ROMAN_DISPERSER_DATA` →
  `$PIXI_PROJECT_ROOT/data` → `./data`. So the **dev pixi env** lands data in
  `tutorials/data` automatically (`pixi run hydrate`); **laptop users** must set
  `ROMAN_DISPERSER_DATA`; **NERSC** already exports it via `.grism_sim_setup`
  (see "NERSC deployment" below).
- **Jupyter kernels do not inherit your shell env.** `ROMAN_DISPERSER_DATA` must
  be set *for the kernel* — via the `kernel.json` `"env"` block (laptop) or the
  NERSC `kernel-helper.sh`. Documented in `docs/SETUP.md`. This is the most
  common "works in the terminal, not in the notebook" trap, and notebook 00
  checks it from inside the kernel.
- Hydration mechanics (manifest/lock, `--only`/`--sca`) live in the disperser's
  `INSTALL.md`; tutorials link to it rather than duplicating.

## NERSC deployment (maintainer)

The shared NERSC envs are built from the generated ymls. Key facts + gotchas,
distilled from the build (`m4943` is the project):

- **Envs:** `/global/common/software/m4943/envs/roman-tutorials-{cpu,gpu}`, built
  with `mamba env create -f environment-{cpu,gpu}.yml -p <prefix>`. Prereqs:
  `gh auth login` (private disperser clones over HTTPS mid-build),
  `CONDA_PKGS_DIRS=$SCRATCH/conda-pkgs` (keep the pkg cache off `$HOME`),
  `chmod -R g+rX` for group read. Build on a login node — no GPU needed to *build*.
- **`.grism_sim_setup`** exports `tutorial_2026_cpu` / `tutorial_2026_gpu` (env
  paths; users `conda activate $tutorial_2026_gpu`) and `ROMAN_DISPERSER_DATA`.
- **Data:** hydrated to `/global/cfs/cdirs/m4943/grismsim/roman-disperser-data`
  (writable); users/kernels read the **`/dvs_ro`** mirror (read-only, like
  `STPSF_PATH`) — runtime only reads. Re-hydrate by overriding
  `ROMAN_DISPERSER_DATA` back to the writable path.
- **Kernels:** auto-named from `$CONDA_PREFIX`, wrapped in `kernel-helper.sh`
  (which propagates `ROMAN_DISPERSER_DATA` to notebooks). See `docs/SETUP.md` §1.

Gotchas:
- **The `jaxlib cpu*` pin is load-bearing.** conda-forge resolves the CUDA jaxlib
  when an env is *built* on a GPU host (the `__cuda` virtual package), so without
  the pin the "CPU" env grabs the shared login GPU and OOMs. Pinned in
  `pixi.toml`'s cpu feature → flows to `environment-cpu.yml`.
- **GPU is only verifiable on a GPU *compute* node** (`salloc -C gpu`); the login
  GPU is shared and OOMs regardless of env.
- The pkg cache on `$SCRATCH` is a different filesystem from the CFS env prefix,
  so conda *copies* (no hardlinks) → envs are self-contained and the scratch
  auto-purge can't break them.

## Validation gate

Local `pixi run check-jax` and a headless `nbconvert --execute` catch gross
breakage, but **they do not match the curated NERSC conda env.** The real gate is
executing every notebook *on NERSC in the actual conda env* before publishing. A
laptop run (pixi or venv) is necessary, not sufficient.

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
- `docs/SETUP.md` — canonical, standalone user setup (NERSC + laptop; the NERSC
  kernel setup is now inline here, not a separate page). The environment check
  is `notebooks/00_environment_check.ipynb` (replaced the old `check_env.sh`).
- `README.md`, `CLAUDE.md` — stay at root (README renders on GitHub; CLAUDE.md
  must be at root to be auto-loaded). User prose lives in `docs/`.
