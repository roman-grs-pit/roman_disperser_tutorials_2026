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

## The audience matrix (drives every env decision)

Three **user** runtime contexts:

| Context     | Manager                | Platform   | GPU | Installs disperser? |
|-------------|------------------------|------------|-----|---------------------|
| NERSC       | conda (shared `$roman`)| linux-64   | yes | no — shared env     |
| RRN         | conda (shared env)     | linux-64   | no  | no — shared env     |
| Own laptop  | pip-into-venv/conda    | osx/linux  | no  | yes — git clone     |

**Only the laptop path clones the disperser.** NERSC and RRN both provide a
curated shared environment with it pre-installed, so they never authenticate.

Two **developer** contexts (Nikhil): laptop (osx-arm64, CPU) and a separate
linux-64 GPU box (CUDA 12).

## Environment strategy

- **Developers use pixi; users do not.** Pixi fights the conda module system
  and `$HOME` quotas on NERSC/RRN, and is more than a tutorial reader should
  have to learn. Pixi is a dev-only tool here.
- **`pixi.toml`** (dev) is the single source of truth: platforms
  `osx-arm64` + `linux-64`, environments `default` (CPU) and `gpu` (CUDA,
  linux-64). **Both bundle romanisim**, so the disperse→wrap pipeline runs in
  one kernel (no kernel switching). One file covers laptop + GPU box.
- The disperser is pulled **from git**, not the `../roman_disperser` sibling
  path, so this repo is reproducible off a laptop (GPU box, CI).
- **The disperser repo is PRIVATE.** Org membership is authorization, not
  authentication — a clone still needs `gh auth login` (HTTPS credential
  helper) or a registered SSH key. This affects only the **laptop** path and
  the **maintainer pixi solve**; NERSC/RRN use pre-installed shared envs.
  `docs/SETUP.md` documents `gh auth login` as a laptop prerequisite.
- **`environment-cpu.yml` / `environment-gpu.yml`** (user) are the two conda
  specs users build from — **generated** from the pixi envs by
  `scripts/export-conda-envs.sh` (which rewrites the disperser URL ssh→https
  and names the conda env). Don't hand-edit; edit `pixi.toml` and rerun.
  `-cpu` = laptop / RRN / NERSC-CPU; `-gpu` = NERSC-GPU / GPU box. NERSC/RRN
  shared envs are built from these too.

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
  `ROMAN_DISPERSER_DATA`; **NERSC/RRN** shared envs must export it (env-maintainer
  TODO, flagged in `docs/SETUP.md` §1/§2).
- **Jupyter kernels do not inherit your shell env.** `ROMAN_DISPERSER_DATA` must
  be set *for the kernel* — via the `kernel.json` `"env"` block (laptop) or the
  NERSC `kernel-helper.sh`. Documented in `docs/SETUP.md` §4. This is the most
  common "works in the terminal, not in the notebook" trap.
- Hydration mechanics (manifest/lock, `--only`/`--sca`) live in the disperser's
  `INSTALL.md`; tutorials link to it rather than duplicating.

## Validation gate

Local `pixi run check-jax` and a headless `nbconvert --execute` catch gross
breakage, but **they do not match the curated NERSC/RRN conda envs.** The real
gate is executing every notebook *on NERSC and RRN in the actual conda env*
before publishing. A laptop pixi run is necessary, not sufficient.

## Files

- `pixi.toml` — dev environment + single source of truth (CPU + gpu, both with
  romanisim). Root (not `docs/`).
- `environment-cpu.yml` / `environment-gpu.yml` — user conda envs, GENERATED
  from `pixi.toml` by `scripts/export-conda-envs.sh`. Root.
- `docs/SETUP.md` — canonical, standalone user setup (all three contexts).
- `docs/activating_conda_environment.md` — NERSC kernel/setup specifics;
  `docs/SETUP.md` links here rather than duplicating.
- `check_env.sh` — verifies expected env vars (CRDS/STPSF/etc.) are set. Root.
- `README.md`, `CLAUDE.md` — stay at root (README renders on GitHub; CLAUDE.md
  must be at root to be auto-loaded). User prose lives in `docs/`.
