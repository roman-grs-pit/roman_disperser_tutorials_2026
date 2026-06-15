# Setup

These tutorials use the [`roman_disperser`](https://github.com/roman-grs-pit/roman_disperser)
library. Pick the path that matches where you're running. All paths end the
same way: a Jupyter kernel that can `import roman_disperser`.

> **One JAX rule for everyone:** `roman_disperser` is JAX-based. The notebooks
> run on **CPU or GPU with no code changes** — only the install differs. A CPU
> build of JAX comes in automatically. If you have a GPU and want to use it,
> add the right JAX build *after* setup, following
> [`roman_disperser/INSTALL.md` → GPU support](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#gpu-support).
> Don't pin JAX yourself — let the disperser pull the floor, then overlay GPU
> if you need it.

On **NERSC and RRN you don't install anything** — a shared environment is
already built for you. Only **laptop** users install the library themselves
(and so are the only ones who need GitHub access to the private repo).

---

## 1. NERSC (GPU available)

A curated shared environment already exists — don't build your own.

```bash
source /global/common/software/m4943/.grism_sim_setup
module load conda
conda activate $roman
```

For a **notebook kernel** (first-time setup, including the kernel-helper that
sets the reference-data paths), follow
[activating_conda_environment.md](activating_conda_environment.md).

> **Env maintainer TODO:** the shared setup should export `ROMAN_DISPERSER_DATA`
> (pointing at the shared reference data) in both `.grism_sim_setup` and the
> kernel-helper — it's new in disperser 0.10.0 and is how the library locates
> its vendored data on NERSC. See §4.

Verify the expected reference-data env vars are set:

```bash
bash ../check_env.sh   # from the docs/ dir; or `bash check_env.sh` from repo root
```

---

## 2. Roman Research Nexus / RRN (CPU)

RRN uses a shared environment too — like NERSC, you activate it rather than
build it. (Activation specifics, and the shared `ROMAN_DISPERSER_DATA` export +
kernel env: **TBD** — fill in once the common RRN install path is finalized.)

RRN has no GPU today; the CPU JAX floor is all you need. You should not need to
clone the disperser or authenticate to GitHub.

---

## 3. Your own laptop (CPU)

This is the only path that installs the disperser from its **private** GitHub
repo, so authenticate to GitHub first — being in the `roman-grs-pit` org grants
access but you still have to prove who you are:

```bash
gh auth login          # easiest: sets up a git credential helper pip will use
# — or — make sure your SSH key is registered with GitHub and use the
#         git+ssh:// form of the URL below.
```

Then create the environment with conda from **`environment-cpu.yml`** (it
bundles romanisim, so the disperse→wrap pipeline runs in one kernel):

```bash
conda env create -f environment-cpu.yml
conda activate roman-disperser-tutorials
```

> A pure `pip`/venv install of *just* the disperser also works if you don't need
> the romanisim wrap (`pip install "roman_disperser[full] @
> git+https://github.com/roman-grs-pit/roman_disperser.git@v0.10.0"` plus
> `jupyterlab ipykernel`) — but romanisim needs conda for `fftw`, so the yml is
> the supported path.

Register a kernel:

```bash
python -m ipykernel install --user \
    --name roman-disperser-tutorials \
    --display-name "Roman Disperser Tutorials"
```

Then **hydrate the reference data and wire `ROMAN_DISPERSER_DATA` into the
kernel** — see [§4](#4-reference-data). (A laptop has no shared data, so this
step is on you.)

For GPU on a workstation with NVIDIA hardware, add the JAX overlay per the
disperser INSTALL link above (Apple-silicon laptops are CPU-only here).

---

## 4. Reference data

`roman_disperser` fetches all its reference data (optical model, sensitivities,
synphot, PSF caches, source catalog) with the `roman-disperser-hydrate` command
— full details in
[`roman_disperser/INSTALL.md` → Reference data](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#reference-data).

**NERSC / RRN** — already hydrated in the shared environment; nothing to do. The
shared env also sets `ROMAN_DISPERSER_DATA` so the library finds it (see §1/§2).

**Laptop** — point `ROMAN_DISPERSER_DATA` at a stable directory, then hydrate:

```bash
export ROMAN_DISPERSER_DATA=~/roman_disperser_data   # any path; add to ~/.bashrc
roman-disperser-hydrate                              # everything (~4.5 GB)
```

For a lighter laptop footprint, fetch only what a given tutorial needs:

```bash
roman-disperser-hydrate --only optical_model,sensitivities,synphot   # essentials (~2 MB)
roman-disperser-hydrate --only catalog                               # source catalog (~155 MB)
roman-disperser-hydrate --only psf --sca 1 2                         # just some PSF SCAs
```

### Making the data visible to the Jupyter kernel

A kernel does **not** inherit your shell's environment, so set
`ROMAN_DISPERSER_DATA` for the kernel itself:

- **Laptop** — add an `env` block to the kernel's `kernel.json` (registered in §3):
  ```bash
  KJ=~/.local/share/jupyter/kernels/roman-disperser-tutorials/kernel.json
  python - "$KJ" <<'PY'
  import json, os, sys
  p = sys.argv[1]; d = json.load(open(p))
  d.setdefault("env", {})["ROMAN_DISPERSER_DATA"] = os.environ["ROMAN_DISPERSER_DATA"]
  json.dump(d, open(p, "w"), indent=2)
  PY
  ```
  (Equivalently, edit `kernel.json` and add `"env": {"ROMAN_DISPERSER_DATA": "/your/path"}`.)
- **NERSC** — the `kernel-helper.sh` wired into the kernel (see
  [activating_conda_environment.md](activating_conda_environment.md)) already
  exports the reference-data paths; `ROMAN_DISPERSER_DATA` belongs there too.

---

## 5. Verify

In any environment — this checks the import, that the reference data resolves
and loads, and the JAX backend:

```bash
python -c "
import jax
from roman_disperser.pipeline import resolve_paths
from roman_disperser.optical_model import RomanOpticalModel
*_, optical_model, _ = resolve_paths()
RomanOpticalModel(str(optical_model))            # fails if data isn't hydrated/found
print('disperser OK; data at', optical_model.parent, '; backend:', jax.default_backend())
"
```

`backend: gpu` confirms a working GPU overlay; `cpu` is expected everywhere
else. A `FileNotFoundError` here means the data isn't hydrated or
`ROMAN_DISPERSER_DATA` isn't set (see §4). Then launch JupyterLab (or open the
notebooks in RRN/NERSC Jupyter) and select the **Roman Disperser Tutorials**
kernel — and confirm the same check passes *inside a notebook cell* (that
exercises the kernel's env, not your shell's).

---

## Developing the tutorials (maintainers)

Authoring uses [pixi](https://pixi.sh) via `pixi.toml` (not for users):

```bash
pixi shell             # laptop / CPU
pixi shell -e cuda     # linux GPU box
pixi run check-jax     # confirm the live backend
```

The disperser is private, so a local pixi solve needs GitHub auth too
(`gh auth login`, or switch the URL to `git+ssh://`). Local runs catch gross
breakage but **do not** match the curated NERSC/RRN conda envs — before
publishing, execute every notebook on NERSC and RRN in the real conda
environment. See [CLAUDE.md](../CLAUDE.md) for the full rationale.
