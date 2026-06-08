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

Verify the expected reference-data env vars are set:

```bash
bash ../check_env.sh   # from the docs/ dir; or `bash check_env.sh` from repo root
```

---

## 2. Roman Research Nexus / RRN (CPU)

RRN uses a shared environment too — like NERSC, you activate it rather than
build it. (Activation specifics: **TBD** — fill in once the common RRN install
path is finalized.)

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

Then either conda or a plain virtualenv:

```bash
# Option A — conda (uses this repo's environment.yml)
conda env create -f environment.yml
conda activate roman-disperser-tutorials

# Option B — virtualenv
python -m venv .venv && source .venv/bin/activate
pip install "roman_disperser[full] @ git+https://github.com/roman-grs-pit/roman_disperser.git"
pip install jupyterlab ipykernel
```

Register a kernel:

```bash
python -m ipykernel install --user \
    --name roman-disperser-tutorials \
    --display-name "Roman Disperser Tutorials"
```

For GPU on a workstation with NVIDIA hardware, add the JAX overlay per the
disperser INSTALL link above (Apple-silicon laptops are CPU-only here).

---

## 4. Data assets

The tutorials may need PSF caches and/or the source catalog. These are public
downloads handled by the disperser's own scripts — see
[`roman_disperser/INSTALL.md` → Data files](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#data-files).
On NERSC/RRN the shared caches are already in place.

---

## 5. Verify

In any environment:

```bash
python -c "import roman_disperser; import jax; print('disperser OK; backend:', jax.default_backend())"
```

`backend: gpu` confirms a working GPU overlay; `cpu` is expected everywhere
else. Then launch JupyterLab (or open the notebooks in RRN/NERSC Jupyter) and
select the **Roman Disperser Tutorials** kernel.

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
