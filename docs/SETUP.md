# Setup

These tutorials use the [`roman_disperser`](https://github.com/roman-grs-pit/roman_disperser)
library and are **standalone**: everything below runs on your own machine, and
ends the same way — a Jupyter kernel that can `import roman_disperser` and find
its reference data. When you're set up, **notebook
`00_environment_check.ipynb` verifies everything** — run it first.

> **One JAX rule for everyone:** `roman_disperser` is JAX-based. The notebooks
> run on **CPU or GPU with no code changes** — only the install differs. A CPU
> build of JAX comes in automatically. If you have a GPU and want to use it,
> add the right JAX build *after* setup, following
> [`roman_disperser/INSTALL.md` → GPU support](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#gpu-support).
> Don't pin JAX yourself — let the disperser pull the floor, then overlay GPU.

---

## 1. Set up the environment

The disperser repo is **public**, so a plain `git+https://` install URL works
with no GitHub auth.

Create the environment. The tutorials need only pip-installable packages
(`roman_disperser[full]` pulls jax, numpy, scipy, matplotlib, pandas, pyarrow,
zarr, astropy, synphot), so a **venv is the lightest path** — no conda required.
You need **Python ≥ 3.12** (`roman_disperser`'s floor; check `python --version`
first — an older interpreter fails at `pip install` with an unhelpful resolver
error rather than a clear message).

**Option A — venv + pip (recommended):**

Run this from the cloned tutorials repo; `.venv` (and `./data` below) are
git-ignored, so the venv, the data, and the notebooks all live in one folder —
the same layout the maintainer pixi setup uses.

```bash
python -m venv .venv          # in the repo root; .venv is git-ignored
source .venv/bin/activate
pip install "roman_disperser[full] @ git+https://github.com/roman-grs-pit/roman_disperser.git@v0.14.2" \
    jupyterlab ipykernel
```

**Option B — conda** (use this if you'll also run the romanisim wrap later — it
needs conda for `fftw`; the current tutorials don't):

```bash
conda env create -f environment-cpu.yml      # no conda? `brew install micromamba` is the lightest
conda activate roman-disperser-tutorials
```

Then register a Jupyter kernel (either option):

```bash
python -m ipykernel install --user --name roman-tutorials \
    --display-name "Roman Disperser Tutorials"
```

**Hydrate the reference data.** The tutorials use **SCA 5**, so fetch just what
they need (a few hundred MB) rather than all 18 SCAs. Notebook 09 uses the
prism, whose assets are separate manifest keys (`*_prism`):

```bash
export ROMAN_DISPERSER_DATA=$PWD/data        # co-located in the repo (./data is git-ignored); or any stable path
roman-disperser-hydrate --only optical_model,sensitivities,synphot   # grism essentials (~2 MB)
roman-disperser-hydrate --only optical_model_prism,sensitivities_prism   # prism essentials (notebook 09)
roman-disperser-hydrate --only psf,psf_prism --sca 5                # PSF caches for SCA 5, both elements
roman-disperser-hydrate --only catalog                              # source catalog (~155 MB; notebook 06)
# (or just `roman-disperser-hydrate` for everything — all 18 SCAs, both elements, ~6.4 GB)
```

Hydration writes `data-versions.lock` into the data directory; since disperser
v0.14.2 that lock is what resolves the optical model at runtime, so always
populate a data dir via hydrate (a hand-assembled dir fails loudly).

**Make the data visible to the kernel.** A Jupyter kernel does **not** inherit
your shell, so add `ROMAN_DISPERSER_DATA` to the kernel's `kernel.json`. The
kernel's location is platform-dependent (macOS `~/Library/Jupyter`, Linux
`~/.local/share/jupyter`), so let Jupyter tell us where it is rather than
guessing — make sure `ROMAN_DISPERSER_DATA` is still exported, then run:

```bash
python - <<'PY'
import json, os
from jupyter_client.kernelspec import KernelSpecManager
p = os.path.join(KernelSpecManager().get_kernel_spec("roman-tutorials").resource_dir, "kernel.json")
d = json.load(open(p))
d.setdefault("env", {})["ROMAN_DISPERSER_DATA"] = os.environ["ROMAN_DISPERSER_DATA"]
json.dump(d, open(p, "w"), indent=2)
print("wired ROMAN_DISPERSER_DATA into", p)
PY
```

(`jupyter kernelspec list` shows the path too, if you'd rather edit
`kernel.json` by hand and add `"env": {"ROMAN_DISPERSER_DATA": "/your/path"}`.)
For GPU on an NVIDIA workstation, add the JAX overlay per the INSTALL link above
(Apple-silicon laptops are CPU-only).

Full hydration details (`--only`, manifests, lock files) are in
[`roman_disperser/INSTALL.md` → Reference data](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#reference-data).

---

## 2. Verify — run notebook 00

Launch JupyterLab from the repo:

```bash
jupyter lab          # opens in your browser
```

Select the **Roman Disperser Tutorials** kernel and run
**`notebooks/00_environment_check.ipynb`** top to bottom. It checks the
import, the JAX backend, that the *kernel* resolves the reference data, and a
smoke dispersion — all with green ✅ markers. If anything is red, fix it here
(most often `ROMAN_DISPERSER_DATA` not reaching the kernel, §1) before moving
on.

---

## Developing the tutorials (maintainers)

Authoring uses [pixi](https://pixi.sh) via `pixi.toml` (not for users):

```bash
pixi shell             # laptop / CPU
pixi shell -e gpu      # linux GPU box
pixi run check-jax     # confirm the live backend
```

Notebooks are committed **without outputs** via an `nbstripout` git filter —
run `pixi run setup-nbstripout` once per clone. Before publishing, execute
every notebook in a clean user-style environment (venv or conda from the
generated ymls), not only the pixi dev env — the user path is what has to
work. See [CLAUDE.md](../CLAUDE.md) for the rationale.
